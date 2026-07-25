#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_all.py — ADME 전면 벤치마크 ★밤샘 무인 오케스트레이터. env: admet (GPU 작업은 교차 env subprocess)
설계 원칙
- (엔드포인트, 축, 변형, seed) 조합 단위로 ★독립 실행. try/except로 실패 격리 → 다음 조합 진행. 전체 중단 없음.
- 매 조합 완료 즉시 results/progress.jsonl append + predictions/ 저장. 재실행 시 ★완료 조합 건너뜀.
- 실패는 logs/failures.jsonl 에 (대상·에러·트레이스) 기록하고 계속.
- 순서 = 가벼운 것부터: 0 준비 → 1 축①G2 → 2 축②특징 → 3 축③멀티태스크 → 4 축①G3 →
  5 ADMET-AI → 6 축②E상류 → 7 축①G4 → 8 축①G5 → 9 리포트
- 기둥(A·D·M·E) 완료마다 results/partial_report.html 갱신.
사용: python run_all.py [--phases 0,1,2] [--only ep]
"""
import argparse, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import common as C
from common import EPS, SEEDS, log, record, fail, done_keys, score, prep, build_X, STACKS

ADMET_AI_PY = "/home/nudge/miniforge3/envs/ADMET_AI/bin/python"
CHEMPROP = "/home/nudge/miniforge3/envs/ADMET_AI/bin/chemprop"
BENCH_PY = "/home/nudge/miniforge3/envs/adme-bench/bin/python"
DONE = set()


def skip(key):
    return key in DONE


def finish(key, **kw):
    DONE.add(key); record(key, **kw)


# ═══════ 0단계 — 준비: chemprop 비호환 분자 + 분할 동일성 실측 ═══════
def phase0():
    key = "setup|splits"
    if skip(key):
        log("0단계 skip(완료)"); return
    import tempfile
    from tdc.benchmark_group import admet_group
    g = admet_group(path=C.TDC_DATA)
    code = ("import sys;from rdkit import Chem,RDLogger;RDLogger.DisableLog('rdApp.*');"
            "[print(l.rstrip('\\n')) for l in open(sys.argv[1]) if l.strip() "
            "and Chem.MolFromSmiles(l.rstrip('\\n')) is None]")
    bad, meta = {}, {}
    for ep in EPS:
        try:
            b = g.get(ep)
            smis = sorted({str(s) for s in list(b["train_val"]["Drug"]) + list(b["test"]["Drug"])})
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
                f.write("\n".join(smis)); p = f.name
            r = subprocess.run([ADMET_AI_PY, "-c", code, p], capture_output=True, text=True, timeout=1800)
            os.unlink(p)
            bad[ep] = sorted(set(r.stdout.split("\n")) - {""})
            meta[ep] = dict(n_total=len(smis), n_bad=len(bad[ep]),
                            n_train_val=len(b["train_val"]), n_test=len(b["test"]))
            log(f"  {ep:<32} 전체 {len(smis):>6} · chemprop 비호환 {len(bad[ep])}")
        except Exception as e:
            bad[ep] = []; fail(f"setup|{ep}", e)
    json.dump(bad, open(f"{C.DATA}/chemprop_incompatible.json", "w"), ensure_ascii=False, indent=1)
    C.BAD = bad
    # 분할 동일성·누수 실측
    lk = {}
    for ep in EPS:
        try:
            b = C.group().get(ep)
            can = lambda s: (lambda m: __import__("rdkit").Chem.MolToSmiles(m) if m else None)(
                __import__("rdkit").Chem.MolFromSmiles(str(s)))
            from rdkit import Chem
            tv = {Chem.MolToSmiles(m) for m in (Chem.MolFromSmiles(str(s)) for s in b["train_val"]["Drug"]) if m}
            te = [Chem.MolFromSmiles(str(s)) for s in b["test"]["Drug"]]
            tec = [Chem.MolToSmiles(m) for m in te if m]
            lk[ep] = dict(n_train_val=len(b["train_val"]), n_test=len(b["test"]),
                          n_test_canonical=len(set(tec)),
                          exact_overlap_trainval_test=sum(1 for c in tec if c in tv),
                          n_bad_excluded=len(bad.get(ep, [])))
        except Exception as e:
            fail(f"setup|leak|{ep}", e)
    json.dump(lk, open(f"{C.RES}/split_leakage.json", "w"), ensure_ascii=False, indent=1)
    nz = [e for e, v in lk.items() if v["exact_overlap_trainval_test"] > 0]
    log(f"0단계 완료 — 분할 {len(lk)}개 · train∩test 정확분자 중복 >0 인 엔드포인트: {nz or '없음'}")
    finish(key, n_eps=len(lk), leakage=lk)


# ═══════ 공용: 트리 모델 학습 1건 ═══════
def fit_tree(model, task, Xtr, ytr, Xte, seed):
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    if model.startswith("rf"):
        M = RandomForestClassifier if task == "cls" else RandomForestRegressor
        m = M(n_estimators=500, n_jobs=C.NJOBS, random_state=seed)
        m.fit(Xtr, ytr.astype(int) if task == "cls" else ytr)
    else:
        p = dict(CFGP(task), random_state=seed, n_jobs=C.NJOBS)
        m = (xgb.XGBClassifier(**p, eval_metric="logloss") if task == "cls" else xgb.XGBRegressor(**p))
        m.fit(Xtr, ytr.astype(int) if task == "cls" else ytr)
    pred = m.predict_proba(Xte)[:, 1] if task == "cls" else m.predict(Xte)
    return pred


def CFGP(task):
    return dict(C.CFG["model"]["cls_params" if task == "cls" else "reg_params"])


def tree_job(ep, model, feats, tag, seed, save_pred_name=None):
    """(엔드포인트·모델·특징스택·seed) 1건 — 반환 metrics dict."""
    info = EPS[ep]; task = info["task"]
    g = C.group(); b = g.get(ep); test = b["test"]
    tr, va = g.split(ep, seed)
    Xtr_r, _ = build_X(ep, feats, tr["Drug"].tolist(), f"tr{seed}")
    Xte_r, _ = build_X(ep, feats, test["Drug"].tolist(), "te")
    Xva_r, _ = build_X(ep, feats, va["Drug"].tolist(), f"va{seed}")
    Xtr, Xte, Xva = prep(Xtr_r.copy(), Xte_r.copy(), Xva_r.copy())
    ytr = tr["Y"].to_numpy(float); yte = test["Y"].to_numpy(float)
    pte = fit_tree(model, task, Xtr, ytr, Xte, seed)
    m = score(yte, pte, task)
    if save_pred_name:
        C.save_pred(save_pred_name, [dict(smiles=str(s), y_true=float(a), y_prob=float(p), seed=seed)
                                     for s, a, p in zip(test["Drug"], yte, pte)])
    return m, pte, yte, test["Drug"].tolist()


# ═══════ 1단계 — 축① G2 세대(트리 3종 × 18개) ═══════
G2_MODELS = [("xgb_physchem", ["phys"]), ("rf_physchem", ["phys"]), ("xgb_ecfp", ["ecfp"])]


def phase1(only=None):
    for ep in EPS:
        if only and ep != only:
            continue
        for model, feats in G2_MODELS:
            preds = {}
            for seed in SEEDS:
                key = f"axis1|G2|{ep}|{model}|s{seed}"
                if skip(key):
                    continue
                try:
                    m, pte, yte, smis = tree_job(ep, model, feats, model, seed)
                    preds[seed] = (smis, yte, pte)
                    finish(key, axis="1", gen="G2", endpoint=ep, model=model, seed=seed,
                           pillar=EPS[ep]["pillar"], task=EPS[ep]["task"], metrics=m)
                except Exception as e:
                    fail(key, e, endpoint=ep, model=model, seed=seed)
            if preds:
                recs = [dict(smiles=s, y_true=float(y), y_prob=float(p), seed=sd)
                        for sd, (S, Y, P) in preds.items() for s, y, p in zip(S, Y, P)]
                C.save_pred(f"{ep}__G2_{model}__test", recs)
                log(f"  [축① G2] {ep:<32}{model:<14} 완료")


# ═══════ 2단계 — 축② 특징 ablation (G2 XGBoost 고정 · 누적 스택) ═══════
def phase2(only=None):
    for ep in EPS:
        if only and ep != only:
            continue
        for tag, feats in STACKS:
            for seed in SEEDS:
                key = f"axis2|{ep}|{tag}|s{seed}"
                if skip(key):
                    continue
                try:
                    m, *_ = tree_job(ep, "xgb", feats, tag, seed)
                    finish(key, axis="2", endpoint=ep, stack=tag, feats=feats, seed=seed,
                           pillar=EPS[ep]["pillar"], task=EPS[ep]["task"], metrics=m)
                except Exception as e:
                    fail(key, e, endpoint=ep, stack=tag, seed=seed)
        log(f"  [축② 특징] {ep:<32} 스택 {len(STACKS)}단계 완료")


# ═══════ 4단계 — 축① G3 정직 D-MPNN ═══════
def chemprop_one(ep, seed, targets=None, extra_train=None):
    """chemprop 1건. targets=None이면 단일과제. 반환 (pred_test df, test df, cols)."""
    import shutil
    info = EPS[ep]; task = info["task"]
    wd = f"{C.ROOT}/work/{ep}_s{seed}"
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd, exist_ok=True)
    g = C.group(); b = g.get(ep); test = b["test"]
    tr, va = g.split(ep, seed)
    for df, nm in ((tr, "train"), (va, "val"), (test, "test")):
        df[["Drug", "Y"]].rename(columns={"Drug": "smiles", "Y": "y"}).to_csv(f"{wd}/{nm}.csv", index=False)
    cmd = [CHEMPROP, "train", "-i", f"{wd}/train.csv", f"{wd}/val.csv", f"{wd}/test.csv",
           "-s", "smiles", "--target-columns", "y",
           "-t", "classification" if task == "cls" else "regression",
           "--epochs", "50", "--accelerator", "gpu", "--devices", "1", "-n", "0", "-b", "50",
           "--metrics", "roc" if task == "cls" else "mae", "--pytorch-seed", str(seed),
           "-o", f"{wd}/model", "-q"] + (["--class-balance"] if task == "cls" else [])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        raise RuntimeError(f"chemprop train 실패: {r.stderr[-500:]}")
    pts = sorted(os.path.join(rt, fn) for rt, _, fs in os.walk(f"{wd}/model") for fn in fs
                 if fn.endswith(".pt") and "best" in fn.lower()) or \
          sorted(os.path.join(rt, fn) for rt, _, fs in os.walk(f"{wd}/model") for fn in fs if fn.endswith(".pt"))
    if not pts:
        raise RuntimeError("체크포인트 없음")
    out = f"{wd}/pred_test.csv"
    r2 = subprocess.run([CHEMPROP, "predict", "-i", f"{wd}/test.csv", "-s", "smiles",
                        "--model-paths", pts[0], "-o", out, "-n", "0", "-q"],
                       capture_output=True, text=True, timeout=3600)
    if r2.returncode != 0:
        raise RuntimeError(f"chemprop predict 실패: {r2.stderr[-400:]}")
    p = pd.read_csv(out)
    shutil.rmtree(f"{wd}/model", ignore_errors=True)
    col = [c for c in p.columns if c != "smiles"][0]
    return p[col].to_numpy(float), test["Y"].to_numpy(float), test["Drug"].tolist()


def phase4(only=None):
    for ep in EPS:
        if only and ep != only:
            continue
        recs = []
        for seed in SEEDS:
            key = f"axis1|G3|{ep}|dmpnn|s{seed}"
            if skip(key):
                continue
            try:
                pte, yte, smis = chemprop_one(ep, seed)
                m = score(yte, pte, EPS[ep]["task"])
                recs += [dict(smiles=s, y_true=float(y), y_prob=float(p), seed=seed)
                         for s, y, p in zip(smis, yte, pte)]
                finish(key, axis="1", gen="G3", endpoint=ep, model="dmpnn_ours", seed=seed,
                       pillar=EPS[ep]["pillar"], task=EPS[ep]["task"], metrics=m)
                log(f"  [축① G3] {ep:<32} s{seed} {m}")
            except Exception as e:
                fail(key, e, endpoint=ep, seed=seed)
        if recs:
            C.save_pred(f"{ep}__G3_dmpnn__test", recs)


# ═══════ 5단계 — ADMET-AI 추론(누수 기준선) ═══════
def phase5():
    key = "axis1|G3ai|all"
    if skip(key):
        log("5단계 skip"); return
    # test 분자 저장 → ADMET_AI env에서 추론
    os.makedirs(f"{C.DATA}/test", exist_ok=True)
    for ep in EPS:
        try:
            b = C.group().get(ep)
            b["test"][["Drug", "Y"]].to_csv(f"{C.DATA}/test/{ep}.csv", index=False)
        except Exception as e:
            fail(f"axis1|G3ai|dump|{ep}", e)
    script = f"""
import json, os, sys
import pandas as pd, numpy as np
from admet_ai import ADMETModel
COL = {json.dumps(C.ADMETAI_COL, ensure_ascii=False)}
D = "{C.DATA}/test"; OUT = "{C.PRED}"
model = ADMETModel(); res = {{}}
for ep, col in COL.items():
    p = os.path.join(D, ep + ".csv")
    if not os.path.exists(p): continue
    try:
        d = pd.read_csv(p); pred = model.predict(smiles=[str(s) for s in d.Drug]).reset_index(drop=True)
        if col not in pred.columns:
            res[ep] = dict(status="미커버"); continue
        with open(os.path.join(OUT, ep + "__G3_admetai__test.jsonl"), "w") as f:
            for s, y, v in zip(d.Drug, d.Y, pred[col].values):
                f.write(json.dumps(dict(smiles=str(s), y_true=float(y), y_prob=float(v), seed=0))+"\\n")
        res[ep] = dict(status="ok", n=len(d))
    except Exception as e:
        res[ep] = dict(status="failed", error=str(e)[:300])
json.dump(res, open("{C.RES}/admetai_status.json","w"), ensure_ascii=False, indent=1)
print(json.dumps({{k:v.get("status") for k,v in res.items()}}, ensure_ascii=False))
"""
    sp = f"{C.ROOT}/src/_admetai_infer.py"
    open(sp, "w").write(script)
    try:
        r = subprocess.run([ADMET_AI_PY, sp], capture_output=True, text=True, timeout=7200)
        log(f"  [ADMET-AI] {r.stdout.strip()[-300:] or r.stderr[-200:]}")
        st = json.load(open(f"{C.RES}/admetai_status.json"))
        for ep, v in st.items():
            if v.get("status") == "ok":
                try:
                    d = pd.read_json(f"{C.PRED}/{ep}__G3_admetai__test.jsonl", lines=True)
                    m = score(d.y_true.to_numpy(float), d.y_prob.to_numpy(float), EPS[ep]["task"])
                    record(f"axis1|G3ai|{ep}", axis="1", gen="G3", endpoint=ep, model="admetai",
                           seed=0, pillar=EPS[ep]["pillar"], task=EPS[ep]["task"], metrics=m,
                           leak_flag="★누수 의심(TDC 전체 사전학습)")
                except Exception as e:
                    fail(f"axis1|G3ai|score|{ep}", e)
        finish(key, n_ok=sum(1 for v in st.values() if v.get("status") == "ok"))
    except Exception as e:
        fail(key, e)


# ═══════ 7·8단계 — G4 Uni-Mol / G5 파운데이션 (교차 env) ═══════
def phase78(which):
    """G4(Uni-Mol)·G5(파운데이션) — adme-bench env에서 스크립트 전체 실행(자체 resume) 후
    원자료(jsonl)를 progress.jsonl 로 흡수한다. 스크립트 자체가 조합별 try/except·이어하기를 가진다."""
    runner = f"{C.ROOT}/src/{'g4_unimol.py' if which == 'g4' else 'g5_finetune.py'}"
    raw = f"{C.RES}/{'finetune2_raw.jsonl' if which == 'g4' else 'finetune_raw.jsonl'}"
    if not os.path.exists(runner):
        log(f"  [{which}] 러너 없음 — skip"); return
    try:
        log(f"  [{which.upper()}] 실행 시작 (장시간)")
        r = subprocess.run([BENCH_PY, runner], capture_output=True, text=True, timeout=64800)
        if r.returncode != 0:
            fail(f"axis1|{which.upper()}|run", RuntimeError(r.stderr[-800:]))
    except Exception as e:
        fail(f"axis1|{which.upper()}|run", e)
    # 원자료 흡수 (부분 완료라도)
    if not os.path.exists(raw):
        log(f"  [{which.upper()}] 원자료 없음"); return
    n = 0
    for line in open(raw):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("status") != "ok":
            continue
        ep, seed = d.get("endpoint"), d.get("seed")
        mdl = d.get("model", which)
        key = f"axis1|{which.upper()}|{ep}|{mdl}|s{seed}"
        if skip(key) or ep not in EPS:
            continue
        te = d.get("test") or {}
        m = {k: (round(float(v), 4) if isinstance(v, (int, float)) else v) for k, v in te.items()}
        gen = "G4" if which == "g4" else "G5"
        finish(key, axis="1", gen=gen, endpoint=ep, model=mdl, seed=seed,
               pillar=EPS[ep]["pillar"], task=EPS[ep]["task"], metrics=m)
        n += 1
    log(f"  [{which.upper()}] progress 흡수 {n}건")


# ═══════ 3단계 — 축③ 멀티태스크·전이 ═══════
def phase3():
    import shutil
    CYP = ["cyp2c9_veith", "cyp2d6_veith", "cyp3a4_veith"]
    SUB = ["cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels",
           "cyp3a4_substrate_carbonmangels"]

    def multitask(name, eps, seed):
        """여러 엔드포인트를 합집합 분자셋 + 다중 타깃으로 chemprop 멀티태스크 학습."""
        wd = f"{C.ROOT}/work/mt_{name}_s{seed}"
        shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd, exist_ok=True)
        g = C.group()
        parts = {}
        for e in eps:
            b = g.get(e); tr, va = g.split(e, seed)
            parts[e] = dict(train=tr, valid=va, test=b["test"])
        # ★★누수 차단(필수) — 이 묶음의 엔드포인트들은 ★같은 분자 라이브러리에 라벨만 다른 경우가 많다
        #   (CYP 3종은 test 분자의 88.9%가 다른 과제의 train에 들어 있었다).
        #   합집합 train/valid에서 ★어느 과제의 test에라도 등장하는 분자를 전부 제외해야
        #   과제별 test 채점이 정직해진다. 이 게이트가 없으면 멀티태스크가 부당하게 이긴다.
        from rdkit import Chem as _Chem

        def _c(x):
            m = _Chem.MolFromSmiles(str(x))
            return _Chem.MolToSmiles(m) if m else str(x)

        test_pool = set()
        for e in eps:
            test_pool |= {_c(x) for x in parts[e]["test"]["Drug"]}
        n_drop = 0
        for e in eps:
            for k in ("train", "valid"):
                d = parts[e][k]
                keep = ~d["Drug"].map(lambda x: _c(x) in test_pool)
                n_drop += int((~keep).sum())
                parts[e][k] = d[keep].reset_index(drop=True)
        log(f"    [{name}] 누수 차단: 합집합 train/valid에서 test 분자 {n_drop}행 제외")
        frames = {}
        for split in ("train", "valid", "test"):
            m = {}
            for e in eps:
                d = parts[e][split]
                for s, y in zip(d["Drug"], d["Y"]):
                    m.setdefault(str(s), {})[e] = float(y)
            rows = [dict(smiles=s, **{e: v.get(e, np.nan) for e in eps}) for s, v in m.items()]
            frames[split] = pd.DataFrame(rows)
            frames[split].to_csv(f"{wd}/{split}.csv", index=False)
        cmd = [CHEMPROP, "train", "-i", f"{wd}/train.csv", f"{wd}/valid.csv", f"{wd}/test.csv",
               "-s", "smiles", "--target-columns", *eps, "-t", "classification",
               "--epochs", "50", "--accelerator", "gpu", "--devices", "1", "-n", "0", "-b", "50",
               "--metrics", "roc", "--pytorch-seed", str(seed), "-o", f"{wd}/model", "-q"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
        if r.returncode != 0:
            r = subprocess.run(cmd + [], capture_output=True, text=True, timeout=14400)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-600:])
        pts = sorted(os.path.join(rt, fn) for rt, _, fs in os.walk(f"{wd}/model") for fn in fs
                     if fn.endswith(".pt"))
        if not pts:
            raise RuntimeError("체크포인트 없음")
        out = f"{wd}/pred_test.csv"
        subprocess.run([CHEMPROP, "predict", "-i", f"{wd}/test.csv", "-s", "smiles",
                        "--model-paths", pts[0], "-o", out, "-n", "0", "-q"],
                       capture_output=True, text=True, timeout=7200)
        p = pd.read_csv(out); t = frames["test"]
        res = {}
        pcols = [c for c in p.columns if c != "smiles"]
        for i, e in enumerate(eps):
            y = pd.to_numeric(t[e], errors="coerce").to_numpy(float)
            v = pd.to_numeric(p[pcols[i]], errors="coerce").to_numpy(float)
            k = ~np.isnan(y) & ~np.isnan(v)
            if k.sum() >= 20 and len(np.unique(y[k])) == 2:
                res[e] = score(y[k], v[k], "cls")
        shutil.rmtree(f"{wd}/model", ignore_errors=True)
        return res

    for name, eps in [("cyp_inhibition", CYP), ("cyp_substrate", SUB), ("all_adme_cls",
                      [e for e in EPS if EPS[e]["task"] == "cls"])]:
        for seed in SEEDS:
            key = f"axis3v2|multitask|{name}|s{seed}"
            if skip(key):
                continue
            try:
                res = multitask(name, eps, seed)
                finish(key, axis="3", kind="multitask", group=name, seed=seed, per_endpoint=res)
                log(f"  [축③ 멀티태스크] {name} s{seed} — {len(res)}개 과제 채점")
            except Exception as e:
                fail(key, e, group=name, seed=seed)


# ═══════ 9단계 — 리포트 ═══════
def phase9():
    try:
        subprocess.run([sys.executable, f"{C.ROOT}/src/build_report.py"], timeout=1800)
    except Exception as e:
        fail("report", e)


PHASES = {0: ("준비(분할·비호환)", phase0), 1: ("축① G2 세대", phase1), 2: ("축② 특징 ablation", phase2),
          3: ("축③ 멀티태스크", phase3), 4: ("축① G3 정직 D-MPNN", phase4), 5: ("ADMET-AI 추론", phase5),
          7: ("축① G4 Uni-Mol", lambda only=None: phase78("g4")),
          8: ("축① G5 파운데이션", lambda only=None: phase78("g5")), 9: ("리포트", lambda only=None: phase9())}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="0,1,2,3,4,5,7,8,9")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    DONE |= done_keys()
    log(f"시작 — 완료된 조합 {len(DONE)}개 이어서 진행")
    for ph in [int(x) for x in a.phases.split(",") if x.strip()]:
        nm, fn = PHASES[ph]
        log(f"══════ {ph}단계: {nm} ══════")
        t0 = time.time()
        try:
            fn(a.only or None) if ph in (1, 2, 4) else fn()
        except Exception as e:
            fail(f"phase{ph}", e)
        log(f"══════ {ph}단계 종료 ({time.time()-t0:.0f}s) ══════")
        try:
            subprocess.run([sys.executable, f"{C.ROOT}/src/build_report.py", "--partial"], timeout=900)
        except Exception:
            pass
    log("전체 종료")
