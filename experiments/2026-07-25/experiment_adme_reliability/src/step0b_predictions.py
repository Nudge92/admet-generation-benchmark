#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step0b_predictions.py — [0단계] 챔피언 분자별 예측 저장(train/valid/test). env: admet · CPU.
★재학습으로 성능 개선 금지 — 원 실험과 동일 config·seed로 ★재현·추론만. HP 변경 없음.
★valid가 필요한 이유: 분류는 t*(MCC 최대), 회귀는 컨포멀 예측구간 분위수를 valid에서 정해야 하므로
  (★test에서 고르면 안 됨).
★재현 검증: 5 seed 평균이 adme_matrix.csv 보고값과 ±0.005 이내인지. 벗어나면 그 엔드포인트 기록 후 계속.
★test 분자 동일성: 개수 + ★Jaccard 실측.
산출: predictions/{ep}__{model}__{split}.jsonl · results/step0b_repro.json
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

ADME = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_adme_full"
NEW = "/home/nudge/Project/ADMET_integrated/2026-07-25/experiment_adme_reliability"
sys.path.insert(0, f"{ADME}/src")
import common as C
from common import EPS, SEEDS, build_X, prep, score
import run_all as RA                       # ★원 실험과 동일 학습 경로(fit_tree)

CH = {r["endpoint"]: r for r in json.load(open(f"{NEW}/results/champions.json"))}
MAT = pd.read_csv(f"{ADME}/results/adme_matrix.csv")
TOL = 0.005
FEATS = {"xgb_physchem": ["phys"], "rf_physchem": ["phys"], "xgb_ecfp": ["ecfp"]}
canon = lambda s: (lambda m: Chem.MolToSmiles(m) if m else None)(Chem.MolFromSmiles(str(s)))
DONE_F = f"{NEW}/results/step0b_repro.json"
prev = json.load(open(DONE_F)) if os.path.exists(DONE_F) else {"checks": []}
done_eps = {c["endpoint"] for c in prev["checks"] if c.get("ok")}

g = C.group()
checks = list(prev["checks"])
print("=" * 100)
print("[0단계] 챔피언 분자별 예측 저장(train/valid/test) + 재현 검증  ★추론만·HP 변경 없음")
print("=" * 100)
for ep, ch in CH.items():
    model = ch["champion"]
    if model == "dmpnn_ours":
        print(f"  {ep:<32} champion=dmpnn_ours → ★별도 처리(GNN·step0c) — skip")
        continue
    if ep in done_eps:
        print(f"  {ep:<32} skip(완료)"); continue
    prim, task = EPS[ep]["primary"], EPS[ep]["task"]
    feats = FEATS[model]
    b = g.get(ep); test = b["test"]
    Xte_r, _ = build_X(ep, feats, test["Drug"].tolist(), "te")
    yte = test["Y"].to_numpy(float)
    per_seed, recs = [], {"train": [], "valid": [], "test": []}
    for seed in SEEDS:
        tr, va = g.split(ep, seed)
        Xtr_r, _ = build_X(ep, feats, tr["Drug"].tolist(), f"tr{seed}")
        Xva_r, _ = build_X(ep, feats, va["Drug"].tolist(), f"va{seed}")
        Xtr, Xte, Xva = prep(Xtr_r.copy(), Xte_r.copy(), Xva_r.copy())
        ytr = tr["Y"].to_numpy(float); yva = va["Y"].to_numpy(float)
        # 원 실험과 동일 학습 → 세 split 모두 추론
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        if model.startswith("rf"):
            M = RandomForestClassifier if task == "cls" else RandomForestRegressor
            m = M(n_estimators=500, n_jobs=C.NJOBS, random_state=seed)
        else:
            p = dict(RA.CFGP(task), random_state=seed, n_jobs=C.NJOBS)
            m = (xgb.XGBClassifier(**p, eval_metric="logloss") if task == "cls"
                 else xgb.XGBRegressor(**p))
        m.fit(Xtr, ytr.astype(int) if task == "cls" else ytr)
        pr = (lambda X: m.predict_proba(X)[:, 1] if task == "cls" else m.predict(X))
        pte, pva, ptr = pr(Xte), pr(Xva), pr(Xtr)
        per_seed.append(score(yte, pte, task)[prim])
        key = "y_prob" if task == "cls" else "y_pred"
        for split, (df, yy, pp) in [("train", (tr, ytr, ptr)), ("valid", (va, yva, pva)),
                                    ("test", (test, yte, pte))]:
            recs[split] += [dict(smiles=str(s), y_true=float(a), **{key: float(v)}, seed=seed)
                            for s, a, v in zip(df["Drug"], yy, pp)]
    for split in recs:
        fn = f"{NEW}/predictions/{ep}__{model}__{split}.jsonl"
        with open(fn, "w", encoding="utf-8") as f:
            for r in recs[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    got = float(np.mean(per_seed))
    rep = float(MAT[(MAT.endpoint == ep) & (MAT.model == model)].iloc[0].value)
    ok = abs(got - rep) <= TOL
    # test 분자 동일성(Jaccard) — 기존 ADME predictions와 대조
    old = f"{ADME}/predictions/{ep}__G2_{model}__test.jsonl"
    jac = None
    if os.path.exists(old):
        a = {canon(json.loads(l)["smiles"]) for l in open(old)} - {None}
        bset = {canon(s) for s in test["Drug"]} - {None}
        jac = round(len(a & bset) / len(a | bset), 6)
    checks.append(dict(endpoint=ep, model=model, metric=prim, reported=rep, reproduced=round(got, 4),
                       diff=round(got - rep, 4), ok=bool(ok), n_seed=len(per_seed),
                       n_test=len(test), jaccard_vs_original=jac))
    print(f"  {ep:<32}{model:<14}{prim} 보고 {rep:.4f} → 재현 {got:.4f} (Δ{got-rep:+.4f}) "
          f"{'OK' if ok else '★불일치'} · Jaccard {jac}")
    json.dump(dict(tolerance=TOL, checks=checks), open(DONE_F, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

bad = [c for c in checks if not c["ok"]]
print(f"\n재현 검증: {len(checks)}건 중 벗어남 {len(bad)}건" + (f" → {[c['endpoint'] for c in bad]}" if bad else " → 전부 통과"))
print(f"저장 → predictions/*.jsonl ({len(glob.glob(f'{NEW}/predictions/*.jsonl'))}개) · results/step0b_repro.json")
