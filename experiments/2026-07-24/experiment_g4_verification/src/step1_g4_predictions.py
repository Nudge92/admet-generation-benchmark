#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step1_g4_predictions.py — [1단계] G4(Uni-Mol) 재현 학습 + ★분자별 예측 저장. env: adme-bench · GPU.
★원 실험(experiment_adme_full/src/g4_unimol.py)과 ★동일 config: EPOCHS 20·BATCH 8·LR 1e-4·
  model_size 84m·use_amp·kfold 1·split scaffold·early_stopping(회귀 3). ★튜닝 금지.
★3D conformer 실패 분자를 기록(logs/conformer_failed.json) — 개수·SMILES 전부. 조용히 빼지 않는다.
★GPU OOM → batch 절반 1회 재시도, 그래도 실패면 그 seed만 skip·기록(전체 중단 금지).
★재현 검증: 5 seed 평균이 adme_matrix.csv 보고값과 ±0.005 안인지 확인.
산출: predictions/{ep}__unimol__seed{n}.jsonl · logs/conformer_failed.json · results/step1_repro.json
"""
import json, os, shutil, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

ADME = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_adme_full"
NEW = "/home/nudge/Project/ADMET_integrated/2026-07-24/experiment_g4_verification"
sys.path.insert(0, f"{ADME}/src")
import common as C
from common import EPS

SEEDS = [1, 2, 3, 4, 5]
EPOCHS, BATCH, LR = 20, 8, 1e-4          # ★원 실험과 동일(바꾸지 말 것)
TOL = 0.005
WORK = f"{NEW}/work"
os.makedirs(WORK, exist_ok=True)
TARGETS = ["half_life_obach", "solubility_aqsoldb"]
MAT = pd.read_csv(f"{ADME}/results/adme_matrix.csv")
RAW = f"{NEW}/results/g4_raw.jsonl"


def reported(ep):
    r = MAT[(MAT.endpoint == ep) & (MAT.model == "unimol")]
    return float(r.iloc[0].value)


def metric(y, p, prim):
    return (float(spearmanr(y, p).correlation) if prim == "Spearman"
            else float(mean_absolute_error(y, p)))


def done_ok():
    s = set()
    if os.path.exists(RAW):
        for l in open(RAW, encoding="utf-8"):
            try:
                d = json.loads(l)
                if d.get("status") == "ok":
                    s.add((d["endpoint"], d["seed"]))
            except Exception:
                pass
    return s


def rec(d):
    with open(RAW, "a", encoding="utf-8") as f:          # ★encoding 필수(한글 사유)
        f.write(json.dumps(d, ensure_ascii=False) + "\n")


def train_one(ep, seed, bs):
    """MolTrain 1회 → (test예측, test df). 원 config 고정."""
    from unimol_tools import MolTrain, MolPredict
    g = C.group(); b = g.get(ep); test = b["test"].reset_index(drop=True)
    tr, va = g.split(ep, seed)
    wd = f"{WORK}/{ep}_s{seed}"
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd, exist_ok=True)
    trcsv, tecsv = f"{wd}/train.csv", f"{wd}/test.csv"
    exp = f"{wd}/exp"
    tr.rename(columns={"Drug": "SMILES", "Y": "TARGET"})[["SMILES", "TARGET"]].to_csv(trcsv, index=False)
    test.rename(columns={"Drug": "SMILES", "Y": "TARGET"})[["SMILES", "TARGET"]].to_csv(tecsv, index=False)
    clf = MolTrain(task="regression", epochs=EPOCHS, batch_size=bs, learning_rate=LR,
                   early_stopping=3, metrics="mae", kfold=1, split="scaffold", save_path=exp,
                   smiles_col="SMILES", target_cols="TARGET", model_size="84m",
                   use_cuda=True, use_amp=True, target_normalize="auto", seed=seed)
    clf.fit(trcsv)
    mp = MolPredict(load_model=exp)
    pred = np.asarray(mp.predict(tecsv), dtype=float).reshape(-1)
    shutil.rmtree(exp, ignore_errors=True)
    return pred, test


# ── ★conformer 생성 실패 분자 기록(양쪽에서 동일 제외하기 위해) ──
def conformer_failures(smiles_list):
    """ETKDG로 3D 생성이 실패하는 분자(Uni-Mol 내부와 같은 RDKit 경로)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    failed = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            failed.append(dict(smiles=str(s), reason="SMILES 파싱 실패")); continue
        try:
            mh = Chem.AddHs(m)
            if AllChem.EmbedMolecule(mh, randomSeed=42, maxAttempts=50) != 0:
                failed.append(dict(smiles=str(s), reason="ETKDG embed 실패"))
        except Exception as e:
            failed.append(dict(smiles=str(s), reason=f"{type(e).__name__}"))
    return failed


DONE = done_ok()
conf_fail = {}
checks = []
print("=" * 96)
print("[1단계] G4(Uni-Mol) 재현 학습 + 분자별 예측 저장 — 원 config 고정(EPOCHS20·BATCH8·LR1e-4·84m)")
print("=" * 96)
for ep in TARGETS:
    prim = EPS[ep]["primary"]
    g = C.group(); test_all = g.get(ep)["test"]
    # conformer 실패 목록(test 기준·전량 기록)
    cf = conformer_failures(test_all["Drug"].tolist())
    conf_fail[ep] = dict(n_test=len(test_all), n_failed=len(cf), failed=cf)
    print(f"\n[{ep}] test {len(test_all)} · ★conformer 실패 {len(cf)}개")
    per_seed = []
    for seed in SEEDS:
        if (ep, seed) in DONE:
            print(f"  s{seed} skip(완료)"); continue
        t0 = time.time(); used_bs = BATCH
        try:
            pred, test = train_one(ep, seed, BATCH)
        except Exception as e:
            oom = "out of memory" in str(e).lower()
            if not oom:
                rec(dict(endpoint=ep, seed=seed, status="failed", reason=str(e)[:200]))
                print(f"  s{seed} ★실패: {str(e)[:90]}"); continue
            try:
                import torch, gc
                gc.collect(); torch.cuda.empty_cache()
            except Exception:
                pass
            used_bs = max(2, BATCH // 2)
            print(f"  s{seed} ★OOM → batch {BATCH}→{used_bs} 재시도")
            try:
                pred, test = train_one(ep, seed, used_bs)
            except Exception as e2:
                rec(dict(endpoint=ep, seed=seed, status="failed",
                         reason=f"OOM 재시도(batch={used_bs}) 실패: {str(e2)[:160]}", oom_retry=True))
                print(f"  s{seed} ★재시도 후 실패 — skip"); continue
        y = test["Y"].to_numpy(float)
        if pred.shape[0] != len(y):
            rec(dict(endpoint=ep, seed=seed, status="failed",
                     reason=f"예측 길이 불일치 {pred.shape[0]} vs {len(y)}"))
            print(f"  s{seed} ★길이 불일치 — skip"); continue
        m = metric(y, pred, prim)
        per_seed.append(m)
        fn = f"{NEW}/predictions/{ep}__unimol__seed{seed}.jsonl"
        with open(fn, "w", encoding="utf-8") as f:
            for s, yt, yp in zip(test["Drug"], y, pred):
                f.write(json.dumps(dict(smiles=str(s), y_true=float(yt), y_pred=float(yp),
                                        seed=seed), ensure_ascii=False) + "\n")
        rec(dict(endpoint=ep, seed=seed, status="ok", metric=prim, value=round(m, 4),
                 batch_size=used_bs, sec=round(time.time() - t0, 1)))
        print(f"  s{seed} {prim}={m:.4f} ({time.time()-t0:.0f}s·bs{used_bs}) → {os.path.basename(fn)}")
    if per_seed:
        got = float(np.mean(per_seed)); repv = reported(ep)
        ok = abs(got - repv) <= TOL
        checks.append(dict(endpoint=ep, metric=prim, reported=repv, reproduced=round(got, 4),
                           diff=round(got - repv, 4), within_tol=bool(ok), n_seed=len(per_seed),
                           per_seed=[round(v, 4) for v in per_seed]))
        print(f"  ★재현: 보고 {repv:.4f} → 재현 {got:.4f} (Δ{got-repv:+.4f}) {'OK' if ok else '★불일치'}")

json.dump(conf_fail, open(f"{NEW}/logs/conformer_failed.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(dict(tolerance=TOL, config=dict(epochs=EPOCHS, batch=BATCH, lr=LR, model_size="84m"),
               checks=checks), open(f"{NEW}/results/step1_repro.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
bad = [c for c in checks if not c["within_tol"]]
print(f"\n재현 검증: {len(checks)}건 중 벗어남 {len(bad)}건" + (f" → {bad}" if bad else " → 전부 통과"))
print(f"저장 → predictions/*.jsonl · logs/conformer_failed.json · results/step1_repro.json")
if bad:
    sys.exit(2)
