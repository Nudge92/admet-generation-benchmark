#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step_logd_g3.py — logD G3(chemprop D-MPNN) 예측 생성. env: admet + chemprop(ADMET_AI env subprocess).
★ADME 세대 실험과 ★동일 config: regression·epochs 50·batch 50·-n 0·pytorch-seed s·5 seed. HP 변경 금지.
★valid 필요 이유: 컨포멀 예측구간 분위수를 valid에서 정해야(test 튜닝 금지).
★재현 검증: 5 seed 평균 MAE가 adme_matrix.csv logD G3(0.4406)과 ±0.005 이내인지. 벗어나면 중단.
산출: predictions/lipophilicity__dmpnn_ours__{split}.jsonl · results/logd_g3_repro.json · logs/logd_chemprop_bad.json
"""
import json, os, shutil, subprocess, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

ADME = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_adme_full"
NEW = "/home/nudge/Project/ADMET_integrated/2026-07-25/experiment_adme_reliability"
sys.path.insert(0, f"{ADME}/src")
import common as C

EP = "lipophilicity_astrazeneca"
SEEDS = [1, 2, 3, 4, 5]
TOL = 0.005
CHEMPROP = "/home/nudge/miniforge3/envs/ADMET_AI/bin/chemprop"
WORK = f"{NEW}/work_logd"
os.makedirs(WORK, exist_ok=True)
MAT = pd.read_csv(f"{ADME}/results/adme_matrix.csv")
REP = float(MAT[(MAT.endpoint == EP) & (MAT.model == "dmpnn_ours")].iloc[0].value)
canon = lambda s: (lambda m: Chem.MolToSmiles(m) if m else None)(Chem.MolFromSmiles(str(s)))
RAW = f"{NEW}/results/logd_g3_raw.jsonl"

# ── 0단계: 분할 확인 ──
g = C.group()
b = g.get(EP); test = b["test"]; tv = b["train_val"]
tv_c = {canon(s) for s in tv["Drug"]} - {None}
overlap = sum(1 for s in test["Drug"] if canon(s) in tv_c)
n_rec = int(json.load(open(f"{NEW}/results/champions.json"))[0].get("n_test", 0)) if False else len(test)
print("=" * 92)
print(f"[0단계] logD 분할 확인 — test {len(test)} · train∩test 정확분자 중복 {overlap}")
if overlap != 0:
    print("★중단 — train∩test 중복 발견"); sys.exit(2)

# chemprop 비호환 분자
ADMET_AI_PY = "/home/nudge/miniforge3/envs/ADMET_AI/bin/python"
code = ("import sys;from rdkit import Chem,RDLogger;RDLogger.DisableLog('rdApp.*');"
        "[print(l.rstrip(chr(10))) for l in open(sys.argv[1]) if l.strip() and Chem.MolFromSmiles(l.rstrip(chr(10))) is None]")
import tempfile
alls = sorted({str(s) for s in list(tv["Drug"]) + list(test["Drug"])})
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write("\n".join(alls)); p = f.name
r = subprocess.run([ADMET_AI_PY, "-c", code, p], capture_output=True, text=True, timeout=600)
os.unlink(p)
bad = sorted(set(r.stdout.split("\n")) - {""})
json.dump(dict(n_total=len(alls), n_bad=len(bad), bad=bad),
          open(f"{NEW}/logs/logd_chemprop_bad.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"  chemprop 비호환 분자: {len(bad)}개")


def train_one(seed):
    tr, va = g.split(EP, seed)
    wd = f"{WORK}/s{seed}"
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd, exist_ok=True)
    for df, nm in ((tr, "train"), (va, "val"), (test, "test")):
        d = df[["Drug", "Y"]].rename(columns={"Drug": "smiles", "Y": "y"})
        d = d[~d["smiles"].astype(str).isin(bad)]           # ★비호환 제외(기록됨)
        d.to_csv(f"{wd}/{nm}.csv", index=False)
    cmd = [CHEMPROP, "train", "-i", f"{wd}/train.csv", f"{wd}/val.csv", f"{wd}/test.csv",
           "-s", "smiles", "--target-columns", "y", "-t", "regression",
           "--epochs", "50", "--accelerator", "gpu", "--devices", "1", "-n", "0", "-b", "50",
           "--metrics", "mae", "--pytorch-seed", str(seed), "-o", f"{wd}/model", "-q"]
    rr = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if rr.returncode != 0:
        raise RuntimeError(f"train 실패: {rr.stderr[-400:]}")
    pts = sorted(os.path.join(rt, fn) for rt, _, fs in os.walk(f"{wd}/model") for fn in fs
                 if fn.endswith(".pt") and "best" in fn.lower()) or \
          sorted(os.path.join(rt, fn) for rt, _, fs in os.walk(f"{wd}/model") for fn in fs if fn.endswith(".pt"))
    out = {}
    for split in ("train", "val", "test"):
        po = f"{wd}/pred_{split}.csv"
        pr = subprocess.run([CHEMPROP, "predict", "-i", f"{wd}/{split}.csv", "-s", "smiles",
                            "--model-paths", pts[0], "-o", po, "-n", "0", "-q"],
                           capture_output=True, text=True, timeout=3600)
        if pr.returncode != 0:
            raise RuntimeError(f"predict {split} 실패: {pr.stderr[-300:]}")
        dfp = pd.read_csv(po); dft = pd.read_csv(f"{wd}/{split}.csv")
        col = [c for c in dfp.columns if c != "smiles"][0]
        out[split] = (dft["smiles"].tolist(), dft["y"].to_numpy(float), dfp[col].to_numpy(float))
    shutil.rmtree(f"{wd}/model", ignore_errors=True)
    return out


# resume
done = set()
if os.path.exists(RAW):
    for l in open(RAW, encoding="utf-8"):
        d = json.loads(l)
        if d.get("status") == "ok":
            done.add(d["seed"])

acc = {"train": [], "val": [], "test": []}
per_seed_mae = []
print("\n[1단계] chemprop D-MPNN 재현 학습 (regression·epochs50·batch50·5seed)")
for seed in SEEDS:
    if seed in done:
        print(f"  s{seed} skip(완료)"); continue
    try:
        out = train_one(seed)
    except Exception as e:
        with open(RAW, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(seed=seed, status="failed", reason=str(e)[:200]), ensure_ascii=False) + "\n")
        print(f"  s{seed} ★실패: {str(e)[:100]}"); continue
    mae = float(mean_absolute_error(out["test"][1], out["test"][2]))
    per_seed_mae.append(mae)
    for split in ("train", "val", "test"):
        S, Y, P = out[split]
        acc[split] += [dict(smiles=str(s), y_true=float(y), y_pred=float(p), seed=seed)
                       for s, y, p in zip(S, Y, P)]
    with open(RAW, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(seed=seed, status="ok", test_mae=round(mae, 4)), ensure_ascii=False) + "\n")
    print(f"  s{seed} test MAE={mae:.4f}")

# 예측 저장(split별)
splitmap = {"train": "train", "val": "valid", "test": "test"}
for split, recs in acc.items():
    if not recs:
        continue
    fn = f"{NEW}/predictions/lipophilicity_astrazeneca__dmpnn_ours__{splitmap[split]}.jsonl"
    with open(fn, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

got = float(np.mean(per_seed_mae)) if per_seed_mae else None
ok = got is not None and abs(got - REP) <= TOL
json.dump(dict(endpoint=EP, model="dmpnn_ours", metric="MAE", reported=REP,
               reproduced=(None if got is None else round(got, 4)),
               diff=(None if got is None else round(got - REP, 4)), within_tol=bool(ok),
               n_seed=len(per_seed_mae), per_seed=[round(v, 4) for v in per_seed_mae],
               n_chemprop_bad=len(bad)),
          open(f"{NEW}/results/logd_g3_repro.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n★재현: 보고 {REP:.4f} → 재현 {got if got is None else round(got,4)} "
      f"(Δ{'—' if got is None else f'{got-REP:+.4f}'}) {'OK' if ok else '★불일치'} · seed {len(per_seed_mae)}/5")
if not ok:
    print("★중단 — 재현 MAE 불일치(다른 모델을 넣는 셈)"); sys.exit(2)
print(f"저장 → predictions/lipophilicity_astrazeneca__dmpnn_ours__*.jsonl · results/logd_g3_repro.json")
