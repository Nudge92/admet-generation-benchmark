#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_admetai_infer.py — ★추론만(학습 아님). env: ADMET_AI.
기존 실험이 저장한 ★동일 고정 test 분자(experiment_tox_benchmark/data/test_<ep>.csv)에
ADMET-AI를 적용해 ★분자별 예측을 admetai_preds.jsonl 로 저장.
(지표는 2026-06-27 admetai_metrics.json 에 이미 있음 → 재현 일치 여부도 함께 검증)
★누수: ADMET-AI 공개모델은 TDC 전체로 사전학습 → 이 test 분자를 이미 학습에 포함(패키지 메타로 실측).
"""
import os, json, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, matthews_corrcoef
from admet_ai import ADMETModel

SRC = "/home/nudge/Project/ADMET_structure/2026-06-27/experiment_tox_benchmark"
OUT = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_generation_matrix/results"
EP2COL = {"dili": "DILI", "herg": "hERG", "ames": "AMES", "ld50_zhu": "LD50_Zhu"}
TASK = {"dili": "cls", "herg": "cls", "ames": "cls", "ld50_zhu": "reg"}

model = ADMETModel()
recs, summ = [], {}
for ep, col in EP2COL.items():
    rows = list(csv.DictReader(open(f"{SRC}/data/test_{ep}.csv")))
    smis = [r["Drug"] for r in rows]
    y = np.array([float(r["Y"]) for r in rows])
    pred = model.predict(smiles=smis).reset_index(drop=True)
    p = pred[col].values.astype(float)
    for s, yy, pp in zip(smis, y, p):
        recs.append(dict(endpoint=ep, smiles=s, y=float(yy), admetai_pred=float(pp)))
    if TASK[ep] == "cls":
        summ[ep] = dict(task="cls", n=len(y),
                        AUROC=round(float(roc_auc_score(y, p)), 4),
                        AUPRC=round(float(average_precision_score(y, p)), 4),
                        MCC=round(float(matthews_corrcoef(y, (p >= 0.5).astype(int))), 4),
                        pos_rate=round(float(y.mean()), 4))
    else:
        summ[ep] = dict(task="reg", n=len(y), MAE=round(float(mean_absolute_error(y, p)), 4))
    print(f"[{ep}] n={len(y)} {summ[ep]}", flush=True)

with open(f"{OUT}/admetai_preds.jsonl", "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(summ, open(f"{OUT}/admetai_recomputed.json", "w"), ensure_ascii=False, indent=1)

# ★재현 검증: 2026-06-27 저장 지표와 일치하는가
old = json.load(open(f"{SRC}/results/admetai_metrics.json"))
print("\n★재현 검증 (2026-06-27 저장값 vs 오늘 재추론)")
for ep in EP2COL:
    for m in ("AUROC", "AUPRC", "MAE"):
        if m in old.get(ep, {}) and m in summ[ep]:
            d = summ[ep][m] - old[ep][m]
            print(f"  {ep:<9}{m:<6} 저장 {old[ep][m]:.4f}  재추론 {summ[ep][m]:.4f}  Δ{d:+.4f}"
                  f"  {'일치' if abs(d) < 1e-3 else '★불일치'}")
print(f"\n저장 → admetai_preds.jsonl ({len(recs)}행) · admetai_recomputed.json")
