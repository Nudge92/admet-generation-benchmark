#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step2_g2_predictions.py — [2단계] G2 챔피언 재현 + ★분자별 예측 저장. env: admet · CPU.
★원 실험(2026-07-22 experiment_adme_full)과 ★동일 config·동일 seed. 튜닝 금지(재현이지 개선 아님).
  반감기 → rf_physchem(RandomForest 500) · 수용해도 → xgb_physchem(config_public cls/reg_params)
  특징: RDKit 210 서술자 → inf→nan → train중앙값 대치 → StandardScaler(train fit) → ±1e6 클리핑
        (전부 common.prep/build_X 재사용)
★재현 검증: 5 seed 평균이 adme_matrix.csv 보고값과 ±0.005 안인지 확인. 벗어나면 중단.
산출: predictions/{ep}__{model}__seed{n}.jsonl · results/step2_repro.json
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ADME = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_adme_full"
NEW = "/home/nudge/Project/ADMET_integrated/2026-07-24/experiment_g4_verification"
sys.path.insert(0, f"{ADME}/src")
import common as C
from common import EPS, SEEDS, build_X, prep, score
sys.path.insert(0, f"{ADME}/src")
import run_all as RA                      # fit_tree 재사용(원 실험과 동일 학습 경로)

TARGET = {"half_life_obach": "rf_physchem", "solubility_aqsoldb": "xgb_physchem"}
TOL = 0.005
MAT = pd.read_csv(f"{ADME}/results/adme_matrix.csv")


def reported(ep, model):
    r = MAT[(MAT.endpoint == ep) & (MAT.model == model)]
    return float(r.iloc[0].value), C.EPS[ep]["primary"]


g = C.group()
checks = []
print("=" * 96)
print("[2단계] G2 챔피언 재현 + 분자별 예측 저장 (원 실험과 동일 config·seed)")
print("=" * 96)
for ep, model in TARGET.items():
    task = EPS[ep]["task"]; prim = EPS[ep]["primary"]
    b = g.get(ep); test = b["test"]
    Xte_r, _ = build_X(ep, ["phys"], test["Drug"].tolist(), "te")
    yte = test["Y"].to_numpy(float)
    per_seed = []
    for seed in SEEDS:
        tr, va = g.split(ep, seed)
        Xtr_r, _ = build_X(ep, ["phys"], tr["Drug"].tolist(), f"tr{seed}")
        Xtr, Xte = prep(Xtr_r.copy(), Xte_r.copy())
        ytr = tr["Y"].to_numpy(float)
        pte = RA.fit_tree(model, task, Xtr, ytr, Xte, seed)     # ★원 실험과 동일 함수
        m = score(yte, pte, task)
        per_seed.append(m[prim])
        # ★분자별 예측 저장 — 이 실험의 핵심 재사용 자산
        fn = f"{NEW}/predictions/{ep}__{model}__seed{seed}.jsonl"
        with open(fn, "w", encoding="utf-8") as f:
            for s, yt, yp in zip(test["Drug"], yte, pte):
                f.write(json.dumps(dict(smiles=str(s), y_true=float(yt), y_pred=float(yp),
                                        seed=seed), ensure_ascii=False) + "\n")
        print(f"  [{ep}/{model}/s{seed}] {prim}={m[prim]:.4f}  → {os.path.basename(fn)}")
    got = float(np.mean(per_seed)); rep, _ = reported(ep, model)
    ok = abs(got - rep) <= TOL
    checks.append(dict(endpoint=ep, model=model, metric=prim, reported=rep,
                       reproduced=round(got, 4), diff=round(got - rep, 4), within_tol=bool(ok),
                       per_seed=[round(v, 4) for v in per_seed], n_test=len(test)))
    print(f"  ★재현: 보고 {rep:.4f} → 재현 {got:.4f} (Δ{got-rep:+.4f}) {'OK' if ok else '★불일치'}\n")

json.dump(dict(tolerance=TOL, checks=checks), open(f"{NEW}/results/step2_repro.json", "w"),
          ensure_ascii=False, indent=1)
bad = [c for c in checks if not c["within_tol"]]
print(f"재현 검증: {len(checks)}건 중 허용오차(±{TOL}) 벗어남 {len(bad)}건" + (f" → {bad}" if bad else " → 전부 통과"))
if bad:
    sys.exit(2)
print("저장 → predictions/*.jsonl · results/step2_repro.json")
