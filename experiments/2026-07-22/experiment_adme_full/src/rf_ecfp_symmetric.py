#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rf_ecfp_symmetric.py — ★비대칭 로스터 교정. env: admet · CPU.
적대검증 확정 HIGH: "물리화학 18/18 ECFP 압승"은 physchem에 2모델(xgb+rf)·ECFP에 1모델만 준
best-of-2 산물이었다. rf_ecfp(빠져 있던 4번째 조합)를 추가해 ★2×2 대칭(모델×특징)으로 재집계 →
진짜 '특징 효과'를 격리한다. 기존 예측·기록은 건드리지 않고 별도 파일에만 쓴다.
산출: results/rf_ecfp_raw.jsonl · results/feature_2x2.csv
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import common as C
from common import EPS, SEEDS, build_X, prep, score, log

RAW = f"{C.RES}/rf_ecfp_raw.jsonl"
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def done():
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


DONE = done()
g = C.group()
for ep, info in EPS.items():
    task = info["task"]
    b = g.get(ep); test = b["test"]
    Xte_r, _ = build_X(ep, ["ecfp"], test["Drug"].tolist(), "te")
    yte = test["Y"].to_numpy(float)
    for seed in SEEDS:
        if (ep, seed) in DONE:
            continue
        try:
            tr, va = g.split(ep, seed)
            Xtr_r, _ = build_X(ep, ["ecfp"], tr["Drug"].tolist(), f"tr{seed}")
            Xtr, Xte = prep(Xtr_r.copy(), Xte_r.copy())
            ytr = tr["Y"].to_numpy(float)
            M = RandomForestClassifier if task == "cls" else RandomForestRegressor
            m = M(n_estimators=500, n_jobs=C.NJOBS, random_state=seed)
            m.fit(Xtr, ytr.astype(int) if task == "cls" else ytr)
            pte = m.predict_proba(Xte)[:, 1] if task == "cls" else m.predict(Xte)
            rec = dict(model="rf_ecfp", endpoint=ep, seed=seed, task=task, status="ok",
                       metrics=score(yte, pte, task))
        except Exception as e:
            rec = dict(model="rf_ecfp", endpoint=ep, seed=seed, status="failed", err=str(e)[:200])
        with open(RAW, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"  rf_ecfp {ep:<32} 완료")

# ── 2×2 집계 ────────────────────────────────────────────────────
rf_ecfp = {}
for l in open(RAW, encoding="utf-8"):
    d = json.loads(l)
    if d.get("status") == "ok":
        rf_ecfp.setdefault(d["endpoint"], []).append(d["metrics"][EPS[d["endpoint"]]["primary"]])
mat = pd.read_csv(f"{C.RES}/adme_matrix.csv")


def val(ep, model):
    r = mat[(mat.endpoint == ep) & (mat.model == model)]
    return float(r.iloc[0].value) if not r.empty else None


rows = []
for ep, info in EPS.items():
    m = info["primary"]; hb = (m != "MAE")
    cell = {"xgb_physchem": val(ep, "xgb_physchem"), "rf_physchem": val(ep, "rf_physchem"),
            "xgb_ecfp": val(ep, "xgb_ecfp"),
            "rf_ecfp": (round(float(np.mean(rf_ecfp[ep])), 4) if ep in rf_ecfp else None)}
    better = lambda a, b: (a > b) if hb else (a < b)
    # ① 동일 XGB 통제: physchem vs ecfp
    xgb_win = better(cell["xgb_physchem"], cell["xgb_ecfp"])
    # ② 동일 RF 통제: physchem vs ecfp
    rf_win = (better(cell["rf_physchem"], cell["rf_ecfp"]) if cell["rf_ecfp"] is not None else None)
    # ③ 2×2 평균: physchem 평균 vs ecfp 평균
    phys_avg = np.mean([cell["xgb_physchem"], cell["rf_physchem"]])
    ecfp_avg = (np.mean([cell["xgb_ecfp"], cell["rf_ecfp"]]) if cell["rf_ecfp"] is not None else None)
    avg_win = (better(phys_avg, ecfp_avg) if ecfp_avg is not None else None)
    rows.append(dict(endpoint=ep, label=info["label"], pillar=info["pillar"], metric=m,
                     xgb_physchem=cell["xgb_physchem"], rf_physchem=cell["rf_physchem"],
                     xgb_ecfp=cell["xgb_ecfp"], rf_ecfp=cell["rf_ecfp"],
                     xgb_physchem_wins=xgb_win, rf_physchem_wins=rf_win,
                     phys_avg=round(phys_avg, 4), ecfp_avg=(None if ecfp_avg is None else round(ecfp_avg, 4)),
                     physchem_avg_wins=avg_win))
D = pd.DataFrame(rows)
D.to_csv(f"{C.RES}/feature_2x2.csv", index=False)

nx = int(D.xgb_physchem_wins.sum()); nrf = int(D.rf_physchem_wins.sum())
navg = int(D.physchem_avg_wins.sum())
print("\n" + "=" * 92)
print("★2×2 대칭 재집계 — physchem vs ECFP (특징 효과 격리)")
print("=" * 92)
print(f"  동일 XGB 통제:  physchem {nx}/18 · ECFP {18-nx}/18")
print(f"  동일 RF  통제:  physchem {nrf}/18 · ECFP {18-nrf}/18")
print(f"  2×2 평균:       physchem {navg}/18 · ECFP {18-navg}/18")
print(f"\n  ECFP가 이긴 엔드포인트:")
for _, r in D.iterrows():
    if not r.xgb_physchem_wins or (r.rf_physchem_wins is not None and not r.rf_physchem_wins):
        who = []
        if not r.xgb_physchem_wins: who.append("XGB")
        if r.rf_physchem_wins is not None and not r.rf_physchem_wins: who.append("RF")
        print(f"    {r.endpoint:<32}{r.metric:<9}{'/'.join(who)}서 ECFP 승 "
              f"(xgb {r.xgb_physchem:.4f}/{r.xgb_ecfp:.4f} · rf {r.rf_physchem:.4f}/{r.rf_ecfp:.4f})")
print(f"\n저장 → results/rf_ecfp_raw.jsonl · feature_2x2.csv")
