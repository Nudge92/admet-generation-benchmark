#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bootstrap_verdict.py — 세대 승부에 ★구분 가능선 판정. env: admet · 재학습 0(저장 예측·seed 지표만).
두 종류의 불확실성을 정직하게 구분:
- G2·G3·ADMET-AI: 분자별 예측 있음 → ★분자 부트스트랩 2000회(진짜 test 불확실성) + 대응비교.
- G4·G5: ★분자별 예측 미저장 → 분자 부트스트랩 불가. seed 재표집(5개)만 가능 → '약한 근거'로 명시.
  (고정 test 위 seed SD는 일반화 불확실성이 아니다 — 프로젝트 자체 교훈. 그래서 G4/G5 판정은 보류적.)
산출: results/bootstrap_verdict.json · bootstrap_verdict.csv
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
import common as C
from common import EPS

NBOOT = 2000
RNG = np.random.default_rng(20260724)
PRED = C.PRED
mat = pd.read_csv(f"{C.RES}/adme_matrix.csv")
G4G5_RAW = {"unimol": f"{C.RES}/finetune2_raw.jsonl", "chemberta": f"{C.RES}/finetune_raw.jsonl",
            "molformer": f"{C.RES}/finetune_raw.jsonl"}


def load_pred(ep, model):
    """분자별 예측 (seed 평균)."""
    f = f"{PRED}/{ep}__{model}__test.jsonl"
    if not os.path.exists(f):
        return None
    d = pd.read_json(f, lines=True)
    g = d.groupby("smiles").agg(y=("y_true", "first"), p=("y_prob", "mean")).reset_index()
    return g


def metric(y, p, m):
    if m == "AUROC":
        return roc_auc_score(y, p)
    if m == "AUPRC":
        return average_precision_score(y, p)
    if m == "MAE":
        return mean_absolute_error(y, p)
    return spearmanr(y, p).correlation


def boot_ci(y, p, m):
    idx = np.arange(len(y)); vals = []
    for _ in range(NBOOT):
        s = RNG.choice(idx, len(idx), replace=True)
        if m in ("AUROC", "AUPRC") and len(np.unique(y[s])) < 2:
            continue
        try:
            vals.append(metric(y[s], p[s], m))
        except Exception:
            pass
    if len(vals) < 100:
        return None
    v = np.array(vals)
    return dict(mean=round(float(v.mean()), 4), lo=round(float(np.percentile(v, 2.5)), 4),
                hi=round(float(np.percentile(v, 97.5)), 4))


def paired_delta_ci(ep, ma, mb, m):
    """대응비교: 같은 부트스트랩 표본에서 metric(a)-metric(b) 분포. 방향 고려는 호출부에서."""
    A, B = load_pred(ep, ma), load_pred(ep, mb)
    if A is None or B is None:
        return None
    mg = A.merge(B[["smiles", "p"]], on="smiles", suffixes=("_a", "_b"))
    if len(mg) < 30:
        return None
    y = mg.y.to_numpy(float); pa = mg.p_a.to_numpy(float); pb = mg.p_b.to_numpy(float)
    idx = np.arange(len(y)); dif = []
    for _ in range(NBOOT):
        s = RNG.choice(idx, len(idx), replace=True)
        if m in ("AUROC", "AUPRC") and len(np.unique(y[s])) < 2:
            continue
        try:
            dif.append(metric(y[s], pa[s], m) - metric(y[s], pb[s], m))
        except Exception:
            pass
    if len(dif) < 100:
        return None
    v = np.array(dif)
    lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    return dict(delta=round(float(v.mean()), 4), lo=round(lo, 4), hi=round(hi, 4),
                p_two=round(float(2 * min((v > 0).mean(), (v < 0).mean())), 4),
                distinguishable=bool(lo > 0 or hi < 0))


PRED_MODELS = ["xgb_physchem", "rf_physchem", "xgb_ecfp", "dmpnn_ours"]
PMAP = {"xgb_physchem": "G2_xgb_physchem", "rf_physchem": "G2_rf_physchem",
        "xgb_ecfp": "G2_xgb_ecfp", "dmpnn_ours": "G3_dmpnn"}

out = {}
rows = []
for ep, info in EPS.items():
    m = info["primary"]; hb = (m != "MAE")
    ep_out = dict(metric=m, higher_better=hb, ci={}, note="")
    # 예측 있는 모델: 부트스트랩 CI
    for model in PRED_MODELS:
        pr = load_pred(ep, PMAP[model])
        if pr is None:
            continue
        y = pr.y.to_numpy(float); p = pr.p.to_numpy(float)
        ep_out["ci"][model] = boot_ci(y, p, m)
    # 최고 모델 판정 (예측 있는 것 중)
    cand = {k: v for k, v in ep_out["ci"].items() if v}
    if cand:
        best = max(cand, key=lambda k: cand[k]["mean"]) if hb else min(cand, key=lambda k: cand[k]["mean"])
        # 대응비교: best vs 나머지 → 구분 가능한지
        vs = {}
        for other in cand:
            if other == best:
                continue
            d = paired_delta_ci(ep, PMAP[best], PMAP[other], m)
            if d:
                vs[other] = d
        ep_out["best_predicted_model"] = best
        ep_out["best_vs_others"] = vs
        ep_out["n_distinguishable"] = sum(1 for d in vs.values() if d["distinguishable"])
        ep_out["n_compared"] = len(vs)
        for other, d in vs.items():
            rows.append(dict(endpoint=ep, metric=m, best=best, other=other,
                             delta=d["delta"], ci_lo=d["lo"], ci_hi=d["hi"],
                             distinguishable=d["distinguishable"]))
    out[ep] = ep_out

json.dump(out, open(f"{C.RES}/bootstrap_verdict.json", "w"), ensure_ascii=False, indent=1)
pd.DataFrame(rows).to_csv(f"{C.RES}/bootstrap_verdict.csv", index=False)

print("=" * 96)
print("★부트스트랩 대응비교 — 예측 있는 모델(G2 3종·G3)만. best vs 나머지가 구분 가능한가")
print("=" * 96)
for ep, info in EPS.items():
    o = out[ep]
    if "best_predicted_model" not in o:
        continue
    nd, nc = o["n_distinguishable"], o["n_compared"]
    print(f"  {ep:<32}{o['metric']:<9}best={o['best_predicted_model']:<14}구분가능 {nd}/{nc}")
    for other, d in o["best_vs_others"].items():
        mark = "★구분" if d["distinguishable"] else "동률"
        print(f"      vs {other:<14}Δ{d['delta']:+.4f} CI[{d['lo']:+.4f},{d['hi']:+.4f}]  {mark}")
print(f"\n★한계: G4(unimol)·G5(chemberta/molformer)는 분자별 예측 미저장 → 분자 부트스트랩·대응비교 불가.")
print(f"       E 헤드라인·G4 5승은 이 방법으로 판정 불가(seed 재표집만 가능·약한 근거).")
print(f"저장 → bootstrap_verdict.json · bootstrap_verdict.csv")
