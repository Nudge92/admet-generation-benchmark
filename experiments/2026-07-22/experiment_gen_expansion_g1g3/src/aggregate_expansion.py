#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aggregate_expansion.py — G1/G2/G3(정직)/G3(ADMET-AI) 원자료 → expansion_metrics.json · overfit.json.
★플래그십 방법론 계승: 구분 가능선은 Hanley-McNeil 해석적 SE(비대응 가정·보수적 참고선).
★실패는 0점 금지 — n_ok와 failed로 기록. 값 없으면 None(=N/A).
"""
import json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_gen_expansion_g1g3"
RES = f"{ROOT}/results"
G2_MODELS = ["xgb_physchem", "rf_physchem", "xgb_ecfp"]


def hm_se(a, npos, nneg):
    if npos < 2 or nneg < 2:
        return None
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    v = (a * (1 - a) + (npos - 1) * (q1 - a * a) + (nneg - 1) * (q2 - a * a)) / (npos * nneg)
    return float(np.sqrt(max(v, 0)))


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


meta = json.load(open(f"{RES}/split_meta.json"))
g1 = pd.read_csv(f"{RES}/g1_summary.csv")
g2r, g3r = jl(f"{RES}/g2_raw.jsonl"), jl(f"{RES}/g3_raw.jsonl")
ai = json.load(open(f"{RES}/admetai_metrics.json")) if os.path.exists(f"{RES}/admetai_metrics.json") else {}

out, overfit, failed = {}, [], dict(g2=[], g3=[])
for r in g2r:
    if r["status"] != "ok":
        failed["g2"].append({k: r[k] for k in ("model", "endpoint", "task", "seed", "status")})
for r in g3r:
    if r["status"] != "ok":
        failed["g3"].append({k: r.get(k) for k in ("endpoint", "seed", "status")})

for ep, m in meta.items():
    out[ep] = {}
    g3ok = [r for r in g3r if r["endpoint"] == ep and r["status"] == "ok"]
    for lab in m["labels"]:
        pl = m["per_label"][lab]["test"]
        n, pos = pl["n"], pl["pos"]
        if n < 20 or pos < 2 or pos == n:
            continue
        rec = dict(n_test=n, n_pos=pos, pos_rate=pl["pos_rate"],
                   resolvable_gap_95=(lambda se: None if se is None else round(1.96 * se * np.sqrt(2), 4))(
                       hm_se(0.80, pos, n - pos)),
                   se_at_auroc_080=(lambda se: None if se is None else round(se, 4))(hm_se(0.80, pos, n - pos)))

        # G1
        g = g1[(g1.endpoint == ep) & (g1.task == lab)]
        rec["G1"] = (None if g.empty else dict(
            metric_form="다름(발화율·정밀도·MCC)", fire_rate=float(g.iloc[0].fire_rate),
            nomatch_rate=float(g.iloc[0].nomatch_rate), precision=float(g.iloc[0].rule_precision),
            recall=float(g.iloc[0].rule_recall), MCC=float(g.iloc[0].rule_MCC),
            verdict=("무력" if abs(g.iloc[0].rule_MCC) < 0.05 else
                     "약한 신호" if abs(g.iloc[0].rule_MCC) < 0.15 else "의미 있는 신호"),
            source="results/g1_summary.csv"))

        # G2 (3종)
        rec["G2"] = {}
        for mk in G2_MODELS:
            rs = [r for r in g2r if r["status"] == "ok" and r["model"] == mk
                  and r["endpoint"] == ep and r["task"] == lab]
            if not rs:
                rec["G2"][mk] = None; continue
            d = {}
            for k in ("AUROC", "AUPRC"):
                te = [r["test"][k] for r in rs]; tr = [r["train"][k] for r in rs]
                d[k] = dict(mean=round(float(np.mean(te)), 4), std=round(float(np.std(te)), 4))
                overfit.append(dict(model=mk, endpoint=ep, task=lab, metric=k,
                                    train=round(float(np.mean(tr)), 4), test=round(float(np.mean(te)), 4),
                                    gap=round(float(np.mean(tr) - np.mean(te)), 4), n_seed=len(rs)))
            d["n_ok"] = len(rs); d["source"] = "results/g2_raw.jsonl"
            rec["G2"][mk] = d

        # G3 정직 D-MPNN
        vals = [r["test"].get(lab if m["n_labels"] > 1 else "Y") for r in g3ok]
        vals = [v for v in vals if v]
        if vals:
            d = {}
            for k in ("AUROC", "AUPRC"):
                te = [v[k] for v in vals]
                trv = [r["train"].get(lab if m["n_labels"] > 1 else "Y") for r in g3ok]
                trv = [v[k] for v in trv if v]
                d[k] = dict(mean=round(float(np.mean(te)), 4), std=round(float(np.std(te)), 4))
                if trv:
                    overfit.append(dict(model="dmpnn_ours", endpoint=ep, task=lab, metric=k,
                                        train=round(float(np.mean(trv)), 4), test=round(float(np.mean(te)), 4),
                                        gap=round(float(np.mean(trv) - np.mean(te)), 4), n_seed=len(te)))
            d["n_ok"] = len(vals)
            d["class_balance"] = bool(g3ok[0].get("class_balance", True))
            d["source"] = "results/g3_raw.jsonl"
            rec["G3_ours"] = d
        else:
            rec["G3_ours"] = None

        # G3 ADMET-AI (누수 기준선)
        a = (ai.get(ep) or {}).get(lab)
        if a and a.get("status") == "ok":
            rec["G3_admetai"] = dict(AUROC=a["AUROC"], AUPRC=a["AUPRC"], n=a["n"],
                                     admetai_train_size=a.get("admetai_train_size"),
                                     leak_flag="★누수 의심(TDC 전체 사전학습)",
                                     source="results/admetai_metrics.json")
            if rec["G3_ours"]:
                rec["leak_premium"] = dict(
                    AUROC=round(a["AUROC"] - rec["G3_ours"]["AUROC"]["mean"], 4),
                    AUPRC=round(a["AUPRC"] - rec["G3_ours"]["AUPRC"]["mean"], 4),
                    note="ADMET-AI − 우리 정직 D-MPNN (같은 아키텍처)")
        else:
            rec["G3_admetai"] = dict(status=(a or {}).get("status", "미커버"),
                                     note="ADMET-AI 미커버 → 이 과제는 정직 G3만")
        out[ep][lab] = rec

out["_meta"] = dict(
    split={ep: m["split"] + " | " + m["source"] for ep, m in meta.items()},
    leakage=json.load(open(f"{RES}/leakage.json")),
    g2_config=dict(models=G2_MODELS, xgb="config_public.json cls_params(n_est400·lr0.05·depth6·sub0.8·col0.8)",
                   rf="RandomForestClassifier(n_estimators=500)",
                   features="RDKit 210 서술자(inf→nan→train중앙값→StandardScaler) / ECFP4 2048bit",
                   seeds=[1, 2, 3, 4, 5], note="★분할 고정·모델 seed만 변주"),
    g3_config=dict(model="Chemprop D-MPNN v2 (순수·외부특징 없음)", epochs=50, batch=50,
                   seeds=[1, 2, 3, 4, 5], multitask="Tox21은 12 타깃 동시(결측 마스킹)",
                   note="플래그십 experiment_g3_dmpnn_seed42 와 동일 config"),
    failed=failed,
    uncertainty_caveat="Hanley-McNeil은 비대응(독립표본) 가정이라 같은 test 위 두 모델 비교엔 ★보수적. "
                       "분자별 예측을 저장했더라도 대응비교(DeLong)는 별도 계산이 필요. "
                       "seed SD는 ★분할 고정 위의 모델 초기화 변동이라 일반화 불확실성이 아니다.")
json.dump(out, open(f"{RES}/expansion_metrics.json", "w"), ensure_ascii=False, indent=1)
json.dump(overfit, open(f"{RES}/overfit.json", "w"), ensure_ascii=False, indent=1)

print(f"{'엔드포인트':<20}{'과제':<15}{'n':>6}{'양성':>7}{'G1 MCC':>9}{'G2 최고':>10}{'G3 정직':>10}{'ADMET-AI':>11}{'프리미엄':>10}{'구분선':>9}")
for ep in meta:
    for lab, r in out[ep].items():
        g2v = [v["AUROC"]["mean"] for v in r["G2"].values() if v]
        g2b = max(g2v) if g2v else None
        g3 = r["G3_ours"]["AUROC"]["mean"] if r["G3_ours"] else None
        aiv = r["G3_admetai"].get("AUROC")
        pr = (r.get("leak_premium") or {}).get("AUROC")
        f = lambda x: "N/A" if x is None else f"{x:.4f}"
        print(f"{ep:<20}{lab:<15}{r['n_test']:>6}{r['pos_rate']:>7.3f}"
              f"{r['G1']['MCC'] if r['G1'] else 0:>+9.3f}{f(g2b):>10}{f(g3):>10}{f(aiv):>11}"
              f"{('' if pr is None else f'{pr:+.4f}'):>10}{f(r['resolvable_gap_95']):>9}")
print(f"\n실패 G2 {len(failed['g2'])}건 · G3 {len(failed['g3'])}건")
print(f"저장 → expansion_metrics.json · overfit.json")
