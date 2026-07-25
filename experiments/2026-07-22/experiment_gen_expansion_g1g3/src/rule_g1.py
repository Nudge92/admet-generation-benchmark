#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rule_g1.py — G1(구조알림)을 확장 3종의 ★고정 test셋에 적용. env: admet · ★학습 없음.
플래그십과 ★같은 규칙셋을 쓴다: T_toxicity/surface/src/featurize.py 의
  BRENK+NIH FilterCatalog + Benigni-Bossa 계열 변이원성 SMARTS 10종 (수정 없이 로드).
★발암성엔 이 알림들이 원래 발암/변이원 계열(방향족아민·nitroso·epoxide·azo 등)이라 의미가 있고,
  ClinTox·Tox21은 매핑이 약할 수 있다 → ★무력이면 무력이라고 그대로 보고(억지 AUROC 금지).
지표: 발화율·무매치율·알림별 정밀도(P(양성|발화))·기저 대비 lift·작동점(정밀도/재현율/MCC).
산출: results/g1_rules.csv (엔드포인트×과제×알림) · results/g1_summary.csv (엔드포인트×과제)
"""
import importlib.util, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score

ROOT = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_gen_expansion_g1g3"
FEAT = "/home/nudge/Project/HITS_portfolio/ADMET/T_toxicity/surface/src/featurize.py"
SPL, RES = f"{ROOT}/splits", f"{ROOT}/results"

spec = importlib.util.spec_from_file_location("feat", FEAT)
feat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat)                     # ★규칙 정의 그대로
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

meta = json.load(open(f"{RES}/split_meta.json"))
rows, summ = [], []
for ep, m in meta.items():
    te = pd.read_csv(f"{SPL}/{ep}_test.csv")
    mols = [Chem.MolFromSmiles(str(s)) for s in te["Drug"]]
    ncat = np.array([len(feat._CATALOG.GetMatches(x)) if x is not None else 0 for x in mols])
    hits = {nm: np.array([bool(x is not None and p is not None and x.HasSubstructMatch(p)) for x in mols])
            for nm, p in feat._MUTAGEN}
    any_mut = np.zeros(len(mols), bool)
    for v in hits.values():
        any_mut |= v
    any_alert = any_mut | (ncat > 0)

    for lab in m["labels"]:
        col = "Y" if m["n_labels"] == 1 else lab
        y = pd.to_numeric(te[col], errors="coerce")
        ok = y.notna().to_numpy()
        yy = y[ok].to_numpy(float)
        if len(yy) < 20 or len(np.unique(yy)) < 2:
            continue
        base = float(yy.mean())
        for nm, v in list(hits.items()) + [("★변이원 알림 ≥1(순수규칙)", any_mut),
                                           ("BRENK/NIH 알림 ≥1", ncat > 0),
                                           ("★전체 알림 ≥1", any_alert)]:
            vv = v[ok]
            n_f = int(vv.sum())
            prec = float(yy[vv].mean()) if n_f else None
            rows.append(dict(endpoint=ep, task=lab, alert=nm, n_eval=len(yy), n_fired=n_f,
                             fire_rate=round(n_f / len(yy), 4),
                             precision=None if prec is None else round(prec, 4),
                             base_rate=round(base, 4),
                             lift=None if (prec is None or base == 0) else round(prec / base, 3)))
        pred = any_mut[ok].astype(int)
        summ.append(dict(endpoint=ep, task=lab, n_eval=len(yy), pos_rate=round(base, 4),
                         fire_rate=round(float(any_mut[ok].mean()), 4),
                         nomatch_rate=round(float((~any_mut[ok]).mean()), 4),
                         fire_rate_all=round(float(any_alert[ok].mean()), 4),
                         rule_precision=round(float(precision_score(yy, pred, zero_division=0)), 4),
                         rule_recall=round(float(recall_score(yy, pred, zero_division=0)), 4),
                         rule_MCC=round(float(matthews_corrcoef(yy, pred)), 4)))

pd.DataFrame(rows).to_csv(f"{RES}/g1_rules.csv", index=False)
s = pd.DataFrame(summ); s.to_csv(f"{RES}/g1_summary.csv", index=False)
print(f"{'엔드포인트':<22}{'과제':<16}{'n':>6}{'양성률':>8}{'발화율':>8}{'무매치':>8}{'정밀도':>8}{'재현율':>8}{'MCC':>8}  판정")
for _, r in s.iterrows():
    verdict = ("★무력" if abs(r.rule_MCC) < 0.05 else "약한 신호" if abs(r.rule_MCC) < 0.15 else "의미 있는 신호")
    print(f"{r.endpoint:<22}{r.task:<16}{r.n_eval:>6}{r.pos_rate:>8.3f}{r.fire_rate:>8.3f}"
          f"{r.nomatch_rate:>8.3f}{r.rule_precision:>8.3f}{r.rule_recall:>8.3f}{r.rule_MCC:>+8.3f}  {verdict}")
print(f"\n저장 → g1_rules.csv({len(rows)}행) · g1_summary.csv({len(summ)}행)")
