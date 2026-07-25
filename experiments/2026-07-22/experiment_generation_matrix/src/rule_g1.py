#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rule_g1.py — G1(구조알림 규칙)을 ★동일 고정 test셋에 적용. env: admet.
★학습 없음 — 기존 T_toxicity/surface/src/featurize.py 의 규칙 정의(BRENK+NIH 카탈로그 +
  Benigni-Bossa 계열 변이원성 SMARTS 10개)를 ★수정 없이 그대로 재사용해 규칙을 적용만 한다.
★지표 형태가 다름: AUROC를 억지로 만들지 않고 ★발화율·무매치율·알림별 정밀도·
  규칙 작동점(정밀도/재현율/MCC)만 보고. (규칙은 확률이 아니라 이진 판정이므로)
산출: results/g1_rules.csv (엔드포인트×알림별) · results/g1_summary.csv (엔드포인트 요약)
"""
import csv, importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score

SRC = "/home/nudge/Project/ADMET_structure/2026-06-27/experiment_tox_benchmark"
FEAT = "/home/nudge/Project/HITS_portfolio/ADMET/T_toxicity/surface/src/featurize.py"
OUT = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_generation_matrix/results"
EPS = {"dili": "cls", "herg": "cls", "ames": "cls", "ld50_zhu": "reg"}

spec = importlib.util.spec_from_file_location("feat", FEAT)
feat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat)                      # ★규칙 정의 그대로 로드
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

rows, summ = [], []
for ep, task in EPS.items():
    rec = list(csv.DictReader(open(f"{SRC}/data/test_{ep}.csv")))
    smis = [r["Drug"] for r in rec]
    y = np.array([float(r["Y"]) for r in rec])
    mols = [Chem.MolFromSmiles(s) for s in smis]
    ok = np.array([m is not None for m in mols])

    ncat = np.array([len(feat._CATALOG.GetMatches(m)) if m is not None else 0 for m in mols])
    hits = {nm: np.array([bool(m is not None and p is not None and m.HasSubstructMatch(p)) for m in mols])
            for nm, p in feat._MUTAGEN}
    any_mut = np.zeros(len(mols), dtype=bool)
    for v in hits.values():
        any_mut |= v
    any_alert = any_mut | (ncat > 0)

    base = float(y.mean()) if task == "cls" else None
    for nm, v in list(hits.items()) + [("★변이원 알림 ≥1(순수규칙)", any_mut),
                                       ("BRENK/NIH 알림 ≥1", ncat > 0),
                                       ("★전체 알림 ≥1", any_alert)]:
        n_f = int(v.sum())
        if task == "cls":
            prec = float(y[v].mean()) if n_f else None          # P(양성 | 알림 발화)
            lift = (prec / base) if (prec is not None and base) else None
            rows.append(dict(endpoint=ep, alert=nm, n_fired=n_f,
                             fire_rate=round(n_f / len(y), 4),
                             precision=None if prec is None else round(prec, 4),
                             base_rate=round(base, 4), lift=None if lift is None else round(lift, 3),
                             mean_LD50_fired=None, mean_LD50_notfired=None))
        else:                                                   # 회귀: 발화군 평균 LD50 대비
            rows.append(dict(endpoint=ep, alert=nm, n_fired=n_f,
                             fire_rate=round(n_f / len(y), 4), precision=None,
                             base_rate=None, lift=None,
                             mean_LD50_fired=None if not n_f else round(float(y[v].mean()), 4),
                             mean_LD50_notfired=None if v.all() else round(float(y[~v].mean()), 4)))

    s = dict(endpoint=ep, task=task, n_test=len(y), n_parse_fail=int((~ok).sum()),
             fire_rate_any=round(float(any_alert.mean()), 4),
             nomatch_rate=round(float((~any_alert).mean()), 4),
             fire_rate_mutagen=round(float(any_mut.mean()), 4),
             nomatch_rate_mutagen=round(float((~any_mut).mean()), 4))
    if task == "cls":                                            # 순수규칙 작동점(AUROC 아님)
        pred = any_mut.astype(int)
        s.update(pos_rate=round(float(y.mean()), 4),
                 rule_precision=round(float(precision_score(y, pred, zero_division=0)), 4),
                 rule_recall=round(float(recall_score(y, pred, zero_division=0)), 4),
                 rule_MCC=round(float(matthews_corrcoef(y, pred)), 4))
    else:
        s.update(pos_rate=None, rule_precision=None, rule_recall=None, rule_MCC=None,
                 LD50_delta=round(float(y[any_mut].mean() - y[~any_mut].mean()), 4))
    summ.append(s)
    print(f"[{ep}] n={len(y)} 발화율(변이원)={s['fire_rate_mutagen']:.3f} "
          f"무매치율={s['nomatch_rate_mutagen']:.3f} " +
          (f"규칙 MCC={s['rule_MCC']:.3f}" if task == "cls" else f"ΔLD50={s['LD50_delta']:+.3f}"), flush=True)

pd.DataFrame(rows).to_csv(f"{OUT}/g1_rules.csv", index=False)
pd.DataFrame(summ).to_csv(f"{OUT}/g1_summary.csv", index=False)
print(f"\n저장 → g1_rules.csv({len(rows)}행) · g1_summary.csv({len(summ)}행)")
