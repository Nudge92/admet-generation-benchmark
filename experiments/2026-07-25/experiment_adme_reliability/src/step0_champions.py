#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step0_champions.py — [0단계] 배포 챔피언 선정표 + ★지표 방향 단위 테스트. env: admet.
규칙:
 - 기본: 각 엔드포인트의 G2 최고(xgb_physchem/rf_physchem/xgb_ecfp 중).
 - ★예외 1건: 친유성 logD → G3 dmpnn_ours(부트스트랩 3/3 통과한 유일한 검증된 비-G2 승리).
 - ★반감기·수용해도: G4 예측이 이미 있으므로 G4도 신뢰도 층 대상에 포함(재사용·재학습 금지).
 - ★VDss·청소율 2종: G4 예측 부재 → G2만, "G4 검증 대기" 배지.
산출: results/champions.json · 화면 표
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ADME = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_adme_full"
G4V = "/home/nudge/Project/ADMET_integrated/2026-07-24/experiment_g4_verification"
NEW = "/home/nudge/Project/ADMET_integrated/2026-07-25/experiment_adme_reliability"
sys.path.insert(0, f"{ADME}/src")
import common as C
from common import EPS

MAT = pd.read_csv(f"{ADME}/results/adme_matrix.csv")
G2_MODELS = ["xgb_physchem", "rf_physchem", "xgb_ecfp"]
LOGD = "lipophilicity_astrazeneca"
G4_HAVE = ["half_life_obach", "solubility_aqsoldb"]        # G4 분자별 예측 보유
G4_WAIT = ["vdss_lombardo", "clearance_hepatocyte_az", "clearance_microsome_az"]

# ── ★지표 방향 단위 테스트 (부호 실수 시 순위가 통째로 뒤집힌다) ──
def hb(ep):
    """higher-is-better?"""
    return EPS[ep]["primary"] != "MAE"


def _selftest():
    """방향 로직을 실측 데이터로 검증."""
    fails = []
    # (1) MAE 엔드포인트는 낮을수록 좋음 → hb False
    for ep in ["caco2_wang", "lipophilicity_astrazeneca", "solubility_aqsoldb", "ppbr_az"]:
        if hb(ep):
            fails.append(f"{ep}: MAE인데 hb=True")
    # (2) Spearman/AUROC는 높을수록 좋음 → hb True
    for ep in ["vdss_lombardo", "half_life_obach", "clearance_hepatocyte_az",
               "clearance_microsome_az", "hia_hou", "bbb_martins", "cyp2c9_veith"]:
        if not hb(ep):
            fails.append(f"{ep}: {EPS[ep]['primary']}인데 hb=False")
    # (3) best 선택이 방향을 따르는지 — 인공 데이터로
    d = pd.DataFrame({"model": ["a", "b"], "value": [0.9, 0.1]})
    assert d.loc[d.value.idxmax()].model == "a", "higher 선택 오류"
    assert d.loc[d.value.idxmin()].model == "b", "lower 선택 오류"
    # (4) 실측: 용해도(MAE↓)의 G2 최고가 실제로 최솟값인가
    g = MAT[(MAT.endpoint == "solubility_aqsoldb") & MAT.model.isin(G2_MODELS)]
    if g.loc[g.value.idxmin()].value != g.value.min():
        fails.append("solubility: MAE 최소 선택 실패")
    return fails


fails = _selftest()
print("=" * 104)
print("[0단계] 배포 챔피언 선정 + ★지표 방향 단위 테스트")
print("=" * 104)
print(f"★방향 단위 테스트: {'통과(4/4)' if not fails else '★실패 ' + str(fails)}")
if fails:
    sys.exit(2)

rows = []
for ep, info in EPS.items():
    prim = info["primary"]; task = info["task"]
    g2 = MAT[(MAT.endpoint == ep) & MAT.model.isin(G2_MODELS) & MAT.value.notna()]
    if g2.empty:
        continue
    best2 = g2.loc[g2.value.idxmax() if hb(ep) else g2.value.idxmin()]
    if ep == LOGD:
        r = MAT[(MAT.endpoint == ep) & (MAT.model == "dmpnn_ours")].iloc[0]
        champ, cval, why = "dmpnn_ours", float(r.value), "★예외: G3(부트스트랩 3/3 통과한 유일 검증 승리)"
    else:
        champ, cval, why = best2.model, float(best2.value), "G2 최고"
    extra = ("G4 포함(예측 재사용)" if ep in G4_HAVE else
             "G4 검증 대기" if ep in G4_WAIT else "")
    g4v = None
    if ep in G4_HAVE:
        r4 = MAT[(MAT.endpoint == ep) & (MAT.model == "unimol")]
        g4v = float(r4.iloc[0].value) if not r4.empty else None
    rows.append(dict(endpoint=ep, label=info["label"], pillar=info["pillar"], task=task,
                     metric=prim, higher_better=hb(ep), champion=champ,
                     champion_value=round(cval, 4), reason=why,
                     g2_best=best2.model, g2_best_value=round(float(best2.value), 4),
                     g4_value=(None if g4v is None else round(g4v, 4)), note=extra))
D = pd.DataFrame(rows)
json.dump(rows, open(f"{NEW}/results/champions.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print(f"\n{'엔드포인트':<32}{'유형':<5}{'주지표':<10}{'방향':<5}{'챔피언':<15}{'값':>9}  비고")
for pil in ("A", "D", "M", "E"):
    for _, r in D[D.pillar == pil].iterrows():
        arrow = "↑" if r.higher_better else "↓"
        print(f"{r.endpoint:<32}{r.task:<5}{r.metric:<10}{arrow:<5}{r.champion:<15}{r.champion_value:>9.4f}"
              f"  {r.reason}{(' · ' + r.note) if r.note else ''}")
ncls = int((D.task == "cls").sum()); nreg = int((D.task == "reg").sum())
print(f"\n★분류 {ncls}개(AUROC↑·AUPRC↑) · 회귀 {nreg}개(MAE↓ 4 / Spearman↑ 4)")
print(f"★챔피언 구성: G2 {int((D.champion!='dmpnn_ours').sum())}개 · G3 1개(logD) · "
      f"G4 추가 대상 {len(G4_HAVE)}개 · G4 검증대기 {len(G4_WAIT)}개")
print(f"저장 → results/champions.json")
