#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step345_bootstrap.py — [3·4·5단계] 공정한 짝짓기 → 짝지은 부트스트랩 → 동등성(TOST). env: admet.
★재학습 0 — predictions/*.jsonl 만 읽는다.

3단계 공정 짝짓기:
  - G4가 conformer 실패로 사실상 못 본 분자가 있으면 ★G2 예측에서도 같은 분자를 제외 → ★교집합으로만 비교.
  - 교집합 기준으로 지표를 다시 계산해 원래 보고값과 병기(순위가 바뀌면 그 자체가 발견).
4단계 짝지은 부트스트랩:
  - ★같은 test 분자를 2000회 재표집. 매 회 두 모델 지표를 ★같은 표본에서 계산 → Δ.
  - 반감기 Δ = Spearman(G4) − Spearman(G2)   (양수면 G4 우세)
  - 용해도  Δ = MAE(G2) − MAE(G4)            (★MAE는 낮을수록 좋으므로 부호 반대)
  - ★seed 처리 2가지 모두 계산: (A) 5 seed 예측 평균 후 부트스트랩  (B) seed별 부트스트랩 후 평균.
5단계 동등성(TOST): 구분 안 되면 실용적 동등 구간(±0.05)으로 적극 판단.
산출: results/g4_verification.csv · results/g4_verification.json
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

NEW = "/home/nudge/Project/ADMET_integrated/2026-07-24/experiment_g4_verification"
ADME = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_adme_full"
PRED = f"{NEW}/predictions"
NBOOT = 2000
RNG = np.random.default_rng(20260724)
EQ_MARGIN = 0.05        # ★실용적 동등 구간(근거는 notes.md): Spearman ±0.05 · MAE ±0.05

PAIR = {   # endpoint: (G4모델, G2모델, 주지표, G4우세방향)
    "half_life_obach":    ("unimol", "rf_physchem",  "Spearman", "higher"),
    "solubility_aqsoldb": ("unimol", "xgb_physchem", "MAE",      "lower"),
}
MAT = pd.read_csv(f"{ADME}/results/adme_matrix.csv")
CF = json.load(open(f"{NEW}/logs/conformer_failed.json")) if os.path.exists(f"{NEW}/logs/conformer_failed.json") else {}


def load(ep, model):
    """seed별 예측 → {seed: DataFrame(smiles,y_true,y_pred)}"""
    out = {}
    for f in sorted(glob.glob(f"{PRED}/{ep}__{model}__seed*.jsonl")):
        seed = int(f.rsplit("seed", 1)[1].split(".")[0])
        out[seed] = pd.read_json(f, lines=True)
    return out


def met(y, p, prim):
    return (float(spearmanr(y, p).correlation) if prim == "Spearman"
            else float(mean_absolute_error(y, p)))


def reported(ep, model):
    r = MAT[(MAT.endpoint == ep) & (MAT.model == model)]
    return None if r.empty else float(r.iloc[0].value)


rows, detail = [], {}
print("=" * 100)
print("[3·4·5단계] 공정 짝짓기 → 짝지은 부트스트랩 → 동등성. ★재학습 0(저장 예측만)")
print("=" * 100)
for ep, (m4, m2, prim, direction) in PAIR.items():
    P4, P2 = load(ep, m4), load(ep, m2)
    seeds = sorted(set(P4) & set(P2))
    if not seeds:
        print(f"\n[{ep}] ★예측 없음 — skip"); continue
    print(f"\n[{ep}] 주지표 {prim} · 공통 seed {seeds}")

    # ── 3단계: 공정 짝짓기 ──
    cf = CF.get(ep, {})
    cf_smis = {d["smiles"] for d in cf.get("failed", [])}
    # 두 모델이 채점한 분자 집합
    s4 = set(P4[seeds[0]].smiles); s2 = set(P2[seeds[0]].smiles)
    inter = s4 & s2
    jac = len(inter) / len(s4 | s2)
    # ★conformer 실패 분자는 양쪽에서 동일 제외
    keep = sorted(inter - cf_smis)
    print(f"  3단계 짝짓기: G4 {len(s4)} · G2 {len(s2)} · 교집합 {len(inter)}(Jaccard {jac:.4f}) "
          f"· conformer 실패 제외 {len(inter & cf_smis)} → ★비교 대상 {len(keep)}")

    def agg(P, use):
        """seed 평균 예측(use 분자만) → (y, pred_mean), seed별 (y,pred) 목록도 반환"""
        per = []
        for sd in seeds:
            d = P[sd].set_index("smiles").loc[use]
            per.append((d.y_true.to_numpy(float), d.y_pred.to_numpy(float)))
        y = per[0][0]
        pmean = np.mean([p for _, p in per], axis=0)
        return y, pmean, per

    y4, p4m, per4 = agg(P4, keep)
    y2, p2m, per2 = agg(P2, keep)
    assert np.allclose(y4, y2), "정답 불일치 — 분자 정렬 오류"
    y = y4

    # 교집합 기준 지표(원래 보고값과 병기)
    v4_full = float(np.mean([met(*pr, prim) for pr in per4]))     # seed별 지표 평균(교집합)
    v2_full = float(np.mean([met(*pr, prim) for pr in per2]))
    v4_mean = met(y, p4m, prim)                                    # 5seed 예측평균(교집합)
    v2_mean = met(y, p2m, prim)
    rep4, rep2 = reported(ep, m4), reported(ep, m2)
    print(f"  교집합 지표 — G4 {v4_full:.4f}(보고 {rep4:.4f}) · G2 {v2_full:.4f}(보고 {rep2:.4f})"
          f"  [seed별 지표평균]")
    print(f"              G4 {v4_mean:.4f} · G2 {v2_mean:.4f}  [5seed 예측평균]")

    def delta(a4, a2):
        return (a4 - a2) if direction == "higher" else (a2 - a4)   # ★양수면 항상 G4 우세

    # ── 4단계: 짝지은 부트스트랩 ──
    idx = np.arange(len(y))

    def boot(pred4, pred2):
        ds = []
        for _ in range(NBOOT):
            s = RNG.choice(idx, len(idx), replace=True)
            try:
                ds.append(delta(met(y[s], pred4[s], prim), met(y[s], pred2[s], prim)))
            except Exception:
                pass
        v = np.array([d for d in ds if np.isfinite(d)])
        return v

    # (A) 5 seed 예측 평균 후 부트스트랩
    vA = boot(p4m, p2m)
    ciA = (float(np.percentile(vA, 2.5)), float(np.percentile(vA, 97.5)))
    dA, frA = float(vA.mean()), float((vA > 0).mean())
    # (B) seed별 부트스트랩 후 평균
    dsB, ciBs = [], []
    for (yy, pp4), (_, pp2) in zip(per4, per2):
        vb = boot(pp4, pp2)
        dsB.append(float(vb.mean())); ciBs.append((float(np.percentile(vb, 2.5)), float(np.percentile(vb, 97.5))))
    dB = float(np.mean(dsB)); ciB = (float(np.mean([c[0] for c in ciBs])), float(np.mean([c[1] for c in ciBs])))
    frB = float(np.mean([(boot(pp4, pp2) > 0).mean() for (yy, pp4), (_, pp2) in zip(per4[:1], per2[:1])]))

    distinct_A = (ciA[0] > 0) or (ciA[1] < 0)
    distinct_B = (ciB[0] > 0) or (ciB[1] < 0)
    print(f"  4단계 부트스트랩(N={NBOOT}) ★주 판정=(A) 5seed 예측평균:")
    print(f"    (A) Δ={dA:+.4f} CI[{ciA[0]:+.4f},{ciA[1]:+.4f}] · Δ>0 비율 {frA:.3f} → "
          f"{'★구분됨(G4 우세)' if distinct_A and dA>0 else '★구분됨(G2 우세)' if distinct_A else '구분 안 됨'}")
    print(f"    (B) seed별 부트스트랩 평균: Δ={dB:+.4f} CI[{ciB[0]:+.4f},{ciB[1]:+.4f}] → "
          f"{'구분됨' if distinct_B else '구분 안 됨'}")

    # ── 5단계: 동등성(TOST) — 구분 안 될 때 적극 결론 ──
    equiv = (ciA[0] > -EQ_MARGIN) and (ciA[1] < EQ_MARGIN)
    eq_txt = ("★실용적 동등(CI가 ±%.2f 안)" % EQ_MARGIN if equiv else
              "동등도 아님(CI가 동등구간 밖까지 뻗음)")
    print(f"  5단계 동등성(±{EQ_MARGIN}): {eq_txt}")

    verdict = ("G4 우세(구분됨)" if distinct_A and dA > 0 else
               "G2 우세(구분됨)" if distinct_A else
               ("구분 안 됨 · " + ("실용적 동등" if equiv else "판정 보류(동등도 미달)")))
    print(f"  ★판정: {verdict}")

    detail[ep] = dict(metric=prim, direction=direction, seeds=seeds,
                      n_g4=len(s4), n_g2=len(s2), n_intersection=len(inter), jaccard=round(jac, 6),
                      n_conformer_failed_in_test=len(cf_smis),
                      n_conformer_failed_excluded=len(inter & cf_smis), n_compared=len(keep),
                      g4_reported=rep4, g2_reported=rep2,
                      g4_intersection_seedmean=round(v4_full, 4), g2_intersection_seedmean=round(v2_full, 4),
                      g4_intersection_predmean=round(v4_mean, 4), g2_intersection_predmean=round(v2_mean, 4),
                      bootstrap_A=dict(delta=round(dA, 4), ci=[round(ciA[0], 4), round(ciA[1], 4)],
                                       frac_gt0=round(frA, 4), distinguishable=bool(distinct_A)),
                      bootstrap_B=dict(delta=round(dB, 4), ci=[round(ciB[0], 4), round(ciB[1], 4)],
                                       distinguishable=bool(distinct_B)),
                      equivalence_margin=EQ_MARGIN, practically_equivalent=bool(equiv),
                      verdict=verdict, primary_method="A(5seed 예측평균 후 부트스트랩)")
    rows.append(dict(endpoint=ep, metric=prim, n_compared=len(keep),
                     g4_reported=rep4, g4_intersection=round(v4_mean, 4),
                     g2_reported=rep2, g2_intersection=round(v2_mean, 4),
                     delta_A=round(dA, 4), ci_lo=round(ciA[0], 4), ci_hi=round(ciA[1], 4),
                     distinguishable=bool(distinct_A), practically_equivalent=bool(equiv),
                     verdict=verdict, source="predictions/*.jsonl(재학습 0)"))

pd.DataFrame(rows).to_csv(f"{NEW}/results/g4_verification.csv", index=False)
json.dump(detail, open(f"{NEW}/results/g4_verification.json", "w"), ensure_ascii=False, indent=1)
print(f"\n저장 → results/g4_verification.csv · g4_verification.json")
