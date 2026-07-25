#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
collect_matrix.py — ★기존 결과 재사용만(새 학습 0). 독성 4엔드포인트 × 세대(G1~G5) 매트릭스 조립.
모든 셀은 원본 결과 파일에서 실제 인용하고 source 컬럼에 파일명을 남긴다. 없으면 '미측정'.
★공정 전제: 본 표의 모든 값은 ★동일한 TDC 공식 scaffold split·동일 고정 test셋 (실측 확인: seed=42, 교집합 100%).
산출 results/gen_matrix.csv · results/matrix_meta.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

S = Path("/home/nudge/Project/ADMET_structure/2026-06-27")
OUT = Path("/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_generation_matrix/results")
NA = "미측정"

bench = pd.read_csv(S / "experiment_tox_benchmark/results/benchmark.csv")
ft = pd.read_csv(S / "experiment_finetune/results/finetune.csv")
ft2 = pd.read_csv(S / "experiment_finetune2/results/finetune2.csv")
pc = pd.read_csv(S / "experiment_physchem/results/physchem.csv")
aim = json.load(open(S / "experiment_tox_benchmark/results/admetai_metrics.json"))
leak = json.load(open(S / "experiment_tox_benchmark/results/leakage.json"))
ourml = json.load(open(S / "experiment_tox_benchmark/results/ourml_metrics.json"))
ov1 = json.load(open(S / "experiment_finetune/results/summary.json"))["overfit"]
ov2 = json.load(open(S / "experiment_finetune2/results/summary2.json"))["overfit"]
g1s = pd.read_csv(OUT / "g1_summary.csv")
G3DIR = Path("/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_g3_dmpnn_seed42/results")
DMPNN = json.load(open(G3DIR / "dmpnn_metrics.json")) if (G3DIR / "dmpnn_metrics.json").exists() else {}
DM_OVERFIT = json.load(open(G3DIR / "overfit.json")) if (G3DIR / "overfit.json").exists() else []
DM_LEAK = json.load(open(G3DIR / "leakage.json")) if (G3DIR / "leakage.json").exists() else {}

LAB = {"dili": "DILI 간독성", "herg": "hERG 차단", "ames": "AMES 변이원성", "ld50_zhu": "LD50 급성독성"}
MET = {"dili": ["AUROC", "AUPRC"], "herg": ["AUROC", "AUPRC"], "ames": ["AUROC", "AUPRC"], "ld50_zhu": ["MAE"]}
SPLIT = "TDC 공식 scaffold split (seed=42) · 5 seed"

# (세대, 세대이름, 방법, 값열, sd열, 원본df, 파일명)
MODELS = [
    ("G2", "고전 ML", "물리화학 서술자 + XGBoost", "xgbpc_mean", "xgbpc_std", pc, "experiment_physchem/results/physchem.csv"),
    ("G2", "고전 ML", "물리화학 서술자 + RandomForest", "rfpc_mean", "rfpc_std", pc, "experiment_physchem/results/physchem.csv"),
    ("G2", "고전 ML", "ECFP4 지문 + XGBoost", "xgb_mean", None, ft, "experiment_finetune/results/finetune.csv"),
    ("G4", "구조·3D", "Uni-Mol (3D conformer, 2억 사전학습)", "unimol_mean", "unimol_std", ft2, "experiment_finetune2/results/finetune2.csv"),
    ("G5", "파운데이션", "ChemBERTa-2 (SMILES transformer, fine-tune)", "chemberta_mean", "chemberta_std", ft, "experiment_finetune/results/finetune.csv"),
    ("G5", "파운데이션", "MoLFormer (SMILES transformer, fine-tune)", "molformer_mean", "molformer_std", ft, "experiment_finetune/results/finetune.csv"),
]


def hm_se(a, npos, nneg):
    """Hanley-McNeil AUROC 표준오차(해석적) — '이 test 크기에서 몇 점 차가 구분 가능한가'의 참고선."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    v = (a * (1 - a) + (npos - 1) * (q1 - a * a) + (nneg - 1) * (q2 - a * a)) / (npos * nneg)
    return float(np.sqrt(max(v, 0)))


rows = []
for ep in LAB:
    ntest = leak[ep]["n_test"]
    pos = ourml[ep].get("pos_frac")
    for met in MET[ep]:
        hb = (met != "MAE")
        # ── G1: 규칙 (지표 형태 다름 — 억지 AUROC 금지)
        g = g1s[g1s.endpoint == ep].iloc[0]
        gnote = (f"발화율 {g.fire_rate_mutagen:.3f}·무매치율 {g.nomatch_rate_mutagen:.3f}·"
                 + (f"규칙 정밀도 {g.rule_precision:.3f}/재현율 {g.rule_recall:.3f}/MCC {g.rule_MCC:.3f}"
                    if str(g.task) == "cls" else f"발화군 LD50 차 {g.LD50_delta:+.3f}"))
        rows.append(dict(endpoint=ep, label=LAB[ep], metric=met, higher_better=hb,
                         gen="G1", gen_label="규칙", method="구조알림 (BRENK+NIH + Benigni-Bossa SMARTS 10)",
                         value=None, sd=None, n_seed=None, split=SPLIT, n_test=ntest,
                         leak_flag="누수 없음(학습 없음·규칙)", metric_form="★다름(발화율·정밀도·MCC)",
                         source="results/g1_summary.csv (규칙 정의 = T_toxicity/surface/src/featurize.py)",
                         note=gnote))
        # ── G2·G4·G5
        for gen, glab, meth, vc, sc, df, src in MODELS:
            r = df[(df.endpoint == ep) & (df.metric == met)]
            if r.empty or vc not in df.columns or pd.isna(r.iloc[0][vc]):
                rows.append(dict(endpoint=ep, label=LAB[ep], metric=met, higher_better=hb, gen=gen,
                                 gen_label=glab, method=meth, value=None, sd=None, n_seed=None, split=SPLIT,
                                 n_test=ntest, leak_flag="누수 0(실측)", metric_form="동일", source=src,
                                 note=NA)); continue
            r = r.iloc[0]
            sd = float(r[sc]) if (sc and sc in df.columns and pd.notna(r[sc])) else None
            if sd is None and vc == "xgb_mean":       # ECFP-XGB의 SD는 벤치마크 원본에 있음
                b = bench[(bench.endpoint == ep) & (bench.metric == met)]
                sd = float(b.iloc[0]["our_std"]) if not b.empty else None
            rows.append(dict(endpoint=ep, label=LAB[ep], metric=met, higher_better=hb, gen=gen,
                             gen_label=glab, method=meth, value=round(float(r[vc]), 4), sd=sd, n_seed=5,
                             split=SPLIT, n_test=ntest, leak_flag="누수 0(실측·exact overlap 0)",
                             metric_form="동일", source=src, note=""))
        # ── G3: 우리가 직접 학습한 D-MPNN (2026-07-22 experiment_g3_dmpnn_seed42 · 정직 학습)
        dm = DMPNN.get(ep, {})
        if met in dm and dm.get("n_ok"):
            a = dm[met]
            rows.append(dict(endpoint=ep, label=LAB[ep], metric=met, higher_better=hb, gen="G3",
                             gen_label="분자 딥러닝", method="chemprop D-MPNN (우리 자체 학습·정직)",
                             value=round(float(a["test_mean"]), 4), sd=round(float(a["test_std"]), 4),
                             n_seed=dm["n_ok"], split=SPLIT, n_test=ntest,
                             leak_flag="누수 0(실측·train만 학습)", metric_form="동일",
                             source="experiment_g3_dmpnn_seed42/results/dmpnn_metrics.json",
                             note="ADMET-AI와 같은 D-MPNN 계열을 train만 보고 학습 → 누수 없는 G3"))
        else:
            rows.append(dict(endpoint=ep, label=LAB[ep], metric=met, higher_better=hb, gen="G3",
                             gen_label="분자 딥러닝", method="chemprop D-MPNN (우리 자체 학습·정직)", value=None,
                             sd=None, n_seed=None, split=SPLIT, n_test=ntest, leak_flag="—", metric_form="동일",
                             source="experiment_g3_dmpnn_seed42/results/dmpnn_metrics.json",
                             note=f"{NA}(학습 실패 또는 미완료)"))
        # ── G3: ADMET-AI (누수)
        v = aim[ep].get(met)
        rows.append(dict(endpoint=ep, label=LAB[ep], metric=met, higher_better=hb, gen="G3",
                         gen_label="분자 딥러닝", method="ADMET-AI (Chemprop D-MPNN, 공개 사전학습)",
                         value=v, sd=None, n_seed=1, split=SPLIT, n_test=ntest,
                         leak_flag="★누수 의심 (TDC 전체 사전학습)", metric_form="동일",
                         source="experiment_tox_benchmark/results/admetai_metrics.json (오늘 재추론으로 Δ0.0000 재현)",
                         note="누수 보정 불가 — 사유는 matrix_meta.json 참조"))
        # ── 참고: TDC 리더보드 SOTA
        b = bench[(bench.endpoint == ep) & (bench.metric == met)]
        if not b.empty and pd.notna(b.iloc[0]["sota_top"]):
            rows.append(dict(endpoint=ep, label=LAB[ep], metric=met, higher_better=hb, gen="참고",
                             gen_label="리더보드", method=f"TDC SOTA — {b.iloc[0]['sota_model']}",
                             value=float(b.iloc[0]["sota_top"]), sd=None, n_seed=None, split=SPLIT,
                             n_test=ntest, leak_flag="—(외부 인용)", metric_form="동일",
                             source="experiment_tox_benchmark/results/benchmark.csv",
                             note="★미재현 — 리더보드 인용값"))

d = pd.DataFrame(rows)
d.to_csv(OUT / "gen_matrix.csv", index=False)

# ── 표본 수준 불확실성(참고선) ─ 분자별 예측 미저장 → 부트스트랩 불가, 해석적 SE로 대체
unc = {}
for ep in LAB:
    n, pf = leak[ep]["n_test"], ourml[ep].get("pos_frac")
    if MET[ep][0] == "AUROC":
        npos, nneg = int(round(n * pf)), n - int(round(n * pf))
        se = hm_se(0.85, npos, nneg)
        unc[ep] = dict(kind="AUROC", n_test=n, n_pos=npos, n_neg=nneg, se_at_auroc_085=round(se, 4),
                       resolvable_gap_95=round(1.96 * se * np.sqrt(2), 4))
    else:
        # ★회귀 구분 가능선: ADMET-AI 잔차로 계산하면 안 된다 — 그 모델은 누수로 잔차가 비정상적으로 작아
        #   기준선이 과소평가된다(적대검증 지적). 우리 모델들의 MAE 수준으로 스케일해 보정한다.
        p = pd.read_json(OUT / "admetai_preds.jsonl", lines=True)
        p = p[p.endpoint == ep]
        e = (p.y - p.admetai_pred).abs()
        se_ai = float(e.std() / np.sqrt(len(e)))
        ai_mae = float(e.mean())
        ours_mae = float(np.mean([v for v in [
            pc[(pc.endpoint == ep) & (pc.metric == "MAE")].iloc[0][c] for c in ("xgbpc_mean", "rfpc_mean")]
            + [ft[(ft.endpoint == ep) & (ft.metric == "MAE")].iloc[0]["xgb_mean"]]]))
        scale = ours_mae / ai_mae if ai_mae else 1.0
        se = se_ai * scale
        unc[ep] = dict(kind="MAE", n_test=n, se_mae_admetai_ref=round(se_ai, 4),
                       admetai_MAE=round(ai_mae, 4), ours_MAE_mean=round(ours_mae, 4),
                       scale_applied=round(scale, 3), se_scaled=round(se, 4),
                       resolvable_gap_95=round(float(1.96 * se * np.sqrt(2)), 4),
                       note="ADMET-AI 잔차 SE를 우리 모델 MAE 수준으로 스케일한 ★근사치. "
                            "우리 모델의 분자별 예측이 저장돼 있지 않아 직접 계산은 불가(재학습 금지).")

meta = dict(
    split=SPLIT,
    split_verified="T_toxicity(seed=1) vs 본 분할 test 교집합 28%/25% → 다른 분할. seed=42는 100% 일치(실측)",
    leak_ours=leak,
    admetai_leak_evidence={
        "출처": "admet_ai 패키지 resources/data/admet.csv (모델이 학습에 쓴 데이터셋 크기)",
        "DILI": {"admetai_train_size": 475, "our_train_val+test": 379 + 96, "일치": True},
        "hERG": {"admetai_train_size": 648, "our_train_val+test": 523 + 132},
        "AMES": {"admetai_train_size": 7255, "our_train_val+test": 5821 + 1457},
        "LD50_Zhu": {"admetai_train_size": 7342, "our_train_val+test": 5907 + 1478},
        "결론": "ADMET-AI 학습셋 크기 ≈ TDC 전체(train+valid+test) → 우리 test 분자가 학습에 포함됨. "
                "따라서 '학습셋에 없는 분자'로 만든 누수 제거 하위셋 = 공집합 → ★누수 보정 불가(실측 근거).",
        "admetai_자체보고_성능": {"hERG": 0.8388, "AMES": 0.8816, "DILI": 0.8815, "LD50_R2": 0.5959, "LD50_MAE": 0.4515},
    },
    uncertainty=unc,
    uncertainty_caveat="분자별 예측이 원본에 저장돼 있지 않고 재학습은 금지 → 부트스트랩 CI 산출 불가. "
                       "seed SD는 ★고정 test셋 위의 변동이라 일반화 불확실성이 아님(분자 재표집 오차가 훨씬 큼). "
                       "Hanley-McNeil SE는 독립표본 가정의 보수적 참고선.",
    overfit=[o for o in (ov1 + ov2 + DM_OVERFIT) if o["model"] in ("unimol", "chemberta", "molformer", "dmpnn_ours")],
    g3_dmpnn=dict(
        config=DMPNN.get("_config"), failed=DMPNN.get("_failed"),
        split_identity={ep: v.get("test_set_identity_vs_report") for ep, v in DM_LEAK.items()},
        leak={ep: {k: v[k] for k in ("exact_canonical_overlap", "full_inchikey_overlap", "inchikey14_overlap")}
              for ep, v in DM_LEAK.items()},
        note="ADMET-AI와 같은 D-MPNN 계열을 ★train만 보고 학습한 누수 없는 G3. "
             "둘의 격차 = 누수(암기) 프리미엄."),
    n_cells=len(d), n_filled=int(d.value.notna().sum()),
)
json.dump(meta, open(OUT / "matrix_meta.json", "w"), ensure_ascii=False, indent=1)

pd.set_option("display.width", 250)
for ep in LAB:
    m = MET[ep][0]
    sub = d[(d.endpoint == ep) & (d.metric == m)]
    print(f"\n{'='*104}\n[{LAB[ep]}] {m} (n_test={leak[ep]['n_test']}) — {'높을수록 좋음' if m!='MAE' else '낮을수록 좋음'}")
    print(f"{'세대':<5}{'방법':<44}{'값':>9}{'±SD':>8}  누수/비고")
    for _, r in sub.iterrows():
        v = f"{r.value:.4f}" if pd.notna(r.value) else "지표 다름" if r.gen == "G1" else NA
        s = f"±{r.sd:.4f}" if pd.notna(r.sd) else ""
        print(f"{r.gen:<5}{r.method[:43]:<44}{v:>9}{s:>8}  {r.leak_flag}")
print(f"\n저장 → gen_matrix.csv ({len(d)}행, 값 있는 셀 {meta['n_filled']}) · matrix_meta.json")
