#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_report.py — 확장 3종(발암성·ClinTox·Tox21) G1~G3 사다리 보고서. 단일 HTML(그림 base64).
★플래그십(experiment_generation_matrix) 템플릿·방법론 계승: 지표 패널·구분 가능선·누수 실측·출처 표기.
★새 계산 없음(조립만).
"""
import base64, io, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path("/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_gen_expansion_g1g3")
RES = ROOT / "results"
FLAG = Path("/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_generation_matrix/results")

for p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
          "/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf"):
    if Path(p).exists():
        fm.fontManager.addfont(p)
plt.rcParams["font.family"] = ["NanumGothic", "NanumSquareRound", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

M = json.load(open(RES / "expansion_metrics.json"))
META = M.pop("_meta")
OV = json.load(open(RES / "overfit.json"))
G1R = pd.read_csv(RES / "g1_rules.csv")
FLAGM = pd.read_csv(FLAG / "gen_matrix.csv") if (FLAG / "gen_matrix.csv").exists() else None
EPLAB = {"Carcinogens_Lagunin": "발암성 (Carcinogens, Lagunin)", "ClinTox": "ClinTox 임상독성",
         "Tox21": "Tox21 12경로 (NR 7 · SR 5)"}
G2LAB = {"xgb_physchem": "물리화학 서술자 + XGBoost", "rf_physchem": "물리화학 서술자 + RandomForest",
         "xgb_ecfp": "ECFP4 지문 + XGBoost"}
CG2, CG3 = "#2a9d8f", "#e76f51"


def b64(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=125, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def img(fig, cap):
    return f'<figure><img src="data:image/png;base64,{b64(fig)}"><figcaption>{cap}</figcaption></figure>'


def ours_vals(r):
    """G2 3종 + G3 정직 → [(gen, label, mean, std)]  (ADMET-AI 제외)"""
    v = []
    for mk, lab in G2LAB.items():
        d = r["G2"].get(mk)
        if d:
            v.append(("G2", lab, d["AUROC"]["mean"], d["AUROC"]["std"]))
    if r["G3_ours"]:
        v.append(("G3", "chemprop D-MPNN (정직)", r["G3_ours"]["AUROC"]["mean"], r["G3_ours"]["AUROC"]["std"]))
    return v


def gen_gap(r):
    """★판정 통계량 = 1위 세대 vs 2위 세대 격차 (플래그십과 동일 정정).
    range(최고−최저)는 '가장 나쁜 세대와의 폭'이라 1위의 우위를 함의하지 않는다."""
    v = ours_vals(r)
    if not v:
        return None, {}
    best = {}
    for g, _, m, _ in v:
        best[g] = max(best.get(g, -1), m)
    if len(best) < 2:
        return 0.0, best
    srt = sorted(best.values(), reverse=True)
    return float(srt[0] - srt[1]), best


H = ['<meta charset="utf-8"><title>독성 세대 사다리 확장 — 발암성·ClinTox·Tox21 (G1~G3)</title>', """<style>
body{font-family:'Nanum Gothic',system-ui,sans-serif;max-width:1150px;margin:0 auto;padding:24px 28px;color:#1d2129;line-height:1.62}
h1{font-size:26px;border-bottom:3px solid #2a9d8f;padding-bottom:10px;margin-bottom:4px}
h2{font-size:20px;margin-top:38px;border-left:5px solid #2a9d8f;padding-left:11px}
h3{font-size:16px;margin-top:24px;color:#264653}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}
th,td{border:1px solid #d8dde3;padding:5px 8px;text-align:left}
th{background:#eef2f5;font-weight:700}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.best{background:#e8f6f1;font-weight:700}
.leak{background:#fff1ec;color:#c1440e;font-weight:700}
.na{color:#98a2ab}
.box{background:#f7f9fb;border:1px solid #dfe5ea;border-radius:8px;padding:14px 18px;margin:16px 0}
.warn{background:#fff8ec;border:1px solid #f0d9a8;border-radius:8px;padding:14px 18px;margin:16px 0}
.crit{background:#fdefea;border:1px solid #f2bda9;border-radius:8px;padding:14px 18px;margin:16px 0}
figure{margin:18px 0;text-align:center}
img{max-width:100%;border:1px solid #e3e8ec;border-radius:6px}
figcaption{font-size:12.5px;color:#5b6670;margin-top:7px}
small,.src{font-size:11.5px;color:#6b7580}
code{background:#eef2f5;padding:1px 5px;border-radius:3px;font-size:12px}
ul{margin:8px 0 8px 4px}li{margin:4px 0}
</style>"""]

H.append("<h1>독성 세대 사다리 확장 — 발암성 · ClinTox · Tox21 (G1~G3)</h1>")
H.append("<p class='src'>작업일 2026-07-22 · 스코프 <b>G1~G3만</b>(G4·G5 파인튜닝 없음) · "
         "플래그십(핵심 4종 DILI/hERG/AMES/LD50)과 <b>동일 방법론</b> — 정직 학습(train만)·ADMET-AI 누수 실측·"
         "구분 가능선·셀별 출처 표기.</p>")

# ── 0. 지표 패널
H.append("<h2>0. 지표를 어떻게 읽는가</h2><div class='box'><ul>"
         "<li><b>AUROC</b> — 무작위 양성/음성 한 쌍에서 양성을 더 높게 줄 확률. 0.5=무작위. 불균형에 둔감해 <b>단독으론 낙관적</b>.</li>"
         "<li><b>AUPRC</b> — 기저선이 <b>양성비율</b>. 이번 확장은 양성이 매우 희박해(Tox21 2.6~25%, ClinTox 8%) "
         "<b>AUPRC를 기저선과 함께</b> 봐야 한다.</li>"
         "<li><b>MCC</b>(G1 전용) — 작동점 기반 상관. 0 근처면 규칙이 사실상 무력.</li>"
         "<li><b>G1은 지표 형태가 다르다</b> — 규칙은 확률이 아니라 이진 판정이라 AUROC를 억지로 만들지 않고 "
         "<b>발화율·무매치율·알림별 정밀도·작동점 MCC</b>로만 보고한다.</li>"
         "<li><b>구분 가능선</b> — 이 test 크기에서 95%로 구분 가능한 최소 AUROC 격차(Hanley-McNeil). "
         "이보다 작은 차이는 <b>순위를 주장하면 안 된다</b>.</li>"
         "<li><b>★누수</b> — ADMET-AI는 TDC 전체로 사전학습돼 이 test 분자를 이미 봤다. 그 값은 실력이 아니다.</li></ul></div>")

# ── 1. 설계
H.append("<h2>1. 무엇을 어떻게 비교했나</h2>")
H.append("<table><tr><th>세대</th><th>모델</th><th>비고</th></tr>"
         "<tr><td><b>G1</b></td><td>구조알림 (BRENK+NIH + Benigni-Bossa SMARTS 10)</td>"
         "<td>학습 없음 · 플래그십과 <b>같은 규칙셋</b> · 지표 형태 다름</td></tr>"
         "<tr><td rowspan=3><b>G2</b></td><td>물리화학 서술자(RDKit 210) + XGBoost</td><td rowspan=3>"
         "플래그십과 동일 3종·동일 하이퍼파라미터 · 5 seed</td></tr>"
         "<tr><td>물리화학 서술자 + RandomForest(500)</td></tr><tr><td>ECFP4(2048) + XGBoost</td></tr>"
         "<tr><td rowspan=2><b>G3</b></td><td><b>chemprop D-MPNN (우리 자체·정직)</b></td>"
         "<td>train만 학습 · valid로만 early stopping · test 1회 · Tox21은 <b>멀티태스크</b>(결측 마스킹)</td></tr>"
         "<tr><td>ADMET-AI (같은 D-MPNN 계열·공개)</td><td><b>★누수 의심</b> — 고정 비교선으로만 사용</td></tr></table>")

lk = META["leakage"]
H.append("<div class='box'><b>분할·공정성(실측)</b><table><tr><th>엔드포인트</th><th>분할</th>"
         "<th class='n'>train/valid/test</th><th class='n'>train∩test 정확분자</th>"
         "<th class='n'>chemprop 비호환 제외</th><th class='n'>재현</th></tr>")
for ep in M:
    v = lk[ep]
    H.append(f"<tr><td>{EPLAB[ep]}</td><td class='src'>{META['split'][ep]}</td>"
             f"<td class='n'>{v['n_train']}/{v['n_valid']}/{v['n_test']}</td>"
             f"<td class='n'><b>{v['exact_canonical_overlap_trainval_test']}</b></td>"
             f"<td class='n'>{sum(v['chemprop_incompatible_dropped'].values())}</td>"
             f"<td class='n'>{'OK' if v['deterministic_regeneration'] else '★실패'}</td></tr>")
H.append("</table><small>분할은 <code>prep_splits.py</code>가 <b>한 번만</b> 만들어 CSV로 고정했고 "
         "G1·G2·G3·ADMET-AI가 <b>모두 그 파일만</b> 읽는다 → 엔드포인트 내 test 동일성이 구조적으로 보장된다. "
         "두 env의 RDKit 버전 차이로 chemprop이 못 읽는 분자는 <b>분할 이후</b> 모든 세대에서 함께 제외했다"
         "(플래그십 <code>chemprop_bad_smiles</code> 방식 계승).</small></div>")
H.append("<div class='warn'><b>★Tox21 분할에 대한 정직한 고지</b> — Tox21 12과제는 서로 다른 분자 부분집합이라 "
         "'과제별 TDC 공식 분할'로는 멀티태스크 D-MPNN과 과제별 G2를 <b>같은 test에서</b> 비교할 수 없다. "
         "그래서 12과제 <b>합집합</b> 분자셋에 TDC 내부 함수 <code>create_scaffold_split(seed=42)</code>를 "
         "<b>한 번</b> 적용했다. 방법론(Bemis-Murcko)은 같지만 <b>TDC 과제별 공식 분할과는 다르므로</b> "
         "이 Tox21 숫자를 TDC 리더보드와 직접 비교하면 안 된다.</div>")

# ── 2. 엔드포인트별 사다리
H.append("<h2>2. 엔드포인트별 세대 사다리</h2>")
concl = []
for ep in M:
    tasks = list(M[ep].keys())
    H.append(f"<h3>{EPLAB[ep]}</h3>")
    multi = len(tasks) > 1

    # 표
    H.append("<table><tr><th>과제</th><th>세대</th><th>모델</th><th class='n'>AUROC ±SD</th>"
             "<th class='n'>AUPRC</th><th class='n'>ADMET-AI 대비</th><th>플래그</th><th>출처</th></tr>")
    for lab in tasks:
        r = M[ep][lab]
        v = ours_vals(r)
        best = max([x[2] for x in v]) if v else None
        ai = r["G3_admetai"].get("AUROC")
        rows = []
        g = r["G1"]
        if g:
            rows.append(("G1", "구조알림 규칙", None, None, None,
                         f"발화율 {g['fire_rate']:.3f}·무매치 {g['nomatch_rate']:.3f}·정밀도 {g['precision']:.3f}"
                         f"·MCC {g['MCC']:+.3f} → <b>{g['verdict']}</b>", "results/g1_summary.csv", "다름"))
        for mk, l2 in G2LAB.items():
            d = r["G2"].get(mk)
            rows.append(("G2", l2, d["AUROC"]["mean"] if d else None, d["AUROC"]["std"] if d else None,
                         d["AUPRC"]["mean"] if d else None, "누수 0(실측)",
                         d["source"] if d else "—", "동일"))
        d = r["G3_ours"]
        rows.append(("G3", "chemprop D-MPNN (정직)", d["AUROC"]["mean"] if d else None,
                     d["AUROC"]["std"] if d else None, d["AUPRC"]["mean"] if d else None,
                     "누수 0(train만 학습)", d["source"] if d else "—", "동일"))
        if ai is not None:
            rows.append(("G3", "ADMET-AI (누수)", ai, None, r["G3_admetai"]["AUPRC"],
                         r["G3_admetai"]["leak_flag"], r["G3_admetai"]["source"], "동일"))
        else:
            rows.append(("G3", "ADMET-AI", None, None, None,
                         r["G3_admetai"].get("note", "미커버"), "—", "동일"))
        for i, (gen, mdl, mv, sd, pr, flag, src, mf) in enumerate(rows):
            first = f"<td rowspan={len(rows)}>{lab}<br><small>n={r['n_test']}·양성 {r['pos_rate']:.3f}<br>구분선 {r['resolvable_gap_95']}</small></td>" if i == 0 else ""
            if mv is None and gen == "G1":
                cells = f"<td class='na' colspan=3>지표 형태 다름 →</td>"
            elif mv is None:
                cells = "<td class='na' colspan=3>N/A</td>"
            else:
                cls = " class='leak'" if "ADMET-AI" in mdl else (" class='best'" if mv == best else "")
                gap = "" if ai is None else f"{mv - ai:+.4f}"
                cells = (f"<td class='n'{cls}>{mv:.4f}" + (f" ±{sd:.4f}" if sd is not None else "") + "</td>"
                         f"<td class='n'>{'' if pr is None else f'{pr:.4f}'}</td><td class='n'>{gap}</td>")
            H.append(f"<tr>{first}<td>{gen}</td><td>{mdl}</td>{cells}<td class='src'>{flag}</td>"
                     f"<td class='src'>{src}</td></tr>")
    H.append("</table>")

    # 그림
    if multi:
        fig, ax = plt.subplots(figsize=(11.5, 4.0))
        x = np.arange(len(tasks)); w = 0.2
        series = [(mk, G2LAB[mk]) for mk in G2LAB] + [("G3", "chemprop D-MPNN (정직)")]
        for j, (mk, l2) in enumerate(series):
            ys = []
            for lab in tasks:
                r = M[ep][lab]
                d = r["G3_ours"] if mk == "G3" else r["G2"].get(mk)
                ys.append(d["AUROC"]["mean"] if d else np.nan)
            ax.bar(x + (j - 1.5) * w, ys, w, label=l2, color=(CG3 if mk == "G3" else None),
                   edgecolor="#2b2b2b", linewidth=.4)
        aiy = [M[ep][lab]["G3_admetai"].get("AUROC", np.nan) for lab in tasks]
        ax.plot(x, aiy, "o--", color="#c1440e", lw=1.6, ms=5, label="ADMET-AI (★누수 의심)")
        ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=8.5)
        ax.set_ylabel("AUROC (높을수록 좋음)", fontsize=9.5); ax.set_ylim(0.4, 1.0)
        ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=.25)
        ax.set_title(f"{EPLAB[ep]} — 과제별 세대 비교", fontsize=11.5)
        H.append(img(fig, f"{EPLAB[ep]}. 막대=우리가 학습한 모델(G2 3종·G3 정직), 붉은 점선=ADMET-AI(★누수 의심). "
                          "G1은 지표 형태가 달라 막대에 넣지 않았다(위 표 참조)."))
    else:
        r = M[ep][tasks[0]]
        v = ours_vals(r)
        fig, ax = plt.subplots(figsize=(8.0, 3.5))
        x = np.arange(len(v))
        ax.bar(x, [t[2] for t in v], yerr=[t[3] for t in v], capsize=3,
               color=[CG2 if t[0] == "G2" else CG3 for t in v], edgecolor="#2b2b2b", linewidth=.6)
        ai = r["G3_admetai"].get("AUROC")
        if ai:
            ax.axhline(ai, ls="--", lw=1.8, color="#c1440e")
            ax.text(len(v) - .45, ai, f" ADMET-AI {ai:.3f}\n ★누수 의심", color="#c1440e",
                    fontsize=8.5, va="bottom", ha="right")
        ax.set_xticks(x); ax.set_xticklabels([f"{t[0]}\n{t[1][:18]}" for t in v], fontsize=8)
        ax.set_ylabel("AUROC (높을수록 좋음)", fontsize=9.5)
        lo = min([t[2] for t in v] + ([ai] if ai else [])) - .06
        hi = max([t[2] for t in v] + ([ai] if ai else [])) + .06
        ax.set_ylim(lo, hi); ax.grid(axis="y", alpha=.25)
        ax.set_title(f"{EPLAB[ep]} — 세대별 AUROC", fontsize=11.5)
        H.append(img(fig, f"{EPLAB[ep]}. 오차막대=5 seed SD. 붉은 점선=ADMET-AI(★누수 의심). "
                          f"이 test 크기의 95% 구분 가능 격차 ≈ {r['resolvable_gap_95']}."))

    # 결론 한 줄
    if multi:
        gaps = [(lab, *gen_gap(M[ep][lab])) for lab in tasks]
        n_res = sum(1 for _, g, _ in gaps if g is not None and g > M[ep][_l]["resolvable_gap_95"]
                    for _l in [_]) if False else sum(
            1 for lab, g, _ in gaps if g is not None and M[ep][lab]["resolvable_gap_95"] and g > M[ep][lab]["resolvable_gap_95"])
        winners = {}
        for lab in tasks:
            v = ours_vals(M[ep][lab])
            if v:
                w = max(v, key=lambda t: t[2])
                winners[w[0]] = winners.get(w[0], 0) + 1
        c = (f"<b>{EPLAB[ep]}</b> — 12과제 중 최고 세대: " +
             " · ".join(f"{k} {n}개" for k, n in sorted(winners.items())) +
             f". 1위·2위 세대 격차가 구분 가능선을 넘는 과제는 <b>{n_res}/{len(tasks)}개</b>. " +
             ("G1 규칙은 <b>12/12 과제 모두 무력</b>(|MCC| &lt; 0.05 수준)." if all(
                 abs(M[ep][lab]["G1"]["MCC"]) < 0.06 for lab in tasks if M[ep][lab]["G1"]) else ""))
    else:
        lab = tasks[0]; r = M[ep][lab]
        g, best = gen_gap(r)
        v = ours_vals(r); w = max(v, key=lambda t: t[2]) if v else None
        rg = r["resolvable_gap_95"]
        c = (f"<b>{EPLAB[ep]}</b> — 우리 모델 중 최고는 <b>{w[0]} {w[1]}</b> ({w[2]:.4f}). "
             f"1위·2위 세대 격차 {g:.3f} (세대별 최고: " + " · ".join(f"{k} {x:.3f}" for k, x in sorted(best.items())) +
             f")는 구분 가능선 {rg:.3f}보다 " +
             ("<b>작다 → 세대 순위 주장 불가</b>. " if (rg and g <= rg) else "<b>크다 → 세대 차이가 실재할 여지</b>. ") +
             f"G1 규칙은 MCC {r['G1']['MCC']:+.3f} = <b>{r['G1']['verdict']}</b>.")
    concl.append(c)
    H.append(f"<div class='box'>{c}</div>")

# ── 3. 누수 프리미엄
H.append("<h2>3. ADMET-AI와의 격차 — <u>누수의 상한</u> (핵심 4종의 0.088이 여기서도 재현되나)</h2>")
H.append("<div class='warn'><b>★두 시스템은 '같은 모델'이 아니다(패키지 실측)</b> — 아래 격차를 전부 누수로 귀속하면 안 된다."
         "<table><tr><th></th><th>ADMET-AI</th><th>우리 D-MPNN</th></tr>"
         "<tr><td>모델 수</td><td class='leak'>5-모델 앙상블</td><td>seed당 단일 모델(지표 평균±SD)</td></tr>"
         "<tr><td>과제 수</td><td class='leak'>분류 31과제 멀티태스크</td><td>발암성·ClinTox 단일 / Tox21 12과제</td></tr>"
         "<tr><td>하이퍼파라미터</td><td class='leak'>튜닝된 배포본</td><td>탐색 없음(epochs 50·batch 50)</td></tr>"
         "<tr><td>학습 분자</td><td class='leak'>TDC 전체(우리 test 포함)</td><td>train 파티션만</td></tr></table>"
         "<small>출처 <code>admet_ai/resources/models/admet_classification/model_0..4.pt</code>(5개) · 체크포인트 "
         "<code>output_columns</code> 길이 31 · <code>_make_ensemble_predictions</code>.</small><br><br>"
         "따라서 격차에는 <b>누수 + 앙상블 + 멀티태스크 + 튜닝</b>이 교락돼 있고 이는 "
         "<b>누수의 크기가 아니라 상한(upper bound)</b>이다. 플래그십에도 같은 정정을 적용했다.</div>")
H.append("<table><tr><th>엔드포인트</th><th>과제</th><th class='n'>우리 D-MPNN(정직)</th>"
         "<th class='n'>ADMET-AI(누수)</th><th class='n'>격차 AUROC(=누수 상한)</th>"
         "<th class='n'>ADMET-AI 학습 size</th></tr>")
prem = []
for ep in M:
    for lab, r in M[ep].items():
        lp = r.get("leak_premium")
        if not lp:
            H.append(f"<tr><td>{EPLAB[ep]}</td><td>{lab}</td><td colspan=4 class='na'>"
                     f"{r['G3_admetai'].get('note', 'N/A')}</td></tr>")
            continue
        prem.append(lp["AUROC"])
        H.append(f"<tr><td>{EPLAB[ep]}</td><td>{lab}</td>"
                 f"<td class='n'>{r['G3_ours']['AUROC']['mean']:.4f}</td>"
                 f"<td class='n leak'>{r['G3_admetai']['AUROC']:.4f}</td>"
                 f"<td class='n leak'>{lp['AUROC']:+.4f}</td>"
                 f"<td class='n'>{r['G3_admetai'].get('admetai_train_size')}</td></tr>")
H.append("</table>")
if prem:
    mp, mn, mx = float(np.mean(prem)), float(np.min(prem)), float(np.max(prem))
    H.append(f"<div class='warn'><b>결과</b> — 확장 {len(prem)}개 과제의 격차(누수 상한) 평균 <b>{mp:+.4f} AUROC</b> "
             f"(범위 {mn:+.4f} ~ {mx:+.4f}). 플래그십 핵심 4종(분류 3개)의 상한 평균 <b>+0.088</b>과 "
             + ("<b>같은 방향으로 재현</b>된다" if mp > 0.02 else "방향은 같으나 크기가 작다") +
             f". 프리미엄이 음수인 과제가 {sum(1 for p in prem if p < 0)}개 있다"
             " — 누수가 항상 이득으로 나타나는 건 아니며(과제 난이도·라벨 정의 차이), 그래서 "
             "<b>이 값은 '누수의 상한 추정'이 아니라 '관측된 격차'로만</b> 읽어야 한다.</div>")

# ── 4. 구분 가능선
H.append("<h2>4. 이 비교로 어디까지 말할 수 있는가</h2>")
H.append("<table><tr><th>엔드포인트</th><th>과제</th><th class='n'>n_test</th><th class='n'>양성수</th>"
         "<th class='n'>구분 가능선(95%)</th><th class='n'>1위−2위 세대 격차</th><th>판정</th></tr>")
for ep in M:
    for lab, r in M[ep].items():
        g, _ = gen_gap(r); rg = r["resolvable_gap_95"]
        ver = "N/A" if (g is None or rg is None) else ("<b>구분 불가</b>" if g <= rg else
              ("경계선" if g < rg * 1.2 else "구분 가능"))
        H.append(f"<tr><td>{EPLAB[ep]}</td><td>{lab}</td><td class='n'>{r['n_test']}</td>"
                 f"<td class='n'>{r['n_pos']}</td><td class='n'>{rg}</td>"
                 f"<td class='n'>{'' if g is None else f'{g:.4f}'}</td><td>{ver}</td></tr>")
H.append("</table>")
H.append(f"<div class='warn'><b>★한계</b> — {META['uncertainty_caveat']} "
         "Hanley-McNeil은 <b>비대응 가정</b>이라 같은 test 위 두 모델 비교엔 지나치게 보수적이다 → "
         "'이보다 작은 차이는 확실히 말할 수 없다'는 <b>하한 경보</b>로만 쓴다. "
         "특히 <b>발암성은 test가 56분자·양성 11개</b>뿐이라 구분 가능선이 매우 크다 → "
         "<b>발암성에서 세대 순위를 주장하면 안 된다</b>.</div>")

# ── 5. 과적합
H.append("<h2>5. 과적합 점검 (train vs test · 새로 학습한 G2·G3 전부)</h2>")
od = pd.DataFrame([o for o in OV if o["metric"] == "AUROC"])
agg = od.groupby("model")[["train", "test", "gap"]].mean().round(4)
H.append("<table><tr><th>모델</th><th class='n'>train 평균</th><th class='n'>test 평균</th>"
         "<th class='n'>평균 격차</th><th class='n'>최대 격차</th><th>해석</th></tr>")
for mk, r in agg.iterrows():
    mx = float(od[od.model == mk]["gap"].max())
    itp = ("★train 거의 완전 암기 — 용량이 데이터를 크게 넘어섬" if r["train"] > 0.99 else
           "과적합 큼" if r["gap"] > 0.15 else "중간" if r["gap"] > 0.05 else "★작음")
    H.append(f"<tr><td>{G2LAB.get(mk, 'chemprop D-MPNN (정직)')}</td><td class='n'>{r['train']:.4f}</td>"
             f"<td class='n'>{r['test']:.4f}</td><td class='n'>{r['gap']:.4f}</td>"
             f"<td class='n'>{mx:.4f}</td><td>{itp}</td></tr>")
H.append("</table><p><small>전체 행은 <code>results/overfit.json</code>. AUROC 기준·과제별 평균.</small></p>")

# ── 6. G1 상세
H.append("<h2>6. G1(구조알림) — 어디서 작동하고 어디서 무력한가</h2>")
H.append("<table><tr><th>엔드포인트</th><th>과제</th><th class='n'>발화율</th><th class='n'>무매치율</th>"
         "<th class='n'>정밀도</th><th class='n'>기저 양성률</th><th class='n'>lift</th>"
         "<th class='n'>MCC</th><th>판정</th></tr>")
for ep in M:
    for lab, r in M[ep].items():
        g = r["G1"]
        if not g:
            continue
        lift = g["precision"] / r["pos_rate"] if r["pos_rate"] else None
        H.append(f"<tr><td>{EPLAB[ep]}</td><td>{lab}</td><td class='n'>{g['fire_rate']:.3f}</td>"
                 f"<td class='n'>{g['nomatch_rate']:.3f}</td><td class='n'>{g['precision']:.3f}</td>"
                 f"<td class='n'>{r['pos_rate']:.3f}</td>"
                 f"<td class='n'>{'' if lift is None else f'{lift:.2f}×'}</td>"
                 f"<td class='n'>{g['MCC']:+.3f}</td><td>{g['verdict']}</td></tr>")
H.append("</table>")
top = G1R[(G1R.endpoint == "Carcinogens_Lagunin") & G1R.precision.notna() & (G1R.n_fired >= 5)] \
    .sort_values("precision", ascending=False).head(6)
H.append("<h3>발암성에서 실제로 작동한 알림 (상위)</h3><table>"
         "<tr><th>알림</th><th class='n'>발화 분자수</th><th class='n'>정밀도</th><th class='n'>기저 대비 lift</th></tr>")
for _, r in top.iterrows():
    H.append(f"<tr><td>{r.alert}</td><td class='n'>{int(r.n_fired)}</td><td class='n'>{r.precision:.3f}</td>"
             f"<td class='n'>{'' if pd.isna(r.lift) else f'{r.lift:.2f}×'}</td></tr>")
H.append("</table><p><small>전체 182행은 <code>results/g1_rules.csv</code>.</small></p>")
H.append("<div class='box'><b>패턴</b> — 같은 규칙셋이 엔드포인트마다 전혀 다르게 작동한다. "
         "Benigni-Bossa 계열은 원래 <b>변이원·발암 알림</b>이라 발암성에서 가장 잘 듣고, "
         "임상 실패(ClinTox)·수용체 경로(Tox21)처럼 <b>기전이 다른 독성</b>에서는 무력하다. "
         "플래그십에서 AMES·DILI엔 듣고 hERG엔 무력했던 것과 <b>같은 이야기</b>다.</div>")

# ── 7. 부록
H.append("<h2>7. 부록 — 기존 실험(다른 분할) <span class='src'>본 표와 직접 비교 불가</span></h2>")
H.append("<div class='warn'>기존 <code>experiment_clintox_benchmark</code>는 train/valid/test 개수가 "
         "<b>1034/147/297로 우리와 완전히 같지만</b>, 분자 단위 교집합은 <b>Jaccard 0.134</b>에 불과하다 "
         "— <b>개수가 같다고 같은 분할이 아니다</b>(플래그십에서도 같은 함정을 겪었다). "
         "따라서 본 표에 섞지 않았다. <code>experiment_tox_expand</code>의 Tox21 결과도 분할 출처가 "
         "본 실험과 다르므로 참고로만 둔다.</div>")

# ── 8. 종합
H.append("<h2>8. 종합 — '세대 ≠ 성능'이 얼마나 넓게 성립하나</h2>")
tot, byg = 0, {}
for ep in M:
    for lab, r in M[ep].items():
        v = ours_vals(r)
        if not v:
            continue
        tot += 1
        w = max(v, key=lambda t: t[2])[0]
        byg[w] = byg.get(w, 0) + 1
flag_line = ""
if FLAGM is not None:
    f4 = []
    for epn, met in [("dili", "AUROC"), ("herg", "AUROC"), ("ames", "AUROC"), ("ld50_zhu", "MAE")]:
        s = FLAGM[(FLAGM.endpoint == epn) & (FLAGM.metric == met) & FLAGM.value.notna()
                  & (FLAGM.gen != "참고") & ~FLAGM.method.str.contains("ADMET-AI")]
        if s.empty:
            continue
        b = s.loc[s.value.idxmax() if met != "MAE" else s.value.idxmin()]
        f4.append(f"{epn}={b.gen}")
    flag_line = ("플래그십 핵심 4종의 최고 세대는 " + " · ".join(f4) +
                 " (G2 3개 · G4 1개) 였다. ")
# ★구분 가능한 세대 효과가 있는 과제를 실측 산출(있으면 방향까지)
_res_tasks, _res_dirs = [], []
for _ep in M:
    for _lab, _r in M[_ep].items():
        _g, _b = gen_gap(_r); _rg = _r["resolvable_gap_95"]
        if _g is None or not _rg or _g <= _rg:
            continue
        _win = max(_b.items(), key=lambda t: t[1])[0]
        _lose = min(_b.items(), key=lambda t: t[1])[0]
        _res_tasks.append(f"{_lab}({_win} {_b[_win]:.3f} &gt; {_lose} {_b[_lose]:.3f})")
        _res_dirs.append((_win, _lose))
_dir_txt = (" · ".join(f"<b>{w}가 {l}를 이김</b> {c}건"
                       for (w, l), c in sorted(__import__("collections").Counter(_res_dirs).items(),
                                               key=lambda t: -t[1])) or "해당 없음")
_res_html = ("<p><b>★이 프로젝트에서 처음으로 '구분 가능한 세대 효과'가 나왔다 — 그런데 방향이 통념과 반대다.</b> "
             f"확장 {tot}개 과제 중 <b>{len(_res_tasks)}개</b>에서 1위·2위 세대 격차가 구분 가능선을 넘는다: "
             + " · ".join(_res_tasks) +
             f". 방향은 {_dir_txt}. "
             "플래그십 핵심 4종에서는 구분 가능한 세대 효과가 0/4였는데, 표본이 큰 Tox21 과제들에서 "
             "비로소 신호가 잡혔고 그 신호는 <b>'최신 세대가 낫다'가 아니라 '고전이 낫다'</b>를 가리킨다. "
             "단 이것도 <b>이 4개 세대·이 하이퍼파라미터 조건에서</b>의 결과이며, "
             "G3에 튜닝을 하면 달라질 여지는 남는다.</p>" if _res_tasks else "")
H.append(f"<div class='box'><p><b>1. 확장 {tot}개 과제에서 최고 세대 분포:</b> " +
         " · ".join(f"<b>{k}</b> {n}개" for k, n in sorted(byg.items())) +
         f". {flag_line}두 실험을 합치면 <b>세대가 올라간다고 성능이 올라가지 않는다</b>는 결론이 "
         f"핵심 4종을 넘어 확장 {tot}개 과제에서도 유지된다.</p>" + _res_html
         + "<p><b>2. 다만 대부분은 여전히 통계적으로 구분 불가다.</b> 위 §4 표에서 1위·2위 세대 격차가 "
         "구분 가능선을 넘는 과제는 소수이고, 특히 <b>발암성은 test 56분자·양성 11개</b>라 "
         "구분 가능선이 0.235로 순위 주장 자체가 불가능하다. "
         "따라서 종합 진술은 <b>'최신 세대가 고전을 이긴다는 증거는 어디에도 없고, "
         "반대 방향(고전 우세)의 증거가 큰 표본 과제 일부에서만 잡힌다'</b>가 된다.</p>"
         "<p><b>3. 엔드포인트가 방법을 정한다.</b> G1 규칙은 <b>발암성에서만</b> 의미 있는 신호를 주고 "
         "ClinTox는 약하며 Tox21 12/12는 무력이다. 이는 플래그십에서 "
         "AMES·DILI엔 듣고 hERG엔 무력했던 패턴의 <b>확장 재현</b>이다.</p>"
         "<p><b>4. ADMET-AI와의 격차도 재현된다(단 누수의 상한).</b> §3 참조 — 정직하게 학습하면 "
         "ADMET-AI보다 낮다(단 앙상블·멀티태스크·튜닝이 교락). <b>공개 모델의 높은 TDC 점수를 실력으로 읽으면 안 된다</b>는 결론이 "
         "새 엔드포인트에서도 유지된다.</p></div>")

H.append("<p class='src'>산출물: <code>results/expansion_metrics.json</code> · "
         "<code>results/admetai_preds.jsonl</code> · <code>results/g1_rules.csv</code> · "
         "<code>results/g1_summary.csv</code> · <code>results/overfit.json</code> · "
         "<code>results/leakage.json</code> · <code>results/split_meta.json</code> · "
         "<code>results/g2_raw.jsonl</code> · <code>results/g3_raw.jsonl</code> · <code>notes.md</code></p>")

(RES / "report_expansion.html").write_text("\n".join(H), encoding="utf-8")
print(f"저장 → results/report_expansion.html ({(RES/'report_expansion.html').stat().st_size/1024:.0f} KB)")
for c in concl:
    print(" ·", c.replace("<b>", "").replace("</b>", ""))
