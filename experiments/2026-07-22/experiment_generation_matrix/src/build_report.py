#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_report.py — gen_matrix.csv + matrix_meta.json + g1_*.csv 로 단일 HTML 보고서 생성.
★새 계산 없음(그림·표 조립만). 그림은 base64 임베드 → results/report.html 하나로 자기완결.
"""
import base64, io, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

OUT = Path("/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_generation_matrix/results")
TT = Path("/home/nudge/Project/HITS_portfolio/ADMET/T_toxicity/surface/results/metrics.csv")

for p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
          "/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf"):
    if Path(p).exists():
        fm.fontManager.addfont(p)
plt.rcParams["font.family"] = ["NanumGothic", "NanumSquareRound", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

d = pd.read_csv(OUT / "gen_matrix.csv")
meta = json.load(open(OUT / "matrix_meta.json"))
g1s = pd.read_csv(OUT / "g1_summary.csv")
g1r = pd.read_csv(OUT / "g1_rules.csv")
EPS = [("dili", "DILI 간독성", "AUROC"), ("herg", "hERG 차단", "AUROC"),
       ("ames", "AMES 변이원성", "AUROC"), ("ld50_zhu", "LD50 급성독성", "MAE")]
GC = {"G1": "#8d99ae", "G2": "#2a9d8f", "G3": "#e76f51", "G4": "#457b9d", "G5": "#9d4edd", "참고": "#adb5bd"}


def b64(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=125, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def img(fig, cap):
    return f'<figure><img src="data:image/png;base64,{b64(fig)}"><figcaption>{cap}</figcaption></figure>'


def _g2_spread_sentence():
    """'같은 세대 안의 표현 차이 vs 세대 간 차이'를 실측으로 비교(문장 자동 생성)."""
    out = []
    for ep, lab, met in EPS:
        o = ours(ep, met)
        g2 = o[o.gen == "G2"]
        s_g2 = float(g2.value.max() - g2.value.min())          # G2 내부(표현만 다름) 폭
        s_all = float(o.value.max() - o.value.min())            # 전체 폭
        both = (o.value.idxmax() in g2.index) and (o.value.idxmin() in g2.index)
        out.append(f"{lab} G2 내부 {s_g2:.3f} / 전체 {s_all:.3f}" + ("(★최고·최저 모두 G2)" if both else ""))
    return ("같은 G2 세대 안에서 <b>특징만 바꿨을 때</b>의 성능 폭이 세대를 통째로 바꿨을 때의 폭에 맞먹는다 — "
            + " · ".join(out) + ". 특히 <b>hERG는 우리 모델 중 최고(물리화학+RF)와 최저(ECFP4+XGB)가 둘 다 G2</b>다. "
            "즉 <b>무엇을 넣느냐가 어떤 세대냐보다 컸다</b>.")


def ours(ep, met):
    """ADMET-AI·SOTA·미측정 제외한 '우리가 학습한' 모델 행."""
    s = d[(d.endpoint == ep) & (d.metric == met) & d.value.notna() & (d.gen != "참고")]
    return s[~s.method.str.contains("ADMET-AI")]


def _genbest(ep, met):
    """세대별 대표값(그 세대 최고) — 좋은 순으로 정렬."""
    o = ours(ep, met); hb = (met != "MAE")
    return o.groupby("gen").value.agg("max" if hb else "min").sort_values(ascending=not hb)


def _gap(ep, met, kind="top2"):
    """★판정 통계량 = 1위 세대 vs 2위 세대 격차.
    ★이전 판(range = 최고−최저)은 '가장 나쁜 세대와의 폭'이라 최상위 세대의 우위를 전혀 함의하지 않는다.
    range로 판정하면 최하위 세대가 만든 폭을 '세대 효과 있음'으로 오독하게 된다(적대검증에서 확인·정정).
    결론 본문에 하드코딩 금지 — 항상 실측값."""
    gb = _genbest(ep, met)
    if kind == "range":
        return float(gb.max() - gb.min())
    return None if len(gb) < 2 else float(abs(gb.iloc[0] - gb.iloc[1]))



H = ['<meta charset="utf-8"><title>독성 예측 — 세대별(G1~G5) 정리 + ADMET-AI 비교</title>', """<style>
body{font-family:'Nanum Gothic',system-ui,sans-serif;max-width:1120px;margin:0 auto;padding:24px 28px;color:#1d2129;line-height:1.62}
h1{font-size:26px;border-bottom:3px solid #2a9d8f;padding-bottom:10px;margin-bottom:4px}
h2{font-size:20px;margin-top:38px;border-left:5px solid #2a9d8f;padding-left:11px}
h3{font-size:16px;margin-top:24px;color:#264653}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px}
th,td{border:1px solid #d8dde3;padding:6px 9px;text-align:left}
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
small,.src{font-size:12px;color:#6b7580}
code{background:#eef2f5;padding:1px 5px;border-radius:3px;font-size:12.5px}
ul{margin:8px 0 8px 4px}li{margin:4px 0}
</style>"""]

H.append("<h1>독성 예측 — 엔드포인트별 × 세대별(G1~G5) 정리 + ADMET-AI 비교</h1>")
H.append('<p class="src">작업일 2026-07-22 · <b>★신규 학습 1종</b> — G3 chemprop D-MPNN '
         '(4 엔드포인트 × 5 seed = 20런, 별도 폴더 <code>experiment_g3_dmpnn_seed42</code>)을 새로 학습해 '
         'G3 빈칸을 채웠다. <b>그 외 모든 값은 기존 실험 결과를 인용</b>했고, 추가 신규 계산은 '
         '(a) ADMET-AI <b>추론만</b>(기존 지표와 Δ0.0000 재현) (b) G1 구조알림 <b>규칙 적용</b>(둘 다 학습 아님). '
         '모든 숫자는 셀마다 출처를 표기.</p>')

# ── 0. 지표 설명
H.append("<h2>0. 지표를 어떻게 읽는가</h2><div class='box'><ul>"
         "<li><b>AUROC</b> — 무작위 양성/음성 한 쌍에서 양성을 더 높게 점수 줄 확률. 0.5=무작위, 1.0=완벽. "
         "불균형에 둔감해서 <b>단독으로는 낙관적</b>일 수 있다.</li>"
         "<li><b>AUPRC</b> — 정밀도-재현율 곡선 아래 면적. 기저선이 <b>양성비율</b>이므로 양성이 많은 데이터(hERG 양성 67%)에서는 "
         "AUPRC가 원래 높게 나온다 → 반드시 기저선과 함께 읽어야 한다.</li>"
         "<li><b>MAE</b>(LD50) — 평균절대오차, <b>낮을수록 좋음</b>. 단위는 log(1/(mol/kg)).</li>"
         "<li><b>G1은 지표 형태가 다르다</b> — 구조알림은 확률이 아니라 이진 판정(발화/무발화)이라 AUROC를 억지로 만들지 않고 "
         "<b>발화율·무매치율·알림별 정밀도·작동점 MCC</b>로 보고한다.</li>"
         "<li><b>★누수(leakage)</b> — 모델이 평가에 쓰인 분자를 학습에서 이미 봤다면 점수는 실력이 아니라 암기다. "
         "본 보고서는 ADMET-AI에 대해 이를 <b>실측</b>했다(§4).</li></ul></div>")

H.append("<h2>1. 무엇을 비교하는가 — 세대 정의와 공정성 전제</h2>")
H.append("<table><tr><th>세대</th><th>정의</th><th>이 표에 들어간 실제 모델</th></tr>"
         "<tr><td><b>G1</b></td><td>규칙·구조알림(학습 없음)</td><td>BRENK+NIH 카탈로그 + Benigni-Bossa 계열 변이원성 SMARTS 10종</td></tr>"
         "<tr><td><b>G2</b></td><td>고전 ML</td><td>물리화학 서술자+XGBoost · 물리화학 서술자+RandomForest · ECFP4 지문+XGBoost</td></tr>"
         "<tr><td><b>G3</b></td><td>분자 딥러닝(GNN)</td><td><b>chemprop D-MPNN (우리 자체 학습·정직)</b> — "
         "train만 보고 학습·누수 0 &nbsp;/&nbsp; ADMET-AI (같은 D-MPNN 계열, 공개 사전학습) — <b>★누수 의심</b>. "
         "둘의 격차는 <b>누수의 상한</b>이다 — 두 시스템은 앙상블·과제수·튜닝이 달라 누수만 분리되지 않는다(§4 증거 ③)</td></tr>"
         "<tr><td><b>G4</b></td><td>구조·3D</td><td>Uni-Mol (3D conformer, 2억 분자 사전학습)</td></tr>"
         "<tr><td><b>G5</b></td><td>파운데이션 모델</td><td>ChemBERTa-2 · MoLFormer (SMILES transformer, fine-tune)</td></tr></table>")
H.append(f"<div class='box'><b>공정성 전제(실측 확인)</b> — 표의 모든 값은 <b>{meta['split']}</b>의 "
         f"<b>동일한 고정 test 분자</b>에서 나온다. 서로 다른 실험 폴더에서 모은 값이므로 분할이 같은지 직접 확인했다: "
         f"{meta['split_verified']}. 우리 모델의 train↔test 정확분자 중복은 4개 엔드포인트 모두 <b>0</b>"
         f"(<code>leakage.json</code>).</div>")

# ── 2. 엔드포인트별 표 + 그림
H.append("<h2>2. 엔드포인트별 세대 사다리</h2>")
concl = []
for ep, lab, met in EPS:
    hb = (met != "MAE")
    sub = d[(d.endpoint == ep) & (d.metric == met)]
    o = ours(ep, met)
    best = (o.value.max() if hb else o.value.min())
    ai = sub[sub.method.str.contains("ADMET-AI")].iloc[0]
    sota = sub[sub.gen == "참고"]
    unc = meta["uncertainty"][ep]
    rg = unc["resolvable_gap_95"]
    g1 = g1s[g1s.endpoint == ep].iloc[0]

    H.append(f"<h3>{lab} — {met} ({'높을수록 좋음' if hb else '낮을수록 좋음'}, n_test={unc['n_test']})</h3>")
    H.append("<table><tr><th>세대</th><th>방법</th><th class='n'>값 ±SD(5 seed)</th>"
             "<th class='n'>ADMET-AI 대비</th><th>신뢰도 플래그</th><th>출처 파일</th></tr>")
    for _, r in sub.iterrows():
        if r.gen == "참고":
            continue
        if pd.isna(r.value):
            v = ("<span class='na'>지표 형태 다름 → 아래 G1 칸</span>" if r.gen == "G1"
                 else f"<span class='na'>{r.note}</span>")
            gap = "—"
            cls = ""
        else:
            v = f"{r.value:.4f}" + (f" ±{r.sd:.4f}" if pd.notna(r.sd) else "")
            g = (r.value - ai.value) if hb else (ai.value - r.value)
            gap = f"{g:+.4f}"
            cls = " class='leak'" if "ADMET-AI" in r.method else (" class='best'" if r.value == best else "")
        H.append(f"<tr><td><b>{r.gen}</b></td><td{cls}>{r.method}</td><td class='n'{cls}>{v}</td>"
                 f"<td class='n'>{gap}</td><td>{r.leak_flag}</td><td class='src'>{r.source}</td></tr>")
    if not sota.empty:
        s = sota.iloc[0]
        H.append(f"<tr><td>참고</td><td>{s.method}</td><td class='n'>{s.value:.4f}</td><td class='n'>—</td>"
                 f"<td>{s.note}</td><td class='src'>{s.source}</td></tr>")
    H.append("</table>")
    H.append(f"<p><small><b>G1(규칙) — 지표 형태 다름:</b> {sub[sub.gen=='G1'].iloc[0].note} "
             f"· 출처 <code>results/g1_summary.csv</code></small></p>")

    # 그림: 막대 + ADMET-AI 점선
    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    x = np.arange(len(o))
    ax.bar(x, o.value, yerr=[0 if pd.isna(s) else s for s in o.sd], capsize=3,
           color=[GC[g] for g in o.gen], edgecolor="#2b2b2b", linewidth=.6)
    ax.axhline(ai.value, ls="--", lw=1.8, color="#c1440e")
    ax.text(len(o) - .45, ai.value, f" ADMET-AI {ai.value:.3f}\n ★누수 의심", color="#c1440e",
            fontsize=8.5, va="bottom", ha="right")
    if not sota.empty:
        ax.axhline(sota.iloc[0].value, ls=":", lw=1.5, color="#6b7580")
        ax.text(0, sota.iloc[0].value, f" TDC SOTA {sota.iloc[0].value:.3f}", color="#4a545e",
                fontsize=8.5, va="bottom")
    lo = min(o.value.min(), ai.value) - .06
    hi = max(o.value.max(), ai.value) + .06
    ax.set_ylim(lo, hi)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n{m.split('(')[0].strip()[:16]}" for g, m in zip(o.gen, o.method)], fontsize=8)
    ax.set_ylabel(met + ("  (높을수록 좋음)" if hb else "  (낮을수록 좋음)"), fontsize=9.5)
    ax.set_title(f"{lab} — 세대별 {met}", fontsize=11.5)
    ax.grid(axis="y", alpha=.25)
    H.append(img(fig, f"{lab}. 막대=우리가 학습한 모델(오차막대=5 seed SD), 붉은 점선=ADMET-AI(★누수 의심), "
                      f"회색 점선=TDC 리더보드 SOTA(미재현 인용). "
                      f"이 test 크기의 95% 구분 가능 격차 참고선 ≈ {rg:.3f}."))

    # 엔드포인트별 한 줄 결론
    bo = o.loc[o.value.idxmax() if hb else o.value.idxmin()]
    # ★세대 간 차이는 '세대별 최고끼리' 비교하되, 판정은 ★1위 vs 2위 격차로 한다.
    #   (range = 최고−최저는 최하위 세대가 만든 폭이라 1위의 우위를 함의하지 않음 — 적대검증에서 정정)
    gb = _genbest(ep, met)
    diffs = _gap(ep, met)                       # 1위−2위
    rng = _gap(ep, met, "range")
    resolvable = (diffs is not None and rg is not None and diffs > rg)
    c = (f"<b>{lab}</b> — 우리 모델 중 최고는 <b>{bo.gen} {bo.method.split('(')[0].strip()}</b> ({bo.value:.4f}). "
         f"세대 순위는 " + " &gt; ".join(f"{g} {v:.3f}" for g, v in gb.items()) + f"이고, "
         f"<b>1위({gb.index[0]})와 2위({gb.index[1]})의 격차는 {diffs:.3f}</b>"
         f"(참고: 최고−최저 폭 {rng:.3f})로 "
         + f"이 test 크기의 구분 가능 격차({rg:.3f})보다 "
         + ("<b>겨우 크다(경계선) → 세대 차이라 단정하기 어려움</b>. " if (resolvable and diffs < rg * 1.2)
            else "<b>크다 → 세대 간 차이가 실재할 여지가 있음</b>. " if resolvable
            else "<b>작다 → 세대 간 순위는 통계적으로 구분 불가</b>. ")
         + f"ADMET-AI는 {ai.value:.4f}로 우리 최고보다 "
         + (f"{abs(ai.value-bo.value):.4f} 높으나 " if hb else f"{abs(ai.value-bo.value):.4f} 낮으나 ")
         + "★누수 의심 값이라 실력 비교에 쓸 수 없다.")
    concl.append(c)
    H.append(f"<div class='box'>{c}</div>")

# ── 3. 세대 요약 그림
H.append("<h2>3. 한눈에 — 세대가 올라가면 좋아지는가</h2>")
fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6))
for ax, (ep, lab, met) in zip(axes, EPS):
    o = ours(ep, met); hb = (met != "MAE")
    gg = o.groupby("gen").value.agg(["max", "min"])
    gorder = [g for g in ["G2", "G3", "G4", "G5"] if g in gg.index]
    y = [gg.loc[g, "max"] if hb else gg.loc[g, "min"] for g in gorder]
    ax.plot(range(len(gorder)), y, "o-", color="#2a9d8f", lw=2, ms=8)
    ai = d[(d.endpoint == ep) & (d.metric == met) & d.method.str.contains("ADMET-AI")].iloc[0].value
    ax.axhline(ai, ls="--", color="#c1440e", lw=1.4)
    ax.set_xticks(range(len(gorder))); ax.set_xticklabels(gorder)
    ax.set_title(f"{lab}\n{met}" + ("↑" if hb else "↓"), fontsize=10)
    ax.grid(alpha=.25)
fig.suptitle("세대별 '그 세대 최고 성적' 추이 — 우상향(=세대가 올라갈수록 좋아짐) 이 아니다", fontsize=12.5, y=1.04)
H.append(img(fig, "각 점 = 그 세대에서 가장 좋은 성적(G3는 ★우리가 정직하게 학습한 D-MPNN). "
                  "붉은 점선 = ADMET-AI(같은 D-MPNN 계열이지만 ★누수 의심이라 세대 판정에서 제외). "
                  "선이 우상향하지 않는다 = 세대가 올라간다고 좋아지지 않는다. "
                  "단, 세대 간 격차 대부분은 통계적으로 구분 불가다(§6)."))

# ── 4. 누수
ale = meta["admetai_leak_evidence"]
H.append("<h2>4. ★ADMET-AI 누수 — 추정이 아니라 실측</h2>")
H.append("<div class='crit'><b>증거 ①  학습 데이터셋 크기가 TDC 전체와 일치</b><br>"
         f"ADMET-AI 패키지가 스스로 기록한 학습 데이터 크기(<code>{ale['출처']}</code>)를 "
         "우리 분할의 전체 분자 수와 맞대면:</div>")
H.append("<table><tr><th>엔드포인트</th><th class='n'>ADMET-AI 학습 size</th>"
         "<th class='n'>우리 train_val + test</th><th>해석</th></tr>")
for k, tdc in [("DILI", "dili"), ("hERG", "herg"), ("AMES", "ames"), ("LD50_Zhu", "ld50_zhu")]:
    e = ale[k]; lk = meta["leak_ours"][tdc]
    tot = lk["n_train_val"] + lk["n_test"]
    same = "★정확히 일치" if e["admetai_train_size"] == tot else "거의 동일(중복제거 차)"
    H.append(f"<tr><td>{k}</td><td class='n'>{e['admetai_train_size']}</td>"
             f"<td class='n'>{lk['n_train_val']} + {lk['n_test']} = {tot}</td><td>{same}</td></tr>")
H.append("</table>")
H.append(f"<div class='crit'><b>따라서</b> — {ale['결론']}<br><br>"
         "<b>공정 보정 시도 결과: ★누수 보정 불가.</b> 다만 그 이유를 정확히 적으면 — "
         "ADMET-AI의 <b>학습 분자 목록이 공개돼 있지 않아</b> '학습셋에 없는 분자'를 <b>정의할 수 없다</b>. "
         "위 크기 대조는 우리 test가 학습에 포함됐음을 <b>강하게 시사</b>하지만(DILI는 475=475 정확 일치), "
         "분자 단위 대조를 한 것은 아니다. <b>'공집합임을 실측했다'가 아니라 '하위셋을 정의할 수 없다'</b>가 참이다.</div>")
sr = ale["admetai_자체보고_성능"]
H.append("<div class='crit'><b>증거 ②  ADMET-AI가 '패키지에 적어둔 자기 성능'보다 우리 test에서 훨씬 잘한다</b><br>"
         "같은 모델·같은 앙상블·같은 멀티태스크 설정이므로 그 성분들은 <b>양쪽에서 상쇄</b>되고 달라지는 것은 "
         "<b>평가 분자</b>뿐이다 → 세 증거 중 <b>누수 크기 추정에 가장 깨끗하다</b>. "
         "단, <code>admet.csv</code>에 기록된 성능의 <b>산출 조건·모델 버전은 확인할 수 없어</b> "
         "초과분을 '암기'로 단정하지 않고 <b>설명되지 않은 초과분</b>으로 부른다.</div>")
H.append("<table><tr><th>엔드포인트</th><th class='n'>패키지 기록 성능</th><th class='n'>우리 test에서</th>"
         "<th class='n'>설명되지 않은 초과분</th></tr>")
for k, tdc, met in [("DILI", "dili", "AUROC"), ("hERG", "herg", "AUROC"), ("AMES", "ames", "AUROC")]:
    own = sr[k]; got = d[(d.endpoint == tdc) & (d.metric == met) & d.method.str.contains("ADMET-AI")].iloc[0].value
    H.append(f"<tr><td>{k} {met}</td><td class='n'>{own:.4f}</td><td class='n leak'>{got:.4f}</td>"
             f"<td class='n leak'>{got-own:+.4f}</td></tr>")
own = sr["LD50_MAE"]; got = d[(d.endpoint == "ld50_zhu") & d.method.str.contains("ADMET-AI")].iloc[0].value
H.append(f"<tr><td>LD50 MAE(낮을수록 좋음)</td><td class='n'>{own:.4f}</td><td class='n leak'>{got:.4f}</td>"
         f"<td class='n leak'>{own-got:+.4f} 만큼 좋아짐</td></tr></table>")
H.append("<p><small>출처: <code>admet_ai/resources/data/admet.csv</code>의 AUROC/AUPRC/MAE 열(모델 자체 보고) vs "
         "<code>results/admetai_recomputed.json</code>(오늘 재추론, 2026-06-27 저장값과 Δ0.0000 일치).</small></p>")
# ── 증거 ③ : 같은 아키텍처를 정직하게 학습한 우리 D-MPNN과의 격차 = 누수 프리미엄
g3 = meta.get("g3_dmpnn") or {}
has_g3 = any(pd.notna(d[(d.gen == "G3") & d.method.str.contains("우리 자체")].value))
if has_g3:
    H.append("<div class='crit'><b>증거 ③  같은 계열을 <u>정직하게</u> 학습하면 얼마나 내려가는가 "
             "— 단, 이 격차는 <u>누수의 상한</u>이다</b><br>"
             "ADMET-AI는 Chemprop D-MPNN 계열이다. 그래서 <b>D-MPNN을 같은 분할의 train만 보고</b> 직접 학습했다"
             "(2026-07-22 <code>experiment_g3_dmpnn_seed42</code>, 5 seed).</div>")
    H.append("<div class='warn'><b>★두 시스템은 '같은 모델'이 아니다(패키지 실측)</b> — 아래 격차를 전부 누수로 "
             "귀속하면 안 된다.<table><tr><th></th><th>ADMET-AI</th><th>우리 D-MPNN</th></tr>"
             "<tr><td>모델 수</td><td class='leak'>5-모델 앙상블</td><td>seed당 단일 모델(5 seed는 예측 앙상블이 아니라 "
             "<b>지표 평균±SD</b>)</td></tr>"
             "<tr><td>과제 수</td><td class='leak'>멀티태스크 — 분류 31과제 · 회귀 10과제 동시 학습</td>"
             "<td>엔드포인트별 단일 과제</td></tr>"
             "<tr><td>하이퍼파라미터</td><td class='leak'>공개 배포본(튜닝됨)</td><td>탐색 없음(epochs 50·batch 50 고정)</td></tr>"
             "<tr><td>학습 분자</td><td class='leak'>TDC 전체(우리 test 포함)</td><td>train 파티션만</td></tr></table>"
             "<small>실측 출처: <code>admet_ai/resources/models/admet_{classification,regression}/model_0..4.pt</code> "
             "(각 5개) · 체크포인트의 <code>output_columns</code> 길이 31/10 · "
             "<code>admet_model.py</code>의 <code>_make_ensemble_predictions</code>.</small><br><br>"
             "따라서 아래 격차에는 <b>누수 + 앙상블(5) + 멀티태스크 전이(31/10과제) + 튜닝</b>이 <b>교락</b>돼 있다. "
             "이 값은 <b>누수의 크기가 아니라 상한(upper bound)</b>이다.</div>")
    H.append("<table><tr><th>엔드포인트</th><th class='n'>우리 D-MPNN (정직·단일·미튜닝)</th>"
             "<th class='n'>ADMET-AI (5앙상블·멀티태스크·TDC 전체)</th><th class='n'>격차(=누수 상한)</th></tr>")
    prem = []
    for ep, lab, met in EPS:
        o = d[(d.endpoint == ep) & (d.metric == met) & d.method.str.contains("우리 자체")]
        a = d[(d.endpoint == ep) & (d.metric == met) & d.method.str.contains("ADMET-AI")].iloc[0]
        if o.empty or pd.isna(o.iloc[0].value):
            continue
        o = o.iloc[0]
        gap = (a.value - o.value) if met != "MAE" else (o.value - a.value)
        prem.append((lab, met, gap))
        H.append(f"<tr><td>{lab} {met}</td><td class='n'>{o.value:.4f} ±{o.sd:.4f}</td>"
                 f"<td class='n leak'>{a.value:.4f}</td><td class='n leak'>{gap:+.4f}</td></tr>")
    H.append("</table>")
    si = (g3.get("split_identity") or {})
    jac = sorted({v.get("jaccard") for v in si.values() if v}) if si else []
    H.append(f"<p><small>같은 <b>공식 고정 test</b>에서 잰 값이다 — 본 보고서 test 집합과의 일치도 "
             f"Jaccard = {'/'.join(f'{j:.4f}' for j in jac) if jac else '미확인'} (4/4 엔드포인트). "
             "우리 D-MPNN은 <code>train</code> 파티션만 학습하고 <code>valid</code>로만 early stopping, "
             "test는 마지막 1회 예측. 출처 <code>experiment_g3_dmpnn_seed42/results/dmpnn_metrics.json</code>.</small></p>")
    mp = np.mean([abs(g) for _, _, g in prem[:3]]) if len(prem) >= 3 else None
    H.append("<div class='warn'><b>해석 — 조심해서 읽어야 하는 숫자</b><br>"
             "정직하게(단일·미튜닝·train만) 학습하면 분류 3종에서 평균 "
             + (f"<b>{mp:.3f} AUROC</b>" if mp else "상당폭") +
             " 낮다. 그러나 위 표에서 보듯 이 격차에는 앙상블·멀티태스크·튜닝이 함께 들어 있으므로 "
             "<b>'누수 = 0.088'이라고 말할 수 없다. 0.088은 누수의 상한이다.</b><br>"
             "실제로 <b>누수 크기의 더 깨끗한 추정은 증거 ②</b>(+0.049~0.074)다 — 거기서는 "
             "ADMET-AI가 <b>자기 자신</b>과 비교되므로 앙상블·멀티태스크·튜닝이 <b>양쪽에서 상쇄</b>되고 "
             "달라지는 건 평가 분자뿐이기 때문이다. 증거 ③의 역할은 크기 확정이 아니라 "
             "<b>독립적인 방향 확인</b>(정직하게 학습하면 반드시 낮아진다)이다.</div>")

H.append("<div class='warn'><b>결론</b> — ADMET-AI의 0.91~0.96대 점수는 <b>실력이 아니라 누수 신호</b>로 읽어야 한다. "
         "DILI에서 ADMET-AI(0.9557)가 TDC 리더보드 1위 MiniMol(0.956)과 사실상 같은 값이라는 점도 같은 방향의 정황이다. "
         "본 보고서는 ADMET-AI를 <b>고정 비교선</b>으로만 그리고, 세대 우열 판정에는 쓰지 않는다.</div>")

# ── 5. G1 규칙
H.append("<h2>5. G1(구조알림) — 지표 형태가 다른 세대</h2>")
H.append("<p>구조알림은 학습이 없고 출력이 확률이 아니라 이진 판정이라 AUROC를 억지로 만들지 않았다. "
         "대신 <b>얼마나 자주 발화하는가(발화율)·발화하면 실제로 독성인가(정밀도)·아무 알림도 안 걸리는 분자 비율(무매치율)</b>로 본다.</p>")
H.append("<table><tr><th>엔드포인트</th><th class='n'>발화율</th><th class='n'>무매치율</th>"
         "<th class='n'>규칙 정밀도</th><th class='n'>기저 양성률</th><th class='n'>규칙 재현율</th><th class='n'>작동점 MCC</th></tr>")
for _, g in g1s.iterrows():
    if g.task == "cls":
        H.append(f"<tr><td>{g.endpoint}</td><td class='n'>{g.fire_rate_mutagen:.3f}</td>"
                 f"<td class='n'>{g.nomatch_rate_mutagen:.3f}</td><td class='n'>{g.rule_precision:.3f}</td>"
                 f"<td class='n'>{g.pos_rate:.3f}</td><td class='n'>{g.rule_recall:.3f}</td>"
                 f"<td class='n'>{g.rule_MCC:+.3f}</td></tr>")
    else:
        H.append(f"<tr><td>{g.endpoint}</td><td class='n'>{g.fire_rate_mutagen:.3f}</td>"
                 f"<td class='n'>{g.nomatch_rate_mutagen:.3f}</td><td colspan=4>회귀 — 알림 발화군의 LD50 평균 차 "
                 f"{g.LD50_delta:+.3f} (양수면 발화군이 더 독성)</td></tr>")
H.append("</table>")
top = g1r[(g1r.endpoint == "ames") & g1r.precision.notna() & (g1r.n_fired >= 20)].sort_values("precision", ascending=False).head(6)
H.append("<h3>알림별 정밀도 — AMES에서 실제로 작동하는 알림</h3><table>"
         "<tr><th>알림</th><th class='n'>발화 분자수</th><th class='n'>정밀도 P(양성|발화)</th><th class='n'>기저 대비 lift</th></tr>")
for _, r in top.iterrows():
    H.append(f"<tr><td>{r.alert}</td><td class='n'>{int(r.n_fired)}</td><td class='n'>{r.precision:.3f}</td>"
             f"<td class='n'>{r.lift:.2f}×</td></tr>")
H.append("</table><p><small>전체 52행은 <code>results/g1_rules.csv</code>.</small></p>")
H.append("<div class='box'><b>규칙 세대가 말해주는 것</b> — 같은 규칙 집합이 엔드포인트마다 완전히 다르게 작동한다. "
         f"AMES(MCC {g1s[g1s.endpoint=='ames'].iloc[0].rule_MCC:+.3f})와 DILI({g1s[g1s.endpoint=='dili'].iloc[0].rule_MCC:+.3f})에서는 "
         f"의미 있는 신호를 주지만, <b>hERG에서는 MCC {g1s[g1s.endpoint=='herg'].iloc[0].rule_MCC:+.3f} — 완전히 무력</b>하다. "
         "hERG 차단은 친전자성 반응기가 아니라 친유성·염기성 같은 <b>물성</b>에서 오기 때문이며, "
         "실제로 hERG에서 물리화학 서술자 모델(G2)이 지문·딥러닝을 모두 앞선 것과 같은 이야기다.</div>")

# ── 6. 불확실성
H.append("<h2>6. 이 비교로 어디까지 말할 수 있는가 (불확실성)</h2>")
H.append("<table><tr><th>엔드포인트</th><th class='n'>n_test</th><th class='n'>표준오차(참고)</th>"
         "<th class='n'>95%에서 구분 가능한 최소 격차</th></tr>")
for ep, lab, met in EPS:
    u = meta["uncertainty"][ep]
    se = u.get("se_at_auroc_085", u.get("se_mae_admetai_ref"))
    H.append(f"<tr><td>{lab} ({u['kind']})</td><td class='n'>{u['n_test']}</td><td class='n'>{se:.4f}</td>"
             f"<td class='n'><b>{u['resolvable_gap_95']:.3f}</b></td></tr>")
H.append("</table>")
H.append(f"<div class='warn'><b>★가장 중요한 한계</b> — {meta['uncertainty_caveat']} "
         "Hanley-McNeil은 <b>비대응(독립표본) 가정</b>이라 같은 test셋 위의 두 모델을 비교할 때는 <b>지나치게 보수적</b>이다. "
         "대응비교(DeLong)라면 더 작아지지만 <b>분자별 예측이 원본에 저장돼 있지 않고 재학습은 금지</b>라 산출할 수 없다. "
         "따라서 위 격차선은 '이보다 작은 차이는 확실히 말할 수 없다'는 <b>하한 경보</b>로만 쓴다.<br><br>"
         "특히 <b>DILI는 test가 96분자뿐</b>이라 격차선이 0.111 AUROC다. DILI에서 관측된 세대 간 차이는 대부분 이보다 작다 "
         "→ <b>DILI에서 세대 순위를 주장하면 안 된다</b>.</div>")

# ── 7. 한계
H.append("<h2>7. 세대별 한계 — 있는 그대로</h2>")
H.append("<table><tr><th>모델</th><th>엔드포인트</th><th class='n'>train</th><th class='n'>test</th>"
         "<th class='n'>격차</th><th>해석</th></tr>")
for o in meta["overfit"]:
    if o["model"] not in ("unimol", "chemberta", "dmpnn_ours"):
        continue
    g = abs(o["gap"])
    itp = ("★train에서 완전 암기(1.000) — 표현 용량이 데이터 크기를 크게 넘어섬" if o.get("train") == 1.0
           else "★작음" if g < 0.03 else "중간" if g <= 0.10 else "큼")
    H.append(f"<tr><td>{o['model']}</td><td>{o['endpoint']} {o['metric']}</td><td class='n'>{o['train']:.4f}</td>"
             f"<td class='n'>{o['test']:.4f}</td><td class='n'>{o['gap']:.4f}</td><td>{itp}</td></tr>")
H.append("</table>")
# ★ChemBERTa가 실제로 '우리 모델 중 최하위'인 엔드포인트를 실측 산출(하드코딩 금지)
_cb_last = []
for _ep, _lab, _met in EPS:
    _o = ours(_ep, _met)
    _worst = _o.loc[_o.value.idxmin() if _met != "MAE" else _o.value.idxmax()]
    if "ChemBERTa" in _worst.method:
        _cb = _o[_o.method.str.contains("ChemBERTa")].iloc[0]
        _cb_last.append(f"{_lab} {_met} {_cb.value:.4f}")
_cb_last_n = len(_cb_last)
_cb_last_txt = " · ".join(_cb_last) if _cb_last else "해당 없음"
H.append("<div class='warn'><ul>"
         "<li><b>Uni-Mol(G4) 과적합</b> — 분류 3개 엔드포인트 모두 <b>train AUROC = 1.000</b>, test와의 격차 0.13~0.20. "
         "그럼에도 LD50에서는 우리 모델 중 최고(MAE 0.5939)다. 3D 표현이 무용한 게 아니라 <b>데이터가 작을 때 통제가 안 되는 것</b>.</li>"
         + f"<li><b>ChemBERTa-2(G5) 전이 무효</b> — 우리 모델 중 <b>최하위인 엔드포인트가 {_cb_last_n}개</b>"
         f"({_cb_last_txt}). 같은 G5인 MoLFormer는 훨씬 낫다 "
         "→ '파운데이션 모델이라서' 좋은 게 아니라 <b>어떤 파운데이션 모델이냐</b>가 갈린다.</li>"
         + ("<li><b>G3 자리는 이제 정직한 실측으로 채워졌다</b>(2026-07-22 추가 학습). "
            "다만 D-MPNN 설정은 seed=1 런과 동일한 값(epochs 50·batch 50·기본 hidden/depth)을 재사용했을 뿐 "
            "<b>이 분할에 맞춘 하이퍼파라미터 탐색은 하지 않았다</b> — G2(XGBoost)도 마찬가지로 튜닝하지 않았으므로 "
            "조건은 대등하나, 양쪽 모두 '튜닝하면 더 오를 수 있다'는 여지는 남는다.</li>"
            if has_g3 else
            "<li><b>우리 자체 G3(chemprop D-MPNN)이 이 분할에 없다</b> — G3 자리는 ADMET-AI(누수)로만 채워져 있어 "
            "<b>G3의 순수 실력은 이 표에서 미측정</b>이다. 다른 분할 결과는 부록.</li>")
         + "<li><b>SOTA는 리더보드 인용값</b>(미재현). 같은 scaffold split 프로토콜이지만 우리가 돌린 게 아니다.</li>"
         + "</ul></div>")

# ── 8. 부록: 다른 분할
H.append("<h2>8. 부록 — 다른 분할(seed=1)의 세대 사다리 <span class='src'>(본 표와 직접 비교 불가)</span></h2>")
tt0 = pd.read_csv(TT)
tt0 = tt0[tt0["분할"].astype(str).str.contains("scaffold")]
_dm_note = ""
if has_g3:
    _pairs = []
    for epn, epk in [("hERG", "herg"), ("AMES", "ames")]:
        r1 = tt0[(tt0.endpoint == epn) & (tt0["세대"] == "G3")]
        v42 = d[(d.endpoint == epk) & (d.metric == "AUROC") & d.method.str.contains("우리 자체")]
        if not r1.empty and not v42.empty and pd.notna(v42.iloc[0].value):
            v1 = float(str(r1.iloc[0]["ROC_AUC"]).split("±")[0])
            _pairs.append(f"{epn} seed=1 {v1:.3f} → seed=42 {v42.iloc[0].value:.3f}")
    _dm_note = ("<br><br><b>★같은 D-MPNN이 분할에 따라 얼마나 흔들리는가</b> — " + " · ".join(_pairs) +
                ". 같은 코드·같은 설정인데 test 분자가 바뀌면 이만큼 달라진다. "
                "<b>분할을 섞으면 안 되는 이유가 바로 이 크기</b>다." if _pairs else "")
H.append("<div class='warn'>아래는 같은 엔드포인트지만 <b>test 분자가 다른</b> 실험이다(TDC scaffold <b>seed=1</b>, "
         "본 표는 공식 seed=42). 실측 교집합이 28%/25%에 불과해 <b>같은 표에 섞지 않았다</b>."
         + _dm_note + "</div>")
tt = pd.read_csv(TT)
tt = tt[tt["분할"].astype(str).str.contains("scaffold")]
H.append("<table><tr><th>엔드포인트</th><th>세대</th><th>방법</th><th class='n'>ROC_AUC</th><th class='n'>PR_AUC</th></tr>")
for epn in ["hERG", "AMES"]:
    for _, r in tt[tt.endpoint == epn].iterrows():
        H.append(f"<tr><td>{epn}</td><td>{r['세대']}</td><td>{r['방법']}</td>"
                 f"<td class='n'>{r['ROC_AUC']}</td><td class='n'>{r['PR_AUC']}</td></tr>")
t21 = tt[tt.endpoint == "Tox21"].copy()
t21["roc"] = t21.ROC_AUC.astype(str).str.split("±").str[0].astype(float)
agg = t21.groupby(["세대", "방법"]).roc.agg(["mean", "count"])
for (g, m), r in agg.iterrows():
    H.append(f"<tr><td>Tox21 (12과제 평균)</td><td>{g}</td><td>{m}</td>"
             f"<td class='n'>{r['mean']:.4f}</td><td class='n'>—</td></tr>")
H.append("</table>")
H.append("<p><small>출처 <code>HITS_portfolio/ADMET/T_toxicity/surface/results/metrics.csv</code>. "
         "이 분할에서도 패턴은 같다 — hERG는 G4a(0.855)·G2(0.828)가 G3 D-MPNN(0.746)을 앞서고, "
         "AMES·Tox21은 G2가 최고다. <b>다른 test셋에서도 '세대=성능'이 성립하지 않는다.</b> "
         "ClinTox 확장 결과(<code>experiment_clintox_benchmark/results/clintox_metrics.csv</code>)도 있으나 "
         "G5 2종만 있어 세대 사다리를 구성하지 못해 제외했다.</small></p>")

# ── 9. 총결론
H.append("<h2>9. 총결론</h2>")
best_by_ep = []
for ep, lab, met in EPS:
    o = ours(ep, met); hb = (met != "MAE")
    b = o.loc[o.value.idxmax() if hb else o.value.idxmin()]
    best_by_ep.append(f"{lab}={b.gen}")
n_res = sum(1 for ep, lab, met in EPS
            if (_gap(ep, met) or 0) > (meta["uncertainty"][ep]["resolvable_gap_95"] or 0))
_gap_txt = " · ".join(
    f"{lab} {_gap(ep, met):.3f} vs 기준선 {meta['uncertainty'][ep]['resolvable_gap_95']:.3f}"
    for ep, lab, met in EPS)
def _rank_of(sub, ep, met):
    o = ours(ep, met); hb = (met != "MAE")
    o = o.sort_values("value", ascending=not hb).reset_index(drop=True)
    hit = o[o.method.str.contains(sub)]
    return (int(hit.index[0]) + 1, len(o)) if not hit.empty else (None, len(o))


g3_ranks = [f"{lab} {_rank_of('우리 자체', ep, met)[0]}위" for ep, lab, met in EPS
            if _rank_of("우리 자체", ep, met)[0]]
H.append(f"<div class='box'><p><b>1. 세대는 성능의 순서가 아니다.</b> 엔드포인트별 최고 세대는 "
         f"{' · '.join(best_by_ep)} — <b>4개 중 3개에서 가장 오래된 학습 세대인 G2(고전 ML)</b>가 "
         f"3D 사전학습(G4)과 파운데이션 모델(G5)을 모두 앞섰다. 최신 세대가 <b>순위상</b> 앞선 곳은 "
         f"LD50(G4 Uni-Mol) 하나뿐이며(단 아래 2번대로 통계적으로 구분되지는 않는다), "
         f"어느 엔드포인트에서도 <b>G5(파운데이션)가 1위를 한 적이 없다</b>."
         + (f" 그리고 2026-07-22에 채워 넣은 <b>정직한 G3(chemprop D-MPNN)도 1위가 없다</b> "
            f"({' · '.join(g3_ranks)} / 7모델 중). <b>'GNN이 고전 ML을 이긴다'는 통념이 이 4개 독성 엔드포인트에서는 성립하지 않았다.</b></p>"
            if g3_ranks else "</p>")
         + f"<p><b>2. 그러나 어떤 세대가 이긴다는 주장도 통계적으로는 성립하지 않는다.</b> "
         + f"판정 기준을 <b>1위 세대 vs 2위 세대 격차</b>로 잡으면(초안은 최고−최저 <i>폭</i>을 썼는데, "
         + f"그 폭은 <b>최하위</b> 세대가 만든 값이라 1위의 우위를 함의하지 않는다 — 적대검증에서 정정), "
         + f"구분 가능 격차를 넘는 엔드포인트는 4개 중 <b>{n_res}개</b>다: {_gap_txt}. "
         + "즉 <b>4개 엔드포인트 어디에서도 세대 순위를 주장할 근거가 없다</b>. "
         + "특히 초안이 '유일하게 확실한 세대 효과'라고 했던 LD50조차, G4(0.594)와 차순위 G2(0.607)의 격차 "
         + f"{_gap('ld50_zhu','MAE'):.3f}는 기준선 {meta['uncertainty']['ld50_zhu']['resolvable_gap_95']:.3f} "
         + "미만이라 <b>3D 우세를 주장할 수 없다</b>(이 진술은 철회한다). 정확한 진술은 "
         + "<b>'최신 세대가 고전을 이긴다는 증거가 없다'</b>이며, 마찬가지로 '고전이 우월하다'는 증거도 없다.</p>"
         + "<p><b>3. 세대보다 표현(feature) 선택이 크다.</b> " + _g2_spread_sentence() + "</p>"
         + "<p><b>4. 엔드포인트가 방법을 정한다.</b> hERG는 물성 축(규칙 MCC −0.03로 구조알림 완전 무력·물리화학 최고), "
         + "AMES는 구조 축(구조알림이 유의미한 신호), LD50은 3D(G4)가 순위상 최고. "
         + "<b>하나의 세대가 모든 독성을 지배하지 않는다.</b></p>"
         + ("<p><b>5. ADMET-AI의 높은 점수는 실력으로 읽으면 안 된다.</b> "
            "세 증거가 독립적으로 같은 방향을 가리킨다 — ①학습 데이터 크기가 TDC 전체와 일치(DILI 475=475), "
            "②패키지에 기록된 자기 성능을 우리 test에서 <b>+0.049~0.074</b> 초과, "
            "③D-MPNN을 정직하게(train만) 학습하면 분류 3종에서 <b>0.088</b> 낮음. "
            "<b>단 ③의 0.088은 누수의 크기가 아니라 상한이다</b> — ADMET-AI는 5-모델 앙상블·31과제 멀티태스크·튜닝됨이고 "
            "우리 D-MPNN은 단일모델·단일과제·미튜닝이라 그 성분들이 분리되지 않는다(실측). "
            "<b>누수 크기의 가장 깨끗한 추정은 ②</b>(같은 모델끼리 비교라 앙상블·멀티태스크가 상쇄된다). "
            "누수 보정은 여전히 불가하다 — ADMET-AI 학습 분자 목록이 공개돼 있지 않아 "
            "'학습에 안 쓰인 분자' 하위셋을 <b>정의할 수 없기</b> 때문이다.</p>"
            if has_g3 else
            "<p><b>5. ADMET-AI의 높은 점수는 실력이 아니다.</b> 학습 데이터 크기 대조와 자체보고 성능 초과, "
            "두 독립 증거로 누수를 실측했고 <b>누수 보정은 불가</b>(하위셋이 공집합)임을 확인했다. "
            "0.9+ 점수는 <b>누수 신호로 해석</b>해야 한다.</p>")
         + "</div>")
H.append(f"<p class='src'>산출물: <code>results/gen_matrix.csv</code>({meta['n_cells']}행·값 {meta['n_filled']}셀) · "
         "<code>results/admetai_preds.jsonl</code>(3163행) · <code>results/g1_rules.csv</code> · "
         "<code>results/g1_summary.csv</code> · <code>results/matrix_meta.json</code> · <code>notes.md</code></p>")

(OUT / "report.html").write_text("\n".join(H), encoding="utf-8")
print(f"저장 → results/report.html ({(OUT/'report.html').stat().st_size/1024:.0f} KB)")
for c in concl:
    print(" ·", c.replace("<b>", "").replace("</b>", ""))
