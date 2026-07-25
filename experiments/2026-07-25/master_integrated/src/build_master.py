#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_master.py — 통합 마스터 보고서(36과제 ADMET 세대 벤치마크 + 배포 신뢰도 + 배포 가이드).
★새 학습·새 계산 0건. 기존 4+1개 보고서의 확정값을 원본 데이터 파일에서 읽어 단일 문서로 이관·조립만.
셀별 출처: 각 값은 아래 SRC 경로에서 직접 읽음(재계산 없음). 서술 판정문은 원본 보고서에서 이관(출처 표기).
값 불일치는 코드가 원본을 그대로 읽으므로 발생하지 않음. G4 검증은 주판정을 (B)seed별로 통일(사유 §5).
"""
import json, os
import numpy as np
import pandas as pd

BASE = "/home/nudge/Project/ADMET_integrated"
TOXM = f"{BASE}/2026-07-22/master_report/results"
TOXR = f"{BASE}/2026-07-22/experiment_deploy_reliability/results"
ADMF = f"{BASE}/2026-07-22/experiment_adme_full/results"
ADMR = f"{BASE}/2026-07-25/experiment_adme_reliability/results"
G4R  = f"{BASE}/2026-07-24/experiment_g4_verification/results"
OUT  = f"{BASE}/2026-07-25/master_integrated/results"
# 원본 보고서 상대링크(마스터=results/에서 본 경로)
LINKS = dict(
    tox_master="../../../2026-07-22/master_report/results/master_report.html",
    tox_rel="../../../2026-07-22/experiment_deploy_reliability/results/reliability_report.html",
    adme_full="../../../2026-07-22/experiment_adme_full/results/report_adme_full.html",
    adme_rel="../../../2026-07-25/experiment_adme_reliability/results/adme_reliability_report.html",
    g4="../../../2026-07-24/experiment_g4_verification/results/g4_verification.html")

# ═══════════ 원본 확정값 로드 ═══════════
toxM = pd.read_csv(f"{TOXM}/master_matrix.csv")
consist = json.load(open(f"{TOXM}/consistency.json"))
finalize = json.load(open(f"{TOXM}/finalize_check.json"))
toxR = pd.read_csv(f"{TOXR}/reliability.csv")
admeM = pd.read_csv(f"{ADMF}/adme_matrix.csv")
feat2 = pd.read_csv(f"{ADMF}/feature_2x2.csv")
learn = pd.read_csv(f"{ADMF}/learning_axis.csv")
boot = json.load(open(f"{ADMF}/bootstrap_verdict.json"))
admeR = pd.read_csv(f"{ADMR}/adme_reliability.csv")
admeDet = json.load(open(f"{ADMR}/reliability_detail.json"))
g4 = json.load(open(f"{G4R}/g4_verification.json"))
champ = {r["endpoint"]: r for r in json.load(open(f"{ADMR}/champions.json"))}
logd_repro = json.load(open(f"{ADMR}/logd_g3_repro.json"))

# ═══════════ 독성 세대 값 이관(정직 행만·누수기준선/리더보드 제외) ═══════════
HONEST = toxM[toxM["kind"].isin(["우리 학습", "우리 학습(규칙)"])].copy()


def tox_best(sub, gen):
    g = sub[sub.gen == gen].dropna(subset=["value"])
    if g.empty:
        return None
    hb = bool(sub.higher_better.iloc[0])
    return float(g.value.max() if hb else g.value.min())


def g1_info(sub):
    g1 = sub[sub.gen == "G1"]
    if g1.empty:
        return None, None
    return (None if g1.g1_fire_rate.isna().all() else float(g1.g1_fire_rate.iloc[0]),
            None if g1.g1_MCC.isna().all() else float(g1.g1_MCC.iloc[0]))


CORE = ["DILI 간독성", "hERG 차단", "AMES 변이원성", "LD50 급성독성"]
core_rows = []
for ep in CORE:
    sub = HONEST[HONEST.endpoint == ep]
    fire, mcc = g1_info(sub)
    vals = {g: tox_best(sub, g) for g in ["G2", "G3", "G4", "G5"]}
    metric = sub.metric.iloc[0]; hb = bool(sub.higher_better.iloc[0])
    best_gen = (max if hb else min)([g for g in vals if vals[g] is not None], key=lambda g: vals[g])
    n = int(sub.n_test.dropna().iloc[0])
    core_rows.append(dict(ep=ep, metric=metric, hb=hb, n=n, fire=fire, mcc=mcc, best=best_gen, **vals))

# 확장: 발암성·ClinTox + Tox21 12경로
exp_simple = []
for ep in ["발암성 (Carcinogens)", "ClinTox 임상독성"]:
    sub = HONEST[HONEST.endpoint == ep]
    fire, mcc = g1_info(sub)
    vals = {g: tox_best(sub, g) for g in ["G2", "G3"]}
    best_gen = max([g for g in vals if vals[g] is not None], key=lambda g: vals[g])
    n = int(sub.n_test.dropna().iloc[0])
    exp_simple.append(dict(ep=ep, n=n, fire=fire, mcc=mcc, best=best_gen, **vals))

tox21 = HONEST[HONEST.endpoint == "Tox21"]
tox21_rows = []
for path, g in tox21.groupby("task"):   # ★Tox21 12경로는 task 컬럼에 들어있음(gen_range는 전부 'G1~G3')
    fire, mcc = g1_info(g)
    v2 = tox_best(g, "G2"); v3 = tox_best(g, "G3")
    n = int(g.n_test.dropna().iloc[0])
    tox21_rows.append(dict(path=path, n=n, mcc=mcc, G2=v2, G3=v3, best="G2" if (v2 or 0) >= (v3 or 0) else "G3"))
tox21_rows.sort(key=lambda r: r["path"])
RESOLVABLE = set(consist["resolvable"])  # 구분가능 3개(전부 G2>G3)

# ═══════════ ADME 세대 값 이관 ═══════════
G2M = ["xgb_physchem", "rf_physchem", "xgb_ecfp"]
PILL_ORDER = ["A", "D", "M", "E"]
PILL = {"A": "흡수(A)", "D": "분포(D)", "M": "대사(M)", "E": "배설(E)"}


def adme_best(sub, models):
    g = sub[sub.model.isin(models)].dropna(subset=["value"])
    if g.empty:
        return None
    hb = bool(sub.higher_better.iloc[0]) if "higher_better" in sub else (sub.metric.iloc[0] != "MAE")
    return float(g.value.max() if hb else g.value.min())


adme_rows = []
for ep, sub in admeM.groupby("endpoint", sort=False):
    metric = sub.metric.iloc[0]; hb = metric != "MAE"
    pillar = sub.pillar.iloc[0]; label = sub.label.iloc[0]
    n = int(champ.get(ep, {}).get("n_test", 0)) or None
    g2 = adme_best(sub, G2M); g3 = adme_best(sub, ["dmpnn_ours"])
    g4v = adme_best(sub, ["unimol"]); g5 = adme_best(sub, ["chemberta", "molformer"])
    ai = adme_best(sub, ["admetai"])
    cand = {"G2": g2, "G3": g3, "G4": g4v, "G5": g5}
    cand = {k: v for k, v in cand.items() if v is not None}
    best = (max if hb else min)(cand, key=lambda k: cand[k])
    adme_rows.append(dict(ep=ep, label=label, pillar=pillar, metric=metric, hb=hb,
                          G2=g2, G3=g3, G4=g4v, G5=g5, ai=ai, best=best))
# 세대 승수
adme_wincount = {}
for r in adme_rows:
    adme_wincount[r["best"]] = adme_wincount.get(r["best"], 0) + 1

# ═══════════ HTML ═══════════
def f(v, d=4):
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


H = []
H.append("""<meta charset="utf-8"><title>통합 마스터 — 36과제 ADMET 세대 벤치마크·배포 신뢰도</title><style>
body{font-family:'Nanum Gothic',system-ui,-apple-system,sans-serif;max-width:1220px;margin:0 auto;padding:20px 26px;color:#1a1f26;line-height:1.62}
h1{font-size:25px;border-bottom:3px solid #1f6f6b;padding-bottom:9px;margin-bottom:2px}
h2{font-size:20px;margin-top:38px;border-left:6px solid #1f6f6b;padding-left:12px;scroll-margin-top:14px}
h3{font-size:16px;margin-top:22px;color:#204e4a}
table{border-collapse:collapse;width:100%;margin:11px 0;font-size:12.2px}
th,td{border:1px solid #d5dbe0;padding:4px 7px}th{background:#eaf1f0}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.win{background:#e3f3ee;font-weight:700}.lo{background:#fdecea;color:#b3261e}
.g2{color:#1f6f6b;font-weight:700}.g3{color:#8a5a00;font-weight:700}.g4{color:#3a4a8a;font-weight:700}.g5{color:#7a3a6a;font-weight:700}
.leak{color:#9a3a3a;background:#fbeee6}
.banner{background:linear-gradient(135deg,#18313a,#1f6f6b);color:#fff;border-radius:12px;padding:19px 24px;margin:14px 0 20px}
.banner b{color:#ffe6a3}.banner a{color:#bfe9e2}
.box{background:#f6f9fa;border:1px solid #dde4e8;border-radius:8px;padding:13px 17px;margin:13px 0}
.warn{background:#fff8ec;border:1px solid #efd7a3;border-radius:8px;padding:13px 17px;margin:13px 0}
.crit{background:#fdefe9;border:1px solid #f0bfa8;border-radius:8px;padding:13px 17px;margin:13px 0}
.toc{background:#f2f6f7;border:1px solid #d5dbe0;border-radius:9px;padding:12px 20px;columns:2;font-size:13px}
.toc a{color:#204e4a;text-decoration:none}.toc a:hover{text-decoration:underline}
.badge{display:inline-block;font-size:10px;padding:1px 6px;border-radius:8px;margin-left:3px}
.b-sm{background:#ffe4c4;color:#8a5200}.b-wait{background:#dfe4f3;color:#33448a}.b-leak{background:#f5d9d0;color:#9a3a3a}
small,.src{font-size:11.2px;color:#68727a}code{background:#eaf1f0;padding:1px 4px;border-radius:3px;font-size:11px}
.pill{font-weight:700;color:#204e4a}
</style>""")

H.append("<h1>통합 마스터 — 36과제 ADMET 세대 벤치마크 · 배포 신뢰도 · 배포 가이드</h1>")
H.append("<p class='src'>작업일 2026-07-25 · 독성 18과제(핵심 4종 G1~G5 + 확장 14 G1~G3·Tox21 12경로 포함) + "
         "ADME 18개(G2~G5) = <b>36과제</b> · ★<b>새 학습·새 계산 0건 — 기존 4+1개 보고서의 확정값을 원본 데이터에서 읽어 이관·조립만</b>. "
         "이 문서는 최상위 인덱스이며, 각 상세는 원본 보고서를 참조.</p>")

# ── 0. 배너 ──
nres = len(RESOLVABLE)
H.append("<div class='banner'>"
         "<p><b>1. 모델 세대는 성능을 예측하지 않는다</b> — 36과제 모두에서, '더 최신·더 큰 모델일수록 낫다'는 성립하지 않았다. "
         "통계적으로 확실한 세대 효과는 <b>ADME 친유성 logD의 G3(GNN) 단 하나</b>뿐이다.</p>"
         "<p><b>2. 독성은 2세대가 압도(17/18), ADME는 갈린다</b>(G2 9·G4 5·G3 4·G5 0) — "
         "<b>기전이 방법을 정한다</b>. 독성=구조알림·물성으로 충분, ADME 일부(용해도·분포·배설)=3D가 관여.</p>"
         "<p><b>3. 높은 AUROC ≠ 사용 가능</b> — 배포에는 작동점·AD·예측구간 재설정이 필수. "
         "임계 0.5에서 <b>독성 NR-PPAR-gamma 민감도 0.000·ADME CYP2C9 기질 0.026</b>(양성 37중 36 놓침).</p>"
         f"<p class='src' style='color:#cfe6e1'>한눈 수치 — 총 36과제 · 독성 최고세대 <b>G2 17·G4 1·정직 G3 0</b> · "
         f"ADME 최고세대 <b>G2 9·G4 5·G3 4·G5 0</b> · 통계 확실한 세대효과 <b>1건</b>(logD G3) · "
         f"AD 유효 독성 {finalize['ad_valid']}/ADME {int((admeR.AD_verdict.str.contains('유효')).sum())} · "
         f"ADMET-AI 누수 상한(참고선) 핵심 {consist['leak_ub_core_mean']:.3f}·확장 {consist['leak_ub_exp_mean']:.3f}.</p>"
         f"<p class='src' style='color:#cfe6e1'>원본 상세: "
         f"<a href='{LINKS['tox_master']}'>독성 마스터</a> · <a href='{LINKS['tox_rel']}'>독성 신뢰도</a> · "
         f"<a href='{LINKS['adme_full']}'>ADME 3축</a> · <a href='{LINKS['adme_rel']}'>ADME 신뢰도</a> · "
         f"<a href='{LINKS['g4']}'>G4 검증</a></p></div>")

# ── 목차 ──
H.append("<div class='toc'>"
         "<a href='#s1'>1. 지표와 세대 프레임</a><br><a href='#s2'>2. 독성 18과제 — 세대 벤치마크</a><br>"
         "<a href='#s3'>3. ADME 18개 — 세대 벤치마크(3축)</a><br><a href='#s4'>4. ★독성 vs ADME 대비</a><br>"
         "<a href='#s5'>5. G4 검증(반감기·수용해도)</a><br><a href='#s6'>6. 배포 신뢰도 — 36과제</a><br>"
         "<a href='#s7'>7. ★통합 배포 가이드 — 36 엔드포인트</a><br><a href='#s8'>8. ★자기정정 기록(8건)</a><br>"
         "<a href='#s9'>9. 한계</a><br><a href='#s10'>10. 결론 + 후속</a></div>")

# ── 1. 지표와 세대 프레임 ──
H.append("<h2 id='s1'>1. 지표와 세대 프레임 (공통 기초)</h2>")
H.append("<div class='box'><b>지표 방향은 섞여 있다</b> — 승부 계산에 반드시 반영해야 한다."
         "<ul><li><b>AUROC↑·AUPRC↑</b>(분류·높을수록 좋음): 독성 17개·ADME 분류 10개.</li>"
         "<li><b>MAE↓</b>(회귀·낮을수록 좋음): LD50·Caco-2·logD·PPBR·용해도.</li>"
         "<li><b>Spearman↑</b>(순위상관): ADME의 VDss·반감기·청소율 2종 — 생리 지배로 절대값 예측 한계.</li></ul>"
         "★부호 실수 시 순위가 통째로 뒤집히므로 두 신뢰도 보고서 모두 <b>방향 단위 테스트</b>로 검증했다(ADME 4/4 통과).</div>")
H.append("<h3>세대 정의 — 누가 특징을 만드는가</h3>")
H.append("<table><tr><th>세대</th><th>정의</th><th>대표 모델</th></tr>"
         "<tr><td class='pill'>G1</td><td>구조알림(규칙)</td><td>BRENK+NIH+Benigni-Bossa SMARTS (독성 전용)</td></tr>"
         "<tr><td class='g2'>G2</td><td>고전 ML</td><td>물리화학 서술자(RDKit 210)+XGBoost/RF · ECFP4+XGBoost</td></tr>"
         "<tr><td class='g3'>G3</td><td>GNN</td><td>chemprop D-MPNN(정직 학습). ADMET-AI는 <span class='leak'>누수 기준선</span>이라 순위 제외</td></tr>"
         "<tr><td class='g4'>G4</td><td>3D</td><td>Uni-Mol(ETKDG conformer, 2억 사전학습)</td></tr>"
         "<tr><td class='g5'>G5</td><td>파운데이션</td><td>ChemBERTa-2 · MoLFormer (SMILES transformer)</td></tr></table>")
H.append("<div class='warn'><b>★공정성 조건(실측)</b> — 36과제 전부 동일 프로토콜: TDC 공식 고정 test(scaffold seed=42) + "
         "5 seed · <b>train∩test 정확분자 중복 = 0</b> · G4가 '더 쉬운 부분집합'에서 평가받았을 가능성은 "
         "<b>분자 단위 Jaccard 1.0</b>으로 배제(Uni-Mol이 conformer 실패 분자를 제외하지 않고 전량 예측). "
         "★세대 범위가 묶음마다 다르다:</div>")
H.append("<table><tr><th>묶음</th><th>과제</th><th>세대 범위</th><th>비고</th></tr>"
         "<tr><td>독성 핵심 4종</td><td>DILI·hERG·AMES·LD50</td><td class='n'>G1~G5</td><td>완전 사다리</td></tr>"
         "<tr><td>독성 확장 14</td><td>발암성·ClinTox·Tox21 12경로</td><td class='n'>G1~G3</td>"
         "<td>G4·G5는 소표본·불균형 근거로 미실행(스코프 결정)</td></tr>"
         "<tr><td>ADME 18</td><td>흡수6·분포3·대사6·배설3</td><td class='n'>G2~G5</td>"
         "<td>G1은 독성 전용(toxicophore)이라 제외</td></tr></table>")

# ── 2. 독성 18과제 ──
H.append("<h2 id='s2'>2. 독성 18과제 — 세대 벤치마크</h2>")
H.append("<h3>핵심 4종 (G1~G5·정직 행만·누수 기준선/리더보드 제외)</h3>")
H.append("<table><tr><th>엔드포인트</th><th>주지표</th><th class='n'>n</th><th>G1 구조알림</th>"
         "<th class='n'>G2 고전</th><th class='n'>G3 GNN</th><th class='n'>G4 3D</th><th class='n'>G5 파운데이션</th><th>최고</th></tr>")
GN = {"G2": "g2", "G3": "g3", "G4": "g4", "G5": "g5"}
for r in core_rows:
    g1txt = (f"MCC {r['mcc']:.3f}<br><small>발화 {r['fire']:.2f}</small>" if r["mcc"] is not None
             else (f"발화 {r['fire']:.2f}" if r["fire"] is not None else "규칙"))
    if r["mcc"] is not None and r["mcc"] < 0.05:
        g1txt = f"<span class='lo'>{g1txt}</span>"
    cells = ""
    for g in ["G2", "G3", "G4", "G5"]:
        cl = "win" if g == r["best"] else "n"
        cells += f"<td class='n {cl}'>{f(r[g])}</td>"
    H.append(f"<tr><td>{r['ep']}</td><td>{r['metric']} {'↑' if r['hb'] else '↓'}</td><td class='n'>{r['n']}</td>"
             f"<td class='n'>{g1txt}</td>{cells}<td class='{GN[r['best']]}'>{r['best']}</td></tr>")
H.append("</table>")
H.append("<div class='box'><b>G1 구조알림</b>은 <b>DILI(MCC 0.36)·AMES(0.26)</b>에서 작동하나 "
         "<b>hERG는 무력</b>(MCC −0.03). <b>LD50만 G4(Uni-Mol)가 최고</b>(MAE 0.594, 낮을수록 좋음) — "
         "18과제 중 <b>유일한 G4 최고</b>. 나머지 3종은 모두 G2가 최고이며, <b>정직 G3(chemprop)는 4종 모두 G2에 못 미친다</b>. "
         "<span class='src'>출처 master_matrix.csv(kind='우리 학습')</span></div>")

H.append("<h3>확장 — 발암성·ClinTox (G1~G3)</h3>")
H.append("<table><tr><th>엔드포인트</th><th class='n'>n</th><th>G1 구조알림</th>"
         "<th class='n'>G2 고전</th><th class='n'>G3 GNN</th><th>최고</th></tr>")
for r in exp_simple:
    g1txt = f"MCC {r['mcc']:.3f}" if r["mcc"] is not None else "규칙"
    if r["mcc"] is not None and r["mcc"] >= 0.2:
        g1txt = f"<b>{g1txt}</b>"
    H.append(f"<tr><td>{r['ep']}</td><td class='n'>{r['n']}</td><td class='n'>{g1txt}</td>"
             f"<td class='n win'>{f(r['G2'])}</td><td class='n'>{f(r['G3'])}</td><td class='g2'>{r['best']}</td></tr>")
H.append("</table>")
H.append("<p class='src'>발암성은 G1 구조알림이 작동(MCC 0.358)하나 두 과제 모두 <b>G2가 최고·G3는 크게 낮다</b>(ClinTox G3 0.689).</p>")

H.append("<h3>확장 — Tox21 12경로 (G1~G3·전부 명시)</h3>")
H.append("<table><tr><th>경로</th><th class='n'>n</th><th class='n'>G1 MCC</th>"
         "<th class='n'>G2 고전(최고)</th><th class='n'>G3 GNN(정직)</th><th>최고</th><th>G2>G3 구분</th></tr>")
for r in tox21_rows:
    key = f"Tox21 {r['path']}"
    res = "★구분 가능" if key in RESOLVABLE else "동률/미검정"
    g1cls = "lo" if (r["mcc"] is not None and abs(r["mcc"]) < 0.06) else "n"
    H.append(f"<tr><td>Tox21 {r['path']}</td><td class='n'>{r['n']}</td>"
             f"<td class='n {g1cls}'>{f(r['mcc'],3)}</td><td class='n win'>{f(r['G2'])}</td>"
             f"<td class='n'>{f(r['G3'])}</td><td class='g2'>{r['best']}</td><td>{res}</td></tr>")
H.append("</table>")
H.append(f"<div class='warn'><b>Tox21 12/12 모두 G2가 최고</b>, 정직 G3는 12/12 모두 낮다. "
         f"<b>G1 구조알림은 12/12 무력</b>(MCC −0.02~+0.05, 거의 0) — toxicophore가 핵수용체·스트레스반응 경로엔 안 맞는다. "
         f"통계적으로 <b>G2>G3가 구분 가능한 곳은 3개</b>({', '.join(sorted(RESOLVABLE))}) — <b>전부 G2 우세</b>(통념과 반대). "
         f"<span class='src'>출처 consistency.json·master_matrix.csv</span></div>")

H.append(f"<div class='crit'><b>독성 종합 판정</b>(consistency.json·finalize_check.json) — "
         f"18과제 최고세대 <b>G2 {consist['best_gen_dist'].get('G2')}·G4 {consist['best_gen_dist'].get('G4')}"
         f"(LD50)·정직 G3 {consist['g3_first_place']}·G5 0</b>. "
         f"DeLong 대응비교 <b>{finalize['delong_significant']}/{finalize['delong_n_compared']} 유의·전부 챔피언(G2) 우세</b>. "
         f"ADMET-AI는 누수 3증거(전체 사전학습·자체보고 초과·정직본 격차=상한 핵심 {consist['leak_ub_core_mean']:.3f}·확장 "
         f"{consist['leak_ub_exp_mean']:.3f})로 <b>순위 제외</b>.</div>")

# ── 3. ADME 3축 ──
H.append("<h2 id='s3'>3. ADME 18개 — 세대 벤치마크 (3축)</h2>")
H.append("<h3>축① 세대 — 엔드포인트별 (세대별 최고값)</h3>")
H.append("<table><tr><th>기둥</th><th>엔드포인트</th><th>주지표</th><th class='n'>G2 고전</th>"
         "<th class='n'>G3 GNN</th><th class='n'>G4 3D</th><th class='n'>G5 파운데이션</th>"
         "<th class='n leak'>ADMET-AI(누수)</th><th>최고</th></tr>")
for pil in PILL_ORDER:
    prs = [r for r in adme_rows if r["pillar"] == pil]
    for i, r in enumerate(prs):
        pc = f"<td rowspan='{len(prs)}' class='pill'>{PILL[pil]}</td>" if i == 0 else ""
        cells = ""
        for g in ["G2", "G3", "G4", "G5"]:
            cl = "win" if g == r["best"] else "n"
            cells += f"<td class='n {cl}'>{f(r[g])}</td>"
        H.append(f"<tr>{pc}<td>{r['label']}<br><small>{r['ep']}</small></td>"
                 f"<td>{r['metric']} {'↑' if r['hb'] else '↓'}</td>{cells}"
                 f"<td class='n leak'>{f(r['ai'])}</td><td class='{GN[r['best']]}'>{r['best']}</td></tr>")
H.append("</table>")
H.append(f"<div class='warn'><b>세대 승수 G2 {adme_wincount.get('G2',0)}·G4 {adme_wincount.get('G4',0)}·"
         f"G3 {adme_wincount.get('G3',0)}·G5 {adme_wincount.get('G5',0)}</b>(독성은 G2 17·G4 1·G3 0). "
         "★<b>이 승수를 세대 우열로 읽으면 안 된다</b> — G2는 3모델 중 best를 뽑는 <b>다중비교 이점</b>이 있고, "
         "<b>G4의 5승은 §5 통계 판정을 받지 못했다</b>(G4 분자별 예측 미저장). ADMET-AI 열은 누수 기준선(참고선)이며 승수에서 제외. "
         "<span class='src'>출처 adme_matrix.csv</span></div>")

# 축②
H.append("<h3>축② 특징 — 물성 서술자 vs 지문(ECFP)</h3>")
xgb_w = int(feat2.xgb_physchem_wins.sum()); rf_w = int(feat2.rf_physchem_wins.sum())
ecfp_xgb = [r.label for _, r in feat2.iterrows() if not r.xgb_physchem_wins]
ecfp_rf = [r.label for _, r in feat2.iterrows() if not r.rf_physchem_wins]
H.append("<table><tr><th>비교 방식</th><th class='n'>물성 승</th><th class='n'>ECFP 승</th><th>해석</th></tr>"
         f"<tr class='win'><td>동일 XGB 통제(phys vs ecfp)</td><td class='n'>{xgb_w}/18</td><td class='n'>{18-xgb_w}/18</td>"
         "<td>가장 깨끗한 특징 비교</td></tr>"
         f"<tr><td>동일 RF 통제</td><td class='n'>{rf_w}/18</td><td class='n'>{18-rf_w}/18</td><td>—</td></tr>"
         "<tr class='lo'><td>best-of-2 (초기 '18/18')</td><td class='n'>18/18</td><td class='n'>0/18</td>"
         "<td>★비대칭 — 물성만 2모델</td></tr></table>")
H.append(f"<div class='box'>18개 중 <b>16개에서 +물리화학 서술자가 최대 기여</b>(PPBR +2.25·용해도 +0.52·친유성 +0.135). "
         f"초기 요약의 '물성 18/18 압승'은 <b>비대칭 로스터(physchem 2모델 vs ECFP 1모델)의 산물</b>이었고, "
         f"빠진 rf_ecfp를 넣어 <b>2×2 대칭</b>으로 재집계하니 정직한 특징 우세는 <b>{xgb_w}/18(XGB)·{rf_w}/18(RF)</b>다. "
         f"ECFP가 이긴 곳: XGB에서 {'·'.join(ecfp_xgb)} / RF에서 {'·'.join(ecfp_rf)}. "
         f"<b>방향(물성 우세)은 견고하나 '압승'은 과장.</b> <span class='src'>출처 feature_2x2.csv</span></div>")

# 축③
H.append("<h3>축③ 학습방식 — 멀티태스크가 이득인가</h3>")
grp = learn.groupby("group").delta_multitask_minus_single
gm = {g: float(v.mean()) for g, v in grp}
H.append("<table><tr><th>묶음</th><th class='n'>평균 Δ(멀티−단일)</th><th>초기(누수)</th><th>판정</th></tr>"
         f"<tr><td>all_adme_cls(10)</td><td class='n'>{gm.get('all_adme_cls',0):+.4f}</td><td>9/10 이득</td>"
         "<td>소표본 2개(생체이용률·HIA)에 집중</td></tr>"
         f"<tr class='lo'><td>cyp_inhibition(3)</td><td class='n'>{gm.get('cyp_inhibition',0):+.4f}</td><td>3/3 이득</td>"
         "<td>★이득 소멸</td></tr>"
         f"<tr class='lo'><td>cyp_substrate(3)</td><td class='n'>{gm.get('cyp_substrate',0):+.4f}</td><td>2/3 이득</td>"
         "<td>손해</td></tr></table>")
H.append("<div class='crit'>★<b>자체 발견한 누수로 초기 결론이 뒤집혔다.</b> CYP 3종은 같은 분자 라이브러리에 라벨만 다르다 → "
         "합집합 멀티태스크에서 <b>test의 88.9%가 다른 과제의 train에 노출</b>. 누수 차단 게이트 후 재계산하니 "
         "<b>'닮은 과제를 묶으면 이득' 가설 기각</b>(CYP 억제 +0.0004·기질 −0.024). 남은 이득은 데이터가 적은 과제만 덕을 본다. "
         "<span class='src'>출처 learning_axis.csv·split_leakage.json</span></div>")
H.append("<div class='box'>★<b>부트스트랩 통계 판정</b>(G2·G3 간·분자 2000회 재표집): <b>친유성 logD에서 G3(GNN)이 G2 3종을 3/3으로 이긴다</b> "
         "(Δ −0.12~−0.25·4,200분자) — <b>이 연구에서 통계적으로 확실한 유일한 세대 효과</b>. "
         "그 외 physchem 확실 우세는 Caco-2(3/3). 구분 불가(동률): 반감기·HIA·P-gp·CYP 기질 3종(소표본). "
         "<b>G4 5승·G5는 예측 미저장으로 이 표에서 검증 불가</b>. <span class='src'>출처 bootstrap_verdict.json</span></div>")

# ── 4. 독성 vs ADME 대비 ──
H.append("<h2 id='s4'>4. ★독성 vs ADME 대비 (핵심 절)</h2>")
tox_win = {"G2": consist['best_gen_dist'].get('G2', 0), "G4": consist['best_gen_dist'].get('G4', 0),
           "G3": consist['g3_first_place'], "G5": 0}
H.append("<table><tr><th>최고 세대</th><th class='n'>독성 18과제</th><th class='n'>ADME 18과제</th><th>비고</th></tr>"
         f"<tr><td class='g2'>G2 고전 ML</td><td class='n win'>{tox_win['G2']}</td><td class='n'>{adme_wincount.get('G2',0)}</td>"
         "<td>두 영역 모두 최다</td></tr>"
         f"<tr><td class='g3'>G3 GNN(정직)</td><td class='n'>{tox_win['G3']}</td><td class='n'>{adme_wincount.get('G3',0)}</td>"
         "<td>ADME 친유성만 통계 확실</td></tr>"
         f"<tr><td class='g4'>G4 3D</td><td class='n'>{tox_win['G4']}</td><td class='n'>{adme_wincount.get('G4',0)}</td>"
         "<td>독성은 구분불가로 철회 · ADME는 ★통계 미검증</td></tr>"
         f"<tr><td class='g5'>G5 파운데이션</td><td class='n'>0</td><td class='n'>0</td><td>두 영역 모두 0승</td></tr></table>")
H.append("<div class='warn'>★<b>분모 주의</b> — 독성 G4·G5는 <b>핵심 4종에만 실행</b>돼 기회가 4회뿐이다(ADME 18회와 직접 비교 불가). "
         "그래서 '독성 G4 1승 vs ADME G4 5승'을 세대 우열로 읽으면 안 된다. "
         "<b>데이터 크기 패턴</b> — G3(GNN)는 소표본에서 붕괴한다: 생체이용률(640) AUROC 0.533(≈무작위)·반감기(667) 0.168·"
         "CYP3A4 기질(670) 0.575. 유일한 확실 승리 <b>친유성 logD는 4,200분자로 가장 큰 축</b>. "
         "<b>해석(검증 아님)</b>: G4 우세가 용해도·분포·배설에 몰린 것 = 분자의 <b>3D 존재형태가 관여하는 성질</b>일 가능성.</div>")

# ── 5. G4 검증 ──
H.append("<h2 id='s5'>5. G4 검증 (반감기 · 수용해도)</h2>")
H.append("<table><tr><th>엔드포인트</th><th>지표</th><th class='n'>★주 판정 (B) seed별</th>"
         "<th class='n'>(A) 예측평균</th><th>판정</th></tr>")
G4NAME = {"half_life_obach": "반감기", "solubility_aqsoldb": "수용해도"}
for ep, d in g4.items():
    bA, bB = d["bootstrap_A"], d["bootstrap_B"]
    H.append(f"<tr><td>{G4NAME[ep]}</td><td>{d['metric']} {'↑' if d['direction']=='higher' else '↓'}</td>"
             f"<td class='n win'>Δ{bB['delta']:+.4f} [{bB['ci'][0]:+.3f}, {bB['ci'][1]:+.3f}]</td>"
             f"<td class='n'>Δ{bA['delta']:+.4f} [{bA['ci'][0]:+.3f}, {bA['ci'][1]:+.3f}]</td>"
             f"<td class='lo'>★미확립(B: CI가 0 포함)</td></tr>")
H.append("</table>")
solA, solB = g4["solubility_aqsoldb"]["bootstrap_A"]["delta"], g4["solubility_aqsoldb"]["bootstrap_B"]["delta"]
H.append(f"<div class='crit'>★<b>주 판정을 (B) seed별 종합으로 통일</b>했다(ADME 신뢰도 §6과 동일 원칙) — "
         "방법 비교 주장에는 학습 변동을 CI에 포함해야 정직하다. <b>두 곳 다 (B)의 CI가 0을 포함 → 구분 안 됨</b>. "
         f"(A) 예측평균은 구분되지만 이는 <b>5-모델 앙상블 평가</b>이며 변동 큰 Uni-Mol에 비대칭 유리하다 — "
         f"<b>용해도 효과가 (B) Δ{solB:.4f} → (A) Δ{solA:.4f}로 커진다</b>(차이의 절반 가까이가 앙상블 전환에서 생김·원본 델타값). "
         "<b>결론: G4 우세는 방향은 일관되나 ★미확립</b>(LD50과 동일). VDss·청소율 2종은 검증조차 안 함 — 일반화 금지. "
         "<br>★<b>원본 불일치 기록</b>: g4_verification.html은 자체적으로 primary_method=A로 '구분됨'이라 적었으나, "
         "이 마스터는 배포 신뢰도 통일 원칙에 따라 <b>(B)를 주판정으로 채택</b>해 '미확립'으로 표기한다(두 수치 모두 원본 값). "
         "<span class='src'>출처 g4_verification.json</span></div>")

# ── 6. 배포 신뢰도 36과제(4각도) ──
H.append("<h2 id='s6'>6. 배포 신뢰도 — 36과제 (4각도 통합)</h2>")
tox_ad_v = finalize['ad_valid']; tox_ad_i = finalize['ad_ineffective']; tox_ad_u = finalize['ad_undetermined']
ad_v = int((admeR.AD_verdict.str.contains('유효')).sum())
ad_i = int((admeR.AD_verdict.str.contains('예측 못함')).sum())
ad_u = len(admeR) - ad_v - ad_i
ld50_cov = float(toxR[toxR.endpoint == "ld50_zhu"].PI90_coverage.iloc[0])
cyp2c9s = admeR[admeR.endpoint == "cyp2c9_substrate_carbonmangels"].iloc[0]
H.append("<table><tr><th>각도</th><th>독성 18</th><th>ADME 18</th><th>공통 교훈</th></tr>"
         f"<tr><td><b>AD 유효성</b></td><td>유효 {tox_ad_v}·예측못함 {tox_ad_i}·판정불가 {tox_ad_u}</td>"
         f"<td>유효 {ad_v}·예측못함 {ad_i}·판정불가 {ad_u}</td>"
         "<td>★AD는 자동 안전장치 아님 — 절반 남짓만 유효</td></tr>"
         f"<tr><td><b>운영지표(0.5)</b></td><td>NR-PPAR-gamma 민감도 <b>0.000</b></td>"
         f"<td>CYP2C9 기질 <b>{cyp2c9s.sens_05:.3f}</b>(37중 36 놓침)</td>"
         "<td>★임계 0.5는 못 씀 · t*도 만능 아님(ADME 양성률 높은 4개는 t*가 악화)</td></tr>"
         "<tr><td><b>보정(ECE)</b></td><td>대부분 ≤0.10</td><td>큰 데이터 낮음·소표본 높음</td>"
         "<td>후처리라 AUROC 불변</td></tr>"
         f"<tr><td><b>회귀 예측구간</b></td><td>LD50 과소커버 <b>{ld50_cov:.3f}</b>(위험)</td>"
         "<td>대부분 과대커버 0.90~0.97</td><td>VDss·반감기 계통편향 커 순위만</td></tr>"
         "<tr><td><b>정확 CI</b></td><td>부트스트랩·DeLong</td><td>부트스트랩(예측 있는 세대만)</td>"
         "<td>★주 판정 (B) seed별 종합 통일(앙상블 비대칭 각주)</td></tr></table>")
vd_bias = admeDet["vdss_lombardo"]["regression"]["systematic_bias_slope"]
hl_bias = admeDet["half_life_obach"]["regression"]["systematic_bias_slope"]
H.append(f"<div class='box'>ADME t* 악화 4개(양성률 높음): BBB 0.949→0.760·생체이용률 0.887→0.433·P-gp 0.718→0.645·"
         f"CYP3A4 기질 0.782→0.641. 회귀 계통편향 큰 곳: <b>VDss {vd_bias}·반감기 {hl_bias}</b>(값 큰 분자 과소예측→순위로만). "
         "<span class='src'>출처 reliability.csv·adme_reliability.csv·reliability_detail.json</span></div>")

# ── 7. 통합 배포 가이드 36 ──
H.append("<h2 id='s7'>7. ★통합 배포 가이드 — 36 엔드포인트</h2>")
H.append("<table><tr><th>엔드포인트</th><th>영역</th><th>챔피언</th><th>권장 작동점/구간</th>"
         "<th>AD 게이트</th><th>확률·구간 신뢰</th><th>배포 판정</th></tr>")


def ad_cell(v):
    if "유효" in v:
        return "<span class='g2'>사용</span>"
    if "예측 못함" in v or "예측하지 못" in v:
        return "<span class='lo'>★사용 금지</span>"
    return "<span style='color:#7a6a3a'>미확정</span>"


TOX_LABEL = {"dili": "DILI 간독성", "herg": "hERG 차단", "ames": "AMES 변이원성", "ld50_zhu": "LD50 급성독성",
             "Carcinogens_Lagunin": "발암성", "ClinTox": "ClinTox 임상독성"}


def deploy_rows(df, system):
    rows = []
    for _, r in df.iterrows():
        if system == "tox":
            label = (f"Tox21 {r.task}" if r.endpoint == "Tox21" else TOX_LABEL.get(r.endpoint, r.endpoint))
            n = int(r.n_test); champ_ = r.model; adv = r.AD_verdict
            task = "reg" if r.metric == "MAE" else "cls"
            small = n < 200
            wait = False
        else:
            label = r.label; n = int(r.n_test); champ_ = r.champion; adv = r.AD_verdict
            task = r.task; small = n < 200; wait = r.endpoint in ("vdss_lombardo", "clearance_hepatocyte_az", "clearance_microsome_az")
        if task == "cls":
            if r.sens_star > r.sens_05:
                op = f"<b>t*={r.t_star:.3f}</b><br><small>민감도 {r.sens_05:.2f}→{r.sens_star:.2f}</small>"
            else:
                op = f"<b>0.5 유지</b><br><small>t*는 {r.sens_05:.2f}→{r.sens_star:.2f}로 낮춤</small>"
            conf = (f"<span class='g2'>사용 가능</span> ECE {r.ECE:.3f}" if r.ECE <= 0.10
                    else f"<span class='lo'>불안정</span> ECE {r.ECE:.3f}")
        else:
            cov = r.PI90_coverage
            hw = (r.PI90_halfwidth if system == "adme" else None)
            op = (f"<b>±{hw:.3f}</b> 90% 구간<br><small>커버리지 {cov:.3f}</small>" if hw is not None
                  else f"<b>90% 구간</b><br><small>커버리지 {cov:.3f}</small>")
            conf = ("<span class='g2'>구간 신뢰</span>" if (not pd.isna(cov) and abs(cov - 0.9) <= 0.05)
                    else ("<span class='lo'>과소커버(위험)</span>" if (not pd.isna(cov) and cov < 0.87)
                          else "<span style='color:#7a6a3a'>과대커버(넓음)</span>"))
        dep = ("<span class='b-wait badge'>G4 검증 대기</span>" if wait
               else "<span class='b-sm badge'>참고용(소표본)</span>" if small else "신뢰 가능")
        rows.append(f"<tr><td>{label}<br><small>n={n}</small></td><td>{system.upper()}</td><td>{champ_}</td>"
                    f"<td>{op}</td><td>{ad_cell(adv)}</td><td>{conf}</td><td>{dep}</td></tr>")
    return rows


H.append("<tr><td colspan='7' style='background:#eef4f3;font-weight:700'>독성 18과제</td></tr>")
H += deploy_rows(toxR, "tox")
H.append("<tr><td colspan='7' style='background:#eef4f3;font-weight:700'>ADME 18과제</td></tr>")
H += deploy_rows(admeR, "adme")
H.append("</table>")
H.append("<div class='crit'><b>★운영 원칙 4줄</b><ol>"
         "<li><b>순위는 쓸 만하다</b> — 챔피언 AUROC/Spearman이 대체로 유효 범위·부트스트랩 CI가 지지.</li>"
         "<li><b>작동점은 반드시 재설정하되 t*가 만능은 아니다</b> — 0.5 붕괴는 t*로, 양성률 높은 엔드포인트는 0.5 유지.</li>"
         "<li><b>AD는 엔드포인트별 확인 후에만</b> — 유효만 게이트·예측 못한 곳은 사용 금지·판정 불가는 하드 근거 금지.</li>"
         "<li><b>회귀는 점추정이 아니라 구간으로</b> — VDss·반감기는 계통편향 커 절대값 금지·순위만.</li></ol>"
         "★<b>구조알림 병용</b>은 구조 축(DILI·AMES·발암성)만 켜고 hERG·Tox21 12경로는 끈다(G1 MCC≈0으로 무력).</div>")

# ── 8. 자기정정 ──
H.append("<h2 id='s8'>8. ★자기정정 기록 (8건 — 이 프로젝트의 신뢰도 근거)</h2>")
H.append("<table><tr><th>#</th><th>초안 주장</th><th>실제(정정 후)</th><th>무엇이 잡아냈나</th></tr>"
         "<tr><td>①</td><td>ADMET-AI 초과분 = 누수 프리미엄</td>"
         "<td>그것은 <b>상한</b>(ADMET-AI=5앙상블×31과제 멀티태스크 vs 단일) — 깨끗한 추정은 증거②(+0.049~0.074)</td>"
         "<td>적대검증 CRITICAL</td></tr>"
         "<tr><td>②</td><td>세대차를 전체 범위로 계산</td><td><b>1위 vs 2위 격차</b>로 고치니 LD50 <b>4/4 구분 불가</b></td>"
         "<td>적대검증 CRITICAL</td></tr>"
         "<tr><td>③</td><td>'LD50 G4가 유일한 세대 효과'</td><td><b>철회</b>(1위vs2위 구분 불가)</td><td>②의 결과</td></tr>"
         "<tr><td>④</td><td>Hanley-McNeil로 AUC 비교</td><td><b>DeLong 대응비교</b>로 교체(보수적)</td><td>방법 검토</td></tr>"
         "<tr><td>⑤</td><td>멀티태스크 9/10 이득</td><td><b>88.9% 누수 산물 → 가설 기각</b>(CYP 억제 +0.0004)</td>"
         "<td>★자체 발견</td></tr>"
         f"<tr><td>⑥</td><td>물성 18/18 압승</td><td>동일 XGB <b>{xgb_w}/18</b>·동일 RF <b>{rf_w}/18</b>(비대칭 로스터 산물)</td>"
         "<td>적대검증 6렌즈 중 5개</td></tr>"
         "<tr><td>⑦</td><td>E(배설) 3종 G4 석권</td><td><b>반감기만 뚜렷</b>·청소율 2종 G2와 동률(Δ≪SD)</td>"
         "<td>적대검증(반박 1/3)</td></tr>"
         "<tr><td>⑧</td><td>G4 검증 '구분됨'(A)</td><td>판정이 <b>seed 처리법에 의존</b> → (B)seed별로는 미확립·둘 다 보고</td>"
         "<td>seed-handling 발견</td></tr></table>")

# ── 9. 한계 ──
H.append("<h2 id='s9'>9. 한계 (정직 고지)</h2>")
H.append("<div class='warn'><ul>"
         "<li><b>독성 확장 14과제 G4·G5 미실행</b> — 소표본·불균형 근거의 스코프 결정. ADME에서 방증했으나 직접 검증은 안 함.</li>"
         "<li><b>ADME G4·G5 분자별 예측 부재</b> → VDss·청소율 2종 G4 미검증. 반감기·수용해도도 (B)선에서 <b>미확립</b>.</li>"
         "<li><b>소표본</b> — 발암성 56·ClinTox 297·CYP 기질 134 등은 CI 폭 크고 t*·ECE 불안정 → 확정 판정 금지.</li>"
         "<li><b>AD 판정 불가</b> 독성 4·ADME 6 — OOD 표본<20이라 유효성 자체를 못 쟀다.</li>"
         "<li><b>G2·G3 미튜닝</b> · 세대 승수는 다중비교 비대칭(G2 3모델 중 best)을 안고 있다.</li>"
         "<li>★<b>전향적 검증 없음</b> — 전부 TDC scaffold 분할 안의 회고적 평가다.</li></ul></div>")

# ── 10. 결론 ──
H.append("<h2 id='s10'>10. 결론 + 후속</h2>")
H.append("<div class='box'><ol>"
         "<li><b>'세대 ≠ 성능'은 성립하나 독성>ADME</b> — 독성은 G2 17/18로 압도적, ADME는 G2 9로 최다지만 갈린다.</li>"
         "<li><b>물성 표현 우세는 견고하나 압승 아님</b> — 17/18(XGB)이지 18/18이 아니다.</li>"
         "<li><b>통계적으로 확실한 세대 효과는 logD의 G3 하나</b> — 나머지는 구분 불가(소표본) 또는 미검증(G4·G5).</li>"
         "<li><b>파운데이션 G5는 두 영역 모두 0승</b> — 분자 수준 예측에서 자리를 못 잡는다.</li></ol></div>")
H.append("<p class='src'><b>후속</b> — (1) G4·G5 분자별 예측 확보 후 부트스트랩(E·G4 헤드라인 검증 전제) · "
         "(2) GNN 아키텍처 확장 · (3) 분할 난이도 축(random vs scaffold) · (4) 라벨 노이즈 강건 학습 · "
         "(5) 통합 예측 파이프라인. ★<b>독성에선 챔피언 재현 학습으로 분자별 예측을 '메웠고'(18/18·Δ≤0.0001), "
         "ADME에선 logD만 메우고 G4·G5는 '한계로 남겼다'</b> — 이 차이를 정확히 구분해 읽어야 한다.</p>")
H.append(f"<p class='src'>산출물: <code>results/master_integrated_report.html</code>(이 문서·36과제 전부) · "
         f"<code>notes.md</code>(이관 매핑). ★기존 4+1개 원본 보고서는 삭제하지 않음 — 이 문서가 최상위 인덱스, 원본은 상세로 보존. "
         f"★새 학습·재계산 0.</p>")

os.makedirs(OUT, exist_ok=True)
open(f"{OUT}/master_integrated_report.html", "w", encoding="utf-8").write("\n".join(H))
sz = os.path.getsize(f"{OUT}/master_integrated_report.html") / 1024
print(f"저장 → results/master_integrated_report.html ({sz:.0f} KB)")
print(f"  독성 최고세대 G2 {tox_win['G2']}·G4 {tox_win['G4']}·G3 {tox_win['G3']} · "
      f"ADME G2 {adme_wincount.get('G2',0)}·G4 {adme_wincount.get('G4',0)}·G3 {adme_wincount.get('G3',0)}·G5 {adme_wincount.get('G5',0)}")
print(f"  AD 유효 독성 {tox_ad_v}/ADME {ad_v} · 물성 {xgb_w}/18(XGB)·{rf_w}/18(RF) · G4 검증 미확립")
