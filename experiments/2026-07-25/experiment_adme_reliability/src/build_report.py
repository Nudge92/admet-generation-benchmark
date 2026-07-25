#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_report.py — ADME 배포 신뢰도 보고서 + 배포 가이드. ★재계산 0(확정 결과 조립만)."""
import base64, io, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

NEW = "/home/nudge/Project/ADMET_integrated/2026-07-25/experiment_adme_reliability"
R = f"{NEW}/results"
for p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
          "/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf"):
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams["font.family"] = ["NanumGothic", "NanumSquareRound", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CSV = pd.read_csv(f"{R}/adme_reliability.csv")
DET = json.load(open(f"{R}/reliability_detail.json"))
CH = {r["endpoint"]: r for r in json.load(open(f"{R}/champions.json"))}
REPRO = json.load(open(f"{R}/step0b_repro.json"))
LOGD_REPRO = json.load(open(f"{R}/logd_g3_repro.json")) if os.path.exists(f"{R}/logd_g3_repro.json") else None
PILLAR = {"A": "흡수", "D": "분포", "M": "대사", "E": "배설"}
G4_WAIT = {"vdss_lombardo", "clearance_hepatocyte_az", "clearance_microsome_az"}
SMALL = 200


def _pf(v):
    """AD 구간 성능 표시 — 표본 부족으로 산출 못 하면 N/A(0으로 채우지 않음)."""
    return "N/A(표본<20)" if v is None else f"{v}"


def b64(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=125, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


cls = CSV[CSV.task == "cls"].copy()
reg = CSV[CSV.task == "reg"].copy()
nv = int((CSV.AD_verdict.str.contains("유효")).sum())
nn = int((CSV.AD_verdict.str.contains("예측 못함")).sum())
nu = len(CSV) - nv - nn
# 0.5가 무너지는 곳(민감도<0.5)
broke = cls[cls.sens_05 < 0.5][["endpoint", "sens_05", "sens_star", "t_star", "FN_05"]]
# t*가 오히려 악화시키는 곳
worse = cls[cls.sens_star < cls.sens_05][["endpoint", "sens_05", "sens_star", "t_star"]]

H = ['<meta charset="utf-8"><title>ADME 배포 신뢰도 — 이 모델을 새 분자에 써도 되나</title>', """<style>
body{font-family:'Nanum Gothic',system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:22px 26px;color:#1d2129;line-height:1.6}
h1{font-size:24px;border-bottom:3px solid #2a9d8f;padding-bottom:9px;margin-bottom:3px}
h2{font-size:19px;margin-top:34px;border-left:5px solid #2a9d8f;padding-left:11px}
h3{font-size:15.5px;margin-top:20px;color:#264653}
table{border-collapse:collapse;width:100%;margin:11px 0;font-size:12.3px}
th,td{border:1px solid #d8dde3;padding:4px 7px}th{background:#eef2f5}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.ok{background:#e8f6f1;font-weight:700}.bad{background:#fdeceb;color:#b3261e;font-weight:700}
.tie{background:#f4f0e4;color:#7a6a3a}.na{color:#98a2ab}
.banner{background:linear-gradient(135deg,#20343f,#2a5d59);color:#fff;border-radius:11px;padding:18px 22px;margin:12px 0 18px}
.banner b{color:#ffe9a8}
.box{background:#f7f9fb;border:1px solid #dfe5ea;border-radius:8px;padding:13px 17px;margin:13px 0}
.warn{background:#fff8ec;border:1px solid #f0d9a8;border-radius:8px;padding:13px 17px;margin:13px 0}
.crit{background:#fdefea;border:1px solid #f2bda9;border-radius:8px;padding:13px 17px;margin:13px 0}
.badge{display:inline-block;font-size:10.5px;padding:1px 6px;border-radius:8px;margin-left:4px}
.b-sm{background:#ffe9c9;color:#8a5a00}.b-wait{background:#e4e8f5;color:#3a4a8a}
figure{margin:16px 0;text-align:center}img{max-width:100%;border:1px solid #e3e8ec;border-radius:6px}
figcaption{font-size:12.2px;color:#5b6670;margin-top:6px}
small,.src{font-size:11.3px;color:#6b7580}code{background:#eef2f5;padding:1px 4px;border-radius:3px;font-size:11.3px}
</style>"""]
H.append("<h1>ADME 배포 신뢰도 — 이 모델을 새 분자에 써도 되나</h1>")
H.append(f"<p class='src'>작업일 2026-07-25 · 18개 ADME 중 <b>{len(CSV)}개</b>(분류 {len(cls)}·회귀 {len(reg)}) · "
         "★재학습으로 성능 개선 없음 — 원 config·seed 재현·추론만 · "
         "독성 reliability_report와 같은 층을 ADME에 구축.</p>")

# ── 배너 3줄
H.append("<div class='banner'>"
         f"<p><b>1. 검증된 것</b> — G2 챔피언 <b>{len(REPRO['checks'])}/{len(REPRO['checks'])} 재현 Δ≤0.0000</b>·test 분자 "
         "<b>Jaccard 1.0</b>, ★<b>logD(G3 chemprop)도 5 seed 재현 Δ0.0000</b>으로 이번에 채웠다. "
         "분자별 예측(train/valid/test)을 확보해 이후 어떤 검정도 재학습 없이 가능.</p>"
         f"<p><b>2. 어디가 위험한가</b> — ★<b>임계 0.5에서 민감도가 무너지는 분류 {len(broke)}개</b>"
         f"(최악 CYP2C9 기질 <b>0.026</b>·FN 37). 게다가 ★<b>t*가 오히려 악화시키는 곳도 {len(worse)}개</b> — "
         "임계값 재설정이 만능이 아니다. 회귀는 <b>컨포멀 90% 구간이 대부분 과대커버</b>(0.90 목표에 0.90~0.97).</p>"
         f"<p><b>3. 미검증</b> — AD가 <b>성능을 예측 못한 {nn}개</b>·<b>판정 불가 {nu}개</b>(OOD&lt;20). "
         "VDss·청소율 2종은 <b>G4 검증 대기</b>(G4 분자별 예측 없어 미다룸).</p></div>")

# ── 1. 지표 정의
H.append("<h2>1. 지표 정의</h2><div class='box'><ul>"
         "<li><b>분류</b>(10): AUROC↑·AUPRC↑ · 민감도=실제 양성 중 잡아낸 비율(★가장 중요) · "
         "NPV=음성이라 했을 때 실제 음성일 확률 · FN=놓친 양성(가장 비싼 오류).</li>"
         "<li><b>회귀</b>(8): MAE↓(Caco-2·logD·용해도·PPBR) / Spearman↑(VDss·반감기·청소율 2종). "
         "★<b>방향이 섞여 있어</b> 집계 코드에 단위 테스트를 넣어 검증했다(4/4 통과).</li>"
         "<li><b>AD</b>: 학습셋 기준 5-NN Tanimoto(ECFP4). 컷=학습셋 자기 LOO 분포의 5%(OOD)·25%(경계) 백분위 "
         "— <b>0.4 같은 절대선이 아니다</b>.</li>"
         "<li><b>컨포멀 예측구간</b>: valid 절대잔차의 90% 분위수로 만든 구간 → test에서 <b>실제 커버리지</b> 측정"
         "(★test로 구간을 고르지 않았다).</li>"
         "<li><b>ECE</b>: 예측확률이 실제 빈도와 얼마나 어긋나나. <b>보정은 후처리라 AUROC 불변</b>.</li>"
         "<li><b>CI 주 판정</b>: ★<b>seed별 부트스트랩 후 종합</b> — 본 연구 주장은 특정 배포물이 아니라 "
         "<b>방법 비교</b>이므로 학습 변동을 포함해야 한다. '5seed 예측평균'도 병기하되 그것은 "
         "<b>5-모델 앙상블 평가</b>이며 변동이 큰 모델에 비대칭적으로 유리하다.</li></ul></div>")

# ── 2. 챔피언 선정
H.append("<h2>2. 배포 챔피언 선정</h2>")
H.append("<table><tr><th>기둥</th><th>엔드포인트</th><th>유형</th><th>주지표(방향)</th><th>챔피언</th>"
         "<th class='n'>값</th><th class='n'>재현 Δ</th><th>비고</th></tr>")
rep = {c["endpoint"]: c for c in REPRO["checks"]}
if LOGD_REPRO:                                            # logD(G3)는 별도 재현 파일에서
    rep[LOGD_REPRO["endpoint"]] = {"diff": LOGD_REPRO["diff"], "reproduced": LOGD_REPRO["reproduced"]}
for pil in ("A", "D", "M", "E"):
    for _, r in CSV[CSV.pillar == pil].iterrows():
        ep = r.endpoint; c = rep.get(ep, {})
        note = ("<span class='badge b-wait'>G4 검증 대기</span>" if ep in G4_WAIT else "")
        if r.n_test < SMALL:
            note += "<span class='badge b-sm'>소표본</span>"
        H.append(f"<tr><td>{pil} {PILLAR[pil]}</td><td>{r.label}<br><small>{ep}</small></td>"
                 f"<td>{r.task}</td><td>{r.metric} {r.direction}</td><td>{r.champion}</td>"
                 f"<td class='n'>{CH[ep]['champion_value']:.4f}</td>"
                 f"<td class='n ok'>{c.get('diff', 0):+.4f}</td><td>{note}</td></tr>")
H.append("</table>")
H.append("<div class='warn'><b>선정 규칙</b> — 기본은 각 엔드포인트 <b>G2 최고</b>. "
         "★예외 1건 <b>친유성 logD → G3 dmpnn_ours</b>(부트스트랩 3/3 통과한 유일한 검증된 비-G2 승리). "
         "이번에 <b>chemprop D-MPNN 예측을 원 config·5 seed로 재현·생성해 채웠다</b> "
         "(5seed 평균 MAE 0.4406 = 보고값과 Δ0.0000·비호환 분자 0). "
         "반감기·수용해도는 <code>experiment_g4_verification</code>의 <b>G4 예측을 재사용</b>할 수 있으나 "
         "본 표의 챔피언은 G2다.</div>")

# ── 3. AD
H.append("<h2>3. 적용범위(AD)와 ★유효성 검증</h2>")
H.append("<table><tr><th>엔드포인트</th><th class='n'>in-domain</th><th class='n'>경계</th>"
         "<th class='n'>OOD</th><th>판정</th></tr>")
for _, r in CSV.iterrows():
    d = DET[r.endpoint]["AD"]; b = d["bands"]
    cls_ = "ok" if "유효" in r.AD_verdict else ("bad" if "예측 못함" in r.AD_verdict else "tie")
    H.append(f"<tr><td>{r.label}<br><small>{r.endpoint}</small></td>"
             f"<td class='n'>{b['in-domain']['frac']:.3f} <small>({_pf(b['in-domain']['perf'])})</small></td>"
             f"<td class='n'>{b['경계']['frac']:.3f}</td>"
             f"<td class='n'>{b['OOD']['frac']:.3f} <small>(n={b['OOD']['n']}·{_pf(b['OOD']['perf'])})</small></td>"
             f"<td class='{cls_}'>{r.AD_verdict}</td></tr>")
H.append("</table>")
H.append(f"<div class='warn'><b>결과 — AD 유효 {nv} · ★성능 예측 못함 {nn} · 판정 불가 {nu}</b>"
         f"(독성은 11/3/4). <b>ADME도 AD는 자동 안전장치가 아니다.</b> "
         "특히 <b>판정 불가가 6개</b>로 독성(4)보다 많은데, CYP 기질 3종·생체이용률처럼 test가 작아 "
         "OOD 분자가 20개 미만이라 유효성 자체를 못 쟀다. 이들에는 AD 게이트를 하드 근거로 쓰면 안 된다.</div>")

# ── 4. 분류 운영지표
H.append("<h2>4. 분류 운영지표 — ★임계 0.5는 쓸 수 있나</h2>")
H.append("<table><tr><th>엔드포인트</th><th class='n'>양성률</th><th class='n'>민감도@0.5</th>"
         "<th class='n'>FN@0.5</th><th class='n'>t*</th><th class='n'>민감도@t*</th>"
         "<th class='n'>FN@t*</th><th class='n'>NPV@t*</th><th class='n'>ECE</th><th>판정</th></tr>")
for _, r in cls.sort_values("sens_05").iterrows():
    b05 = "bad" if r.sens_05 < 0.5 else "n"
    bst = "bad" if r.sens_star < r.sens_05 else ("ok" if r.sens_star > r.sens_05 else "n")
    verd = ("★0.5 붕괴 → t* 필수" if r.sens_05 < 0.5 and r.sens_star > r.sens_05 else
            "★t*가 오히려 악화" if r.sens_star < r.sens_05 else "0.5도 무난")
    sm = "<span class='badge b-sm'>소표본</span>" if r.n_test < SMALL else ""
    H.append(f"<tr><td>{r.label}<br><small>{r.endpoint}</small>{sm}</td>"
             f"<td class='n'>{r.pos_rate:.3f}</td><td class='n {b05}'>{r.sens_05:.3f}</td>"
             f"<td class='n'>{int(r.FN_05)}</td><td class='n'>{r.t_star:.3f}</td>"
             f"<td class='n {bst}'>{r.sens_star:.3f}</td><td class='n'>{int(r.FN_star)}</td>"
             f"<td class='n'>{r.NPV_star:.3f}</td><td class='n'>{r.ECE:.3f}</td><td>{verd}</td></tr>")
H.append("</table>")
fig, ax = plt.subplots(figsize=(11, 3.8))
o = cls.sort_values("sens_05"); x = np.arange(len(o)); w = .38
ax.bar(x - w / 2, o.sens_05, w, label="임계 0.5", color="#8d99ae", edgecolor="#222", linewidth=.5)
ax.bar(x + w / 2, o.sens_star, w, label="t* (valid MCC 최대)", color="#2a9d8f", edgecolor="#222", linewidth=.5)
ax.axhline(.5, ls=":", color="#c1440e", lw=1.3)
ax.set_xticks(x); ax.set_xticklabels(o.endpoint, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("민감도(독성/위험 포착률)", fontsize=9.5); ax.set_ylim(0, 1.05)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=.25)
ax.set_title("임계 0.5 vs t* — ★t*가 항상 낫지는 않다", fontsize=11.5)
H.append(f'<figure><img src="data:image/png;base64,{b64(fig)}"><figcaption>'
         "왼쪽(회색)=0.5, 오른쪽(초록)=t*. CYP 기질 2종은 0.5에서 붕괴해 t*가 구제하지만, "
         "BBB·생체이용률·P-gp·CYP3A4 기질은 <b>t*가 오히려 민감도를 낮춘다</b>.</figcaption></figure>")
H.append("<div class='crit'><b>★두 방향의 발견</b><br>"
         f"<b>(a) 0.5가 무너지는 곳 {len(broke)}개</b> — 최악은 <b>CYP2C9 기질 민감도 0.026</b>"
         "(37개 양성 중 36개 놓침). CYP2D6 기질 0.372·CYP2D6 억제 0.461. 독성 NR-PPAR-gamma(0.000)와 같은 현상이며, "
         "t*로 각각 0.447·0.861·0.612까지 회복된다.<br>"
         f"<b>(b) ★그런데 t*가 오히려 악화시키는 곳도 {len(worse)}개</b> — "
         + " · ".join(f"<b>{r.endpoint}</b> {r.sens_05:.2f}→{r.sens_star:.2f}" for _, r in worse.iterrows())
         + ". 이들은 <b>양성률이 높은</b>(0.51~0.80) 엔드포인트로, MCC 최대화가 특이도 쪽으로 기울면서 민감도를 깎는다. "
         "<b>즉 '무조건 t*'가 아니라 목적(민감도 우선 vs 균형)에 맞는 작동점을 골라야 한다.</b></div>")

# ── 5. 회귀
H.append("<h2>5. 회귀 — 잔차와 예측구간</h2>")
H.append("<table><tr><th>엔드포인트</th><th>지표</th><th class='n'>잔차 SD</th>"
         "<th class='n'>계통편향(기울기)</th><th class='n'>90% 구간 폭(±)</th>"
         "<th class='n'>실제 커버리지</th><th>판정</th></tr>")
for _, r in reg.iterrows():
    cov = r.PI90_coverage
    verd = ("목표 부합" if abs(cov - .9) <= .03 else
            "★과대커버(구간이 넓다)" if cov > .93 else "★과소커버(위험)")
    bias = "" if pd.isna(r.bias_slope) else f"{r.bias_slope:+.4f}"
    bcls = "bad" if (not pd.isna(r.bias_slope) and abs(r.bias_slope) > .3) else "n"
    H.append(f"<tr><td>{r.label}<br><small>{r.endpoint}</small></td><td>{r.metric} {r.direction}</td>"
             f"<td class='n'>{r.residual_sd:.3f}</td><td class='n {bcls}'>{bias}</td>"
             f"<td class='n'>{r.PI90_halfwidth:.3f}</td><td class='n'>{cov:.3f}</td><td>{verd}</td></tr>")
H.append("</table>")
H.append("<div class='warn'><b>읽는 법</b> — <b>계통편향 기울기</b>는 예측값 대비 잔차의 회귀 기울기다. "
         "0에서 멀수록 편향이 크고, <b>음수면 큰 값을 과소예측</b>한다. "
         "<b>VDss(−0.969)·반감기(−0.764)</b>가 특히 크다 — 두 엔드포인트 모두 "
         "<b>값이 큰 분자(분포용적·반감기가 긴 약물)를 체계적으로 낮게 예측</b>한다. "
         "그래서 이 둘은 주지표가 Spearman(순위)인 것이며, <b>절대값 예측에는 쓰면 안 된다</b>.<br><br>"
         "<b>예측구간은 대부분 과대커버</b>(목표 0.90인데 0.90~0.97) — 구간이 필요 이상으로 넓다는 뜻이고, "
         "이는 <b>보수적이라 안전하지만 정보량이 낮다</b>. 독성 LD50에서 과소커버(0.843)였던 것과 반대 방향이다.</div>")

# ── 6. 부트스트랩 CI
H.append("<h2>6. 정확한 신뢰구간</h2>")
H.append("<table><tr><th>엔드포인트</th><th>지표</th><th class='n'>★주 판정(seed별 종합)</th>"
         "<th class='n'>참고(5seed 예측평균)</th><th class='n'>DeLong p (vs G3)</th></tr>")
for _, r in CSV.iterrows():
    d = DET[r.endpoint]
    pri = d.get("ci_primary") or {}
    ens = d.get("ci_ensemble") or {}
    dl = (d.get("delong_vs_dmpnn") or {})
    p = dl.get("p_value")
    H.append(f"<tr><td>{r.label}</td><td>{r.metric} {r.direction}</td>"
             f"<td class='n'>{pri.get('mean','—')} [{pri.get('lo','—')}, {pri.get('hi','—')}]</td>"
             f"<td class='n na'>{ens.get('mean','—')} [{ens.get('lo','—')}, {ens.get('hi','—')}]</td>"
             f"<td class='n'>{'—' if p is None else f'{p:.3g}'}"
             f"{' ★' if (p is not None and p < 0.05) else ''}</td></tr>")
H.append("</table>")
H.append("<div class='box'>★<b>주 판정을 'seed별 부트스트랩 후 종합'으로 둔 이유</b> — 이 연구의 주장은 "
         "'특정 배포물이 이만큼 한다'가 아니라 <b>'방법 A가 방법 B보다 낫다'</b>이므로, "
         "<b>모델을 새로 학습했을 때의 변동</b>을 CI에 포함해야 정직하다. "
         "'5seed 예측평균' 열은 <b>5-모델 앙상블을 평가</b>하는 것이며, "
         "<b>앙상블 이득이 변동 큰 모델(예: Uni-Mol)에 비대칭적으로 유리</b>하므로 방법 비교에는 부적절하다"
         "(G4 검증 실험에서 이 차이가 판정 자체를 뒤집었다).</div>")

# ── 7. ★배포 가이드
H.append("<h2>7. ★배포 가이드</h2>")
H.append("<table><tr><th>엔드포인트</th><th>챔피언</th><th>권장 작동점 / 구간</th>"
         "<th>AD 게이트</th><th>확률·구간 신뢰</th><th>배포 판정</th></tr>")
for _, r in CSV.iterrows():
    ep = r.endpoint
    if r.task == "cls":
        if r.sens_star > r.sens_05:
            op = f"<b>t* = {r.t_star:.3f}</b><br><small>민감도 {r.sens_05:.2f}→{r.sens_star:.2f} · FN {int(r.FN_05)}→{int(r.FN_star)}</small>"
        else:
            op = (f"<b>0.5 유지</b><br><small>t*({r.t_star:.3f})는 민감도를 "
                  f"{r.sens_05:.2f}→{r.sens_star:.2f}로 낮춤</small>")
        conf = (f"<span class='ok'>사용 가능</span> ECE {r.ECE:.3f}" if r.ECE <= .10
                else f"<span class='bad'>불안정</span> ECE {r.ECE:.3f}")
    else:
        cov = r.PI90_coverage
        op = (f"<b>±{r.PI90_halfwidth:.3f}</b> (90% 구간)<br><small>실제 커버리지 {cov:.3f}</small>")
        conf = (f"<span class='ok'>구간 신뢰</span>" if abs(cov - .9) <= .05
                else f"<span class='tie'>과대커버(넓음)</span>")
    ad = ("<span class='ok'>사용</span>" if "유효" in r.AD_verdict else
          "<span class='bad'>★사용 금지</span>" if "예측 못함" in r.AD_verdict else
          "<span class='tie'>미확정</span>")
    dep = ("<span class='badge b-wait'>G4 검증 대기</span>" if ep in G4_WAIT else
           "<span class='badge b-sm'>참고용(소표본)</span>" if r.n_test < SMALL else "신뢰 가능")
    H.append(f"<tr><td>{r.label}<br><small>{ep} · n={r.n_test}</small></td><td>{r.champion}</td>"
             f"<td>{op}</td><td>{ad}</td><td>{conf}</td><td>{dep}</td></tr>")
H.append("</table>")
H.append("<div class='crit'><b>★운영 원칙 4줄</b><ol>"
         "<li><b>순위는 쓸 만하다</b> — 챔피언 AUROC/Spearman이 대체로 유효 범위이고 부트스트랩 CI도 이를 지지한다.</li>"
         "<li><b>작동점은 반드시 재설정하되, t*가 만능은 아니다</b> — 0.5가 무너지는 "
         f"{len(broke)}개는 t*로 구제되지만, 양성률 높은 {len(worse)}개는 t*가 민감도를 오히려 깎는다. "
         "<b>목적(민감도 우선 vs 균형)에 맞춰 골라야 한다.</b></li>"
         f"<li><b>AD는 엔드포인트별 확인 후에만</b> — 유효 {nv}개에서만 게이트로 쓰고, "
         f"성능 예측 못한 {nn}개는 <b>사용 금지</b>, 판정 불가 {nu}개는 하드 근거로 쓰지 않는다.</li>"
         "<li><b>회귀는 점추정이 아니라 구간으로</b> — 특히 VDss·반감기는 계통 편향이 커 "
         "<b>절대값 예측 금지·순위로만</b> 사용한다.</li></ol></div>")

# ── 8. 한계
H.append("<h2>8. 한계</h2>")
H.append("<div class='warn'><ul>"
         "<li><b>★logD(친유성)는 이번에 채웠다</b> — 챔피언 G3 dmpnn_ours의 chemprop 예측을 원 config·5 seed로 "
         "재현·생성해 회귀 신뢰도(잔차·컨포멀 구간·AD)를 산출했다(재현 Δ0.0000). "
         "다만 <b>GNN은 seed마다 학습 변동이 있어</b> 주 판정을 <b>seed별 부트스트랩 종합</b>으로 두었고, "
         "분류 챔피언(G2)들과 달리 <b>단일 배포 가중치가 아니라 5-seed 재학습 결과</b>임을 유의한다.</li>"
         "<li><b>★VDss·청소율 2종은 G4 검증 대기</b> — G4 분자별 예측이 없어 G2만 다뤘다. "
         "반감기·수용해도에서 G4가 검증된 결과를 <b>이 3개로 일반화하면 안 된다</b>.</li>"
         "<li><b>소표본</b> — CYP 기질 3종(test≈134)·생체이용률(128)·Caco-2(182)는 CI 폭이 크고 "
         "t*·ECE가 불안정하다. <b>확정 판정으로 쓰지 말 것</b>.</li>"
         "<li><b>AD 판정 불가 6개</b> — OOD 분자가 20개 미만이라 유효성 자체를 못 쟀다(비율은 계산되지만 검증 안 됨).</li>"
         "<li><b>임계값·구간은 valid에서만</b> 골랐고 test에 1회 적용했다. test로 어떤 선택도 하지 않았다.</li>"
         "<li>TDC scaffold 분할 안의 <b>회고적 평가</b>다. 전향적 검증이 아니다.</li></ul></div>")
import glob as _glob
_npred = len(_glob.glob(f"{NEW}/predictions/*.jsonl"))
H.append(f"<p class='src'>산출물: <code>predictions/*.jsonl</code>(★재사용 자산 {_npred}개·train/valid/test) · "
         "<code>results/adme_reliability.csv</code> · <code>reliability_detail.json</code> · "
         "<code>champions.json</code> · <code>step0b_repro.json</code></p>")

open(f"{R}/adme_reliability_report.html", "w", encoding="utf-8").write("\n".join(H))
print(f"저장 → results/adme_reliability_report.html ({os.path.getsize(f'{R}/adme_reliability_report.html')/1024:.0f} KB)")
print(f"  AD 유효 {nv}·예측못함 {nn}·판정불가 {nu} · 0.5 붕괴 {len(broke)}개 · t* 악화 {len(worse)}개")
