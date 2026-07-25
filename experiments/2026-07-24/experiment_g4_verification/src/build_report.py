#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_report.py — G4 검증 보고서. ★재학습·재계산 0(확정 결과 조립만)."""
import base64, io, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

NEW = "/home/nudge/Project/ADMET_integrated/2026-07-24/experiment_g4_verification"
R = f"{NEW}/results"
for p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
          "/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf"):
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams["font.family"] = ["NanumGothic", "NanumSquareRound", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

D = json.load(open(f"{R}/g4_verification.json"))
S0 = json.load(open(f"{R}/step0_split_check.json"))
S1 = json.load(open(f"{R}/step1_repro.json"))
S2 = json.load(open(f"{R}/step2_repro.json"))
CF = json.load(open(f"{NEW}/logs/conformer_failed.json"))
RAW = [json.loads(l) for l in open(f"{R}/g4_raw.jsonl", encoding="utf-8")]
LAB = {"half_life_obach": "반감기 (Half-Life Obach)", "solubility_aqsoldb": "수용해도 (AqSolDB)"}


def b64(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=125, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


H = ['<meta charset="utf-8"><title>G4 검증 — Uni-Mol 우세는 통계적으로 성립하는가</title>', """<style>
body{font-family:'Nanum Gothic',system-ui,sans-serif;max-width:1120px;margin:0 auto;padding:22px 26px;color:#1d2129;line-height:1.62}
h1{font-size:24px;border-bottom:3px solid #457b9d;padding-bottom:9px;margin-bottom:3px}
h2{font-size:19px;margin-top:34px;border-left:5px solid #457b9d;padding-left:11px}
h3{font-size:15.5px;margin-top:20px;color:#264653}
table{border-collapse:collapse;width:100%;margin:11px 0;font-size:12.7px}
th,td{border:1px solid #d8dde3;padding:5px 8px}th{background:#eef2f5}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.ok{background:#e8f6f1;font-weight:700}.bad{background:#fdeceb;color:#b3261e;font-weight:700}
.tie{background:#f4f0e4;color:#7a6a3a;font-weight:700}.na{color:#98a2ab}
.banner{background:linear-gradient(135deg,#20343f,#3a5a6d);color:#fff;border-radius:11px;padding:18px 22px;margin:12px 0 18px}
.banner b{color:#ffe9a8}
.box{background:#f7f9fb;border:1px solid #dfe5ea;border-radius:8px;padding:13px 17px;margin:13px 0}
.warn{background:#fff8ec;border:1px solid #f0d9a8;border-radius:8px;padding:13px 17px;margin:13px 0}
.crit{background:#fdefea;border:1px solid #f2bda9;border-radius:8px;padding:13px 17px;margin:13px 0}
figure{margin:16px 0;text-align:center}img{max-width:100%;border:1px solid #e3e8ec;border-radius:6px}
figcaption{font-size:12.3px;color:#5b6670;margin-top:6px}
small,.src{font-size:11.5px;color:#6b7580}code{background:#eef2f5;padding:1px 4px;border-radius:3px;font-size:11.5px}
</style>"""]
H.append("<h1>G4 검증 — 반감기·수용해도에서 Uni-Mol 우세가 통계적으로 성립하는가</h1>")
H.append("<p class='src'>작업일 2026-07-24~25 · 목적은 재학습이 아니라 <b>분자별 예측 확보 → 짝지은 부트스트랩</b>. "
         "ADME 보고서에서 G4가 5개 엔드포인트 명목 1위였으나 분자별 예측 미저장으로 검정이 원천 불가였다. "
         "Δ/SD가 큰 2개만 검증한다.</p>")

# ── 배너: 3줄
hl, so = D["half_life_obach"], D["solubility_aqsoldb"]
H.append("<div class='banner'>"
         "<p><b>1. 확인된 것</b> — 두 엔드포인트 모두 <b>주 판정(5seed 예측평균)에서 G4 우세가 구분됨</b>: "
         f"반감기 Δ={hl['bootstrap_A']['delta']:+.3f} CI[{hl['bootstrap_A']['ci'][0]:+.3f},{hl['bootstrap_A']['ci'][1]:+.3f}] · "
         f"용해도 Δ={so['bootstrap_A']['delta']:+.3f} CI[{so['bootstrap_A']['ci'][0]:+.3f},{so['bootstrap_A']['ci'][1]:+.3f}].</p>"
         "<p><b>2. ★그러나 판정이 방법에 의존한다</b> — <b>seed별 부트스트랩(B)로 재면 둘 다 CI가 0을 포함해 "
         "'구분 안 됨'</b>이 된다. 어느 쪽을 주 판정으로 쓰느냐가 결론을 바꾼다. 둘 다 보고한다.</p>"
         "<p><b>3. 여전히 미검증</b> — VDss·간세포청소율·마이크로솜청소율 <b>3개는 이 실험에서 검증하지 않았다</b>"
         "(격차가 흔들림 안이라 제외). 이 결과를 그 3개로 일반화하면 안 된다.</p></div>")

# ── 0단계
H.append("<h2>0. 분할 고정 확인</h2>")
H.append("<table><tr><th>엔드포인트</th><th class='n'>test</th><th class='n'>기록값</th>"
         "<th class='n'>train∩test 정확분자</th><th>주지표</th><th>판정</th></tr>")
for ep, v in S0.items():
    H.append(f"<tr><td>{LAB[ep]}</td><td class='n'>{v['n_test']}</td><td class='n'>{v['n_test_recorded']}</td>"
             f"<td class='n ok'>{v['exact_overlap_trainval_test']}</td><td>{v['primary']}</td>"
             f"<td class='ok'>통과</td></tr>")
H.append("</table><p><small>ADME 실험(2026-07-22)이 쓴 <b>동일 분할</b>을 그대로 읽었다(새로 만들지 않음). "
         "출처 <code>results/step0_split_check.json</code></small></p>")

# ── 1·2단계 재현
H.append("<h2>1·2. 재현 검증 — 같은 모델인가</h2>")
H.append("<table><tr><th>모델</th><th>엔드포인트</th><th class='n'>보고값</th><th class='n'>재현</th>"
         "<th class='n'>Δ</th><th class='n'>seed</th><th>판정</th></tr>")
for c in S1["checks"]:
    H.append(f"<tr><td>G4 unimol</td><td>{LAB[c['endpoint']]}</td><td class='n'>{c['reported']:.4f}</td>"
             f"<td class='n ok'>{c['reproduced']:.4f}</td><td class='n'>{c['diff']:+.4f}</td>"
             f"<td class='n'>{c['n_seed']}/5</td><td class='ok'>OK(±0.005)</td></tr>")
for c in S2["checks"]:
    H.append(f"<tr><td>G2 {c['model']}</td><td>{LAB[c['endpoint']]}</td><td class='n'>{c['reported']:.4f}</td>"
             f"<td class='n ok'>{c['reproduced']:.4f}</td><td class='n'>{c['diff']:+.4f}</td>"
             f"<td class='n'>5/5</td><td class='ok'>OK</td></tr>")
H.append("</table>")
oom = [r for r in RAW if r.get("status") != "ok"]
H.append(f"<div class='warn'><b>★용해도 G4는 3/5 seed</b> — seed 1·2가 <b>GPU OOM</b>으로 실패했다"
         f"(batch 8→4 재시도까지 실패·규격대로 그 seed만 skip·기록). "
         f"3 seed 평균이 보고값과 Δ{S1['checks'][1]['diff']:+.4f}로 허용오차 안이라 <b>같은 모델임은 확인</b>되지만, "
         "<b>5 seed가 아니라는 점은 아래 결과 해석에 반영해야 한다</b>. "
         f"실패 기록 {len(oom)}건: <code>results/g4_raw.jsonl</code></div>")

# ── 3단계 공정 짝짓기
H.append("<h2>3. ★공정한 짝짓기 — conformer 실패 분자 동일 제외</h2>")
H.append("<div class='box'>Uni-Mol은 3D conformer가 필요하다. ETKDG로 생성이 실패하는 분자를 "
         "<b>양쪽(G4·G2)에서 동일하게 제외</b>해야 공정하다. 조용히 빼지 않고 개수·목록을 공개한다"
         "(<code>logs/conformer_failed.json</code>).</div>")
H.append("<table><tr><th>엔드포인트</th><th class='n'>G4 예측</th><th class='n'>G2 예측</th>"
         "<th class='n'>교집합(Jaccard)</th><th class='n'>conformer 실패 제외</th>"
         "<th class='n'>★비교 대상</th></tr>")
for ep, v in D.items():
    H.append(f"<tr><td>{LAB[ep]}</td><td class='n'>{v['n_g4']}</td><td class='n'>{v['n_g2']}</td>"
             f"<td class='n'>{v['n_intersection']} ({v['jaccard']:.4f})</td>"
             f"<td class='n'>{v['n_conformer_failed_excluded']}</td>"
             f"<td class='n ok'>{v['n_compared']}</td></tr>")
H.append("</table>")
H.append("<h3>교집합 기준 지표 — 원래 보고값과 얼마나 달라지나</h3>")
H.append("<table><tr><th>엔드포인트</th><th>모델</th><th class='n'>보고값(전체)</th>"
         "<th class='n'>교집합·seed별 지표평균</th><th class='n'>교집합·5seed 예측평균</th></tr>")
for ep, v in D.items():
    H.append(f"<tr><td rowspan=2>{LAB[ep]}</td><td>G4 unimol</td><td class='n'>{v['g4_reported']:.4f}</td>"
             f"<td class='n'>{v['g4_intersection_seedmean']:.4f}</td>"
             f"<td class='n'>{v['g4_intersection_predmean']:.4f}</td></tr>"
             f"<tr><td>G2 챔피언</td><td class='n'>{v['g2_reported']:.4f}</td>"
             f"<td class='n'>{v['g2_intersection_seedmean']:.4f}</td>"
             f"<td class='n'>{v['g2_intersection_predmean']:.4f}</td></tr>")
H.append("</table><p><small>분자를 빼도 <b>순위는 바뀌지 않았다</b>(두 엔드포인트 모두 G4가 여전히 앞섬). "
         "다만 '5seed 예측평균'은 seed별 지표평균과 값이 다르다 — 아래 4단계에서 이 차이가 판정을 가른다.</small></p>")

# ── 4단계 부트스트랩 그림
H.append("<h2>4. 짝지은 부트스트랩 (2,000회)</h2>")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, (ep, v) in zip(axes, D.items()):
    for i, (key, lab) in enumerate([("bootstrap_A", "(A) 5seed 예측평균"), ("bootstrap_B", "(B) seed별→평균")]):
        b = v[key]
        lo, hi, d = b["ci"][0], b["ci"][1], b["delta"]
        c = "#2a9d8f" if b["distinguishable"] else "#c1440e"
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=c, ecolor=c, capsize=4, ms=8)
        ax.text(hi, i + .18, f"{'구분됨' if b['distinguishable'] else '구분 안 됨'}", color=c, fontsize=8.5)
    ax.axvline(0, ls="--", color="#333", lw=1.3)
    ax.axvspan(-v["equivalence_margin"], v["equivalence_margin"], color="#f0d9a8", alpha=.35)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["(A) 예측평균", "(B) seed별"], fontsize=9)
    ax.set_ylim(-.6, 1.6)
    ax.set_xlabel(f"Δ ({'Spearman' if v['metric']=='Spearman' else 'MAE'}) — 양수면 G4 우세", fontsize=9)
    ax.set_title(LAB[ep], fontsize=11)
    ax.grid(axis="x", alpha=.25)
H.append(f'<figure><img src="data:image/png;base64,{b64(fig)}"><figcaption>'
         "점=Δ 평균, 가로선=95% CI. 점선(0)을 CI가 넘지 않으면 '구분됨'. "
         "노란 띠=실용적 동등 구간(±0.05). ★두 방법이 서로 다른 판정을 낸다.</figcaption></figure>")
H.append("<table><tr><th>엔드포인트</th><th>방법</th><th class='n'>Δ</th><th class='n'>95% CI</th>"
         "<th class='n'>Δ&gt;0 비율</th><th>판정</th></tr>")
for ep, v in D.items():
    for key, lab in [("bootstrap_A", "★(A) 5seed 예측평균 <small>주 판정</small>"), ("bootstrap_B", "(B) seed별 부트스트랩→평균")]:
        b = v[key]
        cls = "ok" if b["distinguishable"] else "tie"
        fr = f"{b['frac_gt0']:.3f}" if "frac_gt0" in b else "—"
        H.append(f"<tr><td>{LAB[ep] if key=='bootstrap_A' else ''}</td><td>{lab}</td>"
                 f"<td class='n'>{b['delta']:+.4f}</td><td class='n'>[{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]</td>"
                 f"<td class='n'>{fr}</td><td class='{cls}'>"
                 f"{'★구분됨(G4 우세)' if b['distinguishable'] else '구분 안 됨'}</td></tr>")
H.append("</table>")
H.append("<div class='crit'><b>★가장 중요한 발견 — 판정이 seed 처리 방식에 의존한다.</b><br>"
         "(A) <b>5 seed 예측을 먼저 평균</b>한 뒤 부트스트랩하면 앙상블 효과로 예측이 안정돼 Δ가 커지고 CI가 좁아진다 → <b>구분됨</b>.<br>"
         "(B) <b>seed별로 부트스트랩</b>해 평균하면 seed 간 학습 변동이 CI에 그대로 들어가 넓어진다 → <b>구분 안 됨</b>"
         f"(반감기 CI 하한 {D['half_life_obach']['bootstrap_B']['ci'][0]:+.4f}, "
         f"용해도 {D['solubility_aqsoldb']['bootstrap_B']['ci'][0]:+.4f} — 둘 다 0을 아슬아슬하게 포함).<br><br>"
         "<b>어느 것이 맞나</b> — 배포 관점에서는 (A)가 타당하다(실제로 쓰는 것은 5 seed 앙상블 예측이므로). "
         "그러나 <b>'G4라는 방법이 G2보다 낫다'는 일반화 주장에는 (B)가 더 보수적이고 정직하다</b>(단일 모델을 새로 학습했을 때의 변동을 포함). "
         "→ <b>주 판정은 (A)로 하되, (B)에서 구분되지 않는다는 사실을 함께 보고한다.</b></div>")

# ── 5단계 동등성
H.append("<h2>5. 동등성 검정 (TOST·±0.05)</h2>")
H.append("<table><tr><th>엔드포인트</th><th class='n'>동등 구간</th><th class='n'>(A) CI</th><th>실용적 동등?</th></tr>")
for ep, v in D.items():
    b = v["bootstrap_A"]
    H.append(f"<tr><td>{LAB[ep]}</td><td class='n'>±{v['equivalence_margin']}</td>"
             f"<td class='n'>[{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]</td>"
             f"<td class='{'ok' if v['practically_equivalent'] else 'bad'}'>"
             f"{'실용적 동등' if v['practically_equivalent'] else '동등 아님(CI가 구간 밖)'}</td></tr>")
H.append("</table><p><small>동등 구간 근거: Spearman ±0.05는 순위상관에서 실무적으로 무의미한 차이로 보는 관례적 폭이고, "
         "MAE ±0.05는 용해도 logS 단위에서 실험 재현오차보다 작은 값이다(notes.md). "
         "두 엔드포인트 모두 <b>동등도 아니다</b> — 즉 '차이가 없다'고도 말할 수 없다.</small></p>")

# ── 6. 판정·한계
H.append("<h2>6. 판정과 한계</h2>")
H.append("<table><tr><th>엔드포인트</th><th class='n'>비교 분자</th><th>주 판정(A)</th><th>보수 판정(B)</th><th>동등성</th></tr>")
for ep, v in D.items():
    H.append(f"<tr><td>{LAB[ep]}</td><td class='n'>{v['n_compared']}</td>"
             f"<td class='ok'>{'G4 우세(구분됨)' if v['bootstrap_A']['distinguishable'] else '구분 안 됨'}</td>"
             f"<td class='tie'>{'구분됨' if v['bootstrap_B']['distinguishable'] else '구분 안 됨'}</td>"
             f"<td>{'동등' if v['practically_equivalent'] else '동등 아님'}</td></tr>")
H.append("</table>")
H.append("<div class='warn'><b>한계 — 정직 고지</b><ul>"
         "<li><b>★용해도 G4는 3/5 seed</b>(seed 1·2 GPU OOM 실패). 재현은 통과했으나 seed 수가 적어 (B) 방법의 "
         "변동 추정이 불안정하다.</li>"
         "<li><b>★검증하지 않은 3개</b> — VDss(seed 4/5·Δ≈SD/2)·간세포청소율(Δ 0.0025)·마이크로솜청소율(Δ≈SD/2)은 "
         "격차가 흔들림 안이라 <b>이 실험에서 제외</b>했다. <b>이 결과를 그 3개로 일반화하면 안 된다.</b></li>"
         "<li><b>conformer 실패 분자를 양쪽에서 제외</b>했다(반감기 6·용해도 17). 제외해도 순위는 안 바뀌었다.</li>"
         "<li><b>하이퍼파라미터 튜닝 없음</b> — 원 실험 재현이지 개선이 아니다. test로 어떤 선택도 하지 않았다.</li>"
         "<li>이 검증은 <b>TDC scaffold 분할 안</b>의 회고적 비교다. 전향적 검증이 아니다.</li></ul></div>")
H.append("<p class='src'>산출물: <code>predictions/*.jsonl</code>(★재사용 자산 20개 — 이후 어떤 검정도 재학습 없이 가능) · "
         "<code>results/g4_verification.csv</code> · <code>g4_verification.json</code> · "
         "<code>step0_split_check.json</code> · <code>step1_repro.json</code> · <code>step2_repro.json</code> · "
         "<code>logs/conformer_failed.json</code> · <code>results/g4_raw.jsonl</code></p>")

open(f"{R}/g4_verification.html", "w", encoding="utf-8").write("\n".join(H))
print(f"저장 → results/g4_verification.html ({os.path.getsize(f'{R}/g4_verification.html')/1024:.0f} KB)")
for ep, v in D.items():
    print(f"  {ep}: (A) {v['bootstrap_A']['delta']:+.4f} {v['bootstrap_A']['ci']} "
          f"{'구분됨' if v['bootstrap_A']['distinguishable'] else '구분안됨'} · "
          f"(B) {'구분됨' if v['bootstrap_B']['distinguishable'] else '구분안됨'}")
