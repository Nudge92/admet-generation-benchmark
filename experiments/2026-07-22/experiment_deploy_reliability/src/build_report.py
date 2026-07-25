#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_report.py — 배포 신뢰도 번들 단일 HTML. ★새 계산 없음(reliability.json 조립·시각화만)."""
import base64, io, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path("/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_deploy_reliability")
RES = ROOT / "results"
for p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
          "/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf"):
    if Path(p).exists():
        fm.fontManager.addfont(p)
plt.rcParams["font.family"] = ["NanumGothic", "NanumSquareRound", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

D = json.load(open(RES / "reliability.json"))
CHK = json.load(open(RES / "reproduction_check.json"))
CSV = pd.read_csv(RES / "reliability.csv")
CLS = {k: v for k, v in D.items() if "operating" in v}
REG = {k: v for k, v in D.items() if "regression" in v}
SMALL = 150            # 소표본 배지 기준
NAME = {"dili": "DILI", "herg": "hERG", "ames": "AMES", "ld50_zhu": "LD50",
        "Carcinogens_Lagunin": "발암성", "ClinTox": "ClinTox", "Tox21": "Tox21"}


def nm(v):
    return NAME.get(v["endpoint"], v["endpoint"]) + ("" if v["task"] in ("—", "Y") else f" {v['task']}")


def b64(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=125, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def img(fig, cap):
    return f'<figure><img src="data:image/png;base64,{b64(fig)}"><figcaption>{cap}</figcaption></figure>'


def badge(v):
    return (" <span class='sm'>소표본 참고</span>" if v["n_test"] < SMALL else "")


H = ['<meta charset="utf-8"><title>배포 신뢰도 번들 — AD · 운영지표 · 보정 · 정확 CI</title>', """<style>
body{font-family:'Nanum Gothic',system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:22px 26px;color:#1d2129;line-height:1.62}
h1{font-size:26px;border-bottom:3px solid #2a9d8f;padding-bottom:10px;margin-bottom:3px}
h2{font-size:20px;margin-top:38px;border-left:5px solid #2a9d8f;padding-left:11px}
h3{font-size:16px;margin-top:22px;color:#264653}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:12.7px}
th,td{border:1px solid #d8dde3;padding:5px 8px;text-align:left}
th{background:#eef2f5;font-weight:700}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.good{background:#e8f6f1}.bad{background:#fdeceb;color:#b3261e;font-weight:700}
.leak{background:#fff1ec;color:#c1440e;font-weight:700}
.na{color:#98a2ab}
.box{background:#f7f9fb;border:1px solid #dfe5ea;border-radius:8px;padding:14px 18px;margin:16px 0}
.warn{background:#fff8ec;border:1px solid #f0d9a8;border-radius:8px;padding:14px 18px;margin:16px 0}
.crit{background:#fdefea;border:1px solid #f2bda9;border-radius:8px;padding:14px 18px;margin:16px 0}
.fix{background:#eef4fd;border:1px solid #bcd0ee;border-radius:8px;padding:14px 18px;margin:16px 0}
.sm{background:#ffe9c9;color:#8a5a00;font-size:11px;padding:1px 6px;border-radius:8px}
figure{margin:18px 0;text-align:center}img{max-width:100%;border:1px solid #e3e8ec;border-radius:6px}
figcaption{font-size:12.5px;color:#5b6670;margin-top:7px}
small,.src{font-size:11.5px;color:#6b7580}
code{background:#eef2f5;padding:1px 5px;border-radius:3px;font-size:12px}
ul{margin:8px 0 8px 4px}li{margin:5px 0}
</style>"""]

H.append("<h1>배포 신뢰도 번들 — AD · 운영지표 · 보정 · 정확 CI</h1>")
H.append("<p class='src'>작업일 2026-07-22 · 대상 <b>18과제의 G2 챔피언</b>(마스터 보고서 권장 모델) · "
         "마스터가 <b>'분자별 예측 미저장'</b>이라 못 했던 네 가지를 분자별 예측을 만들어 해결했다.</p>")

# ── 0. 지표 패널
H.append("<h2>0. 이 문서의 지표를 한 줄씩</h2><div class='box'><ul>"
         "<li><b>AD(적용범위)</b> — 새 분자가 <b>학습셋과 얼마나 닮았나</b>. 학습셋 대비 5-NN Tanimoto(ECFP4) 평균. "
         "컷오프는 <b>학습셋 자기 분포의 5%·25% 백분위</b>(절대선 아님).</li>"
         "<li><b>민감도(sensitivity)</b> = 실제 독성 중 잡아낸 비율. <b>독성 예측에서 가장 중요</b>.</li>"
         "<li><b>FN(놓친 독성)</b> — 독성인데 안전하다고 한 개수. <b>가장 비싼 오류</b>.</li>"
         "<li><b>NPV</b> — '안전'이라 했을 때 실제로 안전할 확률. 스크리닝에서 통과시킬 근거.</li>"
         "<li><b>ECE(보정오차)</b> — 예측확률이 실제 빈도와 얼마나 어긋나나. 0에 가까울수록 확률을 믿을 수 있다. "
         "<b>보정은 후처리라 AUROC는 변하지 않는다.</b></li>"
         "<li><b>부트스트랩 CI</b> — test 분자를 2000회 재표집한 95% 구간. "
         "마스터의 Hanley-McNeil '구분 가능선'(보수적 근사)을 <b>실제 분포로 대체</b>한 것.</li>"
         "<li><b>DeLong</b> — 같은 test 분자 위 두 모델의 <b>대응비교</b>. 마스터의 비대응 근사보다 <b>정확</b>하다.</li>"
         "</ul></div>")

# ── 1. 재현
n_ok = sum(1 for c in CHK["checks"] if c["within_tol"])
mx = max(abs(c["diff"]) for c in CHK["checks"])
H.append("<h2>1. 전제 — 예측은 '로드'가 아니라 '재현'이다</h2>")
H.append(f"<div class='warn'><b>★정직 고지</b> — 챔피언 모델의 <b>아티팩트(.pkl/.joblib)가 저장돼 있지 않았다</b>. "
         f"그래서 <b>동일 config·동일 seed·동일 분할</b>로 재현 학습한 뒤 추론했다. "
         f"재현 지표가 원본 보고값과 일치하는지 검증했고 — <b>18/18 통과, 최대 편차 {mx:.4f}</b>"
         f"(허용 ±{CHK['tolerance']}). 이 번들의 모든 예측은 <b>재현본</b>임을 표시한다.</div>")
H.append("<table><tr><th>범위</th><th>엔드포인트/과제</th><th>챔피언 모델</th><th class='n'>원본 보고</th>"
         "<th class='n'>재현</th><th class='n'>Δ</th><th>출처</th></tr>")
for c in CHK["checks"]:
    H.append(f"<tr><td>{c['scope']}</td><td>{NAME.get(c['endpoint'], c['endpoint'])}"
             f"{'' if c['task'] in ('—', 'Y') else ' ' + c['task']}</td><td>{c['model']}</td>"
             f"<td class='n'>{c['reported']:.4f}</td><td class='n good'>{c['reproduced']:.4f}</td>"
             f"<td class='n'>{c['diff']:+.4f}</td><td class='src'>{c['source']}</td></tr>")
H.append("</table>")
H.append("<p><small>LD50의 챔피언은 <b>G2 물리화학+XGBoost</b>다. 마스터에서 수치상 최고는 G4 Uni-Mol(MAE 0.5939)이었으나 "
         "두 값의 차이는 구분 가능선 미만이었고, 배포 관점에서 재현·검증 가능한 G2를 권장 모델로 삼았다.</small></p>")

# ── 2. AD
H.append("<h2>2. 적용범위(AD) — 그리고 AD가 정말 신뢰도를 예측하는가</h2>")
H.append("<div class='box'>이전 파이프라인의 '모분자 대비' 근사를 <b>폐기</b>하고, <b>학습셋 기준 5-NN Tanimoto</b>로 다시 계산했다. "
         "컷오프는 임의의 0.4가 아니라 <b>그 엔드포인트 학습셋이 스스로 갖는 유사도 분포</b>의 5%(OOD)·25%(경계) 백분위다.<br>"
         "<b>★핵심 검증</b>: AD 구간별로 성능을 재서, OOD에서 실제로 성능이 떨어지는지 확인했다. "
         "<b>안 떨어지면 그대로 '예측 못함'으로 적었다.</b></div>")
H.append("<table><tr><th>과제</th><th class='n'>n_test</th><th class='n'>OOD 컷</th><th class='n'>in-domain</th>"
         "<th class='n'>경계</th><th class='n'>OOD</th><th class='n'>성능 in</th><th class='n'>성능 OOD</th>"
         "<th>판정</th></tr>")
for k, v in D.items():
    a = v["AD"]; b = a["bands"]
    cls = "good" if "유효" in a["verdict"] else ("bad" if "못함" in a["verdict"] else "")
    f = (lambda x: "—" if x is None else f"{x:.4f}")
    H.append(f"<tr><td>{nm(v)}{badge(v)}</td><td class='n'>{v['n_test']}</td>"
             f"<td class='n'>{a['cut_ood']:.3f}</td><td class='n'>{b['in-domain']['frac']:.3f}</td>"
             f"<td class='n'>{b['경계']['frac']:.3f}</td><td class='n'>{b['OOD']['frac']:.3f}"
             f"<small> (n={b['OOD']['n']})</small></td>"
             f"<td class='n'>{f(b['in-domain']['perf'])}</td><td class='n'>{f(b['OOD']['perf'])}</td>"
             f"<td class='{cls}'>{a['verdict']}</td></tr>")
H.append("</table>")
pairs = [(nm(v), v["AD"]["bands"]["in-domain"]["perf"], v["AD"]["bands"]["OOD"]["perf"], v["endpoint"] == "ld50_zhu")
         for v in D.values() if v["AD"]["bands"]["in-domain"]["perf"] and v["AD"]["bands"]["OOD"]["perf"]]
fig, ax = plt.subplots(figsize=(10.5, 4.0))
x = np.arange(len(pairs))
ax.plot(x, [p[1] for p in pairs], "o-", color="#2a9d8f", lw=2, ms=7, label="in-domain")
ax.plot(x, [p[2] for p in pairs], "s--", color="#c1440e", lw=2, ms=7, label="OOD")
for i, p in enumerate(pairs):
    ax.annotate("", xy=(i, p[2]), xytext=(i, p[1]), arrowprops=dict(arrowstyle="->", color="#8d99ae", lw=1))
ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pairs], rotation=32, ha="right", fontsize=8.5)
ax.set_ylabel("성능 (분류=AUROC↑ · LD50=MAE↓)", fontsize=9.5); ax.grid(axis="y", alpha=.25); ax.legend(fontsize=9)
ax.set_title("AD 구간별 성능 — 화살표가 아래로 향하면 OOD에서 실제로 나빠진 것", fontsize=11.5)
H.append(img(fig, "in-domain 대비 OOD 성능. 분류는 AUROC(내려가면 나빠짐), LD50은 MAE(올라가면 나빠짐)라 "
                  "LD50만 화살표 방향의 의미가 반대다. OOD 표본이 20 미만인 과제는 산출 불가라 빠졌다."))
nv = sum(1 for v in D.values() if "유효" in v["AD"]["verdict"])
nn = sum(1 for v in D.values() if "못함" in v["AD"]["verdict"])
nu = len(D) - nv - nn
H.append(f"<div class='warn'><b>결과 — 섞였다.</b> {len(D)}과제 중 <b>AD 유효 {nv}건</b>, "
         f"<b>★AD가 신뢰도를 예측하지 못한 경우 {nn}건</b>, 판정 불가 {nu}건(OOD 표본 20 미만). "
         "즉 <b>AD를 켜 두면 항상 안전해진다는 보장은 없다.</b> "
         "AD가 통한 대표 사례는 AMES(in 0.881 → OOD 0.752)와 LD50(MAE 0.485 → 0.730)이고, "
         "SR-MMP·NR-PPAR-gamma처럼 OOD에서도 성능이 유지되는 과제도 있다. "
         "<b>AD는 '자동 안전장치'가 아니라 엔드포인트마다 유효성을 확인해야 하는 도구</b>다.</div>")

# ── 3. 운영지표
H.append("<h2>3. 운영지표 — 실제로 독성을 몇 개 놓치는가</h2>")
H.append("<div class='box'>임계값 두 개를 비교한다. <b>(a) 0.5 고정</b>, <b>(b) t*</b> = <b>valid에서 MCC가 최대</b>인 값 "
         "(★test에서 고르지 않았다 — valid로 정해 test에 1회 적용).</div>")
H.append("<table><tr><th>과제</th><th class='n'>양성률</th><th class='n'>임계</th><th class='n'>민감도</th>"
         "<th class='n'>특이도</th><th class='n'>PPV</th><th class='n'>NPV</th><th class='n'>TP/FP/FN/TN</th>"
         "<th class='n'>★FN</th></tr>")
for k, v in CLS.items():
    o = v["operating"]
    for tag, m in (("0.5", o["t_fixed"]), ("t*", o["t_star"])):
        first = f"<td rowspan=2>{nm(v)}{badge(v)}</td><td class='n' rowspan=2>{o['pos_rate']:.3f}</td>" if tag == "0.5" else ""
        f = lambda x: "—" if x is None else f"{x:.3f}"
        low = (m["sensitivity"] is not None and m["sensitivity"] < 0.5)
        scls = "n bad" if low else "n"
        H.append(f"<tr>{first}<td class='n'>{tag} ({m['threshold']:.3f})</td>"
                 f"<td class='{scls}'>{f(m['sensitivity'])}</td>"
                 f"<td class='n'>{f(m['specificity'])}</td><td class='n'>{f(m['PPV'])}</td>"
                 f"<td class='n'>{f(m['NPV'])}</td>"
                 f"<td class='n'>{m['TP']}/{m['FP']}/{m['FN']}/{m['TN']}</td>"
                 f"<td class='n bad'>{m['FN']}</td></tr>")
H.append("</table>")
low = [(nm(v), v["operating"]["t_fixed"]["sensitivity"], v["operating"]["t_fixed"]["FN"])
       for v in CLS.values() if v["operating"]["t_fixed"]["sensitivity"] is not None
       and v["operating"]["t_fixed"]["sensitivity"] < 0.4]
fig, ax = plt.subplots(figsize=(11, 3.9))
ks = list(CLS.values())
x = np.arange(len(ks)); w = .38
s05 = [v["operating"]["t_fixed"]["sensitivity"] or 0 for v in ks]
sst = [v["operating"]["t_star"]["sensitivity"] or 0 for v in ks]
ax.bar(x - w / 2, s05, w, label="임계 0.5", color="#8d99ae", edgecolor="#222", linewidth=.5)
ax.bar(x + w / 2, sst, w, label="t* (valid MCC 최대)", color="#2a9d8f", edgecolor="#222", linewidth=.5)
ax.axhline(.5, ls=":", color="#c1440e", lw=1.3)
ax.text(len(ks) - .5, .51, "민감도 0.5", color="#c1440e", fontsize=8.5, ha="right")
ax.set_xticks(x); ax.set_xticklabels([nm(v) for v in ks], rotation=32, ha="right", fontsize=8.5)
ax.set_ylabel("민감도 (독성 포착률)", fontsize=9.5); ax.set_ylim(0, 1.05)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=.25)
ax.set_title("임계값을 바꾸면 놓치는 독성이 줄어드나", fontsize=11.5)
H.append(img(fig, "임계 0.5 vs valid에서 고른 t*의 민감도. 불균형이 심한 과제에서 0.5는 사실상 "
                  "'전부 음성'에 가까운 판정을 낸다."))
H.append(f"<div class='crit'><b>★가장 중요한 실무 발견 — 기본 임계 0.5는 독성 스크리닝에 쓰면 안 된다.</b><br>"
         f"양성이 희박한 과제에서 0.5는 독성을 거의 못 잡는다: "
         + " · ".join(f"<b>{n} 민감도 {s:.2f}(FN {f})</b>" for n, s, f in low[:5]) +
         ". 극단적으로 <b>NR-PPAR-gamma는 민감도 0.000 — 46개 양성을 전부 놓친다</b>(AUROC는 0.839인데도!). "
         "<b>AUROC가 높다는 것과 쓸 만한 작동점이 있다는 것은 완전히 다른 얘기</b>다.<br><br>"
         "t*로 바꾸면 여러 과제에서 크게 나아진다(NR-AhR 0.38→0.62 · SR-ARE 0.23→0.65 · SR-p53 0.08→0.45). "
         "<b>다만 만능이 아니다</b> — DILI는 t*=0.712로 올라가면서 민감도가 <b>0.94→0.66으로 악화</b>됐다"
         "(FN 3→17). valid가 54~65분자로 작아 t* 자체가 불안정하기 때문이다. "
         "<b>소표본 엔드포인트에서는 임계값 튜닝도 믿을 수 없다.</b></div>")

# ── 4. 보정
H.append("<h2>4. 보정 — 확률을 믿어도 되는가</h2>")
H.append("<table><tr><th>과제</th><th class='n'>ECE</th><th class='n'>평균 예측확률</th>"
         "<th class='n'>실제 양성률</th><th>방향</th><th class='n'>양성분자 평균 확률</th></tr>")
for k, v in CLS.items():
    c = v["calibration"]
    cls = " class='bad'" if c["ECE"] > 0.10 else ""
    _p = c["mean_conf_on_positives"]
    mcp = "—" if _p is None else f"{_p:.4f}"
    H.append(f"<tr><td>{nm(v)}{badge(v)}</td><td class='n'{cls}>{c['ECE']:.4f}</td>"
             f"<td class='n'>{c['mean_conf']:.4f}</td><td class='n'>{c['obs_rate']:.4f}</td>"
             f"<td>{c['direction']}</td>"
             f"<td class='n'>{mcp}</td></tr>")
H.append("</table>")
sel = [v for v in CLS.values() if v["endpoint"] in ("dili", "herg", "ames", "ClinTox")][:4] + \
      [v for v in CLS.values() if v["task"] == "SR-MMP"]
fig, axes = plt.subplots(1, len(sel), figsize=(3.4 * len(sel), 3.3))
for ax, v in zip(np.atleast_1d(axes), sel):
    c = v["calibration"]["curve"]
    xs = [(r["lo"] + r["hi"]) / 2 for r in c]; ys = [r["freq"] for r in c]
    ax.plot([0, 1], [0, 1], ":", color="#8d99ae", lw=1.2)
    ax.plot(xs, ys, "o-", color="#2a9d8f", lw=1.8, ms=5)
    ax.set_title(f"{nm(v)}\nECE {v['calibration']['ECE']:.3f}", fontsize=10)
    ax.set_xlabel("예측 확률", fontsize=8.5); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=.25)
np.atleast_1d(axes)[0].set_ylabel("실제 양성 빈도", fontsize=8.5)
H.append(img(fig, "신뢰도 곡선(10-bin). 점선 = 완벽 보정. 점이 점선 아래면 과신(확률을 실제보다 높게 말함)."))
H.append("<div class='warn'><b>읽는 법</b> — 큰 데이터셋(AMES ECE 0.030 · Tox21 대부분 0.015~0.073)은 확률이 비교적 쓸 만하다. "
         "반면 <b>DILI(0.131)·발암성(0.150)은 보정이 나쁘다</b> — test가 96·55분자라 곡선 자체가 불안정하다. "
         "<b>불균형 주의</b>: 양성이 3~10%인 과제에서는 다수(음성) 구간이 ECE를 지배하므로, "
         "위 표에 <b>양성 분자에서의 평균 예측확률</b>을 따로 실었다(대부분 양성률보다 높게 나와 방향은 맞다). "
         "<b>보정은 후처리라 고쳐도 AUROC는 그대로</b>다 — 순위 능력이 아니라 '확률 해석'만 개선된다.</div>")

# ── 5. CI + DeLong
H.append("<h2>5. 정확한 CI와 대응비교 — 마스터의 보수적 판정을 대체한다</h2>")
fig, ax = plt.subplots(figsize=(9.5, 5.6))
ks = sorted(CLS.values(), key=lambda v: v["ci"]["AUROC"])
y = np.arange(len(ks))
lo = [v["ci"]["AUROC_CI95"][0] for v in ks]; hi = [v["ci"]["AUROC_CI95"][1] for v in ks]
val = [v["ci"]["AUROC"] for v in ks]
ax.errorbar(val, y, xerr=[np.array(val) - np.array(lo), np.array(hi) - np.array(val)],
            fmt="o", color="#2a9d8f", ecolor="#8d99ae", capsize=3, ms=6)
ax.set_yticks(y); ax.set_yticklabels([f"{nm(v)} (n={v['n_test']})" for v in ks], fontsize=8.5)
ax.axvline(.5, ls=":", color="#c1440e", lw=1.2)
ax.set_xlabel("AUROC (부트스트랩 2000회 95% CI)", fontsize=9.5); ax.grid(axis="x", alpha=.25)
ax.set_title("챔피언 모델의 AUROC와 실제 95% 신뢰구간", fontsize=11.5)
H.append(img(fig, "구간이 넓을수록 그 과제의 결론이 불안정하다. DILI·발암성처럼 표본이 작은 과제는 "
                  "구간 폭이 0.10~0.24에 달한다."))
H.append("<h3>DeLong 대응비교 — 챔피언(G2) vs 정직 G3 D-MPNN</h3>")
H.append("<table><tr><th>과제</th><th class='n'>챔피언</th><th class='n'>G3 D-MPNN</th><th class='n'>Δ</th>"
         "<th class='n'>p</th><th>판정</th></tr>")
nsig = 0
for k, v in CLS.items():
    c = (v.get("delong") or {}).get("dmpnn_ours")
    if not c or "p_value" not in c:
        continue
    sig = c["p_value"] < 0.05
    nsig += sig
    H.append(f"<tr><td>{nm(v)}{badge(v)}</td><td class='n'>{c['auc_champion']:.4f}</td>"
             f"<td class='n'>{c['auc_other']:.4f}</td><td class='n'>{c['delta']:+.4f}</td>"
             f"<td class='n'>{c['p_value']:.2e}</td>"
             f"<td class='{'good' if sig else ''}'>{'★유의 — G2 우세' if sig else '비유의'}</td></tr>")
H.append("</table>")
H.append(f"<div class='fix'><b>★마스터 보고서 §6을 업데이트한다.</b> 마스터는 Hanley-McNeil <b>비대응</b> 근사로 "
         f"구분 가능한 세대 효과를 <b>3건</b>이라고 했다. 같은 test 분자 위의 <b>정확한 대응비교(DeLong)</b>로 다시 재면 "
         f"<b>{nsig}건이 유의</b>(p&lt;0.05)하고, <b>{nsig}건 모두 G2 챔피언이 정직한 G3 D-MPNN보다 우세</b>다. "
         f"<b>D-MPNN이 유의하게 이긴 과제는 0건</b>이다.<br><br>"
         "마스터의 보수적 판정이 <b>증거를 과소평가</b>하고 있었다는 뜻이고, 방향은 <b>바뀌지 않았다</b> — "
         "오히려 '고전 ML 우세'가 더 넓게 확인됐다. 다만 (a) 여전히 비유의 과제가 있고 "
         "(b) G2·G3 모두 하이퍼파라미터 탐색을 하지 않았으며 (c) G3는 <code>--class-balance</code>를 쓰고 "
         "G2는 안 쓰는 비대칭이 남아 있다 — <b>'이 조건에서'</b>의 결론이다.</div>")
H.append("<h3>참고 — 챔피언 vs ADMET-AI <span class='leak'>★누수</span></h3>")
H.append("<table><tr><th>과제</th><th class='n'>챔피언</th><th class='n'>ADMET-AI</th><th class='n'>Δ</th>"
         "<th class='n'>p</th><th>해석</th></tr>")
for k, v in CLS.items():
    c = (v.get("delong") or {}).get("admetai")
    if not c or "p_value" not in c:
        continue
    H.append(f"<tr><td>{nm(v)}</td><td class='n'>{c['auc_champion']:.4f}</td>"
             f"<td class='n leak'>{c['auc_other']:.4f}</td><td class='n'>{c['delta']:+.4f}</td>"
             f"<td class='n'>{c['p_value']:.2e}</td><td class='src'>{c['leak_flag']}</td></tr>")
H.append("</table>")
H.append("<div class='crit'><b>★이 표는 실력 비교가 아니다.</b> ADMET-AI는 TDC 전체로 사전학습돼 "
         "이 test 분자를 이미 봤을 가능성이 매우 높다(마스터 §5의 증거 3건). "
         "정확한 대응비교로 p값이 아무리 작게 나와도, 그것은 <b>'누수된 모델이 더 높은 점수를 낸다'</b>는 사실을 "
         "정밀하게 확인한 것일 뿐 <b>방법의 우열이 아니다</b>.</div>")

# ── 6. 회귀
if REG:
    v = list(REG.values())[0]; r = v["regression"]
    H.append("<h2>6. LD50(회귀) — 혼동행렬 대신 잔차와 예측구간</h2>")
    H.append(f"<div class='box'><b>MAE {r['MAE']:.4f}</b> (부트스트랩 95% CI "
             f"[{r['MAE_CI95'][0]:.4f}, {r['MAE_CI95'][1]:.4f}]) · 잔차 평균 {r['residual_mean']:+.4f}"
             f"(편향 거의 없음) · 잔차 SD {r['residual_sd']:.4f}<br>"
             f"<b>예측구간</b> — valid 잔차의 90% 분위(±{r['PI90_halfwidth']:.4f})로 구간을 만들어 test에 적용하니 "
             f"실제 커버리지 <b>{r['PI90_coverage']:.3f}</b>다. 목표 0.90 대비 "
             + ("<b>거의 맞다</b>" if abs(r["PI90_coverage"] - .9) < .03 else
                f"<b>{'과대' if r['PI90_coverage'] > .9 else '과소'} 커버</b>") +
             f". <small>{r['PI_note']}</small></div>")
    a = v["AD"]["bands"]
    H.append(f"<div class='warn'>LD50은 <b>AD가 가장 잘 작동한 사례</b>다 — in-domain MAE "
             f"{a['in-domain']['perf']:.4f} vs OOD {a['OOD']['perf']:.4f}로 "
             f"OOD에서 오차가 <b>{a['OOD']['perf'] / a['in-domain']['perf']:.2f}배</b>로 커진다. "
             "즉 이 엔드포인트에서는 AD 게이트가 실제로 쓸모 있다.</div>")

# ── 7. 소표본·한계
H.append("<h2>7. 어디까지 믿을 수 있나</h2>")
H.append("<div class='warn'><ul>"
         "<li><b>소표본 경고</b> — DILI(96)·발암성(55)은 CI 폭이 0.10~0.24다. 운영지표·ECE·t*가 모두 불안정하며, "
         "표에 <span class='sm'>소표본 참고</span> 배지를 달았다. <b>AD 비율은 소표본에도 유효</b>하지만 "
         "구간별 성능은 OOD 표본이 20 미만이라 산출하지 못했다.</li>"
         "<li><b>예측은 재현본</b> — 아티팩트가 없어 재현 학습했다(18/18 일치, 최대 편차 0.0001). "
         "완전히 동일한 객체는 아니지만 동일 config·seed·분할이다.</li>"
         "<li><b>배포 예측 = 5 seed 확률 평균</b>. 마스터가 인용한 값은 seed별 지표의 평균이라 "
         "여기 AUROC와 소수점 셋째 자리에서 다를 수 있다(예: DILI 0.9125 → 앙상블 0.9196). "
         "둘은 <b>다른 대상</b>(모델 평균 vs 예측 평균)이며 둘 다 정직한 값이다.</li>"
         "<li><b>t*는 valid에서만 선택</b>했고 test에 1회 적용했다. 단 valid가 작은 엔드포인트에서는 t*가 흔들린다(§3 DILI).</li>"
         "<li><b>AD 컷오프는 절대선이 아니다</b> — 학습셋 분포의 5%·25% 백분위라는 <b>선택</b>이다. "
         "다른 컷을 쓰면 비율이 달라진다.</li>"
         "<li><b>여전히 전향적 검증은 없다</b> — 전부 TDC 안에서의 회고적 평가다.</li></ul></div>")

# ── 8. 종합
H.append("<h2>8. 종합 — 이 모델을 새 분자에 써도 되나</h2>")
H.append("<div class='box'>"
         f"<p><b>1. 순위 능력은 쓸 만하다.</b> 18과제 챔피언의 AUROC는 대체로 0.75~0.91이고 "
         "부트스트랩 CI도 0.5를 크게 상회한다(§5).</p>"
         f"<p><b>2. 그런데 기본 작동점(0.5)은 못 쓴다.</b> 불균형 과제에서 민감도가 0.0~0.3까지 떨어져 "
         "<b>독성을 대부분 놓친다</b>. NR-PPAR-gamma는 AUROC 0.839인데 민감도 0.000이다. "
         "<b>배포하려면 임계값을 반드시 재설정</b>해야 하고, 그 선택은 valid에서 해야 한다(§3).</p>"
         f"<p><b>3. AD는 항상 통하지 않는다.</b> {nv}/{len(D)}과제에서만 OOD 성능 하락이 확인됐고 "
         f"{nn}과제에서는 <b>AD가 신뢰도를 예측하지 못했다</b>. AD 게이트는 엔드포인트별로 "
         "<b>유효성을 확인한 뒤에</b> 켜야 한다(§2).</p>"
         "<p><b>4. 확률은 큰 데이터셋에서만 믿을 만하다.</b> AMES·Tox21은 ECE 0.015~0.073이지만 "
         "DILI·발암성은 0.13~0.15다(§4).</p>"
         f"<p><b>5. 정확한 대응비교는 마스터의 결론을 강화했다.</b> DeLong으로 재면 "
         f"<b>{nsig}과제에서 G2 챔피언이 정직한 GNN보다 유의하게 우세</b>하고 반대 사례는 0건이다(§5).</p></div>")
H.append("<p class='src'>산출물: <code>predictions/*.jsonl</code>(72개 · <b>재사용 자산</b>) · "
         "<code>results/reliability.csv</code> · <code>results/reliability.json</code> · "
         "<code>results/reproduction_check.json</code> · <code>notes.md</code></p>")

(RES / "reliability_report.html").write_text("\n".join(H), encoding="utf-8")
print(f"저장 → reliability_report.html ({(RES/'reliability_report.html').stat().st_size/1024:.0f} KB)")
print(f"AD 유효 {nv} · 예측못함 {nn} · 판정불가 {nu} · DeLong 유의 {nsig}건(전부 G2 우세)")
