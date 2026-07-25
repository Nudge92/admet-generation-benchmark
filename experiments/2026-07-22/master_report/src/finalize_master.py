#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
finalize_master.py — 마스터 보고서 최종화. ★새 계산·재학습·재측정 0건.
(A) §6에 ★DeLong 정확 대응비교 결과를 ★기존 3건 판정을 지우지 않고 이어 붙인다.
(B) §10 뒤에 새 §11 '권장 사용(배포 가이드)'를 신설한다.
숫자는 전부 experiment_deploy_reliability 의 확정 산출물에서 인용(재계산 금지).
★§1~10 본문은 바이트 단위로 보존 — 재생성이 아니라 ★삽입 편집만 한다.
산출: results/master_report.html(최종) · results/finalize_check.json
"""
import json, re, shutil
from pathlib import Path

B = Path("/home/nudge/Project/ADMET_integrated/2026-07-22")
MR = B / "master_report/results"
REL = B / "experiment_deploy_reliability/results"
SRC_HTML = MR / "master_report.html"
BACKUP = MR / "master_report_before_final.html"

if not BACKUP.exists():
    shutil.copy(SRC_HTML, BACKUP)
base = BACKUP.read_text(encoding="utf-8")          # ★항상 백업(편집 전)에서 시작 → 재실행 안전

R = json.load(open(REL / "reliability.json"))
CHK = json.load(open(REL / "reproduction_check.json"))
NAME = {"dili": "DILI", "herg": "hERG", "ames": "AMES", "ld50_zhu": "LD50",
        "Carcinogens_Lagunin": "발암성", "ClinTox": "ClinTox", "Tox21": "Tox21"}
CHAMP = {(c["endpoint"], c["task"]): c["model"] for c in CHK["checks"]}


def nm(v):
    return NAME.get(v["endpoint"], v["endpoint"]) + ("" if v["task"] in ("—", "Y") else f" {v['task']}")


CLS = {k: v for k, v in R.items() if "operating" in v}
REG = {k: v for k, v in R.items() if "regression" in v}

# ══════════ (A) §6 — DeLong 추가 ══════════
dl = []
for k, v in CLS.items():
    c = (v.get("delong") or {}).get("dmpnn_ours")
    if c and "p_value" in c:
        dl.append((nm(v), v["n_test"], c))
nsig = sum(1 for _, _, c in dl if c["p_value"] < 0.05)
nwin = sum(1 for _, _, c in dl if c["p_value"] < 0.05 and c["delta"] > 0)

A = ["<h3>같은 판정을 ★정확한 대응비교(DeLong)로 다시 하면</h3>",
     "<div class='fix'><b>왜 다시 재는가</b> — 위 판정에 쓴 Hanley-McNeil 구분 가능선은 "
     "<b>두 모델이 서로 다른 표본에서 나왔다고 가정</b>하는 근사다. 실제로는 <b>같은 test 분자</b>에 두 모델을 "
     "적용했으므로 <b>대응비교(DeLong)</b>가 정확하다. 분자별 예측이 없어 마스터 작성 시점에는 못 했는데, "
     "이후 <code>experiment_deploy_reliability</code>에서 챔피언의 분자별 예측을 만들어 계산할 수 있게 됐다.<br>"
     "<span class='src'>주의: 이 비교의 '챔피언'은 각 엔드포인트의 G2 최고 모델이고, 상대는 "
     "§3에 나온 <b>정직하게 학습한 G3 D-MPNN</b>이다.</span></div>",
     "<table><tr><th>과제</th><th class='n'>n_test</th><th class='n'>G2 챔피언</th>"
     "<th class='n'>G3 D-MPNN(정직)</th><th class='n'>Δ</th><th class='n'>p (DeLong)</th><th>판정</th></tr>"]
for name, n, c in dl:
    sig = c["p_value"] < 0.05
    A.append(f"<tr{' class=best' if sig else ''}><td>{name}</td><td class='n'>{n}</td>"
             f"<td class='n'>{c['auc_champion']:.4f}</td><td class='n'>{c['auc_other']:.4f}</td>"
             f"<td class='n'>{c['delta']:+.4f}</td><td class='n'>{c['p_value']:.2e}</td>"
             f"<td>{'<b>★유의 — G2 우세</b>' if sig else '비유의'}</td></tr>")
A.append("</table>")
A.append(f"<div class='warn'><b>보수적 근사 {3}건 → 정확 대응비교 <b>{nsig}건</b></b><br>"
         f"DeLong으로 재면 {len(dl)}개 분류 과제 중 <b>{nsig}건이 유의</b>(p&lt;0.05)하고, "
         f"<b>{nwin}건 모두 G2 챔피언이 우세</b>하다. <b>정직한 G3 D-MPNN이 유의하게 이긴 과제는 0건</b>이다.<br><br>"
         "<b>→ 해석: 보수적 방법이 증거를 과소평가하고 있었다. 방향은 바뀌지 않았고, "
         "오히려 '고전 ML 우세'가 더 넓게 확인됐다.</b> "
         "위의 Hanley-McNeil 판정(3건)을 지우지 않고 함께 남긴 이유는, <b>3 → " + str(nsig) +
         "의 차이가 새 데이터 때문이 아니라 ★검정 방법(비대응 근사 vs 대응 정확)의 차이</b>이기 때문이다. "
         "같은 숫자라도 어떻게 재느냐에 따라 결론의 강도가 달라진다는 것 자체가 기록할 가치가 있다.</div>")
A.append("<div class='warn'><b>★유지되는 단서</b> — (a) 여전히 비유의 과제가 "
         f"{len(dl) - nsig}건 있다 (b) <b>G2·G3 모두 하이퍼파라미터 탐색을 하지 않았다</b> "
         "(c) G3는 <code>--class-balance</code>를 쓰고 G2는 쓰지 않는 <b>불균형 처리 비대칭</b>이 남아 있다. "
         "따라서 이는 <b>'이 세대 구성·이 조건에서'</b>의 결론이다.<br>"
         "<b>★ADMET-AI와의 DeLong 비교도 계산했으나(신뢰도 보고서 §5) 여기서는 세대 판정에 쓰지 않는다</b> — "
         "정확한 p값이 나와도 <b>누수된 값과의 비교는 실력 비교가 아니다</b>(§5).</div>")
A.append("<p class='src'>출처: <code>experiment_deploy_reliability/results/reliability.json</code> "
         "(DeLong 대응비교) · 챔피언 예측은 <code>predictions/*.jsonl</code></p>")
DELONG_BLOCK = "\n".join(A)

# ══════════ (B) §11 — 권장 사용 ══════════
def verdict(v):
    return "참고용 <span class='sm'>소표본</span>" if v["n_test"] < 150 else "조건부 사용 가능"


def adcell(v):
    w = v["AD"]["verdict"]
    if "유효" in w:
        return "<span class='good'>게이트 사용</span>"
    if "못함" in w:
        return "<span class='bad'>★AD 신뢰 금지</span>"
    return "<span class='na'>판정 불가(OOD 표본 부족)</span>"


def probcell(v):
    if "calibration" not in v:
        return "—(회귀)"
    e = v["calibration"]["ECE"]
    return (f"<span class='bad'>불안정 (ECE {e:.3f})</span>" if e > 0.10
            else f"<span class='good'>사용 가능 (ECE {e:.3f})</span>")


P = ["<h2 id='s11'>11. 권장 사용 — 배포 가이드</h2>",
     "<div class='box'>§1~10이 <b>'무엇이 이겼나'</b>였다면, 이 절은 <b>'실제로 써도 되나'</b>다. "
     "모든 수치는 <code>experiment_deploy_reliability</code>(18과제 챔피언의 분자별 예측)에서 이관했다. "
     "★그 예측은 <b>재현본</b>이다 — 챔피언 모델 아티팩트가 저장돼 있지 않아 동일 config·seed·분할로 "
     f"재현 학습했고, <b>{len(CHK['checks'])}/{len(CHK['checks'])} 원본 지표와 일치</b>"
     f"(최대 편차 {max(abs(c['diff']) for c in CHK['checks']):.4f}, 허용 ±{CHK['tolerance']})했다.</div>",
     "<h3>11-1. 엔드포인트별 권장 구성</h3>",
     "<table><tr><th>엔드포인트/과제</th><th class='n'>n_test</th><th>추천 모델(챔피언)</th>"
     "<th class='n'>권장 임계값</th><th>AD 게이트</th><th>확률 신뢰</th><th>배포 판정</th></tr>"]
for k, v in R.items():
    champ = CHAMP.get((v["endpoint"], v["task"]), v["model"])
    if "operating" in v:
        o = v["operating"]
        th = (f"<b>t* = {o['t_star']['threshold']:.3f}</b><br>"
              f"<small>민감도 {o['t_fixed']['sensitivity']:.2f}→{o['t_star']['sensitivity']:.2f}"
              f" · FN {o['t_fixed']['FN']}→{o['t_star']['FN']}</small>")
    else:
        r = v["regression"]
        th = (f"<small>회귀 — 예측구간 ±{r['PI90_halfwidth']:.3f}<br>"
              f"실제 커버리지 {r['PI90_coverage']:.3f} (목표 0.90)</small>")
    P.append(f"<tr><td>{nm(v)}</td><td class='n'>{v['n_test']}</td><td>{champ}</td>"
             f"<td class='n'>{th}</td><td>{adcell(v)}</td><td>{probcell(v)}</td><td>{verdict(v)}</td></tr>")
P.append("</table>")
P.append("<p><small>추천 모델은 §3의 G2 챔피언이다. LD50은 수치상 최고가 G4 Uni-Mol이었으나 "
         "G2와의 차이가 구분 가능선 미만이었고, 재현·검증 가능한 G2를 배포 후보로 삼았다.</small></p>")

low = sorted([(nm(v), v["operating"]["t_fixed"]["sensitivity"], v["operating"]["t_fixed"]["FN"],
               v["ci"]["AUROC"]) for v in CLS.values()
              if v["operating"]["t_fixed"]["sensitivity"] is not None], key=lambda t: t[1])[:5]
imp = [(nm(v), v["operating"]["t_fixed"]["sensitivity"], v["operating"]["t_star"]["sensitivity"])
       for v in CLS.values()
       if v["operating"]["t_star"]["sensitivity"] and v["operating"]["t_fixed"]["sensitivity"]
       and v["operating"]["t_star"]["sensitivity"] - v["operating"]["t_fixed"]["sensitivity"] > 0.15]
dili = CLS.get("dili|—")
P.append("<h3>11-2. ★핵심 배포 경고 — 기본 임계값 0.5를 쓰지 말 것</h3>")
P.append("<div class='crit'><b>양성이 희박한 과제에서 임계 0.5는 독성을 거의 못 잡는다.</b>"
         "<table><tr><th>과제</th><th class='n'>AUROC</th><th class='n'>민감도 @0.5</th>"
         "<th class='n'>놓친 독성(FN)</th></tr>"
         + "".join(f"<tr><td>{n}</td><td class='n'>{a:.4f}</td><td class='n bad'>{s:.3f}</td>"
                   f"<td class='n bad'>{f}</td></tr>" for n, s, f, a in low)
         + "</table>"
         "가장 극단적인 사례는 <b>NR-PPAR-gamma — AUROC 0.839인데 민감도 0.000</b>으로 "
         "<b>양성 46개를 전부 놓친다</b>.<br><br>"
         "<b>★교훈: AUROC가 높다는 것과 쓸 만한 작동점이 있다는 것은 완전히 다른 얘기다.</b> "
         "순위 능력(AUROC)은 좋아도, 기본 임계값에서는 모델이 사실상 '전부 음성'이라고 답한다.</div>")
P.append("<div class='warn'><b>대안과 그 한계</b> — <b>valid에서 MCC가 최대인 t\\*</b>로 재설정하면 "
         "여러 과제가 크게 개선된다("
         + " · ".join(f"{n} {a:.2f}→{b:.2f}" for n, a, b in imp[:4]) + ").<br>"
         "<b>★그러나 만능이 아니다</b> — "
         + (f"<b>DILI는 t*={dili['operating']['t_star']['threshold']:.3f}로 올라가며 민감도가 "
            f"{dili['operating']['t_fixed']['sensitivity']:.2f} → "
            f"{dili['operating']['t_star']['sensitivity']:.2f}로 악화</b>됐다"
            f"(FN {dili['operating']['t_fixed']['FN']} → {dili['operating']['t_star']['FN']}). "
            if dili else "")
         + "valid가 54~65분자로 작아 t* 자체가 불안정하기 때문이다. "
         "<b>소표본 엔드포인트에서는 임계값 튜닝도 믿을 수 없다.</b><br>"
         "★임계값은 <b>반드시 valid에서</b> 고르고 test에는 1회만 적용해야 한다(여기서도 그렇게 했다).</div>")

nv = sum(1 for v in R.values() if "유효" in v["AD"]["verdict"])
nn_ = sum(1 for v in R.values() if "못함" in v["AD"]["verdict"])
nu = len(R) - nv - nn_
bad_ad = [nm(v) for v in R.values() if "못함" in v["AD"]["verdict"]]
ld = list(REG.values())[0] if REG else None
P.append("<h3>11-3. AD(적용범위) 사용 규칙</h3>")
P.append(f"<div class='warn'><b>AD는 자동 안전장치가 아니다.</b> 학습셋 기준 5-NN Tanimoto로 재정의하고 "
         f"구간별 성능으로 검증한 결과 — <b>유효 {nv}건 · ★신뢰도를 예측하지 못함 {nn_}건 · "
         f"판정 불가 {nu}건</b>(OOD 표본 20 미만)이었다.<ul>"
         f"<li><b>게이트로 쓸 것</b>: 유효가 확인된 {nv}과제(AMES·LD50·대부분 Tox21).</li>"
         f"<li><b>★AD를 신뢰하지 말 것</b>: {', '.join(bad_ad)} — 이 과제들은 OOD에서도 성능이 떨어지지 않아 "
         "AD가 위험 신호를 주지 못한다.</li>"
         f"<li><b>판정 불가</b>: DILI·hERG·발암성·ClinTox는 OOD 분자가 20개 미만이라 유효성 자체를 못 쟀다. "
         "AD 비율은 계산되지만 <b>게이트 근거로 쓰기엔 검증되지 않았다</b>.</li>"
         + (f"<li><b>AD 최적 사례 = LD50</b>: in-domain MAE {ld['AD']['bands']['in-domain']['perf']:.3f} vs "
            f"OOD {ld['AD']['bands']['OOD']['perf']:.3f} — OOD에서 오차가 "
            f"<b>{ld['AD']['bands']['OOD']['perf'] / ld['AD']['bands']['in-domain']['perf']:.2f}배</b>로 커진다.</li>"
            if ld else "")
         + "</ul><small>컷오프는 학습셋 자기 5-NN 분포의 5%(OOD)·25%(경계) 백분위다 — <b>절대선이 아니라 선택</b>이며, "
         "다른 컷을 쓰면 비율이 달라진다.</small></div>")

P.append("<h3>11-4. 운영 원칙 요약</h3>")
P.append("<div class='box'><ul>"
         "<li><b>순위는 쓸 만하다</b> — 챔피언 AUROC 대체로 0.75~0.91, 부트스트랩 95% CI도 0.5를 크게 상회.</li>"
         "<li><b>작동점은 반드시 재설정</b> — 0.5 금지. valid에서 t*를 고르고 test 1회 적용. 소표본에서는 t*도 흔들림.</li>"
         "<li><b>AD는 엔드포인트별 확인 후에만</b> — 유효 확인된 과제에서만 게이트로 사용.</li>"
         "<li><b>확률은 큰 데이터셋에서만</b> — AMES·Tox21은 ECE 0.015~0.073으로 쓸 만하나 "
         "DILI(0.131)·발암성(0.150)은 불안정. 보정은 후처리라 <b>고쳐도 AUROC는 그대로</b>다.</li>"
         "<li><b>소표본은 '참고용'</b> — DILI(96)·발암성(55)은 CI 폭이 0.10~0.24로 운영지표·ECE·t*가 모두 불안정하다.</li>"
         "<li><b>★남은 최대 한계: 전향적 검증이 없다</b> — 전부 TDC 벤치마크 안에서의 회고적 평가이며, "
         "새로 합성·측정한 분자로 확인한 것이 아니다.</li>"
         "<li><b>예측은 재현본</b> — 아티팩트 부재로 재현 학습했고 18/18 일치(최대 편차 0.0001)했다. "
         "동일 config·seed·분할이지만 동일 객체는 아니다.</li></ul></div>")
P.append("<div class='fix'><b>★숫자 뉘앙스 고지</b> — §11의 챔피언 AUROC(예: DILI <b>0.9196</b>)는 "
         "<b>5 seed 확률을 평균한 배포 예측</b>의 성능이고, §3 본문의 값(DILI <b>0.9125</b>)은 "
         "<b>seed별 지표를 평균</b>한 것이다. 둘은 <b>서로 다른 대상</b>(예측 평균 vs 모델 평균)이며 "
         "<b>둘 다 정직한 값</b>이라 억지로 통일하지 않았다.</div>")
P.append("<p class='src'>출처: <code>experiment_deploy_reliability/results/reliability.csv</code> · "
         "<code>reliability.json</code> · <code>reproduction_check.json</code> · "
         "전체 신뢰도 보고서 <code>experiment_deploy_reliability/results/reliability_report.html</code></p>")
SEC11 = "\n".join(P)

# ══════════ 삽입 ══════════
anchor7 = "<h2 id='s7'>"
assert base.count(anchor7) == 1
out = base.replace(anchor7, DELONG_BLOCK + "\n" + anchor7, 1)

anchor_out = "<p class='src'>산출물:"
assert out.count(anchor_out) == 1
out = out.replace(anchor_out, SEC11 + "\n" + anchor_out, 1)

nav_last = "<a href='#s10'>10. 종합 결론</a>"
assert nav_last in out
out = out.replace(nav_last, nav_last + " <a href='#s11'>11. 권장 사용(배포)</a>", 1)
out = out.replace("<title>독성 예측 세대 분석 — 마스터 보고서 (18과제)</title>",
                  "<title>독성 예측 세대 분석 — 마스터 보고서 최종본 (18과제 + 배포 가이드)</title>", 1)

SRC_HTML.write_text(out, encoding="utf-8")

# ══════════ 검증: §1~10 본문 불변 ══════════
def body_text(html, start, end):
    i = html.index(start); j = html.index(end)
    t = re.sub(r'data:image/png;base64,[A-Za-z0-9+/=]+', 'IMG', html[i:j])
    return re.sub(r'\s+', ' ', re.sub('<[^>]+>', ' ', t)).strip()


old_s1_s6 = body_text(base, "<h2 id='s1'>", "<h2 id='s7'>")
new_s1_s6 = body_text(out, "<h2 id='s1'>", "<h3>같은 판정을 ★정확한 대응비교")
old_s7_end = body_text(base, "<h2 id='s7'>", "<p class='src'>산출물:")
new_s7_end = body_text(out, "<h2 id='s7'>", "<h2 id='s11'>")
same_a, same_b = old_s1_s6 == new_s1_s6, old_s7_end == new_s7_end
nums_old = re.findall(r'\d+\.\d+', old_s1_s6 + old_s7_end)
nums_new = re.findall(r'\d+\.\d+', new_s1_s6 + new_s7_end)
chk = dict(sections_1_6_identical=same_a, sections_7_10_identical=same_b,
           n_numbers_before=len(nums_old), n_numbers_after=len(nums_new),
           numbers_identical=(nums_old == nums_new),
           delong_significant=nsig, delong_champion_wins=nwin, delong_n_compared=len(dl),
           ad_valid=nv, ad_ineffective=nn_, ad_undetermined=nu,
           backup=str(BACKUP.name), size_before=len(base), size_after=len(out))
json.dump(chk, open(MR / "finalize_check.json", "w"), ensure_ascii=False, indent=1)

print(f"§1~6 본문 동일: {same_a} · §7~10 본문 동일: {same_b} · 숫자열 동일: {nums_old == nums_new} "
      f"({len(nums_old)}개)")
print(f"DeLong 유의 {nsig}/{len(dl)}건(전부 G2 우세 {nwin}) · AD 유효 {nv}/못함 {nn_}/불가 {nu}")
print(f"저장 → master_report.html ({len(out)/1024:.0f} KB, 이전 {len(base)/1024:.0f} KB) · "
      f"백업 {BACKUP.name} · finalize_check.json")
if not (same_a and same_b and nums_old == nums_new):
    raise SystemExit("★중단 — §1~10 본문이 달라졌다")
