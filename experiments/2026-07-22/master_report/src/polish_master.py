#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
polish_master.py — 마스터 최종본 정리 패스. ★순수 편집(새 계산·재학습·재측정 0).
(A) §3-A 4개 표의 행을 ★세대 오름차순(G1→G2→G3→G4→G5→참고)으로 재정렬 — 셀 내용은 한 글자도 안 바꿈.
    표 순서가 바뀌므로 §3-A 막대그림 4장도 ★master_matrix.csv의 같은 값으로 세대 순서로 다시 그린다.
(B) §8 과적합 표의 chemberta·molformer 중복 블록 제거(모델×엔드포인트당 1행).
(C) 맨 위 배너·§10 결론 #2를 §6과 정합("보수적 3 → 정확 DeLong 10").
★값·SD·p·격차·구분선은 어떤 것도 변경 금지 — 위치·중복·요약 문구만.
산출: results/master_report.html · master_report_before_polish.html(백업) · results/polish_check.json
"""
import base64, io, json, re, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

MR = Path("/home/nudge/Project/ADMET_integrated/2026-07-22/master_report/results")
SRC = MR / "master_report.html"
BAK = MR / "master_report_before_polish.html"
if not BAK.exists():
    shutil.copy(SRC, BAK)
base = BAK.read_text(encoding="utf-8")          # ★항상 백업에서 시작 → 재실행 안전

for p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
          "/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf"):
    if Path(p).exists():
        fm.fontManager.addfont(p)
plt.rcParams["font.family"] = ["NanumGothic", "NanumSquareRound", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MM = pd.read_csv(MR / "master_matrix.csv")
GORD = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5, "참고": 9}
MORD = {"물리화학 서술자 + XGBoost": 1, "물리화학 서술자 + RandomForest": 2, "ECFP4 지문 + XGBoost": 3,
        "chemprop D-MPNN (우리 자체 학습·정직)": 1, "ADMET-AI (Chemprop D-MPNN, 공개 사전학습)": 2,
        "ChemBERTa-2 (SMILES transformer, fine-tune)": 1, "MoLFormer (SMILES transformer, fine-tune)": 2}
out = base
report = {}

# ══════════ (A) §3-A 표 재정렬 ══════════
def rowkey(tr):
    m = re.search(r"<tr>\s*<td><b>([^<]+)</b></td>\s*<td[^>]*>(.*?)</td>", tr, re.S)
    if not m:
        return (99, 99, "")
    gen = m.group(1).strip()
    model = re.sub("<[^>]+>", "", m.group(2)).strip()
    return (GORD.get(gen, 8), MORD.get(model, 5), model)


reordered = []
for lab in ("DILI 간독성", "hERG 차단", "AMES 변이원성", "LD50 급성독성"):
    i = out.index(f"{lab} — ")
    ts = out.index("<table>", i); te = out.index("</table>", ts)
    seg = out[ts:te]
    trs = re.findall(r"<tr>.*?</tr>", seg, re.S)
    head, body = trs[0], trs[1:]
    before = [rowkey(t)[2] for t in body]
    body_sorted = sorted(body, key=rowkey)
    after = [rowkey(t)[2] for t in body_sorted]
    assert sorted(before) == sorted(after) and set(body) == set(body_sorted), f"{lab}: 행 집합 변형"
    new_seg = "<table>" + head + "".join(body_sorted)
    out = out[:ts] + new_seg + out[te:]
    reordered.append(dict(endpoint=lab, n_rows=len(body), before=before, after=after,
                          rowset_identical=True))
report["A_reorder"] = reordered

# ── §3-A 막대그림 4장을 세대 순서로 다시 그림(값은 master_matrix.csv 그대로) ──
FLAB = {"DILI 간독성": ("AUROC", True), "hERG 차단": ("AUROC", True),
        "AMES 변이원성": ("AUROC", True), "LD50 급성독성": ("MAE", False)}


def draw(lab):
    met, hb = FLAB[lab]
    sub = MM[(MM.endpoint == lab) & (MM.metric == met) & MM.value.notna()]
    ours = sub[sub.kind == "우리 학습"].copy()
    ours["o1"] = ours.gen.map(GORD); ours["o2"] = ours.model.map(lambda m: MORD.get(m, 5))
    ours = ours.sort_values(["o1", "o2"])
    ai = sub[sub.kind == "누수 기준선"]
    so = sub[sub.kind == "리더보드(미재현)"]
    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    x = np.arange(len(ours))
    GC = {"G2": "#2a9d8f", "G3": "#e76f51", "G4": "#457b9d", "G5": "#9d4edd"}
    ax.bar(x, ours.value, yerr=[0 if pd.isna(s) else s for s in ours.sd], capsize=3,
           color=[GC.get(g, "#8d99ae") for g in ours.gen], edgecolor="#2b2b2b", linewidth=.6)
    vals = list(ours.value)
    if not ai.empty:
        a = float(ai.iloc[0].value); vals.append(a)
        ax.axhline(a, ls="--", lw=1.8, color="#c1440e")
        ax.text(len(ours) - .45, a, f" ADMET-AI {a:.3f}\n ★누수 의심", color="#c1440e",
                fontsize=8.5, va="bottom", ha="right")
    if not so.empty:
        s = float(so.iloc[0].value)
        ax.axhline(s, ls=":", lw=1.5, color="#6b7580")
        ax.text(0, s, f" TDC SOTA {s:.3f}", color="#4a545e", fontsize=8.5, va="bottom")
    ax.set_ylim(min(vals) - .06, max(vals) + .06)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n{m.split('(')[0].strip()[:16]}" for g, m in zip(ours.gen, ours.model)],
                       fontsize=8)
    ax.set_ylabel(met + ("  (높을수록 좋음)" if hb else "  (낮을수록 좋음)"), fontsize=9.5)
    ax.set_title(f"{lab} — 세대별 {met}", fontsize=11.5)
    ax.grid(axis="y", alpha=.25)
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=125, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode(), list(ours.gen), list(ours.value)


figs = re.findall(r'<figure><img src="data:image/png;base64,([A-Za-z0-9+/=]+)"><figcaption>(.*?)</figcaption></figure>',
                  out, re.S)
i3a = out.index("3-A. 핵심 4종"); i3b = out.index("3-B. 확장 14과제")
blk = out[i3a:i3b]
old_figs = re.findall(r'<figure><img src="data:image/png;base64,([A-Za-z0-9+/=]+)"><figcaption>(.*?)</figcaption></figure>',
                      blk, re.S)
fig_log = []
labs = ["DILI 간독성", "hERG 차단", "AMES 변이원성", "LD50 급성독성"]
for (ob64, cap), lab in zip(old_figs, labs):
    nb64, gens, vals = draw(lab)
    newcap = re.sub(r"\s*<b>\[플래그십 보고서에서 이관\]</b>", "", cap).strip()
    newcap += (" <b>[표와 같은 세대 순서로 다시 그림 — 값은 master_matrix.csv 그대로, "
               "재계산 없음]</b>")
    blk = blk.replace(f'<figure><img src="data:image/png;base64,{ob64}"><figcaption>{cap}</figcaption></figure>',
                      f'<figure><img src="data:image/png;base64,{nb64}"><figcaption>{newcap}</figcaption></figure>', 1)
    fig_log.append(dict(endpoint=lab, bar_order=gens, values=[round(float(v), 4) for v in vals]))
out = out[:i3a] + blk + out[i3b:]
report["A_figures"] = fig_log

# ══════════ (B) §8 중복 제거 ══════════
i8 = out.index("<h2 id='s8'>")
ts = out.index("<table>", i8); te = out.index("</table>", ts)
seg = out[ts:te]
trs = re.findall(r"<tr>.*?</tr>", seg, re.S)
head, body = trs[0], trs[1:]
seen, uniq = set(), []
for t in body:
    k = re.sub(r"\s+", " ", re.sub("<[^>]+>", "|", t)).strip()
    if k in seen:
        continue
    seen.add(k); uniq.append(t)
out = out[:ts] + "<table>" + head + "".join(uniq) + out[te:]
report["B_dedupe"] = dict(rows_before=len(body), rows_after=len(uniq), removed=len(body) - len(uniq),
                          rowset_identical=(set(body) == set(uniq)))

# ══════════ (C) 배너·§10 정합 ══════════
old_banner = "<span class='k'>구분 가능 세대 효과 <b>3</b>건(전부 G2&gt;G3)</span>"
assert old_banner in out
out = out.replace(old_banner,
                  "<span class='k'>구분 가능 세대 효과: 보수적 <b>3</b>건 → 정확(DeLong) <b>10</b>건"
                  " · 전부 G2&gt;G3</span>", 1)
old_c2 = ("<b>3/18</b>뿐이고 핵심 4종은 <b>0/4</b>다. 정확한 진술은 "
          "<b>'최신 세대가 고전을 이긴다는 증거가 없다'</b>이며, 반대 방향의 증거도 큰 표본 일부에서만 잡힌다. ")
if old_c2 not in out:                                   # 문구가 조금 달라도 잡히도록
    m = re.search(r"<b>3/18</b>[^<]*<b>0/4</b>다\.", out)
    assert m, "§10 결론 #2 문장을 찾지 못함"
    old_c2 = m.group(0)
    new_c2 = old_c2 + (" <b>단 이는 보수적 근사(Hanley-McNeil)에 따른 것이고, 같은 test 분자 위 "
                       "정확한 대응비교(DeLong)로 재면 17개 분류 과제 중 <b>10건</b>이 유의하며 "
                       "<b>전부 G2가 G3를 앞선다</b>(§6). 방향은 바뀌지 않았고 '고전 우세'가 더 넓게 확인됐다.</b>")
else:
    new_c2 = old_c2 + ("<b>단 이는 보수적 근사(Hanley-McNeil)이고, 정확한 대응비교(DeLong)로는 "
                       "17개 분류 과제 중 10건이 유의하며 전부 G2가 G3를 앞선다(§6) — "
                       "방향은 불변, '고전 우세'가 더 넓게 확인됐다.</b> ")
out = out.replace(old_c2, new_c2, 1)
report["C_summary"] = dict(banner_updated=True, conclusion2_updated=True)

SRC.write_text(out, encoding="utf-8")

# ══════════ 검증 ══════════
def table_values(html, marker, is_anchor=False):
    i = html.index(marker)
    ts = html.index("<table>", i); te = html.index("</table>", ts)
    cells = re.findall(r"<td[^>]*>(.*?)</td>", html[ts:te], re.S)
    return sorted(re.sub(r"\s+", " ", re.sub("<[^>]+>", "", c)).strip() for c in cells)


ok = {}
for lab in labs:
    ok[f"§3-A {lab} 셀집합 동일"] = (table_values(base, f"{lab} — ") == table_values(out, f"{lab} — "))
ok["§8 셀집합 동일(중복 제외)"] = (sorted(set(table_values(base, "<h2 id='s8'>")))
                            == sorted(set(table_values(out, "<h2 id='s8'>"))))
# ★숫자 검증 — (B)가 중복 행을 지우므로 multiset은 줄어드는 게 정상이다.
#   따라서 (1) 새로 생긴 값 0 (2) 고유값 집합 동일 (3) 줄어든 분량이 ★전부 §8 표에서만 나온 것
#   — 이 세 가지로 "값은 안 바뀌고 중복만 빠졌다"를 확인한다.
import collections as _c
strip_img = lambda s: re.sub(r'data:image/png;base64,[A-Za-z0-9+/=]+', 'IMG', s)
nb = _c.Counter(re.findall(r"\d+\.\d{3,}", strip_img(base)))
na = _c.Counter(re.findall(r"\d+\.\d{3,}", strip_img(out)))


def _t8(h):
    i = h.index("<h2 id='s8'>"); ts = h.index("<table>", i); te = h.index("</table>", ts)
    return _c.Counter(re.findall(r"\d+\.\d{3,}", h[ts:te]))


lost, gained = nb - na, na - nb
ok["새로 생긴 숫자 없음"] = (len(gained) == 0)
ok["고유 숫자값 집합 동일"] = (set(nb) == set(na))
ok["감소분이 §8 중복분과 정확히 일치"] = (lost == (_t8(base) - _t8(out)))
ok["§6 수치와 정합(3·10)"] = ("보수적 <b>3</b>건 → 정확(DeLong) <b>10</b>건" in out)
report["verification"] = ok
report["counts"] = dict(numbers_before=sum(nb.values()), numbers_after=sum(na.values()),
                        numbers_removed_as_duplicates=sum(lost.values()),
                        size_before=len(base), size_after=len(out))
json.dump(report, open(MR / "polish_check.json", "w"), ensure_ascii=False, indent=1)

for k, v in ok.items():
    print(f"  {'OK ' if v else '★실패'} {k}")
print(f"\n(A) 4개 표 재정렬 · 그림 4장 세대순 재작성")
print(f"(B) §8 {report['B_dedupe']['rows_before']}행 → {report['B_dedupe']['rows_after']}행 "
      f"(중복 {report['B_dedupe']['removed']}행 제거)")
print(f"(C) 배너·§10 결론 #2 정합 완료")
print(f"저장 → master_report.html ({len(out)/1024:.0f} KB) · 백업 {BAK.name} · polish_check.json")
if not all(ok.values()):
    raise SystemExit("★중단 — 값이 변했다")
