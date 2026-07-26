#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""추가 그림 8장 (A~H) — 확정된 CSV/JSON 값만 사용. 새 계산·새 학습 없음.

`make_figures.py`(그림 1~3 · sc03 · sc06 · sc08)와 같은 스타일 계약을 쓴다.
세대 색·저장 방식·제목 톤을 바꾸지 말 것 — 독자가 그림 사이를 색으로 따라간다.

사용법:  python3 scripts/make_figures2.py     (저장소 루트에서)

값의 출처는 각 함수 머리말에 적었다. 재계산으로 확인한 것은 `[검증]`으로 표시했다.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "Noto Sans CJK JP",
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#c9d1d9",
    "axes.labelcolor": "#24292f",
    "text.color": "#24292f",
    "xtick.color": "#57606a",
    "ytick.color": "#57606a",
    "font.size": 10,
    "figure.dpi": 160,
})

GC = {"G1": "#9aa0a6", "G2": "#1f6feb", "G3": "#f0883e", "G4": "#2da44e", "G5": "#8250df"}
GL = {"G1": "G1 규칙", "G2": "G2 고전 ML", "G3": "G3 GNN", "G4": "G4 3D", "G5": "G5 파운데이션"}
RED, GREEN, GRAY, INK = "#d1242f", "#2da44e", "#8b949e", "#24292f"

SRC = os.environ.get("ADMET_RESULTS", "results")
OUT = os.environ.get("ADMET_ASSETS", "docs/assets")
MAN = os.environ.get("ADMET_SPLITS", "splits/_manifest.csv")
EXP = os.environ.get("ADMET_EXPERIMENTS", "experiments")
OUTSC = os.path.join(OUT, "selfcorrection")
os.makedirs(OUTSC, exist_ok=True)

TOXNAME = {  # master_matrix 표기 → 짧은 라벨
    "AMES 변이원성": "AMES 변이원성", "DILI 간독성": "DILI 간독성", "hERG 차단": "hERG 차단",
    "LD50 급성독성": "LD50 급성독성", "ClinTox 임상독성": "ClinTox 임상독성",
    "발암성 (Carcinogens)": "발암성",
}


def short(ep):
    return TOXNAME.get(ep, ep.replace("Tox21__", "Tox21 "))


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  저장 %-52s %6.0f KB" % (path, os.path.getsize(path) / 1024))


# ══════════════════════════════════════════════════════════════════
#  A. 36 엔드포인트 지도
#     출처 results/figure_data.csv (1위 세대·격차·SD) + splits/_manifest.csv (n_test)
#          + results/adme_matrix.csv (기둥 A/D/M/E)
#     n_total = n_test ÷ 0.2 — make_figures.py의 기존 규약과 동일(TDC 고정 20% test)
# ══════════════════════════════════════════════════════════════════
def fig4_endpoint_map():
    fd = pd.read_csv(f"{SRC}/figure_data.csv")
    man = pd.read_csv(MAN)
    adme = pd.read_csv(f"{SRC}/adme_matrix.csv")
    pil = dict(zip(adme["endpoint"], adme["pillar"]))
    lab = dict(zip(adme["endpoint"], adme["label"]))

    # 독성은 manifest 이름과 표기가 달라 n_test를 따로 잇는다
    tox_key = {"AMES 변이원성": "ames", "DILI 간독성": "dili", "hERG 차단": "herg",
               "LD50 급성독성": "ld50_zhu", "ClinTox 임상독성": "ClinTox",
               "발암성 (Carcinogens)": "Carcinogens_Lagunin"}
    ntest = dict(zip(man["endpoint"], man["n_test"]))

    rows = []
    for _, r in fd.iterrows():
        ep = r["endpoint"]
        if r["domain"] == "독성":
            key = tox_key.get(ep, ep)
            rows.append(dict(block="독성", name=short(ep), n_test=ntest[key],
                             gen=r["gen"], ratio=r["ratio"]))
        else:
            rows.append(dict(block=pil[ep], name=lab[ep], n_test=ntest[ep],
                             gen=r["gen"], ratio=r["ratio"]))
    D = pd.DataFrame(rows)
    D["n_total"] = D["n_test"] / 0.2
    D["resolved"] = D["ratio"] >= 1

    blocks = [("독성", "독성 18과제"), ("A", "ADME · A 흡수"), ("D", "ADME · D 분포"),
              ("M", "ADME · M 대사"), ("E", "ADME · E 배설")]

    ys, ylab, seps, heads = [], [], [], []
    y = 0.0
    plot = []
    for b, title in blocks:
        g = D[D["block"] == b].sort_values("n_total", ascending=False)
        heads.append((y + 0.70, title, len(g)))
        for _, r in g.iterrows():
            ys.append(y); ylab.append(r["name"]); plot.append((y, r))
            y -= 1.0
        seps.append(y + 0.5)
        y -= 0.85

    fig, ax = plt.subplots(figsize=(10.6, 10.2))
    for s in seps[:-1]:
        ax.axhline(s, color="#eaeef2", lw=1.1)

    smin, smax = D["n_test"].min(), D["n_test"].max()
    def msize(n):
        return 42 + 190 * (np.sqrt(n) - np.sqrt(smin)) / (np.sqrt(smax) - np.sqrt(smin))

    for yy, r in plot:
        c = GC[r["gen"]]
        ax.scatter(r["n_total"], yy, s=msize(r["n_test"]),
                   facecolor=c if r["resolved"] else "white",
                   edgecolor=c, linewidth=1.7, zorder=3)

    # 유일하게 확정된 세대 효과
    logd_y = [yy for yy, r in plot if r["name"] == "친유성 logD"][0]
    logd_x = [r["n_total"] for yy, r in plot if r["name"] == "친유성 logD"][0]
    ax.axhspan(logd_y - .45, logd_y + .45, color="#fff8c5", zorder=0)
    ax.annotate("부트스트랩 3/3 — 36개 중 유일하게\n재표집을 견딘 세대 효과",
                xy=(logd_x, logd_y), xytext=(logd_x * 2.7, logd_y - 2.4),
                fontsize=8.8, color=INK,
                bbox=dict(boxstyle="round,pad=.42", fc="#fff8c5", ec="#d4a72c", lw=.8),
                arrowprops=dict(arrowstyle="->", color="#8b949e", lw=.9))

    ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels(ylab, fontsize=8.7)
    ax.set_ylim(y + 0.6, 1.5)
    ax.set_xlim(180, 40000)
    ax.set_xlabel("데이터셋 전체 크기 (log) — test 집합 ÷ 0.2")
    ax.tick_params(axis="y", length=0)
    for yy, t, n in heads:
        ax.text(197, yy, "%s  (%d)" % (t, n), fontsize=9.6, fontweight="bold",
                color="#57606a", va="center")

    gens = [g for g in ["G2", "G3", "G4"] if g in set(D["gen"])]
    h = [Line2D([], [], marker="o", ls="", ms=8, mfc=GC[g], mec=GC[g], label=GL[g]) for g in gens]
    h += [Line2D([], [], marker="o", ls="", ms=8, mfc=INK, mec=INK, label="1·2위 구분 가능"),
          Line2D([], [], marker="o", ls="", ms=8, mfc="white", mec=INK, mew=1.7,
                 label="구분 불가 (격차 < seed SD)")]
    ax.legend(handles=h, loc="lower right", frameon=True, fontsize=8.8,
              facecolor="white", edgecolor="#d0d7de", ncol=1)

    n_open = int((~D["resolved"]).sum())
    ax.set_title("36개 전부 — 색은 1위 세대, 속 빈 원은 1위·2위를 구분할 수 없는 곳 (%d개)" % n_open,
                 fontsize=12.5, fontweight="bold", loc="left", pad=14)
    fig.text(.005, -.018,
             "점 크기 ∝ test 집합 크기.  세로는 기둥별 묶음이고 각 묶음 안에서는 데이터가 큰 순서다.  "
             "누수 기준선(ADMET-AI)은 순위에서 제외돼 있다.",
             fontsize=8.5, color="#57606a")
    save(fig, f"{OUT}/fig4_endpoint_map.png")
    print("    · 구분 불가 %d/36 · 1위 세대 %s" % (n_open, dict(D["gen"].value_counts())))


# ══════════════════════════════════════════════════════════════════
#  B. 배포 작동점 — 0.5 vs t*
#     출처 results/reliability.csv (sens_05 · sens_star · t_star · FN_05 · FN_star)
# ══════════════════════════════════════════════════════════════════
def fig5_operating_point():
    DISP = {"dili": "DILI 간독성", "herg": "hERG 차단", "ames": "AMES 변이원성",
            "ClinTox": "ClinTox 임상독성", "Carcinogens_Lagunin": "발암성"}
    r = pd.read_csv(f"{SRC}/reliability.csv").dropna(subset=["sens_05"]).copy()
    r["name"] = np.where(r["endpoint"] == "Tox21",
                         "Tox21 " + r["task"].astype(str),
                         r["endpoint"].map(lambda e: DISP.get(e, e)))
    r["d"] = r["sens_star"] - r["sens_05"]
    r = r.sort_values("d")

    fig, ax = plt.subplots(figsize=(10.4, 6.6))
    y = np.arange(len(r))
    for i, (_, s) in enumerate(r.iterrows()):
        up = s["d"] > 1e-9
        dn = s["d"] < -1e-9
        c = GREEN if up else RED if dn else GRAY
        if abs(s["d"]) > 1e-9:
            ax.annotate("", xy=(s["sens_star"], i), xytext=(s["sens_05"], i),
                        arrowprops=dict(arrowstyle="-|>,head_width=.28,head_length=.5",
                                        color=c, lw=2.4, shrinkA=0, shrinkB=0))
        ax.plot(s["sens_05"], i, "o", ms=7.5, mfc="white", mec="#57606a", mew=1.6, zorder=4)
        ax.plot(s["sens_star"], i, "o", ms=7.5, color=c, mec="white", mew=1.1, zorder=5)
        ax.text(1.045, i, "t*=%.3f" % s["t_star"], fontsize=8, color="#57606a", va="center")
        ax.text(1.175, i, "놓친 양성 %d→%d" % (s["FN_05"], s["FN_star"]),
                fontsize=8, color=c if abs(s["d"]) > 1e-9 else "#57606a", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(r["name"], fontsize=8.8)
    ax.set_ylim(-.8, len(r) - .2)
    ax.set_xlim(-.03, 1.03)
    ax.set_xlabel("민감도 (양성을 실제로 잡아낸 비율)")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#eaeef2", lw=.9)
    ax.set_axisbelow(True)

    i_pp = int(np.where(r["name"] == "Tox21 NR-PPAR-gamma")[0][0])
    ax.annotate("AUROC 0.839인데 0.5에서는 민감도 0.000\n— 양성 46건을 하나도 못 잡는다",
                xy=(0.0, i_pp), xytext=(.60, i_pp + 1.05), fontsize=8.6, color=INK,
                va="center",
                bbox=dict(boxstyle="round,pad=.4", fc="#fff8c5", ec="#d4a72c", lw=.8),
                arrowprops=dict(arrowstyle="->", color="#8b949e", lw=.9))
    i_dili = int(np.where(r["name"] == "DILI 간독성")[0][0])
    ax.annotate("DILI는 반대로 악화된다 0.94→0.66\nvalid 54~65분자라 t* 자체가 불안정",
                xy=(0.72, i_dili + .18), xytext=(.30, i_dili + 3.0), fontsize=8.6, color=RED,
                va="center", ha="center",
                bbox=dict(boxstyle="round,pad=.4", fc="#ffebe9", ec="#ffb3ab", lw=.8),
                arrowprops=dict(arrowstyle="->", color="#8b949e", lw=.9))

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=8, mfc="white", mec="#57606a", mew=1.6,
               label="기본 임계값 0.5"),
        Line2D([], [], marker="o", ls="", ms=8, color=GREEN, label="권고 작동점 t* — 개선"),
        Line2D([], [], marker="o", ls="", ms=8, color=RED, label="권고 작동점 t* — 악화"),
    ], loc="upper center", bbox_to_anchor=(.42, -.085), ncol=3,
        frameon=False, fontsize=9)

    n_up = int((r["d"] > 1e-9).sum()); n_dn = int((r["d"] < -1e-9).sum())
    ax.set_title("기본 임계값 0.5는 쓸 수 없다 — 다만 권고 작동점도 만능은 아니다",
                 fontsize=12.5, fontweight="bold", loc="left", pad=14)
    fig.text(.005, -.075,
             "독성 17개 분류 과제(LD50은 회귀라 제외).  t*로 개선 %d · 악화 %d · 변화 없음 %d.  "
             "출처 results/reliability.csv" % (n_up, n_dn, len(r) - n_up - n_dn),
             fontsize=8.5, color="#57606a")
    save(fig, f"{OUT}/fig5_operating_point.png")
    print("    · 개선 %d · 악화 %d · 무변화 %d" % (n_up, n_dn, len(r) - n_up - n_dn))


# ══════════════════════════════════════════════════════════════════
#  C. AD 판정 분포
#     출처 results/reliability.csv (독성) [검증 11/3/4 = finalize_check.json]
#          experiments/2026-07-25/.../adme_reliability.csv (ADME) [검증 9/3/6]
# ══════════════════════════════════════════════════════════════════
def fig6_ad_verdict():
    tox = pd.read_csv(f"{SRC}/reliability.csv")["AD_verdict"]
    ad = pd.read_csv(f"{EXP}/2026-07-25/experiment_adme_reliability/results/"
                     "adme_reliability.csv")["AD_verdict"]

    def tri(s):
        return (int(s.str.contains("AD 유효").sum()),
                int(s.str.contains("예측").sum()),
                int(s.str.contains("판정 불가").sum()))

    T, A = tri(tox), tri(ad)
    chk = json.load(open(f"{SRC}/finalize_check.json"))
    assert T == (chk["ad_valid"], chk["ad_ineffective"], chk["ad_undetermined"]), T
    assert A == (9, 3, 6), A

    cats = [("AD 유효", GREEN), ("예측 못함", RED), ("판정 불가", GRAY)]
    rows = [("독성 18과제", T), ("ADME 18과제", A)]

    fig, ax = plt.subplots(figsize=(10.4, 3.0))
    for yi, (name, v) in enumerate(rows):
        left = 0
        for (lab, c), n in zip(cats, v):
            w = n / 18 * 100
            ax.barh(yi, w, left=left, height=.56, color=c, edgecolor="white", linewidth=1.6)
            ax.text(left + w / 2, yi, str(n), ha="center", va="center",
                    color="white", fontweight="bold", fontsize=13)
            if yi == 0:  # 범례 대신 위쪽 막대에 직접 라벨
                ax.text(left + w / 2, -.52, lab, ha="center", va="center",
                        fontsize=9.6, color=c, fontweight="bold")
            left += w
    ax.set_yticks(range(2))
    ax.set_yticklabels([n for n, _ in rows], fontsize=11)
    ax.set_ylim(1.62, -.95)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("엔드포인트 비율")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("적용 도메인은 자동 안전장치가 아니다",
                 fontsize=12.5, fontweight="bold", loc="left", pad=26)
    fig.text(.005, -.25,
             "「AD 유효」는 OOD 구간에서 성능이 실제로 떨어진 경우, 「예측 못함」은 AD 안이든 밖이든 성능이 같았던 경우, "
             "「판정 불가」는 OOD 구간 표본이 20개에 못 미친 경우다.\n"
             "36개 중 AD가 실제로 신뢰도를 예측한 건 20개뿐이다.  "
             "출처 results/reliability.csv · experiments/2026-07-25/.../adme_reliability.csv",
             fontsize=8.5, color="#57606a")
    save(fig, f"{OUT}/fig6_ad_verdict.png")
    print("    · 독성 %s · ADME %s" % (T, A))


# ══════════════════════════════════════════════════════════════════
#  D. 자기정정 ⑦ — 배설 3종
#     출처 results/adme_matrix.csv  [검증] Δ = G4(unimol) − G2 최고
#     ★ SD는 G2 최고 모델과 G4의 것을 둘 다 그린다 (notes 표의 SD 열이 혼재)
# ══════════════════════════════════════════════════════════════════
def sc07_excretion():
    a = pd.read_csv(f"{SRC}/adme_matrix.csv")
    eps = [("half_life_obach", "반감기"),
           ("clearance_hepatocyte_az", "간세포 청소율"),
           ("clearance_microsome_az", "마이크로솜 청소율")]
    rows = []
    for ep, name in eps:
        g = a[(a["endpoint"] == ep) & (a["leak_flag"] == "누수 0")]
        g4 = g[g["gen"] == "G4"].iloc[0]
        g2 = g[g["gen"] == "G2"].sort_values("value", ascending=False).iloc[0]
        rows.append(dict(name=name, g4=g4["value"], sd4=g4["sd"], g2=g2["value"],
                         sd2=g2["sd"], m2=g2["model"], delta=g4["value"] - g2["value"]))
    E = pd.DataFrame(rows)
    assert np.allclose(E["delta"], [0.1558, 0.0025, 0.0258], atol=5e-5), E["delta"].tolist()

    fig, ax = plt.subplots(figsize=(9.8, 4.0))
    y = np.arange(len(E))[::-1]
    for i, (_, s) in zip(y, E.iterrows()):
        clear = s["delta"] > max(s["sd2"], s["sd4"])
        c = GREEN if clear else GRAY
        sdmax = max(s["sd2"], s["sd4"])
        ax.barh(i, sdmax, height=.60, color=RED, alpha=.09, zorder=1)
        ax.barh(i, s["delta"], height=.38, color=c, alpha=.95, zorder=3)
        for sd in (s["sd2"], s["sd4"]):
            ax.plot([sd, sd], [i - .30, i + .30], color=RED, lw=1.9, zorder=5)
        ax.text(sdmax + .0035, i + .40, "seed 흔들림 SD   G2 %.4f · G4 %.4f"
                % (s["sd2"], s["sd4"]), fontsize=8.1, color=RED, va="center")
        ax.text(max(s["delta"], sdmax) + .0035, i - .04, "Δ %.4f" % s["delta"],
                fontsize=10, fontweight="bold", color=c, va="center")
        ax.text(1.015, i, "Δ ≫ SD\n뚜렷" if clear else "Δ < SD\n동률",
                transform=ax.get_yaxis_transform(), fontsize=9.4, color=c,
                fontweight="bold", va="center", linespacing=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(E["name"], fontsize=10.4)
    ax.set_xlim(0, .175)
    ax.set_ylim(-.72, len(E) - .28)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("G4(Uni-Mol) − G2 최고  ·  Spearman  ·  분홍 띠 = seed 흔들림 범위")
    ax.set_title("자기정정 ⑦  “배설 3종 석권” → 반감기만 뚜렷\n"
                 "격차가 seed 흔들림보다 작으면 순위표의 1위는 아무것도 말하지 않는다",
                 fontsize=11.5, fontweight="bold", loc="left", pad=12)
    fig.text(.005, -.10,
             "간세포 청소율의 격차는 자기 흔들림의 1/7~1/20이다.  어느 쪽 SD를 기준으로 잡아도 판정은 같다 — "
             "그래서 둘 다 그렸다.  출처 results/adme_matrix.csv",
             fontsize=8.5, color="#57606a")
    save(fig, f"{OUTSC}/sc07_excretion.png")
    print("    · Δ %s" % [round(v, 4) for v in E["delta"]])


# ══════════════════════════════════════════════════════════════════
#  E. 자기정정 타임라인
#     출처 notes/evidence.md 의 1차 근거 경로 (날짜 = experiments/ 폴더)
# ══════════════════════════════════════════════════════════════════
def sc00_timeline():
    COLS = ["2026-07-22", "2026-07-24", "2026-07-25"]
    items = [
        (0, "experiment_generation_matrix", [
            ("①", "누수 프리미엄 → 누수 상한", "배포 패키지 실측"),
            ("②", "세대 range → 1위·2위 격차", "지표 정의 감사"),
            ("③", "LD50 4세대 승 → 철회", "②의 직접 결과")]),
        (0, "experiment_adme_full", [
            ("⑤", "멀티태스크 이득 → 가설 기각", "★자체 발견"),
            ("⑥", "물성 18/18 → 17·16/18", "적대검증 → 재실행"),
            ("⑦", "배설 3종 → 반감기만", "효과크기 vs SD")]),
        (0, "master_report", [
            ("④", "Hanley-McNeil → DeLong", "검정 자체를 가능하게 만듦")]),
        (1, "experiment_g4_verification", [
            ("⑧", "판정이 seed 처리에 의존", "집계 방식 민감도")]),
        (2, "experiment_adme_reliability", [
            ("⑧", "주 판정을 A → B로 교체", "방법 비교엔 학습 변동 포함")]),
    ]

    fig, ax = plt.subplots(figsize=(11.8, 6.0))
    XC = [0.3, 4.5, 8.6]
    W = [3.9, 3.0, 3.2]
    ax.axhline(6.05, color="#d0d7de", lw=1.2)
    for i, c in enumerate(COLS):
        ax.plot(XC[i] + W[i] / 2, 6.05, "o", ms=10, color=INK, zorder=4)
        ax.text(XC[i] + W[i] / 2, 6.42, c, ha="center", fontsize=11.5,
                fontweight="bold", color=INK)

    anchors = {}
    ytop = 5.55
    ycur = [ytop, ytop, ytop]
    for col, expname, ent in items:
        h = 0.62 + 0.62 * len(ent)
        y0 = ycur[col] - h
        ax.add_patch(plt.Rectangle((XC[col], y0), W[col], h, fc="#f6f8fa",
                                   ec="#d0d7de", lw=.9, zorder=1,
                                   joinstyle="round"))
        ax.text(XC[col] + .16, ycur[col] - .34, expname, fontsize=8.6,
                color="#57606a", style="italic", va="center")
        for j, (num, what, how) in enumerate(ent):
            yy = ycur[col] - .96 - .62 * j
            ax.text(XC[col] + .22, yy, num, fontsize=13, fontweight="bold",
                    color=GC["G2"], va="center")
            ax.text(XC[col] + .62, yy + .11, what, fontsize=9.3, color=INK, va="center")
            ax.text(XC[col] + .62, yy - .17, how, fontsize=7.9, color="#57606a", va="center")
            anchors[(col, num)] = (XC[col] + W[col], yy)
        ycur[col] = y0 - 0.42

    a1 = anchors[(1, "⑧")]
    a2 = anchors[(2, "⑧")]
    ax.add_patch(FancyArrowPatch((a1[0] + .08, a1[1]), (XC[2] - .10, a2[1]),
                                 connectionstyle="arc3,rad=-.55", lw=2.2,
                                 color="#8250df", mutation_scale=17,
                                 arrowstyle="-|>", zorder=6))
    ax.text((a1[0] + XC[2]) / 2, a1[1] - 1.28,
            "발견은 07-24 · 주 판정 교체는 07-25\n하루를 두고 스스로 뒤집은 유일한 항목",
            ha="center", fontsize=8.8, color="#8250df", fontweight="bold")

    # 오른쪽 빈 자리 — 8건에서 반복해서 나타난 것 (notes/self_corrections.md 맺음)
    bx, by, bw, bh = XC[1], -1.49, 6.8, 2.35
    ax.add_patch(plt.Rectangle((bx, by), bw, bh, fc="white", ec="#d0d7de",
                               lw=.9, ls=(0, (4, 3)), zorder=1))
    ax.text(bx + .28, by + bh - .38, "8건에서 반복해서 나타난 것",
            fontsize=9.6, fontweight="bold", color=INK, va="center")
    for j, t in enumerate([
            "격차는 흔들림과 함께 봐야 한다 — ③⑦이 같은 실수의 두 사례",
            "비교 대상의 수를 맞춰야 한다 — ⑥. 세대 축에는 아직 남아 있다",
            "집계 방식이 결론을 만든다 — ⑧. 데이터도 모델도 그대로인데 뒤집혔다"]):
        ax.text(bx + .42, by + bh - .90 - .46 * j, "· " + t, fontsize=8.8,
                color="#57606a", va="center")

    ax.set_xlim(0, 11.9); ax.set_ylim(-1.75, 7.0)
    ax.axis("off")
    ax.set_title("검증 장치가 작동한 순서 — 자기정정 8건이 언제 어느 실험에서 일어났나",
                 fontsize=12.5, fontweight="bold", loc="left", pad=10)
    fig.text(.005, .012,
             "날짜는 notes/evidence.md 가 가리키는 1차 근거 파일의 experiments/ 폴더다.  "
             "07-22의 7건은 하룻밤 무인 실행과 그 뒤의 적대검증에서 한꺼번에 나왔다.",
             fontsize=8.5, color="#57606a")
    save(fig, f"{OUTSC}/sc00_timeline.png")


# ══════════════════════════════════════════════════════════════════
#  F. 자기정정 ⑤ — 멀티태스크
#     출처 experiments/2026-07-22/experiment_adme_full/results/learning_axis.csv
#          초기(누수) 관측은 .../src/build_final.py:234 에 문자로 기록돼 있다
# ══════════════════════════════════════════════════════════════════
def sc05_multitask():
    L = pd.read_csv(f"{EXP}/2026-07-22/experiment_adme_full/results/learning_axis.csv")
    meta = [("cyp_inhibition", "CYP 억제 3종\n닮은 과제", "3/3 이득"),
            ("cyp_substrate", "CYP 기질 3종\n작은 과제", "2/3 이득"),
            ("all_adme_cls", "ADME 분류 10개\n잡탕", "9/10 이득")]
    assert abs(L[L.group == "cyp_inhibition"].delta_multitask_minus_single.mean() - 0.0004) < 5e-5
    assert abs(L[L.group == "cyp_substrate"].delta_multitask_minus_single.mean() + 0.024) < 5e-5

    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    XLO, XHI = -.075, .19
    ax.axvline(0, color=INK, lw=1.3, zorder=2)
    ax.axvspan(XLO, 0, color=RED, alpha=.035, zorder=0)
    tr = ax.get_yaxis_transform()
    ax.text(1.035, len(meta) - .40, "초기(누수)", fontsize=8.8, color="#8b949e",
            transform=tr, ha="center", fontweight="bold")
    ax.text(1.20, len(meta) - .40, "차단 후", fontsize=8.8, color="#57606a",
            transform=tr, ha="center", fontweight="bold")
    for i, (grp, name, init) in enumerate(meta):
        g = L[L.group == grp]
        d = g["delta_multitask_minus_single"].values
        mean = d.mean(); npos = int((d > 0).sum())
        rng = np.random.RandomState(7 + i)
        jit = rng.uniform(-.13, .13, len(d))
        ax.scatter(d, i + jit, s=46, facecolor="white",
                   edgecolor=[GREEN if v > 0 else RED for v in d], linewidth=1.6, zorder=4)
        ax.plot([mean, mean], [i - .30, i + .30], lw=3.4,
                color=GREEN if mean > 0 else RED, zorder=5, solid_capstyle="butt")
        ax.text(mean, i + .40, "평균 %+.4f" % mean, ha="center", fontsize=9.2,
                fontweight="bold", color=GREEN if mean > 0 else RED)
        ax.text(1.035, i, init, fontsize=9, color="#8b949e", transform=tr,
                va="center", ha="center")
        ax.text(1.20, i, "%d/%d" % (npos, len(d)), fontsize=9.6, color="#57606a",
                transform=tr, va="center", ha="center", fontweight="bold")

    ax.set_yticks(range(len(meta)))
    ax.set_yticklabels([m[1] for m in meta], fontsize=9.6)
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(-.7, len(meta) - .25)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Δ AUROC  (멀티태스크 − 단일 과제)  ·  점 하나 = 엔드포인트 하나")
    ax.set_title("자기정정 ⑤  “멀티태스크 9/10 이득” → 가설 기각\n"
                 "닮은 과제를 묶어서 이득이 난 게 아니라, 과제들이 서로의 답을 갖고 있었다",
                 fontsize=11.5, fontweight="bold", loc="left", pad=12)
    fig.text(.005, -.075,
             "CYP 2C9·2D6·3A4는 같은 분자 라이브러리에 라벨만 다른 데이터셋이다.  합집합으로 묶으면 "
             "train에 test 분자의 88.9%가 섞인다 — 이건 이득 중 누수의 비율이 아니라 섞인 분자의 비율이다.\n"
             "차단 게이트(어느 과제의 test에라도 등장하는 분자를 합집합 train/valid에서 전부 제외) 후 "
             "남은 이득은 소표본 2개(생체이용률·HIA)에 몰려 있다.",
             fontsize=8.5, color="#57606a")
    save(fig, f"{OUTSC}/sc05_multitask.png")


# ══════════════════════════════════════════════════════════════════
#  G. 누수 3증거 — 상한이지 크기가 아니다
#     출처 experiments/2026-07-22/master_report/results/consistency.json
#            (leak_ub_core_mean 0.0876 · leak_ub_exp_mean 0.1491) [검증: 재계산 일치]
#          experiment_generation_matrix/notes.md:56-61 (증거② 표)
# ══════════════════════════════════════════════════════════════════
def fig7_leakage_evidence():
    cj = json.load(open(f"{EXP}/2026-07-22/master_report/results/consistency.json"))
    ub_core, ub_exp = cj["leak_ub_core_mean"], cj["leak_ub_exp_mean"]
    assert (ub_core, ub_exp) == (0.0876, 0.1491)
    clean = [("DILI", 0.8815, 0.9557), ("hERG", 0.8388, 0.9113), ("AMES", 0.8816, 0.9303)]
    cd = [b - a for _, a, b in clean]

    fig, ax = plt.subplots(figsize=(10.6, 4.7))
    ax.set_xlim(0, .215); ax.set_ylim(-1.15, 2.75)

    # 증거 ① — 수치 없음
    ax.text(.002, 2.42, "증거 ①", fontsize=10.4, fontweight="bold", color=INK)
    ax.text(.021, 2.42, "학습셋 크기가 TDC 전체와 일치 (DILI 475 = 379+96 정확 일치)",
            fontsize=9.4, color=INK, va="center")
    ax.text(.021, 2.13, "→ 방향만 말한다. 학습 분자 목록이 비공개라 분자 단위 대조는 불가능하다.",
            fontsize=8.5, color="#57606a", va="center")

    # 증거 ② — 가장 깨끗한 추정
    ax.text(.002, 1.62, "증거 ②", fontsize=10.4, fontweight="bold", color=GREEN)
    ax.text(.021, 1.62, "패키지에 기록된 자기 성능을 우리 test에서 초과",
            fontsize=9.4, color=INK, va="center")
    lo, hi = min(cd), max(cd)
    ax.add_patch(plt.Rectangle((lo, .78), hi - lo, .30, fc=GREEN, alpha=.20,
                               ec=GREEN, lw=1.4, zorder=3))
    for v in cd:
        ax.plot(v, .93, "o", ms=8, color=GREEN, mec="white", mew=1.2, zorder=5)
    ax.text(hi + .013, 1.03, "가장 깨끗한 추정  +0.049 ~ +0.074",
            fontsize=9.8, fontweight="bold", color=GREEN, va="center")
    ax.text(hi + .013, .77, "  ·  ".join("%s %+.3f" % (nm, v)
                                         for (nm, _a, _b), v in zip(clean, cd)),
            fontsize=8.4, color=GREEN, va="center")
    ax.text(.021, 1.34, "같은 모델·같은 앙상블·같은 멀티태스크라 그 성분들이 양쪽에서 상쇄된다 — "
            "달라지는 건 평가 분자뿐이다.", fontsize=8.5, color="#57606a", va="center")

    # 증거 ③ — 상한
    ax.text(.002, .30, "증거 ③", fontsize=10.4, fontweight="bold", color=RED)
    ax.text(.021, .30, "누수 없이 정직하게 학습한 같은 계열(D-MPNN)과의 격차",
            fontsize=9.4, color=INK, va="center")
    for i, (nm, v) in enumerate([("핵심 4종 (AUROC 3개)", ub_core), ("확장 14과제", ub_exp)]):
        yy = -.10 - .38 * i
        ax.add_patch(plt.Rectangle((0, yy - .11), v, .22, fc=RED, alpha=.13,
                                   ec=RED, lw=1.2, ls=(0, (4, 2)), zorder=3))
        ax.plot([v, v], [yy - .13, yy + .13], color=RED, lw=2.4, zorder=5)
        ax.text(v + .004, yy, "≤ %.4f   %s" % (v, nm), fontsize=9.2, color=RED, va="center")
    ax.text(.0035, -.90, "이 상자 안에는 누수 + 5모델 앙상블 + 31과제 멀티태스크 + 튜닝이 전부 교락돼 있다 "
            "→ 크기가 아니라 상한", fontsize=8.6, color="#57606a", va="center")

    ax.set_yticks([])
    ax.set_xlabel("ADMET-AI 초과분 (AUROC)")
    for s in ["left", "top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#eaeef2", lw=.9)
    ax.set_axisbelow(True)
    ax.set_title("누수의 크기가 아니라 상한이다",
                 fontsize=12.5, fontweight="bold", loc="left", pad=14)
    fig.text(.005, -.055,
             "점선 상자는 「여기까지일 수 있다」는 뜻이지 「이만큼이다」가 아니다.  "
             "증거 ③은 크기를 확정하지 못하고 방향을 독립적으로 확인할 뿐이다.\n"
             "출처 consistency.json (leak_ub_core_mean · leak_ub_exp_mean) · generation_matrix/notes.md 증거② 표",
             fontsize=8.5, color="#57606a")
    save(fig, f"{OUT}/fig7_leakage_evidence.png")


# ══════════════════════════════════════════════════════════════════
#  H. DeLong 결과
#     출처 results/reliability.csv (delong_vs_dmpnn_p) · results/finalize_check.json
#          효과크기는 results/master_matrix.csv (챔피언 − 정직 D-MPNN)
# ══════════════════════════════════════════════════════════════════
def fig8_delong():
    m = pd.read_csv(f"{SRC}/master_matrix.csv")
    man = pd.read_csv(MAN)
    r = pd.read_csv(f"{SRC}/reliability.csv").dropna(subset=["delong_vs_dmpnn_p"]).copy()
    r["key"] = np.where(r["endpoint"] == "Tox21", "Tox21__" + r["task"].astype(str),
                        r["endpoint"])
    n2a = dict(zip(man["n_test"], man["endpoint"]))
    tox_key = {"AMES 변이원성": "ames", "DILI 간독성": "dili", "hERG 차단": "herg",
               "ClinTox 임상독성": "ClinTox", "발암성 (Carcinogens)": "Carcinogens_Lagunin"}
    m["key"] = np.where(m["endpoint"] == "Tox21", m["n_test"].map(n2a),
                        m["endpoint"].map(lambda e: tox_key.get(e, e)))
    eff = {}
    for k, g in m[m["kind"] == "우리 학습"].groupby("key"):
        gg = g.dropna(subset=["value"])
        if gg["metric"].iloc[0] != "AUROC":
            continue
        d3 = gg[gg["model"].str.contains("D-MPNN")]
        if d3.empty:
            continue
        eff[k] = gg["value"].max() - d3["value"].iloc[0]
    DISP = {"dili": "DILI 간독성", "herg": "hERG 차단", "ames": "AMES 변이원성",
            "ClinTox": "ClinTox 임상독성", "Carcinogens_Lagunin": "발암성"}
    r["delta"] = r["key"].map(eff)
    r["name"] = np.where(r["endpoint"] == "Tox21", "Tox21 " + r["task"].astype(str),
                         r["endpoint"].map(lambda e: DISP.get(e, e)))
    r = r.dropna(subset=["delta"])
    chk = json.load(open(f"{SRC}/finalize_check.json"))
    assert len(r) == chk["delong_n_compared"] == 17, len(r)
    sig = r["delong_vs_dmpnn_p"] < .05
    assert int(sig.sum()) == chk["delong_significant"] == 10
    assert int((r.loc[sig, "delta"] > 0).sum()) == chk["delong_champion_wins"] == 10

    HL = {"Tox21 NR-Aromatase", "Tox21 SR-ARE", "Tox21 SR-MMP"}
    # 라벨이 겹치는 곳만 손으로 밀어준다 (offset points)
    NUDGE = {"Tox21 SR-p53": (9, -14), "Tox21 NR-ER": (9, 5),
             "Tox21 NR-PPAR-gamma": (9, 6), "Tox21 NR-AR-LBD": (9, -15),
             "Tox21 NR-AR": (9, -14), "ClinTox 임상독성": (10, -14)}
    fig, ax = plt.subplots(figsize=(10.6, 5.9))
    ax.axvspan(-.06, 0, color="#eaeef2", alpha=.7, zorder=0)
    ax.text(-.030, 3.05, "여기가 GNN이 이긴 영역이다\n— 17개 중 하나도 없다",
            ha="center", fontsize=9.4, color="#57606a", fontweight="bold")
    ax.axvline(0, color=INK, lw=1.3, zorder=2)
    ax.axhline(-np.log10(.05), color=RED, ls="--", lw=1.3, zorder=2)
    ax.text(-.058, -np.log10(.05) + .12, "p = 0.05", fontsize=9, color=RED)

    for _, s in r.iterrows():
        p = s["delong_vs_dmpnn_p"]; yv = -np.log10(max(p, 1e-12)); x = s["delta"]
        hi = s["name"] in HL
        is_sig = p < .05
        ax.scatter(x, yv, s=132 if hi else 74,
                   facecolor=GC["G2"] if is_sig else "white",
                   edgecolor=GC["G2"] if is_sig else GRAY,
                   linewidth=2.1 if hi else 1.5, zorder=5 if hi else 4,
                   marker="D" if hi else "o")
        if hi or is_sig:
            off = NUDGE.get(s["name"], (9, 8 if hi else 4))
            ax.annotate(s["name"], (x, yv), textcoords="offset points",
                        xytext=off, fontsize=8.3,
                        ha="right" if off[0] < 0 else "left",
                        color=INK if hi else "#57606a",
                        fontweight="bold" if hi else "normal")

    ax.set_xlim(-.062, .152)
    ax.set_xlabel("챔피언(G2 고전 ML) − 정직 학습 G3 D-MPNN  ·  AUROC 차")
    ax.set_ylabel("−log10 (DeLong p)")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=8, color=GC["G2"], label="유의 (p < 0.05) — 10건"),
        Line2D([], [], marker="o", ls="", ms=8, mfc="white", mec=GRAY, mew=1.5,
               label="유의하지 않음 — 7건"),
        Line2D([], [], marker="D", ls="", ms=8, mfc=GC["G2"], mec=GC["G2"],
               label="구분 가능선도 넘은 3건"),
    ], loc="upper left", frameon=True, fontsize=8.8, facecolor="white", edgecolor="#d0d7de")
    ax.set_title("유의한 차이 10건은 전부 고전 ML 쪽이었다",
                 fontsize=12.5, fontweight="bold", loc="left", pad=14)
    fig.text(.005, -.035,
             "17개 분류 과제 전부에서 점추정이 오른쪽(G2 우세)이다.  LD50은 회귀라 ROC 검정 대상이 아니어서 분모가 18이 아닌 17이다.\n"
             "Hanley-McNeil 근사로는 유의 3건이었다 — 같은 test를 쓰는 두 모델의 예측 상관을 무시해 검정력을 잃었기 때문이다(자기정정 ④).",
             fontsize=8.5, color="#57606a")
    save(fig, f"{OUT}/fig8_delong.png")
    print("    · 유의 %d/17 · 전부 G2 우세 확인" % int(sig.sum()))


if __name__ == "__main__":
    print("A 36 엔드포인트 지도");   fig4_endpoint_map()
    print("B 배포 작동점");          fig5_operating_point()
    print("C AD 판정 분포");         fig6_ad_verdict()
    print("D 자기정정 ⑦ 배설");      sc07_excretion()
    print("E 자기정정 타임라인");     sc00_timeline()
    print("F 자기정정 ⑤ 멀티태스크"); sc05_multitask()
    print("G 누수 3증거");           fig7_leakage_evidence()
    print("H DeLong");              fig8_delong()
    print("\n완료 — 8장")
