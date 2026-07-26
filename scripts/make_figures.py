#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""README 그림 생성 — 확정된 CSV 값만 사용. 새 계산·새 학습 없음.

사용법:  python3 scripts/make_figures.py     (저장소 루트에서)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
SRC = os.environ.get("ADMET_RESULTS", "results")
OUT = os.environ.get("ADMET_ASSETS", "docs/assets")
MAN = os.environ.get("ADMET_SPLITS", "splits/_manifest.csv")
OUTSC = os.path.join(OUT, "selfcorrection")
os.makedirs(OUTSC, exist_ok=True)
LOWER = {"MAE"}


# ── 데이터 ───────────────────────────────────────────────
def load():
    tox = pd.read_csv(f"{SRC}/master_matrix.csv")
    adme = pd.read_csv(f"{SRC}/adme_matrix.csv")
    man = pd.read_csv(MAN)
    n2a = dict(zip(man["n_test"], man["endpoint"]))
    ntest = dict(zip(man["endpoint"], man["n_test"]))

    own = tox[tox["kind"].isin(["우리 학습", "우리 학습(규칙)"])].copy()
    own["ep"] = np.where(own["endpoint"] == "Tox21",
                         own["n_test"].map(lambda n: n2a.get(n, f"Tox21_{n}")),
                         own["endpoint"])
    trow = []
    for ep, g in own.groupby("ep"):
        gg = g.dropna(subset=["value"])
        if gg.empty:
            continue
        hb = bool(gg["higher_better"].iloc[0])
        gg = gg.sort_values("value", ascending=not hb)
        w, r = gg.iloc[0], gg.iloc[1]
        trow.append(dict(domain="독성", endpoint=ep, metric=w["metric"], gen=w["gen"],
                         v1=w["value"], sd1=w["sd"], v2=r["value"],
                         gap=abs(w["value"] - r["value"])))
    T = pd.DataFrame(trow)

    clean = adme[adme["leak_flag"] == "누수 0"]
    arow = []
    for ep, g in clean.groupby("endpoint"):
        m = g["metric"].iloc[0]
        asc = m in LOWER
        gg = g.dropna(subset=["value"]).sort_values("value", ascending=asc)
        w, r = gg.iloc[0], gg.iloc[1]
        g2 = g[g["gen"] == "G2"].dropna(subset=["value"])["value"]
        g3 = g[g["model"] == "dmpnn_ours"].dropna(subset=["value"])["value"]
        b2 = g2.min() if asc else g2.max()
        rel = ((b2 - g3.iloc[0]) / abs(b2) * 100) if asc else ((g3.iloc[0] - b2) / abs(b2) * 100)
        arow.append(dict(domain="ADME", endpoint=ep, label=w["label"], metric=m, gen=w["gen"],
                         v1=w["value"], sd1=w["sd"], v2=r["value"],
                         gap=abs(w["value"] - r["value"]),
                         n_total=ntest[ep] / 0.2, rel_gnn=rel))
    A = pd.DataFrame(arow)
    return T, A, tox


T, A, TOXRAW = load()
ALL = pd.concat([T, A], ignore_index=True)
ALL["ratio"] = ALL["gap"] / ALL["sd1"]


# ── 그림 1 : 세대별 1위 ──────────────────────────────────
def fig1():
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    order = ["G2", "G3", "G4", "G5"]
    doms = ["독성 18과제", "ADME 18과제"]
    counts = {
        "독성 18과제": T["gen"].value_counts().reindex(order).fillna(0).astype(int),
        "ADME 18과제": A["gen"].value_counts().reindex(order).fillna(0).astype(int),
    }
    for yi, d in enumerate(doms):
        left = 0
        for g in order:
            n = counts[d][g]
            if n == 0:
                continue
            ax.barh(yi, n, left=left, height=.52, color=GC[g],
                    edgecolor="white", linewidth=1.4)
            ax.text(left + n / 2, yi, str(n), ha="center", va="center",
                    color="white", fontweight="bold", fontsize=11)
            left += n
    ax.set_yticks(range(len(doms)))
    ax.set_yticklabels(doms, fontsize=11)
    ax.set_xlim(0, 18.6)
    ax.set_xlabel("1위 획득 엔드포인트 수 (누수 기준선 제외)")
    ax.invert_yaxis()
    ax.legend(handles=[Patch(facecolor=GC[g], label=GL[g]) for g in order],
              loc="lower center", bbox_to_anchor=(.5, -.52), ncol=4,
              frameon=False, fontsize=9)
    ax.annotate("이 1건(LD50)은 통계 검정에서 철회됨 → 자기정정 ③",
                xy=(17.5, 0), xytext=(12.4, -.62), fontsize=8.4, color="#57606a",
                arrowprops=dict(arrowstyle="->", color="#8b949e", lw=.9))
    ax.set_title("세대가 올라간다고 이기지는 않는다 — 그리고 영역마다 다르다",
                 fontsize=12, fontweight="bold", pad=26, loc="left")
    fig.savefig(f"{OUT}/fig1_generation_wins.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── 그림 2 : 격차 / 불확실성 ─────────────────────────────
def fig2():
    d = ALL.dropna(subset=["ratio"]).sort_values("ratio")
    fig, ax = plt.subplots(figsize=(8.4, 8.2))
    y = np.arange(len(d))
    cols = [GC[g] for g in d["gen"]]
    ax.hlines(y, 0, d["ratio"], color=cols, lw=2.2, alpha=.85)
    ax.scatter(d["ratio"], y, color=cols, s=42, zorder=3)
    ax.axvline(1, color="#d1242f", ls="--", lw=1.3)
    ax.text(1.06, len(d) - .6, "격차 = 흔들림(SD)", color="#d1242f", fontsize=9)
    ax.axvspan(0, 1, color="#d1242f", alpha=.05)
    lab = [e.replace("Tox21__", "Tox21 ").replace("_carbonmangels", "")
            .replace("_astrazeneca", "").replace("_", " ") for e in d["endpoint"]]
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=8.6)
    ax.set_ylim(-1, len(d))
    ax.set_xlabel("1위–2위 격차 ÷ 1위 표준편차  (1보다 작으면 seed 흔들림 안)")
    n_in = int((d["ratio"] < 1).sum())
    ax.set_title("36개 중 %d개는 1위와 2위를 구분할 수 없다"
                 % n_in, fontsize=12, fontweight="bold", loc="left", pad=14)
    top = d.iloc[-1]
    ax.annotate("친유성 logD — 부트스트랩 3/3\n유일하게 확인된 세대 효과",
                xy=(top["ratio"], len(d) - 1), xytext=(top["ratio"] * .42, len(d) - 7),
                fontsize=9, color="#24292f",
                bbox=dict(boxstyle="round,pad=.4", fc="#fff8c5", ec="#d4a72c", lw=.8),
                arrowprops=dict(arrowstyle="->", color="#8b949e"))
    ax.legend(handles=[Patch(facecolor=GC[g], label=GL[g])
                       for g in ["G2", "G3", "G4"]],
              loc="lower right", frameon=False, fontsize=9)
    fig.savefig(f"{OUT}/fig2_gap_vs_uncertainty.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── 그림 3 : 데이터 크기 vs GNN ──────────────────────────
def fig3():
    d = A.dropna(subset=["rel_gnn"]).sort_values("n_total")
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    c = ["#2da44e" if v > 0 else "#d1242f" for v in d["rel_gnn"]]
    ax.axhline(0, color="#57606a", lw=1)
    ax.vlines(d["n_total"], 0, d["rel_gnn"], color=c, lw=1.6, alpha=.55)
    ax.scatter(d["n_total"], d["rel_gnn"], c=c, s=58, zorder=3, edgecolor="white", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("데이터셋 전체 크기 (log)")
    ax.set_ylabel("G3 GNN의 G2 최고 대비 상대 성능 (%)")
    ax.axvspan(1500, 2000, color="#8b949e", alpha=.12)
    ax.text(1730, ax.get_ylim()[1] * .9, "1500\n~2000", ha="center",
            fontsize=8, color="#57606a")
    for _, r in d.iterrows():
        if abs(r["rel_gnn"]) > 15:
            ax.annotate(r["label"], (r["n_total"], r["rel_gnn"]),
                        textcoords="offset points", xytext=(0, 9 if r["rel_gnn"] > 0 else -16),
                        ha="center", fontsize=8.6, color="#24292f")
    ax.set_title("GNN의 성패를 가른 건 아키텍처가 아니라 데이터 크기",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    fig.savefig(f"{OUT}/fig3_datasize_gnn.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── 자기정정 ③ : LD50 철회 ──────────────────────────────
def sc03():
    r = TOXRAW[(TOXRAW["endpoint"] == "LD50 급성독성") & (TOXRAW["kind"] == "우리 학습")]
    r = r.dropna(subset=["value"]).sort_values("value")
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    y = np.arange(len(r))[::-1]
    ax.errorbar(r["value"], y, xerr=r["sd"], fmt="none", elinewidth=2.2,
                capsize=5, ecolor="#8b949e")
    ax.scatter(r["value"], y, s=70, zorder=4,
               c=[GC[g] for g in r["gen"]], edgecolor="white", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}  {m[:26]}" for g, m in zip(r["gen"], r["model"])], fontsize=9)
    a, b = r.iloc[0], r.iloc[1]
    ax.annotate("", xy=(a["value"], .38), xytext=(b["value"], .38),
                arrowprops=dict(arrowstyle="<->", color="#d1242f", lw=1.5))
    ax.text((a["value"] + b["value"]) / 2, .52,
            "격차 %.4f" % abs(a["value"] - b["value"]),
            ha="center", fontsize=9.5, color="#d1242f", fontweight="bold")
    ax.set_xlabel("LD50 급성독성 — MAE (낮을수록 좋음) · 오차막대 = seed 표준편차")
    ax.set_title("자기정정 ③  “LD50은 4세대 승” → 철회\n"
                 "1위와 2위의 격차가 1위 자신의 seed 흔들림보다 작다",
                 fontsize=11.5, fontweight="bold", loc="left", pad=12)
    fig.savefig(f"{OUTSC}/sc03_ld50_withdrawn.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


fig1(); fig2(); fig3(); sc03()

ALL.to_csv(f"{SRC}/figure_data.csv", index=False)
print("생성 완료")
for root, _d, fs in os.walk(OUT):
    for f in sorted(fs):
        p = os.path.join(root, f)
        print("  %-46s %6.0f KB" % (p.replace(OUT, "assets"), os.path.getsize(p) / 1024))
print()
print("36개 중 격차<SD:", int((ALL['ratio'] < 1).sum()))


# ── 자기정정 ⑥ : 비대칭 로스터 ──────────────────────────
def sc06():
    f = pd.read_csv(f"{SRC}/feature_2x2.csv")
    n = len(f)
    asc = f["metric"].isin(LOWER)
    ph_best = np.where(asc, f[["xgb_physchem", "rf_physchem"]].min(axis=1),
                       f[["xgb_physchem", "rf_physchem"]].max(axis=1))
    orig = int(np.where(asc, ph_best < f["xgb_ecfp"], ph_best > f["xgb_ecfp"]).sum())
    xgb = int(f["xgb_physchem_wins"].sum())
    rf = int(f["rf_physchem_wins"].sum())

    bars = [
        ("원래 로스터 — 물성 2모델 중 최고  vs  ECFP 1모델", orig, "#d1242f",
         "철회된 비교 — 물성 쪽에만 모델이 하나 더 있었다"),
        ("동일 XGB끼리 — xgb_physchem vs xgb_ecfp", xgb, "#1f6feb",
         "진 곳: " + ", ".join(f[~f["xgb_physchem_wins"]]["label"])),
        ("동일 RF끼리 — rf_physchem vs rf_ecfp", rf, "#1f6feb",
         "진 곳: " + ", ".join(f[~f["rf_physchem_wins"]]["label"])),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    for i, (lab, v, c, note) in enumerate(bars):
        ax.barh(i, v, height=.46, color=c, alpha=.9)
        ax.barh(i, n - v, left=v, height=.46, color="#eaeef2")
        ax.text(v - .25, i, "%d/%d" % (v, n), ha="right", va="center",
                color="white", fontweight="bold", fontsize=11)
        ax.text(0, i - .42, lab, fontsize=9.4, color="#24292f", va="center")
        ax.text(n + .25, i, note, fontsize=8.4, color="#57606a", va="center")
    ax.set_yticks([]); ax.set_xlim(0, n)
    ax.set_ylim(len(bars) - .5, -.75)
    ax.set_xlabel("물성 서술자가 ECFP 지문을 이긴 엔드포인트 수 (ADME 18)")
    ax.spines["left"].set_visible(False)
    ax.set_title("자기정정 ⑥  “물성 18/18” → 17/18 · 16/18\n"
                 "차이를 만든 건 표현이 아니라 로스터가 비대칭이었다는 사실",
                 fontsize=11.5, fontweight="bold", loc="left", pad=14)
    fig.savefig(f"{OUTSC}/sc06_asymmetric_roster.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("⑥ 원래 %d/%d · XGB %d/%d · RF %d/%d" % (orig, n, xgb, n, rf, n))


sc06()


# ── 자기정정 ⑧ : seed 처리에 따른 판정 뒤집힘 ────────────
def sc08():
    import json
    d = json.load(open(f"{SRC}/g4_verification.json"))
    names = {"half_life_obach": "반감기 (Spearman ↑)",
             "solubility_aqsoldb": "수용해도 (MAE ↓)"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    for ax, (ep, v) in zip(axes, d.items()):
        rows = [("A  5개 seed 예측을 평균낸 뒤 비교", v["bootstrap_A"], "#2da44e"),
                ("B  seed별로 비교", v["bootstrap_B"], "#d1242f")]
        for i, (lab, b, c) in enumerate(rows):
            lo, hi = b["ci"]
            ax.plot([lo, hi], [i, i], lw=3.2, color=c, alpha=.55,
                    solid_capstyle="butt")
            ax.plot([lo, lo], [i - .12, i + .12], lw=2, color=c)
            ax.plot([hi, hi], [i - .12, i + .12], lw=2, color=c)
            ax.plot(b["delta"], i, "o", ms=9, color=c, zorder=4,
                    mec="white", mew=1.2)
            ax.text(hi, i + .3, "구분됨" if b["distinguishable"] else "구분 안 됨",
                    fontsize=9, color=c, fontweight="bold", ha="right")
        ax.axvline(0, color="#24292f", lw=1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([r[0] for r in rows], fontsize=9)
        ax.set_ylim(1.6, -.5)
        ax.set_xlabel("G4 − G2 차이  ·  95% 부트스트랩 구간")
        dA, dB = v["bootstrap_A"]["delta"], v["bootstrap_B"]["delta"]
        ax.set_title("%s\n앙상블이 설명하는 몫 %.0f%%" % (names[ep], (dA - dB) / dA * 100),
                     fontsize=10.5, fontweight="bold", loc="left", pad=8)
    fig.suptitle("자기정정 ⑧  같은 데이터, 같은 모델 — seed를 어떻게 묶느냐로 판정이 뒤집힌다",
                 fontsize=12, fontweight="bold", x=.012, ha="left", y=1.12)
    fig.text(.012, -.10,
             "예측을 먼저 평균내면 변동이 큰 Uni-Mol(G4)이 더 이득을 본다. "
             "수용해도에서 G4의 앙상블 이득 +0.0356 vs G2 +0.0170.\n"
             "두 판정을 모두 보고하는 이유 — 어느 하나만 쓰면 결론이 달라진다.",
             fontsize=8.6, color="#57606a")
    fig.savefig(f"{OUTSC}/sc08_seed_flip.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


sc08()
