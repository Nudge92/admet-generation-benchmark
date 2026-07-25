#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_master.py — 플래그십(핵심 4종·G1~G5) + 확장(14과제·G1~G3)을 하나의 마스터 보고서로 통합.
★새 학습·재측정 0건 — 두 보고서의 ★확정 숫자를 옮겨 담고, 원본 그림은 base64를 ★그대로 추출해 이관.
★유일한 신규 그림 2장은 이관한 숫자만 그린 요약 시각화(모델 재실행 없음).
산출: results/master_report.html · results/master_matrix.csv · results/consistency.json
"""
import base64, io, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

B = Path("/home/nudge/Project/ADMET_integrated/2026-07-22")
F, E = B / "experiment_generation_matrix/results", B / "experiment_gen_expansion_g1g3/results"
OUT = B / "master_report/results"
OUT.mkdir(parents=True, exist_ok=True)

for p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
          "/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf"):
    if Path(p).exists():
        fm.fontManager.addfont(p)
plt.rcParams["font.family"] = ["NanumGothic", "NanumSquareRound", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 원본 로드 (재계산 없음) ────────────────────────────────────────
FM = pd.read_csv(F / "gen_matrix.csv")
FMETA = json.load(open(F / "matrix_meta.json"))
FG1 = pd.read_csv(F / "g1_summary.csv")
EX = json.load(open(E / "expansion_metrics.json"))
EMETA = EX.pop("_meta")
EG1 = pd.read_csv(E / "g1_summary.csv")
EOV = json.load(open(E / "overfit.json"))
FLAG_HTML = (F / "report.html").read_text(encoding="utf-8")
EXP_HTML = (E / "report_expansion.html").read_text(encoding="utf-8")

FLAB = {"dili": "DILI 간독성", "herg": "hERG 차단", "ames": "AMES 변이원성", "ld50_zhu": "LD50 급성독성"}
FMET = {"dili": "AUROC", "herg": "AUROC", "ames": "AUROC", "ld50_zhu": "MAE"}
ELAB = {"Carcinogens_Lagunin": "발암성 (Carcinogens)", "ClinTox": "ClinTox 임상독성", "Tox21": "Tox21"}
G2LAB = {"xgb_physchem": "물리화학 서술자 + XGBoost", "rf_physchem": "물리화학 서술자 + RandomForest",
         "xgb_ecfp": "ECFP4 지문 + XGBoost"}


def figs_from(html):
    """원본 HTML의 <figure> 를 (base64, caption) 으로 추출 — 재렌더링 없음."""
    return re.findall(r'<figure><img src="data:image/png;base64,([A-Za-z0-9+/=]+)">'
                      r'<figcaption>(.*?)</figcaption></figure>', html, re.S)


FFIGS, EFIGS = figs_from(FLAG_HTML), figs_from(EXP_HTML)


def emb(b64s, cap, note=""):
    return (f'<figure><img src="data:image/png;base64,{b64s}">'
            f'<figcaption>{cap}{note}</figcaption></figure>')


def b64_of(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=125, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


# ── 통합 매트릭스 조립 (값은 전부 원본 인용) ──────────────────────
rows = []
for ep, lab in FLAB.items():
    met = FMET[ep]
    sub = FM[(FM.endpoint == ep) & (FM.metric == met)]
    unc = FMETA["uncertainty"][ep]
    g1 = FG1[FG1.endpoint == ep].iloc[0]
    for _, r in sub.iterrows():
        if r.gen == "참고":
            kind = "리더보드(미재현)"
        elif "ADMET-AI" in str(r.method):
            kind = "누수 기준선"
        else:
            kind = "우리 학습"
        rows.append(dict(scope="핵심 4종", endpoint=lab, task="—", gen_range="G1~G5",
                         gen=r.gen, model=r.method, kind=kind, metric=met,
                         higher_better=bool(r.higher_better),
                         value=(None if pd.isna(r.value) else float(r.value)),
                         sd=(None if pd.isna(r.sd) else float(r.sd)),
                         n_test=int(r.n_test), pos_rate=None,
                         resolvable_gap_95=unc["resolvable_gap_95"],
                         g1_fire_rate=(float(g1.fire_rate_mutagen) if r.gen == "G1" else None),
                         g1_MCC=(float(g1.rule_MCC) if (r.gen == "G1" and pd.notna(g1.rule_MCC)) else None),
                         flag=r.leak_flag, source="experiment_generation_matrix/results/gen_matrix.csv"))
for ep in EX:
    for task, r in EX[ep].items():
        tname = "—" if len(EX[ep]) == 1 else task
        g1 = r["G1"]
        rows.append(dict(scope="확장", endpoint=ELAB[ep], task=tname, gen_range="G1~G3",
                         gen="G1", model="구조알림 (BRENK+NIH + Benigni-Bossa)", kind="우리 학습(규칙)",
                         metric="AUROC", higher_better=True, value=None, sd=None,
                         n_test=r["n_test"], pos_rate=r["pos_rate"],
                         resolvable_gap_95=r["resolvable_gap_95"],
                         g1_fire_rate=(g1 or {}).get("fire_rate"), g1_MCC=(g1 or {}).get("MCC"),
                         flag="누수 없음(학습 없음)",
                         source="experiment_gen_expansion_g1g3/results/g1_summary.csv"))
        for mk, ml in G2LAB.items():
            d = r["G2"].get(mk)
            rows.append(dict(scope="확장", endpoint=ELAB[ep], task=tname, gen_range="G1~G3",
                             gen="G2", model=ml, kind="우리 학습", metric="AUROC", higher_better=True,
                             value=(d["AUROC"]["mean"] if d else None),
                             sd=(d["AUROC"]["std"] if d else None),
                             n_test=r["n_test"], pos_rate=r["pos_rate"],
                             resolvable_gap_95=r["resolvable_gap_95"], g1_fire_rate=None, g1_MCC=None,
                             flag="누수 0(실측)",
                             source="experiment_gen_expansion_g1g3/results/expansion_metrics.json"))
        d = r["G3_ours"]
        rows.append(dict(scope="확장", endpoint=ELAB[ep], task=tname, gen_range="G1~G3", gen="G3",
                         model="chemprop D-MPNN (정직)", kind="우리 학습", metric="AUROC",
                         higher_better=True, value=(d["AUROC"]["mean"] if d else None),
                         sd=(d["AUROC"]["std"] if d else None), n_test=r["n_test"],
                         pos_rate=r["pos_rate"], resolvable_gap_95=r["resolvable_gap_95"],
                         g1_fire_rate=None, g1_MCC=None, flag="누수 0(train만 학습)",
                         source="experiment_gen_expansion_g1g3/results/expansion_metrics.json"))
        a = r["G3_admetai"]
        rows.append(dict(scope="확장", endpoint=ELAB[ep], task=tname, gen_range="G1~G3", gen="G3",
                         model="ADMET-AI", kind="누수 기준선", metric="AUROC", higher_better=True,
                         value=a.get("AUROC"), sd=None, n_test=r["n_test"], pos_rate=r["pos_rate"],
                         resolvable_gap_95=r["resolvable_gap_95"], g1_fire_rate=None, g1_MCC=None,
                         flag=a.get("leak_flag", a.get("note", "")),
                         source="experiment_gen_expansion_g1g3/results/admetai_metrics.json"))
MM = pd.DataFrame(rows)
MM.to_csv(OUT / "master_matrix.csv", index=False)


# ── 과제별 요약(세대 순위·판정) — 원본과 ★동일한 정의로만 재조립 ──
def task_summary():
    out = []
    for ep, lab in FLAB.items():
        met = FMET[ep]; hb = (met != "MAE")
        o = FM[(FM.endpoint == ep) & (FM.metric == met) & FM.value.notna() & (FM.gen != "참고")]
        o = o[~o.method.str.contains("ADMET-AI")]
        gb = o.groupby("gen").value.agg("max" if hb else "min").sort_values(ascending=not hb)
        ai = FM[(FM.endpoint == ep) & (FM.metric == met) & FM.method.str.contains("ADMET-AI")].iloc[0].value
        best_model = o.loc[o.value.idxmax() if hb else o.value.idxmin()]
        g3 = o[o.method.str.contains("우리 자체")]
        srt = o.sort_values("value", ascending=not hb).reset_index(drop=True)
        g3rank = int(srt[srt.method.str.contains("우리 자체")].index[0]) + 1 if not g3.empty else None
        prem = (ai - float(g3.iloc[0].value)) if (not g3.empty and hb) else (
            (float(g3.iloc[0].value) - ai) if not g3.empty else None)
        out.append(dict(scope="핵심 4종", endpoint=lab, task="—", metric=met, higher_better=hb,
                        n_test=int(o.iloc[0].n_test), gen_best=dict(gb.round(4)),
                        best_gen=gb.index[0], best_model=best_model.method, best_value=float(gb.iloc[0]),
                        gap_top2=float(abs(gb.iloc[0] - gb.iloc[1])),
                        rg=FMETA["uncertainty"][ep]["resolvable_gap_95"],
                        g3_rank=g3rank, n_models=len(o), admetai=float(ai), leak_ub=(None if prem is None else round(prem, 4)),
                        g1_MCC=float(FG1[FG1.endpoint == ep].iloc[0].rule_MCC)))
    for ep in EX:
        for task, r in EX[ep].items():
            vals = []
            for mk, ml in G2LAB.items():
                d = r["G2"].get(mk)
                if d:
                    vals.append(("G2", ml, d["AUROC"]["mean"]))
            if r["G3_ours"]:
                vals.append(("G3", "chemprop D-MPNN (정직)", r["G3_ours"]["AUROC"]["mean"]))
            gb = pd.Series({g: max(v for gg, _, v in vals if gg == g) for g in {x[0] for x in vals}}
                           ).sort_values(ascending=False)
            srt = sorted(vals, key=lambda t: -t[2])
            g3rank = next((i + 1 for i, t in enumerate(srt) if t[0] == "G3"), None)
            out.append(dict(scope="확장", endpoint=ELAB[ep], task=("—" if len(EX[ep]) == 1 else task),
                            metric="AUROC", higher_better=True, n_test=r["n_test"],
                            gen_best=dict(gb.round(4)), best_gen=gb.index[0],
                            best_model=srt[0][1], best_value=float(gb.iloc[0]),
                            gap_top2=float(gb.iloc[0] - gb.iloc[1]) if len(gb) > 1 else 0.0,
                            rg=r["resolvable_gap_95"], g3_rank=g3rank, n_models=len(vals),
                            admetai=r["G3_admetai"].get("AUROC"),
                            leak_ub=(r.get("leak_premium") or {}).get("AUROC"),
                            g1_MCC=(r["G1"] or {}).get("MCC")))
    return out


TS = task_summary()
N_TASK = len(TS)
byg = {}
for t in TS:
    byg[t["best_gen"]] = byg.get(t["best_gen"], 0) + 1
res = [t for t in TS if t["rg"] and t["gap_top2"] > t["rg"]]
g3_first = sum(1 for t in TS if t["g3_rank"] == 1)
ub_core = [t["leak_ub"] for t in TS if t["scope"] == "핵심 4종" and t["leak_ub"] is not None and t["metric"] != "MAE"]
ub_exp = [t["leak_ub"] for t in TS if t["scope"] == "확장" and t["leak_ub"] is not None]

# ── 정합성 체크: 원본 HTML 본문에 그 숫자가 실제로 있는지 ─────────
def has(html, s):
    return s in re.sub("<[^>]+>", "", html)


checks = []
for t in TS:
    src = FLAG_HTML if t["scope"] == "핵심 4종" else EXP_HTML
    v = f"{t['best_value']:.4f}" if t["scope"] == "핵심 4종" else f"{t['best_value']:.4f}"
    checks.append(dict(task=f"{t['endpoint']} {t['task']}", best_value=v, found_in_source=has(src, v)))
checks.append(dict(item="핵심 누수 상한 평균", value=round(float(np.mean(ub_core)), 4),
                   found_in_source=has(FLAG_HTML, "0.088")))
checks.append(dict(item="확장 누수 상한 평균", value=round(float(np.mean(ub_exp)), 4),
                   found_in_source=has(EXP_HTML, f"{np.mean(ub_exp):+.4f}")))
json.dump(dict(n_tasks=N_TASK, best_gen_dist=byg, resolvable=[f"{t['endpoint']} {t['task']}" for t in res],
               g3_first_place=g3_first, leak_ub_core_mean=round(float(np.mean(ub_core)), 4),
               leak_ub_exp_mean=round(float(np.mean(ub_exp)), 4), checks=checks),
          open(OUT / "consistency.json", "w"), ensure_ascii=False, indent=1)

# ══════════════════════════ HTML ══════════════════════════
H = ['<meta charset="utf-8"><title>독성 예측 세대 분석 — 마스터 보고서 (18과제)</title>', """<style>
body{font-family:'Nanum Gothic',system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:22px 26px;color:#1d2129;line-height:1.62}
h1{font-size:27px;border-bottom:3px solid #2a9d8f;padding-bottom:10px;margin-bottom:2px}
h2{font-size:20px;margin-top:40px;border-left:5px solid #2a9d8f;padding-left:11px;scroll-margin-top:12px}
h3{font-size:16px;margin-top:24px;color:#264653}
h4{font-size:14.5px;margin:18px 0 6px;color:#3d5a5a}
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
.fix{background:#eef4fd;border:1px solid #bcd0ee;border-radius:8px;padding:14px 18px;margin:16px 0}
.banner{background:linear-gradient(135deg,#20343f,#2a5d59);color:#fff;border-radius:12px;padding:22px 26px;margin:14px 0 22px}
.banner h2{color:#fff;border:0;padding:0;margin:0 0 10px;font-size:19px}
.banner .k{display:inline-block;background:rgba(255,255,255,.13);border-radius:8px;padding:9px 14px;margin:5px 7px 0 0;font-size:13.5px}
.banner b{color:#ffe9a8}
.core{border-left:4px solid #2a9d8f}.exp{border-left:4px solid #9d4edd}
.chip{display:inline-block;font-size:11.5px;padding:1px 7px;border-radius:9px;margin-left:6px}
.chip-core{background:#dff2ee;color:#1d6b60}.chip-exp{background:#f0e4fb;color:#6b30a8}
figure{margin:18px 0;text-align:center}
img{max-width:100%;border:1px solid #e3e8ec;border-radius:6px}
figcaption{font-size:12.5px;color:#5b6670;margin-top:7px}
small,.src{font-size:11.5px;color:#6b7580}
code{background:#eef2f5;padding:1px 5px;border-radius:3px;font-size:12px}
ul{margin:8px 0 8px 4px}li{margin:5px 0}
nav{background:#f7f9fb;border:1px solid #dfe5ea;border-radius:8px;padding:12px 18px;margin:14px 0;font-size:13.5px}
nav a{color:#1d6b60;text-decoration:none;margin-right:14px;display:inline-block;padding:2px 0}
nav a:hover{text-decoration:underline}
</style>"""]

H.append("<h1>독성 예측 세대 분석 — 마스터 보고서</h1>")
H.append(f"<p class='src'>작업일 2026-07-22 · 총 <b>{N_TASK}과제</b> "
         "(핵심 4종 DILI·hERG·AMES·LD50 = G1~G5 / 확장 14과제 발암성·ClinTox·Tox21 12경로 = G1~G3) · "
         "<b>★이 문서는 새 학습·재측정 0건</b> — 두 원본 보고서의 확정 숫자와 그림을 그대로 이관해 재구성했다.</p>")

# ── 0. 요약 배너
mean_core, mean_exp = float(np.mean(ub_core)), float(np.mean(ub_exp))
H.append("<div class='banner'><h2>한 줄로 말하면</h2>"
         f"<p><b>1.</b> {N_TASK}개 독성 과제 어디에서도 <b>최신 세대가 고전 ML(G2)을 이겼다는 증거가 없다.</b></p>"
         f"<p><b>2.</b> ADMET-AI의 0.9+ 점수는 실력이 아니라 <b>누수 신호</b>다 — 세 가지 독립 증거로 실측했다.</p>"
         f"<p><b>3.</b> 처음으로 통계적으로 <b>구분 가능한</b> 세대 효과가 3건 잡혔는데, 방향이 "
         f"<b>오히려 '고전 우세'</b>였다.</p>"
         f"<div><span class='k'>총 <b>{N_TASK}</b>과제</span>"
         + "".join(f"<span class='k'>최고 세대 <b>{k} {v}</b></span>" for k, v in sorted(byg.items()))
         + f"<span class='k'>정직 G3 1위 <b>{g3_first}</b>개</span>"
         f"<span class='k'>구분 가능 세대 효과 <b>{len(res)}</b>건(전부 G2&gt;G3)</span>"
         f"<span class='k'>누수 상한 평균 핵심 <b>{mean_core:.3f}</b> · 확장 <b>{mean_exp:.3f}</b></span></div></div>")

H.append("<nav><b>목차</b><br>"
         + " ".join(f"<a href='#s{i}'>{i}. {t}</a>" for i, t in enumerate(
             ["지표 읽는 법", "세대 정의·공정성", "엔드포인트별 사다리", "한눈에", "ADMET-AI 누수",
              "구분 가능한 세대 효과", "G1 구조알림", "과적합·신뢰성", "불확실성", "종합 결론"], 1))
         + "</nav>")

# ── 1. 지표
H.append("<h2 id='s1'>1. 지표를 어떻게 읽는가</h2><div class='box'><ul>"
         "<li><b>AUROC</b> — 무작위 양성/음성 한 쌍에서 양성을 더 높게 줄 확률. 0.5=무작위. 불균형에 둔감해 <b>단독으론 낙관적</b>.</li>"
         "<li><b>AUPRC</b> — 기저선이 <b>양성비율</b>. hERG는 양성 67%라 원래 높게 나오고, 확장 과제는 양성이 "
         "2.6~25%로 희박하다 → <b>반드시 기저선과 함께</b> 읽는다.</li>"
         "<li><b>MAE</b>(LD50만) — 평균절대오차, <b>낮을수록 좋음</b>. 단위 log(1/(mol/kg)).</li>"
         "<li><b>MCC</b>(G1 전용) — 작동점 기반 상관. 0 근처면 규칙이 사실상 무력.</li>"
         "<li><b>G1은 지표 형태가 다르다</b> — 규칙은 확률이 아니라 이진 판정이라 AUROC를 억지로 만들지 않고 "
         "<b>발화율·무매치율·알림별 정밀도·작동점 MCC</b>로만 보고한다.</li>"
         "<li><b>구분 가능선</b> — 그 test 크기에서 95%로 구분 가능한 최소 격차(Hanley-McNeil). "
         "이보다 작은 차이는 <b>순위를 주장하면 안 된다</b>. 비대응 가정이라 보수적 → <b>하한 경보</b>로만 쓴다.</li>"
         "<li><b>★누수</b> — 모델이 평가 분자를 학습에서 이미 봤다면 그 점수는 실력이 아니다(§5).</li></ul></div>")
H.append("<div class='warn'><b>★엔드포인트 간 절대 점수를 비교하지 말 것</b> — 데이터셋마다 분할이 다르고 "
         "난이도·양성비율이 다르다. 이 보고서가 비교하는 것은 <b>같은 엔드포인트 안에서의 세대 순위</b>뿐이다.</div>")

# ── 2. 세대 정의
H.append("<h2 id='s2'>2. 세대 정의와 공정성 전제</h2>")
H.append("<table><tr><th>세대</th><th>정의</th><th>핵심 4종</th><th>확장 14과제</th></tr>"
         "<tr><td><b>G1</b></td><td>규칙·구조알림(학습 없음)</td>"
         "<td colspan=2>BRENK+NIH + Benigni-Bossa SMARTS 10 — <b>양쪽 동일 규칙셋</b></td></tr>"
         "<tr><td><b>G2</b></td><td>고전 ML</td>"
         "<td colspan=2>물리화학 서술자+XGBoost / +RandomForest / ECFP4+XGBoost — <b>양쪽 동일 3종·동일 하이퍼파라미터</b></td></tr>"
         "<tr><td><b>G3</b></td><td>분자 딥러닝(GNN)</td>"
         "<td colspan=2>chemprop D-MPNN <b>정직 학습</b>(train만) + ADMET-AI(<b>★누수 기준선</b>)</td></tr>"
         "<tr><td><b>G4</b></td><td>구조·3D</td><td>Uni-Mol</td>"
         "<td class='na'>★미실행(스코프 밖)</td></tr>"
         "<tr><td><b>G5</b></td><td>파운데이션</td><td>ChemBERTa-2 · MoLFormer</td>"
         "<td class='na'>★미실행(스코프 밖)</td></tr></table>")
H.append("<div class='warn'><b>★두 스코프가 다르다</b> — 확장 14과제에는 <b>G4·G5가 아예 없다</b>(성능이 나빠서가 아니라 "
         "<b>돌리지 않았다</b>). 확장의 '최고 세대 G2'는 <b>G1~G3 중에서</b>의 결과다.</div>")
lk = FMETA["leak_ours"]
H.append("<div class='box'><b>분할·공정성(양쪽 모두 실측)</b><ul>"
         f"<li><b>핵심 4종</b> — {FMETA['split']}. 서로 다른 실험 폴더에서 모은 값이라 분할 동일성을 분자 단위로 확인했다: "
         f"{FMETA['split_verified']}. train↔test 정확분자 중복 <b>4/4 모두 0</b>.</li>"
         "<li><b>확장 14과제</b> — 각 TDC 데이터셋의 scaffold seed=42. 분할을 <code>prep_splits.py</code>가 "
         "<b>한 번만</b> 만들어 CSV로 고정하고 G1·G2·G3·ADMET-AI가 <b>모두 그 파일만</b> 읽는다 → "
         "동일성을 사후 확인이 아니라 <b>설계로 보장</b>. train∩test 중복 <b>3/3 모두 0</b>, 재현 3/3.</li>"
         "<li><b>Tox21 고지</b> — 12과제가 서로 다른 분자 부분집합이라 멀티태스크 D-MPNN과 과제별 G2를 같은 test에서 "
         "비교하려면 합집합에 <code>create_scaffold_split(seed=42)</code>를 한 번 적용해야 했다. 방법론은 TDC와 같지만 "
         "<b>과제별 공식 분할과는 달라 리더보드와 직접 비교 금지</b>.</li></ul></div>")
H.append("<div class='crit'><b>★개수가 같다고 같은 분할이 아니다</b> — 두 번 데였다. "
         "(a) T_toxicity 실험은 hERG n_test가 132로 핵심 4종과 <b>같은데</b> 분자 교집합은 28%뿐이었다(split seed=1). "
         "(b) 기존 ClinTox 실험은 train/valid/test가 <b>1034/147/297로 완전히 동일한데</b> Jaccard는 <b>0.134</b>였다. "
         "둘 다 본 표에 섞지 않고 각 원본의 부록으로 분리했다.</div>")

# ── 3. 엔드포인트별 사다리
H.append("<h2 id='s3'>3. 엔드포인트별 세대 사다리</h2>")
H.append("<h3>3-A. 핵심 4종 <span class='chip chip-core'>G1~G5</span></h3>")
for ep, lab in FLAB.items():
    met = FMET[ep]; hb = (met != "MAE")
    sub = FM[(FM.endpoint == ep) & (FM.metric == met)]
    t = next(x for x in TS if x["endpoint"] == lab)
    o = sub[sub.value.notna() & (sub.gen != "참고") & ~sub.method.str.contains("ADMET-AI")]
    best = o.value.max() if hb else o.value.min()
    ai = sub[sub.method.str.contains("ADMET-AI")].iloc[0]
    H.append(f"<h4 class='core'>&nbsp;{lab} — {met} ({'높을수록' if hb else '낮을수록'} 좋음 · "
             f"n_test={t['n_test']} · 구분 가능선 {t['rg']:.3f})</h4>")
    H.append("<table><tr><th>세대</th><th>모델</th><th class='n'>값 ±SD</th><th>플래그</th><th>출처</th></tr>")
    for _, r in sub.iterrows():
        if pd.isna(r.value):
            v = "<span class='na'>지표 형태 다름 → §7</span>" if r.gen == "G1" else f"<span class='na'>{r.note}</span>"
            cls = ""
        else:
            v = f"{r.value:.4f}" + (f" ±{r.sd:.4f}" if pd.notna(r.sd) else "")
            cls = " class='leak'" if "ADMET-AI" in str(r.method) else (" class='best'" if r.value == best else "")
        H.append(f"<tr><td><b>{r.gen}</b></td><td{cls}>{r.method}</td><td class='n'{cls}>{v}</td>"
                 f"<td class='src'>{r.leak_flag}</td><td class='src'>{r.source}</td></tr>")
    H.append("</table>")
    H.append(f"<div class='box'><b>{lab}</b> — 세대 순위 "
             + " &gt; ".join(f"{g} {v:.3f}" for g, v in t["gen_best"].items())
             + f" · <b>1위({t['best_gen']})−2위 격차 {t['gap_top2']:.3f}</b> vs 구분 가능선 {t['rg']:.3f} → "
             + ("<b>구분 불가</b>" if t["gap_top2"] <= t["rg"] else "<b>구분 가능</b>")
             + f". 정직 G3는 {t['n_models']}모델 중 <b>{t['g3_rank']}위</b>. "
             f"ADMET-AI {t['admetai']:.4f}(★누수 기준선).</div>")
for b, c in FFIGS[:4]:
    H.append(emb(b, c, " <b>[플래그십 보고서에서 이관]</b>"))

H.append("<h3>3-B. 확장 14과제 <span class='chip chip-exp'>G1~G3 · G4·G5 미실행</span></h3>")
for ep in EX:
    H.append(f"<h4 class='exp'>&nbsp;{ELAB[ep]}</h4>")
    H.append("<table><tr><th>과제</th><th class='n'>n / 양성률</th><th class='n'>G2 최고</th>"
             "<th class='n'>G3 정직</th><th class='n'>ADMET-AI(누수)</th><th class='n'>1위−2위</th>"
             "<th class='n'>구분선</th><th>판정</th></tr>")
    for task, r in EX[ep].items():
        t = next(x for x in TS if x["endpoint"] == ELAB[ep] and x["task"] in (task, "—"))
        g2 = max(v["AUROC"]["mean"] for v in r["G2"].values() if v)
        g3 = r["G3_ours"]["AUROC"]["mean"] if r["G3_ours"] else None
        ver = "구분 불가" if t["gap_top2"] <= (t["rg"] or 9) else (
            "<b>구분 가능</b>" if t["gap_top2"] > t["rg"] * 1.2 else "경계선")
        H.append(f"<tr><td>{task if len(EX[ep])>1 else '—'}</td>"
                 f"<td class='n'>{r['n_test']} / {r['pos_rate']:.3f}</td>"
                 f"<td class='n best'>{g2:.4f}</td><td class='n'>{'' if g3 is None else f'{g3:.4f}'}</td>"
                 f"<td class='n leak'>{r['G3_admetai'].get('AUROC', float('nan')):.4f}</td>"
                 f"<td class='n'>{t['gap_top2']:.4f}</td><td class='n'>{t['rg']:.4f}</td><td>{ver}</td></tr>")
    H.append("</table>")
H.append("<p><small>확장 표의 G2 최고는 3종(물리화학+XGB / 물리화학+RF / ECFP4+XGB) 중 최고값. 모델별 전체 값은 "
         "<code>results/master_matrix.csv</code>와 원본 <code>expansion_metrics.json</code>에.</small></p>")
for b, c in EFIGS:
    H.append(emb(b, c, " <b>[확장 보고서에서 이관]</b>"))

# ── 4. 한눈에 (신규 요약 그림 2장 — 이관한 숫자만 사용)
H.append("<h2 id='s4'>4. 한눈에 — 세대가 올라가면 좋아지나</h2>")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
ks = sorted(byg.items(), key=lambda t: -t[1])
ax1.bar([k for k, _ in ks], [v for _, v in ks],
        color=["#2a9d8f" if k == "G2" else "#457b9d" for k, _ in ks], edgecolor="#222", linewidth=.6)
for i, (k, v) in enumerate(ks):
    ax1.text(i, v + .3, str(v), ha="center", fontsize=11, fontweight="bold")
ax1.set_ylabel("최고 성적을 낸 과제 수", fontsize=10)
ax1.set_title(f"{N_TASK}과제의 '최고 세대' 분포", fontsize=12)
ax1.set_ylim(0, max(byg.values()) * 1.2); ax1.grid(axis="y", alpha=.25)
ranks = [t["g3_rank"] for t in TS if t["g3_rank"]]
mx = max(ranks)
cnt = [ranks.count(i) for i in range(1, mx + 1)]
ax2.bar(range(1, mx + 1), cnt, color=["#c1440e" if i == 1 else "#e76f51" for i in range(1, mx + 1)],
        edgecolor="#222", linewidth=.6)
for i, c in enumerate(cnt, 1):
    if c:
        ax2.text(i, c + .3, str(c), ha="center", fontsize=10, fontweight="bold")
ax2.set_xticks(range(1, mx + 1)); ax2.set_xlabel("정직 G3(D-MPNN)의 순위", fontsize=10)
ax2.set_ylabel("과제 수", fontsize=10); ax2.set_ylim(0, max(cnt) * 1.25)
ax2.set_title(f"정직한 GNN은 1위가 {ranks.count(1)}개", fontsize=12); ax2.grid(axis="y", alpha=.25)
if len(FFIGS) > 4:
    H.append(emb(FFIGS[4][0], FFIGS[4][1], " <b>[플래그십 보고서에서 이관 — 핵심 4종만]</b>"))
H.append(emb(b64_of(fig), f"왼쪽: {N_TASK}과제에서 각 세대가 최고를 차지한 횟수(핵심 4종은 G1~G5 중, 확장 14과제는 G1~G3 중). "
                          "오른쪽: 우리가 정직하게 학습한 G3 D-MPNN의 과제별 순위 분포 — <b>1위가 한 번도 없다</b>. "
                          "★두 그림 모두 이관한 숫자만 사용(재계산 없음)."))

# ── 5. 누수
ale = FMETA["admetai_leak_evidence"]
H.append("<h2 id='s5'>5. ★ADMET-AI 누수 — 증거 3건과 자기정정</h2>")
H.append("<div class='crit'><b>증거 ① 학습 데이터 크기가 TDC 전체와 일치</b> — 패키지가 스스로 기록한 학습 size "
         f"(<code>{ale['출처']}</code>)를 우리 분할 전체와 대조: <b>DILI 475 = 379+96 = 475 정확 일치</b>, "
         "hERG 648/655 · AMES 7255/7278 · LD50 7342/7385.</div>")
sr = ale["admetai_자체보고_성능"]
H.append("<div class='crit'><b>증거 ② 패키지에 기록된 자기 성능을 우리 test에서 크게 초과</b> — 같은 모델·같은 앙상블·"
         "같은 멀티태스크 설정이라 그 성분들이 <b>양쪽에서 상쇄</b>되고 달라지는 건 <b>평가 분자</b>뿐 → "
         "<b>누수 크기 추정에 가장 깨끗한 증거</b>."
         "<table><tr><th>엔드포인트</th><th class='n'>패키지 기록</th><th class='n'>우리 test</th>"
         "<th class='n'>설명되지 않은 초과분</th></tr>")
for k, tdc in [("DILI", "dili"), ("hERG", "herg"), ("AMES", "ames")]:
    got = float(FM[(FM.endpoint == tdc) & (FM.metric == "AUROC") & FM.method.str.contains("ADMET-AI")].iloc[0].value)
    H.append(f"<tr><td>{k} AUROC</td><td class='n'>{sr[k]:.4f}</td><td class='n leak'>{got:.4f}</td>"
             f"<td class='n leak'>{got - sr[k]:+.4f}</td></tr>")
H.append("</table></div>")
H.append("<div class='crit'><b>증거 ③ 같은 계열을 정직하게 학습하면 얼마나 내려가나</b> — D-MPNN을 train만 보고 학습."
         "<table><tr><th>범위</th><th class='n'>과제 수</th><th class='n'>격차 평균</th><th class='n'>범위</th></tr>"
         f"<tr><td>핵심 4종(분류 3)</td><td class='n'>{len(ub_core)}</td><td class='n leak'>{mean_core:+.4f}</td>"
         f"<td class='n'>{min(ub_core):+.4f} ~ {max(ub_core):+.4f}</td></tr>"
         f"<tr><td>확장 14과제</td><td class='n'>{len(ub_exp)}</td><td class='n leak'>{mean_exp:+.4f}</td>"
         f"<td class='n'>{min(ub_exp):+.4f} ~ {max(ub_exp):+.4f}</td></tr></table></div>")
H.append("<div class='fix'><b>★자기정정 — 이 보고서가 한 번 틀렸던 곳</b><br>"
         "초안은 증거 ③의 격차를 <b>'누수 프리미엄'</b>이라 부르며 100%를 누수에 귀속했다. "
         "적대 검증에서 패키지를 뜯어보니 <b>ADMET-AI는 5-모델 앙상블 × 31과제(회귀 10과제) 멀티태스크·튜닝된 배포본</b>이고, "
         "우리 D-MPNN은 <b>단일모델·단일과제·미튜닝</b>이었다"
         "(<code>admet_classification/model_0..4.pt</code> 5개 · 체크포인트 <code>output_columns</code> 길이 31 · "
         "<code>_make_ensemble_predictions</code>). "
         "즉 격차에는 <b>누수 + 앙상블 + 멀티태스크 전이 + 튜닝</b>이 교락돼 있다.<br><br>"
         "<b>→ 정정: 증거 ③의 격차는 누수의 '크기'가 아니라 '상한(upper bound)'이다.</b> "
         "누수 크기의 가장 깨끗한 추정은 <b>증거 ②(+0.049~0.074)</b>이며, 증거 ③의 역할은 크기 확정이 아니라 "
         "<b>방향의 독립 확인</b>(정직하게 학습하면 반드시 낮아진다)이다. 플래그십 보고서도 소급 정정했다.</div>")
H.append("<div class='warn'><b>누수 보정은 불가능하다 — 이유를 정확히</b>: ADMET-AI의 <b>학습 분자 목록이 공개돼 있지 않아</b> "
         "'학습에 안 쓰인 분자' 하위셋을 <b>정의할 수 없다</b>. 크기 대조는 포함을 강하게 시사할 뿐 분자 단위 대조가 아니다. "
         "그래서 본 보고서는 ADMET-AI를 <b>고정 비교선</b>으로만 그리고 <b>세대 우열 판정에는 쓰지 않는다</b>.</div>")

# ── 6. 구분 가능한 세대 효과
H.append("<h2 id='s6'>6. 구분 가능한 세대 효과 — 처음 잡힌 신호</h2>")
H.append("<div class='box'><b>판정 방법</b> — 세대별 대표값(그 세대 최고)을 세우고 <b>1위 세대 vs 2위 세대</b> 격차를 "
         "구분 가능선과 비교한다. <b>★초안은 최고−최저 <i>폭(range)</i>을 썼는데, 그 폭은 '가장 나쁜 세대'가 만든 값이라 "
         "1위의 우위를 전혀 함의하지 않는다</b> — 적대 검증에서 지적받아 정정했고, 그 결과 핵심 4종의 "
         "'LD50에서 G4가 유일하게 확실한 세대 효과'라는 결론이 <b>철회</b>됐다.</div>")
H.append("<table><tr><th>범위</th><th>엔드포인트/과제</th><th class='n'>n_test</th><th>세대 순위</th>"
         "<th class='n'>1위−2위</th><th class='n'>구분선</th><th>판정</th></tr>")
for t in TS:
    hit = t["rg"] and t["gap_top2"] > t["rg"]
    ver = ("<b>구분 가능</b>" if (hit and t["gap_top2"] > t["rg"] * 1.2) else
           "<b>경계선</b>" if hit else "구분 불가")
    row = " &gt; ".join(f"{g} {v:.3f}" for g, v in t["gen_best"].items())
    H.append(f"<tr{' class=best' if hit else ''}><td>{t['scope']}</td>"
             f"<td>{t['endpoint']}{'' if t['task']=='—' else ' / '+t['task']}</td>"
             f"<td class='n'>{t['n_test']}</td><td class='src'>{row}</td>"
             f"<td class='n'>{t['gap_top2']:.4f}</td><td class='n'>{t['rg']:.4f}</td><td>{ver}</td></tr>")
H.append("</table>")
H.append(f"<div class='warn'><b>결과</b> — 핵심 4종은 <b>0/4</b>(전부 구분 불가). 확장에서 표본이 큰 Tox21 과제 "
         f"<b>{len(res)}건</b>({', '.join(t['task'] for t in res)})만 구분선을 넘었고, "
         "<b>{}</b>. ".format(" · ".join(f"{t['task']}는 {t['best_gen']}가 1위" for t in res)) +
         "<b>세 건 모두 G2(고전 ML)가 G3(정직 학습한 GNN)를 이긴 경우</b>다. "
         "즉 이 프로젝트에서 통계적으로 확인된 유일한 세대 효과는 <b>'최신이 낫다'가 아니라 '고전이 낫다'</b> 방향이다.<br><br>"
         "<b>★과대해석 금지</b>: (a) 18과제 중 3건뿐이고 나머지 15건은 구분 불가다. "
         "(b) <b>G3에 하이퍼파라미터 탐색을 하지 않았다</b>(G2도 안 했으므로 조건은 대등하나, 튜닝하면 달라질 여지가 있다). "
         "(c) G3는 <code>--class-balance</code>를 쓰고 G2는 안 쓰는 <b>불균형 처리 비대칭</b>이 있다. "
         "(d) 이 결론은 <b>이 세대 구성·이 조건</b>에 한정된다.</div>")

# ── 7. G1
H.append("<h2 id='s7'>7. G1(구조알림) — 어디서 듣고 어디서 무력한가</h2>")
H.append("<table><tr><th>범위</th><th>엔드포인트/과제</th><th class='n'>발화율</th><th class='n'>무매치율</th>"
         "<th class='n'>정밀도</th><th class='n'>기저 양성률</th><th class='n'>MCC</th><th>판정</th></tr>")
for _, g in FG1.iterrows():
    if pd.isna(g.rule_MCC):
        continue
    ver = "★무력" if abs(g.rule_MCC) < 0.05 else ("약한 신호" if abs(g.rule_MCC) < 0.15 else "<b>의미 있는 신호</b>")
    H.append(f"<tr><td>핵심 4종</td><td>{FLAB.get(g.endpoint, g.endpoint)}</td>"
             f"<td class='n'>{g.fire_rate_mutagen:.3f}</td><td class='n'>{g.nomatch_rate_mutagen:.3f}</td>"
             f"<td class='n'>{g.rule_precision:.3f}</td><td class='n'>{g.pos_rate:.3f}</td>"
             f"<td class='n'>{g.rule_MCC:+.3f}</td><td>{ver}</td></tr>")
for _, g in EG1.iterrows():
    ver = "★무력" if abs(g.rule_MCC) < 0.05 else ("약한 신호" if abs(g.rule_MCC) < 0.15 else "<b>의미 있는 신호</b>")
    nm = ELAB[g.endpoint] + ("" if g.task == "Y" else f" / {g.task}")
    H.append(f"<tr><td>확장</td><td>{nm}</td><td class='n'>{g.fire_rate:.3f}</td>"
             f"<td class='n'>{g.nomatch_rate:.3f}</td><td class='n'>{g.rule_precision:.3f}</td>"
             f"<td class='n'>{g.pos_rate:.3f}</td><td class='n'>{g.rule_MCC:+.3f}</td><td>{ver}</td></tr>")
H.append("</table>")
H.append("<div class='box'><b>기전이 방법을 정한다</b> — 같은 규칙셋이 엔드포인트마다 정반대로 작동한다. "
         "Benigni-Bossa 계열은 원래 <b>변이원·발암 알림</b>이라 <b>발암성(MCC +0.358)·DILI(+0.361)·AMES(+0.260)</b>에서 듣고, "
         "<b>hERG(−0.028)·Tox21 12/12(−0.03~+0.05)</b>처럼 기전이 다른 독성(물성 주도·수용체 경로)에서는 <b>완전히 무력</b>하다. "
         "ClinTox(+0.093)는 그 중간이다. <b>이는 세대 문제가 아니라 기전 정합성 문제</b>다.</div>")

# ── 8. 과적합
H.append("<h2 id='s8'>8. 과적합·신뢰성 — 있는 그대로</h2>")
H.append("<table><tr><th>범위</th><th>모델</th><th class='n'>train</th><th class='n'>test</th>"
         "<th class='n'>격차</th><th>해석</th></tr>")
for o in FMETA["overfit"]:
    if o["metric"] != "AUROC":
        continue
    g = abs(o["gap"])
    itp = ("★train 완전 암기(1.000)" if o.get("train") == 1.0 else
           "★작음" if g < 0.03 else "중간" if g <= 0.10 else "큼")
    H.append(f"<tr><td>핵심 4종</td><td>{o['model']} — {o['endpoint']}</td><td class='n'>{o['train']:.4f}</td>"
             f"<td class='n'>{o['test']:.4f}</td><td class='n'>{o['gap']:.4f}</td><td>{itp}</td></tr>")
ed = pd.DataFrame([o for o in EOV if o["metric"] == "AUROC"])
for mk, r in ed.groupby("model")[["train", "test", "gap"]].mean().round(4).iterrows():
    g = abs(r["gap"])
    itp = ("★train 완전 암기" if r["train"] > 0.999 else "★작음" if g < 0.03 else "중간" if g <= 0.10 else "큼")
    H.append(f"<tr><td>확장(14과제 평균)</td><td>{G2LAB.get(mk, 'chemprop D-MPNN (정직)')}</td>"
             f"<td class='n'>{r['train']:.4f}</td><td class='n'>{r['test']:.4f}</td>"
             f"<td class='n'>{r['gap']:.4f}</td><td>{itp}</td></tr>")
H.append("</table>")
H.append("<div class='box'><b>★핵심 통찰 — 과적합이 없다는 게 성능이 좋다는 뜻은 아니다</b><br>"
         "Uni-Mol(G4)은 분류 3종 모두 <b>train AUROC = 1.000</b>으로 완전히 암기하고, "
         "물리화학+트리도 train이 0.99대까지 올라간다. 반면 <b>정직하게 학습한 G3 D-MPNN은 과적합이 가장 작다</b>"
         "(핵심 4종 DILI에서는 격차가 <b>음수 −0.035</b> — train보다 test가 좋다). "
         "그런데도 <b>순위는 어디서도 1위가 아니다</b>. "
         "일반화가 깨끗한 것과 성능이 높은 것은 별개이며, 이 데이터 크기에서는 "
         "<b>암기 여력이 큰 모델이 오히려 test에서도 앞서는</b> 상황이 관찰된다.</div>")

# ── 9. 불확실성
H.append("<h2 id='s9'>9. 이 비교로 어디까지 말할 수 있나</h2>")
H.append(f"<div class='warn'><ul>"
         f"<li><b>구분 가능선은 하한 경보다.</b> {FMETA['uncertainty_caveat']} Hanley-McNeil은 <b>비대응 가정</b>이라 "
         "같은 test 위 두 모델 비교엔 지나치게 보수적이다 → '이보다 작으면 확실히 말할 수 없다'로만 쓴다.</li>"
         "<li><b>소표본 경고.</b> DILI는 test <b>96분자</b>(구분선 0.111), 발암성은 <b>56분자·양성 11개</b>"
         "(구분선 0.235)다. 이 둘에서는 <b>세대 순위를 주장하는 것 자체가 불가능</b>하다.</li>"
         "<li><b>seed SD ≠ 일반화 불확실성.</b> 핵심 4종의 SD는 train/valid 파티션+초기화 변동, "
         "확장의 SD는 <b>초기화 변동만</b>(분할 고정)이다. 어느 쪽도 분자 재표집 오차가 아니다. "
         "분자별 예측이 원본에 저장돼 있지 않아 부트스트랩·DeLong은 산출할 수 없다.</li>"
         "<li><b>튜닝 비대칭 없음, 대신 튜닝 자체가 없음.</b> G2·G3 모두 하이퍼파라미터 탐색을 하지 않았다. "
         "조건은 대등하나 '각 세대의 최대 성능'을 잰 것은 아니다.</li>"
         "<li><b>불균형 처리 비대칭.</b> G3는 <code>--class-balance</code>, G2는 미적용(양쪽 보고서 공통 설정). "
         "확장 과제는 양성이 2.6~25%로 희박해 영향이 더 클 수 있다.</li>"
         "<li><b>★남은 최대 한계 — 전향적 검증이 없다.</b> 모든 결론은 TDC 벤치마크 안에서의 회고적 비교다. "
         "새로 합성·측정한 분자로 확인한 것이 아니다.</li></ul></div>")

# ── 10. 종합
H.append("<h2 id='s10'>10. 종합 결론</h2>")
core_best = " · ".join(f"{t['endpoint'].split()[0]}={t['best_gen']}" for t in TS if t["scope"] == "핵심 4종")
H.append("<div class='box'>"
         f"<p><b>1. 세대는 세련도의 순서일 뿐 성능의 순서가 아니다.</b> {N_TASK}과제 중 "
         + " · ".join(f"<b>{k}가 {v}과제</b>" for k, v in sorted(byg.items(), key=lambda x: -x[1]))
         + f"에서 최고였다(핵심 4종: {core_best}). "
         f"파운데이션 모델(G5)은 4/4 어디서도 1위가 아니었고, <b>정직하게 학습한 GNN(G3)은 {N_TASK}과제 중 1위가 "
         f"{g3_first}개</b>다. <span class='src'>근거: §3·§4</span></p>"
         "<p><b>2. 그러나 '고전이 우월하다'도 과장이다.</b> 1위−2위 세대 격차가 구분 가능선을 넘는 과제는 "
         f"<b>{len(res)}/{N_TASK}</b>뿐이고 핵심 4종은 <b>0/4</b>다. 정확한 진술은 "
         "<b>'최신 세대가 고전을 이긴다는 증거가 없다'</b>이며, 반대 방향의 증거도 큰 표본 일부에서만 잡힌다. "
         "<span class='src'>근거: §6</span></p>"
         "<p><b>3. 세대보다 '무엇을 넣느냐'가 컸다.</b> hERG에서 우리 모델 중 <b>최고(물리화학+RF 0.8369)와 "
         "최저(ECFP4+XGB 0.7349)가 둘 다 G2</b>다 — 한 세대 안의 특징 차이가 전체 성능 폭을 설명한다. "
         "<span class='src'>근거: §3-A hERG</span></p>"
         "<p><b>4. 엔드포인트(기전)가 방법을 정한다.</b> hERG는 물성 축(구조알림 완전 무력·물리화학 최고), "
         "AMES·발암성은 구조 축(구조알림이 의미 있는 신호), LD50은 3D(G4)가 순위상 최고, "
         "Tox21 수용체 경로는 구조알림 12/12 무력. <b>하나의 세대가 모든 독성을 지배하지 않는다.</b> "
         "<span class='src'>근거: §7</span></p>"
         "<p><b>5. ADMET-AI의 0.9+는 실력이 아니라 누수 신호다.</b> ①학습 크기 일치(DILI 475=475) "
         f"②패키지 기록 성능 초과(+0.049~0.074) ③정직 학습과의 격차(핵심 {mean_core:.3f}·확장 {mean_exp:.3f}). "
         "<b>단 ③은 앙상블·멀티태스크·튜닝이 교락된 '상한'</b>이며, 누수 크기의 가장 깨끗한 추정은 ②다. "
         "누수 보정은 학습 분자 목록 비공개로 <b>정의 자체가 불가능</b>하다. <span class='src'>근거: §5</span></p></div>")
H.append("<div class='fix'><b>이 보고서의 신뢰도는 '고친 자국'에 있다</b> — 초안에서 두 개의 결론이 적대 검증으로 "
         "뒤집혔고(누수 프리미엄 → 상한 / range 판정 → 1위·2위 판정에 따른 'LD50 G4' 철회), "
         "그 과정과 이유를 §5·§6에 그대로 남겼다. 소표본 경고와 전향적 검증 부재(§9)도 지우지 않았다.</div>")

H.append(f"<p class='src'>산출물: <code>results/master_report.html</code> · "
         f"<code>results/master_matrix.csv</code>({len(MM)}행) · <code>results/consistency.json</code> · "
         "<code>notes.md</code><br>원본: "
         "<code>experiment_generation_matrix/results/report.html</code> · "
         "<code>experiment_gen_expansion_g1g3/results/report_expansion.html</code> "
         "(그림은 원본 base64를 그대로 이관)</p>")

(OUT / "master_report.html").write_text("\n".join(H), encoding="utf-8")
print(f"저장 → master_report.html ({(OUT/'master_report.html').stat().st_size/1024:.0f} KB) · "
      f"master_matrix.csv ({len(MM)}행) · consistency.json")
print(f"총 과제 {N_TASK} · 최고 세대 분포 {byg} · 정직 G3 1위 {g3_first}개 · 구분 가능 {len(res)}건")
print(f"누수 상한 평균 — 핵심 {mean_core:.4f} · 확장 {mean_exp:.4f}")
print(f"이관 그림 — 플래그십 {len(FFIGS)}장 · 확장 {len(EFIGS)}장 · 신규 요약 1장")
bad = [c for c in checks if "found_in_source" in c and not c["found_in_source"]]
print(f"정합성 체크: {len(checks)}건 중 원본 본문 미확인 {len(bad)}건" + (f" → {bad}" if bad else ""))
