#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_report.py — progress.jsonl 을 읽어 3축 표·CSV·HTML 생성. ★부분 결과에도 동작(밤샘 중간 리포트).
산출: results/adme_matrix.csv · feature_ablation.csv · learning_axis.csv
      results/partial_report.html (또는 report_adme_full.html)
"""
import argparse, base64, io, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import common as C
from common import EPS

ap = argparse.ArgumentParser(); ap.add_argument("--partial", action="store_true")
A = ap.parse_args()
OUTNAME = "partial_report.html" if A.partial else "report_adme_full.html"

rows = []
if os.path.exists(C.PROGRESS):
    for line in open(C.PROGRESS):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
P = pd.DataFrame(rows)
fails = []
if os.path.exists(C.FAILURES):
    for line in open(C.FAILURES):
        try:
            fails.append(json.loads(line))
        except Exception:
            pass
F = pd.DataFrame(fails)

PILLAR = {"A": "흡수", "D": "분포", "M": "대사", "E": "배설"}
GENLAB = {"G2": "고전 ML", "G3": "GNN", "G4": "3D", "G5": "파운데이션"}


def prim(ep):
    return EPS[ep]["primary"]


def agg_axis1():
    """엔드포인트 × 모델 → 주지표 평균±SD."""
    if P.empty or "axis" not in P.columns:
        return pd.DataFrame()
    d = P[(P.axis == "1") & P.metrics.notna()].copy()
    if d.empty:
        return pd.DataFrame()
    out = []
    for (ep, model), g in d.groupby(["endpoint", "model"]):
        m = prim(ep)
        vals = [x.get(m) for x in g.metrics if isinstance(x, dict) and x.get(m) is not None]
        if not vals:
            continue
        gen = g.iloc[0].get("gen", "")
        sec = [x.get("AUPRC" if EPS[ep]["task"] == "cls" else "MAE") for x in g.metrics
               if isinstance(x, dict)]
        sec = [v for v in sec if v is not None]
        out.append(dict(endpoint=ep, label=EPS[ep]["label"], pillar=EPS[ep]["pillar"],
                        task=EPS[ep]["task"], metric=m, gen=gen, model=model,
                        value=round(float(np.mean(vals)), 4), sd=round(float(np.std(vals)), 4),
                        secondary=(round(float(np.mean(sec)), 4) if sec else None),
                        n_seed=len(vals),
                        leak_flag=("★누수 의심(TDC 전체 사전학습)" if model == "admetai" else "누수 0"),
                        source="results/progress.jsonl"))
    return pd.DataFrame(out)


def agg_axis2():
    if P.empty or "axis" not in P.columns:
        return pd.DataFrame()
    d = P[(P.axis == "2") & P.metrics.notna()].copy()
    if d.empty:
        return pd.DataFrame()
    out = []
    for (ep, stack), g in d.groupby(["endpoint", "stack"]):
        m = prim(ep)
        vals = [x.get(m) for x in g.metrics if isinstance(x, dict) and x.get(m) is not None]
        if not vals:
            continue
        out.append(dict(endpoint=ep, label=EPS[ep]["label"], pillar=EPS[ep]["pillar"],
                        task=EPS[ep]["task"], metric=m, stack=stack,
                        value=round(float(np.mean(vals)), 4), sd=round(float(np.std(vals)), 4),
                        n_seed=len(vals), source="results/progress.jsonl"))
    df = pd.DataFrame(out)
    order = {s: i for i, (s, _) in enumerate(C.STACKS)}
    df["ord"] = df["stack"].map(order)
    df = df.sort_values(["endpoint", "ord"])
    # 증분(직전 스택 대비)
    inc = []
    for ep, g in df.groupby("endpoint"):
        g = g.sort_values("ord"); prev = None; hb = (EPS[ep]["primary"] != "MAE")
        for _, r in g.iterrows():
            delta = None if prev is None else round(float((r.value - prev) if hb else (prev - r.value)), 4)
            inc.append(delta); prev = r.value
    df["delta_vs_prev"] = inc
    return df.drop(columns=["ord"])


def agg_axis3():
    if P.empty or "axis" not in P.columns:
        return pd.DataFrame()
    d = P[(P.axis == "3")].copy()
    if d.empty:
        return pd.DataFrame()
    out = []
    for (grp,), g in d.groupby(["group"]):
        per = {}
        for rec in g.per_endpoint:
            if not isinstance(rec, dict):
                continue
            for ep, m in rec.items():
                per.setdefault(ep, []).append(m.get(prim(ep)))
        for ep, vals in per.items():
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            out.append(dict(group=grp, endpoint=ep, label=EPS[ep]["label"], metric=prim(ep),
                            multitask=round(float(np.mean(vals)), 4),
                            multitask_sd=round(float(np.std(vals)), 4), n_seed=len(vals)))
    df = pd.DataFrame(out)
    if df.empty:
        return df
    single = agg_axis1()
    if not single.empty:
        s = single[single.model == "dmpnn_ours"][["endpoint", "value"]].rename(
            columns={"value": "single_dmpnn"})
        df = df.merge(s, on="endpoint", how="left")
        df["delta_multitask_minus_single"] = (df.multitask - df.single_dmpnn).round(4)
    return df


M1, M2, M3 = agg_axis1(), agg_axis2(), agg_axis3()
for df, nm in ((M1, "adme_matrix.csv"), (M2, "feature_ablation.csv"), (M3, "learning_axis.csv")):
    if not df.empty:
        df.to_csv(f"{C.RES}/{nm}", index=False)

# ── HTML ──
H = ['<meta charset="utf-8"><title>ADME 전면 벤치마크 — 세대·특징·학습방식</title>', """<style>
body{font-family:'Nanum Gothic',system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:22px 26px;color:#1d2129;line-height:1.6}
h1{font-size:25px;border-bottom:3px solid #2a9d8f;padding-bottom:9px}
h2{font-size:19px;margin-top:34px;border-left:5px solid #2a9d8f;padding-left:11px}
h3{font-size:15.5px;margin-top:20px;color:#264653}
table{border-collapse:collapse;width:100%;margin:11px 0;font-size:12.5px}
th,td{border:1px solid #d8dde3;padding:4px 7px}th{background:#eef2f5}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.best{background:#e8f6f1;font-weight:700}.leak{background:#fff1ec;color:#c1440e;font-weight:700}
.pos{color:#1d6b60;font-weight:700}.neg{color:#b3261e;font-weight:700}.na{color:#98a2ab}
.box{background:#f7f9fb;border:1px solid #dfe5ea;border-radius:8px;padding:13px 17px;margin:14px 0}
.warn{background:#fff8ec;border:1px solid #f0d9a8;border-radius:8px;padding:13px 17px;margin:14px 0}
small,.src{font-size:11.5px;color:#6b7580}code{background:#eef2f5;padding:1px 4px;border-radius:3px;font-size:11.5px}
</style>"""]
H.append("<h1>ADME 전면 벤치마크 — 세대 × 특징 × 학습방식</h1>")
done_n = len(P); fail_n = len(F)
H.append(f"<p class='src'>작업일 2026-07-22-23 · 18개 ADME 엔드포인트 · TDC admet_group 공식 분할(=scaffold seed=42)·5 seed · "
         f"완료 조합 <b>{done_n}</b> · 실패 <b>{fail_n}</b>"
         + (" · <b>★중간 리포트(실행 중)</b>" if A.partial else "") + "</p>")

# 진행 현황
if not P.empty and "axis" in P.columns:
    st = []
    for ax, nm in (("1", "축① 세대"), ("2", "축② 특징"), ("3", "축③ 학습방식")):
        d = P[P.axis == ax]
        eps_done = d.endpoint.nunique() if "endpoint" in d.columns and not d.empty else 0
        st.append(f"<tr><td>{nm}</td><td class='n'>{len(d)}</td><td class='n'>{eps_done}</td></tr>")
    H.append("<div class='box'><b>진행 현황</b><table><tr><th>축</th><th class='n'>완료 조합</th>"
             "<th class='n'>엔드포인트 수</th></tr>" + "".join(st) + "</table></div>")

# 분할·누수
if os.path.exists(f"{C.RES}/split_leakage.json"):
    lk = json.load(open(f"{C.RES}/split_leakage.json"))
    nz = [e for e, v in lk.items() if v.get("exact_overlap_trainval_test", 0) > 0]
    nb = {e: v["n_bad_excluded"] for e, v in lk.items() if v.get("n_bad_excluded", 0) > 0}
    H.append(f"<div class='box'><b>분할·공정성(실측)</b> — 18개 엔드포인트 모두 TDC admet_group 공식 고정 test. "
             f"train↔test 정확분자 중복 &gt;0 인 엔드포인트: <b>{nz or '없음'}</b>. "
             f"chemprop 비호환으로 전 조합 공통 제외한 분자: <b>{nb or '없음'}</b>(분할 이후 제외). "
             f"<span class='src'>출처 results/split_leakage.json</span></div>")

# 축①
H.append("<h2>축① 세대 (G2~G5)</h2>")
H.append("<div class='warn'><b>G1(규칙)은 세대에서 제외</b> — 구조알림(toxicophore)은 독성 전용이라 "
         "Caco-2·청소율 등에 부적용이다. 대신 의약화학 규칙(GSE·CNS MPO·이온화)을 <b>축②의 특징</b>으로 넣었다.</div>")
if M1.empty:
    H.append("<p class='na'>아직 결과 없음</p>")
else:
    for pil in ("A", "D", "M", "E"):
        sub = M1[M1.pillar == pil]
        if sub.empty:
            continue
        H.append(f"<h3>{pil} {PILLAR[pil]}</h3>")
        H.append("<table><tr><th>엔드포인트</th><th>주지표</th><th>세대</th><th>모델</th>"
                 "<th class='n'>값 ±SD</th><th class='n'>보조지표</th><th>플래그</th></tr>")
        for ep, g in sub.groupby("endpoint"):
            hb = (EPS[ep]["primary"] != "MAE")
            ours = g[g.model != "admetai"]
            best = (ours.value.max() if hb else ours.value.min()) if not ours.empty else None
            for _, r in g.sort_values(["gen", "model"]).iterrows():
                cls = (" class='leak'" if r.model == "admetai"
                       else (" class='best'" if best is not None and r.value == best else ""))
                H.append(f"<tr><td>{r.label}<br><small>{ep}</small></td><td>{r.metric}</td>"
                         f"<td>{r.gen} <small>{GENLAB.get(r.gen,'')}</small></td><td{cls}>{r.model}</td>"
                         f"<td class='n'{cls}>{r.value:.4f} ±{r.sd:.4f}</td>"
                         f"<td class='n'>{'' if r.secondary is None else f'{r.secondary:.4f}'}</td>"
                         f"<td class='src'>{r.leak_flag}</td></tr>")
        H.append("</table>")
    # 세대 요약
    win = {}
    for ep, g in M1[M1.model != "admetai"].groupby("endpoint"):
        hb = (EPS[ep]["primary"] != "MAE")
        b = g.loc[g.value.idxmax() if hb else g.value.idxmin()]
        win[ep] = b.gen
    from collections import Counter
    cnt = Counter(win.values())
    H.append("<div class='box'><b>세대 요약(현재까지)</b> — 최고 세대 분포: "
             + " · ".join(f"<b>{k}</b> {v}개" for k, v in sorted(cnt.items()))
             + f" (집계 엔드포인트 {len(win)}개). "
             "<span class='src'>독성 18과제에서는 G2 17 · G4 1 이었다.</span></div>")

# 축②
H.append("<h2>축② 특징 ablation (G2 XGBoost · 누적 스택)</h2>")
if M2.empty:
    H.append("<p class='na'>아직 결과 없음</p>")
else:
    H.append("<table><tr><th>기둥</th><th>엔드포인트</th><th>주지표</th>"
             + "".join(f"<th class='n'>{s}</th>" for s, _ in C.STACKS) + "</tr>")
    for pil in ("A", "D", "M", "E"):
        for ep, g in M2[M2.pillar == pil].groupby("endpoint"):
            cells = []
            for s, _ in C.STACKS:
                r = g[g["stack"] == s]
                if r.empty:
                    cells.append("<td class='na'>—</td>"); continue
                r = r.iloc[0]
                d = r.delta_vs_prev
                dd = ("" if d is None or pd.isna(d) else
                      f"<br><small class='{'pos' if d>0 else 'neg'}'>{d:+.4f}</small>")
                cells.append(f"<td class='n'>{r.value:.4f}{dd}</td>")
            H.append(f"<tr><td>{pil}</td><td>{EPS[ep]['label']}<br><small>{ep}</small></td>"
                     f"<td>{EPS[ep]['primary']}</td>" + "".join(cells) + "</tr>")
    H.append("</table><p><small>각 칸 아래 작은 숫자 = 직전 스택 대비 증분(주지표 기준·양수가 개선). "
             "출처 <code>results/feature_ablation.csv</code></small></p>")

# 축③
H.append("<h2>축③ 학습방식 (멀티태스크·전이)</h2>")
if M3.empty:
    H.append("<p class='na'>아직 결과 없음</p>")
else:
    H.append("<table><tr><th>묶음</th><th>엔드포인트</th><th>주지표</th><th class='n'>멀티태스크</th>"
             "<th class='n'>단일과제 D-MPNN</th><th class='n'>차이</th><th>판정</th></tr>")
    for _, r in M3.iterrows():
        d = r.get("delta_multitask_minus_single")
        v = ("<span class='na'>단일과제 결과 대기</span>" if d is None or pd.isna(d)
             else f"<span class='{'pos' if d>0 else 'neg'}'>{d:+.4f}</span>")
        j = ("—" if d is None or pd.isna(d) else ("이득" if d > 0.005 else ("손해" if d < -0.005 else "무차")))
        H.append(f"<tr><td>{r.group}</td><td>{r.label}</td><td>{r.metric}</td>"
                 f"<td class='n'>{r.multitask:.4f} ±{r.multitask_sd:.4f}</td>"
                 f"<td class='n'>{'' if pd.isna(r.get('single_dmpnn', np.nan)) else f'{r.single_dmpnn:.4f}'}</td>"
                 f"<td class='n'>{v}</td><td>{j}</td></tr>")
    H.append("</table><p><small>주지표가 MAE인 회귀는 부호가 반대이므로 차이 해석에 주의. "
             "출처 <code>results/learning_axis.csv</code></small></p>")

# 실패
if not F.empty:
    H.append("<h2>실패 조합 (숨기지 않고 그대로)</h2>")
    g = F.groupby(F.key.str.split("|").str[0]).size().to_dict()
    H.append("<table><tr><th>단계</th><th class='n'>실패 수</th></tr>"
             + "".join(f"<tr><td>{k}</td><td class='n'>{v}</td></tr>" for k, v in g.items()) + "</table>")
    H.append("<table><tr><th>대상</th><th>에러</th></tr>"
             + "".join(f"<tr><td>{r.key}</td><td class='src'>{str(r.error)[:200]}</td></tr>"
                       for _, r in F.head(25).iterrows()) + "</table>"
             "<p><small>전체는 <code>logs/failures.jsonl</code>. 실패 조합은 표에서 값 대신 빈칸(—)이며 "
             "<b>0으로 채우지 않았다</b>.</small></p>")

H.append(f"<p class='src'>산출물: <code>results/adme_matrix.csv</code> · "
         "<code>feature_ablation.csv</code> · <code>learning_axis.csv</code> · "
         "<code>progress.jsonl</code> · <code>logs/failures.jsonl</code> · "
         "<code>predictions/*.jsonl</code></p>")
open(f"{C.RES}/{OUTNAME}", "w", encoding="utf-8").write("\n".join(H))
print(f"저장 → results/{OUTNAME} (완료 {done_n} · 실패 {fail_n} · 축① {len(M1)}행 · 축② {len(M2)}행 · 축③ {len(M3)}행)")
