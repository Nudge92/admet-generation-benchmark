#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
g4_partial_view.py — G4(Uni-Mol) ★중간 스냅샷 뷰. 읽기 전용·비침습.
- 실행 중인 오케스트레이터를 방해하지 않는다: ★새 학습·재계산 0, 기록 파일에 ★쓰지 않는다.
- 읽기 전에 /tmp 로 스냅샷 복사(쓰기 중 반쪽 읽기 방지).
- status=="ok" 인 레코드만 집계(실패를 완료로 세지 않는다 — done_keys 버그 교훈).
- 비교를 ★두 벌 낸다: (a) 각 모델의 전체 seed 평균 (b) ★G4가 성공한 seed로만 맞춘 평균.
산출(새 파일에만): results/g4_partial_view.html · results/g4_partial.csv
재실행하면 그 시점 스냅샷으로 갱신된다.
"""
import json, os, shutil, sys, tempfile
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import EPS, RES, PROGRESS, FAILURES, LOGS

G4_RAW = f"{RES}/finetune2_raw.jsonl"
SNAP = tempfile.mkdtemp(prefix="g4snap_")
NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
SEEDS = [1, 2, 3, 4, 5]
PILLAR = {"A": "흡수", "D": "분포", "M": "대사", "E": "배설"}
G2_MODELS = ["xgb_physchem", "rf_physchem", "xgb_ecfp"]


def snap(src):
    """★쓰기 중인 파일을 그대로 읽지 않는다 — /tmp 로 복사한 뒤 읽는다."""
    if not os.path.exists(src):
        return None
    dst = os.path.join(SNAP, os.path.basename(src))
    shutil.copy2(src, dst)
    return dst


def jl(path):
    """부분 기록된 마지막 줄은 버린다(쓰는 중일 수 있음)."""
    out = []
    if not path:
        return out
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


P = jl(snap(PROGRESS))
G4 = jl(snap(G4_RAW))
FA = jl(snap(FAILURES))
if not P:
    print("★중단 — progress.jsonl 을 못 찾음"); sys.exit(1)

hb = lambda ep: EPS[ep]["primary"] != "MAE"          # True면 높을수록 좋음


# ── G4: (endpoint, seed) → 주지표 ─────────────────────────────────
g4_ok, g4_fail = {}, {}
for r in G4:
    ep, sd = r.get("endpoint"), r.get("seed")
    if ep not in EPS:
        continue
    if r.get("status") == "ok":
        v = (r.get("test") or {}).get(EPS[ep]["primary"])
        if v is not None:
            g4_ok.setdefault(ep, {})[sd] = float(v)
    else:
        g4_fail.setdefault(ep, []).append(dict(seed=sd, reason=str(r.get("reason", ""))[:120]))

# ── 축①(G2·G3·ADMET-AI): (endpoint, model, seed) → 주지표 ────────
base = {}
for r in P:
    if r.get("axis") != "1":
        continue
    ep, mdl, sd = r.get("endpoint"), r.get("model"), r.get("seed")
    if ep not in EPS or not isinstance(r.get("metrics"), dict):
        continue
    v = r["metrics"].get(EPS[ep]["primary"])
    if v is not None:
        base.setdefault(ep, {}).setdefault(mdl, {})[sd] = float(v)


def agg(d, seeds=None):
    """seeds 지정 시 그 seed로만 평균. 반환 (mean, sd, n)."""
    if not d:
        return None, None, 0
    ks = [k for k in d if (seeds is None or k in seeds)]
    if not ks:
        return None, None, 0
    v = np.array([d[k] for k in ks], float)
    return float(v.mean()), float(v.std()), len(v)


rows = []
for ep, info in EPS.items():
    m = info["primary"]
    ok = g4_ok.get(ep, {})
    n_ok = len(ok)
    matched = set(ok.keys()) if n_ok else None
    status = ("완주 (5/5)" if n_ok == 5 else f"부분 (seed {n_ok}/5)" if n_ok else "미실행/대기")
    for mdl in G2_MODELS + ["dmpnn_ours", "unimol", "admetai"]:
        if mdl == "unimol":
            a_mean, a_sd, a_n = agg(ok)
            b_mean, b_sd, b_n = a_mean, a_sd, a_n          # G4는 정의상 동일
            gen = "G4"
        else:
            d = (base.get(ep) or {}).get(mdl, {})
            a_mean, a_sd, a_n = agg(d)
            b_mean, b_sd, b_n = agg(d, matched) if matched else (None, None, 0)
            gen = "G3" if mdl in ("dmpnn_ours", "admetai") else "G2"
        rows.append(dict(
            endpoint=ep, label=info["label"], pillar=info["pillar"], task=info["task"],
            metric=m, higher_better=hb(ep), gen=gen, model=mdl,
            n_ok=(n_ok if mdl == "unimol" else a_n),
            value_all_seeds=(None if a_mean is None else round(a_mean, 4)),
            sd_all_seeds=(None if a_sd is None else round(a_sd, 4)),
            value_seed_matched=(None if b_mean is None else round(b_mean, 4)),
            n_seed_matched=b_n,
            g4_status=status,
            leak_flag=("★누수 의심(TDC 전체 사전학습)·순위 제외" if mdl == "admetai" else "누수 0"),
            source=("results/finetune2_raw.jsonl" if mdl == "unimol" else "results/progress.jsonl")))
D = pd.DataFrame(rows)
D.to_csv(f"{RES}/g4_partial.csv", index=False)

# ── 순위 (정직 세대만: G2 3종 + G3 dmpnn + G4) ────────────────────
HONEST = G2_MODELS + ["dmpnn_ours", "unimol"]


def rank_of(ep, key):
    sub = D[(D.endpoint == ep) & D.model.isin(HONEST) & D[key].notna()]
    if sub.empty or ep not in g4_ok:
        return None, None, None
    s = sub.sort_values(key, ascending=not hb(ep)).reset_index(drop=True)
    hit = s[s.model == "unimol"]
    if hit.empty:
        return None, None, len(s)
    return int(hit.index[0]) + 1, s.iloc[0].model, len(s)


n_full = sum(1 for ep in EPS if len(g4_ok.get(ep, {})) == 5)
n_part = sum(1 for ep in EPS if 0 < len(g4_ok.get(ep, {})) < 5)
n_none = len(EPS) - n_full - n_part
avail = [ep for ep in EPS if ep in g4_ok]
g4_first_a = sum(1 for ep in avail if rank_of(ep, "value_all_seeds")[0] == 1)
g4_first_b = sum(1 for ep in avail if rank_of(ep, "value_seed_matched")[0] == 1)

# ══════════════════════ HTML ══════════════════════
H = ['<meta charset="utf-8"><title>G4 Uni-Mol 중간 스냅샷 — 실행 중</title>', """<style>
body{font-family:'Nanum Gothic',system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:22px 26px;color:#1d2129;line-height:1.6}
h1{font-size:24px;border-bottom:3px solid #457b9d;padding-bottom:9px;margin-bottom:4px}
h2{font-size:19px;margin-top:32px;border-left:5px solid #457b9d;padding-left:11px}
table{border-collapse:collapse;width:100%;margin:11px 0;font-size:12.5px}
th,td{border:1px solid #d8dde3;padding:4px 7px}th{background:#eef2f5}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.g4{background:#e7eef6;font-weight:700}.leak{background:#fff1ec;color:#c1440e}
.na{color:#98a2ab}.best{background:#e8f6f1;font-weight:700}
.badge{display:inline-block;font-size:10.5px;padding:1px 6px;border-radius:8px;margin-left:4px}
.b-full{background:#dff2ee;color:#1d6b60}.b-part{background:#ffe9c9;color:#8a5a00}
.b-none{background:#eceff2;color:#6b7580}.b-uns{background:#fdeceb;color:#b3261e}
.banner{background:#fff4e0;border:2px solid #f0c078;border-radius:10px;padding:15px 19px;margin:14px 0}
.warn{background:#fff8ec;border:1px solid #f0d9a8;border-radius:8px;padding:13px 17px;margin:14px 0}
.box{background:#f7f9fb;border:1px solid #dfe5ea;border-radius:8px;padding:13px 17px;margin:14px 0}
small,.src{font-size:11.5px;color:#6b7580}code{background:#eef2f5;padding:1px 4px;border-radius:3px;font-size:11.5px}
</style>"""]
H.append("<h1>G4 (Uni-Mol) 중간 스냅샷 — 다른 세대와의 잠정 비교</h1>")
H.append(f"<div class='banner'><b>★실행 중 중간 스냅샷 — 최종 결과가 아닙니다.</b><br>"
         f"스냅샷 시각 <b>{NOW}</b> · G4 성공 런 <b>{sum(len(v) for v in g4_ok.values())}/90</b>"
         f" · 완주 {n_full} · 부분 {n_part} · 미실행 {n_none} (18개 중)<br>"
         "<b>G5(파운데이션)는 아직 미실행</b>이므로 세대 최종 결론을 낼 수 없습니다. "
         "아래 모든 서술은 <b>잠정 관찰</b>이며, 구분 가능선·신뢰구간을 계산하지 않았으므로 "
         "<b>어느 세대가 이겼다고 판정하지 않습니다</b>.</div>")

# ① 진행 현황
H.append("<h2>① G4 진행 현황 (정직한 커버리지)</h2>")
H.append("<table><tr><th>기둥</th><th>엔드포인트</th><th>주지표</th><th class='n'>성공 seed</th>"
         "<th class='n'>실패</th><th>상태</th><th>실패 사유(요약)</th></tr>")
for pil in ("A", "D", "M", "E"):
    for ep, info in EPS.items():
        if info["pillar"] != pil:
            continue
        n = len(g4_ok.get(ep, {})); fl = g4_fail.get(ep, [])
        bd = ("<span class='badge b-full'>완주 5/5</span>" if n == 5
              else f"<span class='badge b-part'>부분 {n}/5</span>" if n
              else "<span class='badge b-none'>미실행/대기</span>")
        why = ("; ".join(sorted({f['reason'][:60] for f in fl})) if fl else
               ("" if n == 5 else "<span class='na'>대기(아직 실행 전)</span>" if n == 0 else "<span class='na'>진행 중</span>"))
        H.append(f"<tr><td>{pil} {PILLAR[pil]}</td><td>{info['label']}<br><small>{ep}</small></td>"
                 f"<td>{info['primary']} {'↑' if hb(ep) else '↓'}</td><td class='n'>{n}</td>"
                 f"<td class='n'>{len(fl)}</td><td>{bd}</td><td class='src'>{why}</td></tr>")
H.append("</table>")
H.append(f"<div class='box'><b>요약</b> — 18개 중 <b>완주 {n_full} · 부분 {n_part} · 미실행 {n_none}</b>. "
         "미실행 셀은 <b>0이 아니라 N/A(아직 실행 전)</b>이며, 실패 셀은 OOM 재시도 대기 상태다"
         "(재시도 로직은 `status==ok` 만 완료로 인정하도록 수정됨).</div>")

# ② 비교표
H.append("<h2>② 세대 비교 — (a) 전체 seed 평균 · (b) ★G4 성공 seed로 맞춘 평균</h2>")
H.append("<div class='warn'><b>왜 두 벌인가</b> — G4는 아직 일부 seed만 끝났다. "
         "(a)는 각 모델의 원래 5-seed 평균이라 <b>비교 대상 seed가 서로 다르고</b>, "
         "(b)는 G4가 성공한 <b>바로 그 seed로만</b> G2·G3를 다시 평균해 조건을 맞춘 것이다. "
         "(a)와 (b)가 크게 다르면 그 엔드포인트는 <b>seed 변동이 크다</b>는 뜻이다.</div>")
for ep in avail:
    info = EPS[ep]; m = info["primary"]; n = len(g4_ok[ep])
    sub = D[D.endpoint == ep]
    unstable = " <span class='badge b-uns'>SD 불안정(n≤2)</span>" if n <= 2 else ""
    ra, wa, _ = rank_of(ep, "value_all_seeds")
    rb, wb, tot = rank_of(ep, "value_seed_matched")
    H.append(f"<h3>{info['label']} <small>({ep} · {info['pillar']} {PILLAR[info['pillar']]} · "
             f"{m} {'높을수록 좋음' if hb(ep) else '낮을수록 좋음'})</small>"
             f"<span class='badge b-part'>G4 seed {n}/5</span>{unstable}</h3>")
    H.append("<table><tr><th>세대</th><th>모델</th><th class='n'>(a) 전체 seed</th><th class='n'>n</th>"
             "<th class='n'>(b) seed 맞춤</th><th class='n'>n</th><th class='n'>(a)−(b) 차</th>"
             "<th>비고</th></tr>")
    hon = sub[sub.model.isin(HONEST) & sub.value_seed_matched.notna()]
    best_b = (hon.value_seed_matched.max() if hb(ep) else hon.value_seed_matched.min()) if not hon.empty else None
    for _, r in sub.iterrows():
        if r.value_all_seeds is None and r.value_seed_matched is None:
            continue
        cls = (" class='g4'" if r.model == "unimol"
               else " class='leak'" if r.model == "admetai"
               else (" class='best'" if (best_b is not None and r.value_seed_matched == best_b) else ""))
        f = lambda x: "<span class='na'>—</span>" if x is None or pd.isna(x) else f"{x:.4f}"
        diff = ("" if (r.value_all_seeds is None or r.value_seed_matched is None
                       or pd.isna(r.value_all_seeds) or pd.isna(r.value_seed_matched))
                else f"{r.value_all_seeds - r.value_seed_matched:+.4f}")
        note = (r.leak_flag if r.model == "admetai" else
                f"±{r.sd_all_seeds:.4f}" if r.sd_all_seeds is not None and not pd.isna(r.sd_all_seeds) else "")
        H.append(f"<tr><td>{r.gen}</td><td{cls}>{r.model}</td><td class='n'{cls}>{f(r.value_all_seeds)}</td>"
                 f"<td class='n'>{int(r.n_ok) if r.model=='unimol' else int(r.n_ok)}</td>"
                 f"<td class='n'{cls}>{f(r.value_seed_matched)}</td><td class='n'>{int(r.n_seed_matched)}</td>"
                 f"<td class='n'>{diff}</td><td class='src'>{note}</td></tr>")
    H.append(f"<tr><td colspan=8 class='src'>G5(파운데이션): <b>대기</b> — 아직 실행 전</td></tr></table>")
    H.append(f"<p><small>정직 세대 {tot}개 중 G4 순위 — (a) 기준 <b>{ra}위</b> · (b) 기준 <b>{rb}위</b>"
             f" · (b)에서 1위는 <b>{wb}</b>. ADMET-AI는 누수라 순위에서 뺐다. "
             "<b>이 순위는 잠정</b>이며 구분 가능선을 계산하지 않았다.</small></p>")

# ③ 잠정 관찰
H.append("<h2>③ 잠정 관찰</h2>")
obs = []
obs.append(f"집계 가능한 <b>{len(avail)}개</b> 엔드포인트에서 G4가 1위인 경우는 "
           f"(a) 전체 seed 기준 <b>{g4_first_a}개</b> · (b) seed 맞춤 기준 <b>{g4_first_b}개</b>다. "
           "<b>이겼다/졌다로 읽지 말 것</b> — 미완료·소seed 예비 관측이다.")
big = []
for ep in avail:
    r = D[(D.endpoint == ep) & (D.model == "unimol")].iloc[0]
    if r.value_all_seeds is not None and r.value_seed_matched is not None:
        pass
    hon = D[(D.endpoint == ep) & D.model.isin(G2_MODELS + ["dmpnn_ours"])]
    d = hon.dropna(subset=["value_all_seeds", "value_seed_matched"])
    if d.empty:
        continue
    mx = float((d.value_all_seeds - d.value_seed_matched).abs().max())
    big.append((ep, mx))
big.sort(key=lambda x: -x[1])
if big:
    obs.append("(a)와 (b)의 차가 가장 큰 엔드포인트는 "
               + " · ".join(f"<b>{EPS[e]['label']}</b> {v:.4f}" for e, v in big[:3])
               + " 로, 그만큼 <b>seed 변동이 크다</b>. 이런 곳에서는 seed 수가 다른 비교를 특히 조심해야 한다.")
obs.append("<b>G5(ChemBERTa-2·MoLFormer)가 미실행</b>이므로 '세대 사다리'는 아직 완성되지 않았다. "
           "지금 표는 G2·G3·G4만의 부분 그림이다.")
obs.append("<b>구분 가능선·신뢰구간을 계산하지 않았다.</b> 격차가 커 보여도 "
           "<b>차이 관찰</b>까지만이고 통계적 판정이 아니다. 최종 보고서에서 부트스트랩·DeLong으로 판정한다.")
obs.append("엔드포인트마다 <b>분할·지표·난이도가 다르므로 절대값을 가로로 비교하지 말 것</b>. "
           "세대 순위는 <b>같은 엔드포인트 안에서만</b> 의미가 있다.")
H.append("<div class='warn'><ol>" + "".join(f"<li>{o}</li>" for o in obs) + "</ol></div>")
H.append(f"<p class='src'>출처: <code>results/finetune2_raw.jsonl</code>(G4) · "
         "<code>results/progress.jsonl</code>(G2·G3·ADMET-AI) — 둘 다 <b>/tmp 스냅샷을 통해 읽기 전용</b>으로 접근했다. "
         "이 문서는 새 파일에만 쓴다(<code>g4_partial_view.html</code> · <code>g4_partial.csv</code>). "
         f"재실행하면 그 시점으로 갱신된다: <code>python src/g4_partial_view.py</code></p>")

open(f"{RES}/g4_partial_view.html", "w", encoding="utf-8").write("\n".join(H))
shutil.rmtree(SNAP, ignore_errors=True)
print(f"저장 → results/g4_partial_view.html · g4_partial.csv")
print(f"  G4 성공 {sum(len(v) for v in g4_ok.values())}/90 · 완주 {n_full} · 부분 {n_part} · 미실행 {n_none}")
print(f"  집계 가능 {len(avail)}개 · G4 1위: (a) {g4_first_a}개 / (b) {g4_first_b}개  ※잠정")
for ep in avail:
    ra, _, tot = rank_of(ep, "value_all_seeds"); rb, wb, _ = rank_of(ep, "value_seed_matched")
    print(f"    {ep:<32}seed {len(g4_ok[ep])}/5 · G4 순위 (a){ra}/{tot} (b){rb}/{tot} · (b)1위 {wb}")
