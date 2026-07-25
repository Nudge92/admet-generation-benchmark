#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_final.py — ADME 전면 벤치마크 ★최종 보고서. ★새 학습·재계산 0 — 확정값 이관 + 정정 반영만.
입력: adme_matrix.csv · feature_ablation.csv · feature_2x2.csv · learning_axis.csv ·
      bootstrap_verdict.json · split_leakage.json · finetune2_raw.jsonl
산출: results/report_adme_full.html
★적대검증 확정 정정 4건 반영: 18/18→17·16/18 / E석권→분해 / 승수 비대칭 병기 / 축③ 누수 기각.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import common as C
from common import EPS

RES = C.RES
MAT = pd.read_csv(f"{RES}/adme_matrix.csv")
FA = pd.read_csv(f"{RES}/feature_ablation.csv")
F2 = pd.read_csv(f"{RES}/feature_2x2.csv")
LX = pd.read_csv(f"{RES}/learning_axis.csv")
BV = json.load(open(f"{RES}/bootstrap_verdict.json"))
LK = json.load(open(f"{RES}/split_leakage.json"))
G4RAW = [json.loads(l) for l in open(f"{RES}/finetune2_raw.jsonl", encoding="utf-8")]
NSIZE = {e: v["n_train_val"] + v["n_test"] for e, v in LK.items()}
PILLAR = {"A": "흡수", "D": "분포", "M": "대사", "E": "배설"}
GENLAB = {"G2": "고전 ML", "G3": "GNN", "G4": "3D", "G5": "파운데이션"}
HON = ["xgb_physchem", "rf_physchem", "xgb_ecfp", "dmpnn_ours", "unimol", "chemberta", "molformer"]
GEN = {"xgb_physchem": "G2", "rf_physchem": "G2", "xgb_ecfp": "G2", "dmpnn_ours": "G3",
       "unimol": "G4", "chemberta": "G5", "molformer": "G5"}
hb = lambda ep: EPS[ep]["primary"] != "MAE"

# G4 seed 수(정직 표기)
g4n = {}
for r in G4RAW:
    if r.get("status") == "ok":
        g4n[r["endpoint"]] = g4n.get(r["endpoint"], 0) + 1


def val(ep, model):
    r = MAT[(MAT.endpoint == ep) & (MAT.model == model)]
    return (None if r.empty or pd.isna(r.iloc[0].value) else float(r.iloc[0].value),
            None if r.empty or pd.isna(r.iloc[0].sd) else float(r.iloc[0].sd))


# 세대 승수(비대칭) + 최고 세대
win = {}; best_gen = {}
for ep in EPS:
    g = MAT[(MAT.endpoint == ep) & MAT.model.isin(HON) & MAT.value.notna()]
    if g.empty:
        continue
    b = g.loc[g.value.idxmax() if hb(ep) else g.value.idxmin()]
    best_gen[ep] = (GEN[b.model], b.model, float(b.value))
    win[GEN[b.model]] = win.get(GEN[b.model], 0) + 1

# 부트스트랩 판정 요약
distinguishable = {}
for ep, v in BV.items():
    if "best_predicted_model" in v:
        distinguishable[ep] = (v["best_predicted_model"], v["n_distinguishable"], v["n_compared"],
                               v.get("best_vs_others", {}))


def fmt(x, d=4):
    return "—" if x is None else f"{x:.{d}f}"


H = ['<meta charset="utf-8"><title>ADME 전면 벤치마크 — 최종 보고서 (18과제·3축)</title>', """<style>
body{font-family:'Nanum Gothic',system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:22px 26px;color:#1d2129;line-height:1.62}
h1{font-size:25px;border-bottom:3px solid #457b9d;padding-bottom:9px;margin-bottom:3px}
h2{font-size:19px;margin-top:36px;border-left:5px solid #457b9d;padding-left:11px;scroll-margin-top:12px}
h3{font-size:15.5px;margin-top:22px;color:#264653}
table{border-collapse:collapse;width:100%;margin:11px 0;font-size:12.5px}
th,td{border:1px solid #d8dde3;padding:5px 7px}th{background:#eef2f5}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.best{background:#e8f6f1;font-weight:700}.leak{background:#fff1ec;color:#c1440e;font-weight:700}
.dist{background:#dff2ee;color:#1d6b60;font-weight:700}.tie{background:#f4f0e4;color:#7a6a3a}
.na{color:#98a2ab}.pos{color:#1d6b60;font-weight:700}.neg{color:#b3261e;font-weight:700}
.banner{background:linear-gradient(135deg,#20343f,#3a5a6d);color:#fff;border-radius:12px;padding:20px 24px;margin:12px 0 20px}
.banner h2{color:#fff;border:0;padding:0;margin:0 0 8px;font-size:18px}
.banner .k{display:inline-block;background:rgba(255,255,255,.13);border-radius:8px;padding:8px 13px;margin:5px 6px 0 0;font-size:13px}
.banner b{color:#ffe9a8}
.box{background:#f7f9fb;border:1px solid #dfe5ea;border-radius:8px;padding:13px 17px;margin:14px 0}
.warn{background:#fff8ec;border:1px solid #f0d9a8;border-radius:8px;padding:13px 17px;margin:14px 0}
.crit{background:#fdefea;border:1px solid #f2bda9;border-radius:8px;padding:13px 17px;margin:14px 0}
.fix{background:#eef4fd;border:1px solid #bcd0ee;border-radius:8px;padding:13px 17px;margin:14px 0}
.badge{display:inline-block;font-size:10.5px;padding:1px 6px;border-radius:8px;margin-left:4px}
.b-part{background:#ffe9c9;color:#8a5a00}
nav{background:#f7f9fb;border:1px solid #dfe5ea;border-radius:8px;padding:11px 16px;margin:13px 0;font-size:13px}
nav a{color:#2a5d6d;text-decoration:none;margin-right:13px;display:inline-block;padding:2px 0}
small,.src{font-size:11.5px;color:#6b7580}code{background:#eef2f5;padding:1px 4px;border-radius:3px;font-size:11.5px}
ul,ol{margin:7px 0 7px 4px}li{margin:5px 0}
</style>"""]

H.append("<h1>ADME 전면 벤치마크 — 최종 보고서</h1>")
H.append("<p class='src'>작업일 2026-07-22~24 · 18개 ADME 엔드포인트(A흡수6·D분포3·M대사6·E배설3) × "
         "4세대(G2·G3·G4·G5) + ADMET-AI · 3축(세대·특징·학습방식) · TDC admet_group 공식 분할·5 seed · "
         "★적대 검증 6렌즈 정정 반영 · 독성 마스터 보고서와 나란히 읽는 구조.</p>")

# ── 0. 배너
H.append("<div class='banner'><h2>한 줄로</h2>"
         "<p><b>1.</b> ADME는 독성과 다르다 — 독성에선 고전 ML(G2)이 압도적이었지만 ADME에선 세대 우열이 갈린다.</p>"
         "<p><b>2.</b> 물성 표현(서술자)이 지문(ECFP)을 이기는 건 견고하나, <b>18/18 '압승'이 아니라 17/18</b>다"
         "(동일 XGB 통제; '18/18'은 비대칭 로스터의 산물이었음 — §8 정정).</p>"
         "<p><b>3.</b> 통계적으로 <b>확실한 세대 효과는 친유성에서 G3(GNN) 하나뿐</b>. 나머지는 대부분 구분 불가거나 미검증.</p>"
         f"<div><span class='k'>18과제</span>"
         + "".join(f"<span class='k'>{k} <b>{v}</b>승</span>" for k, v in sorted(win.items()))
         + "<span class='k'>물성vs지문 <b>17/18</b>(XGB)·16/18(RF)</span>"
         "<span class='k'>부트스트랩 구분가능 <b>23/54</b>쌍</span>"
         "<span class='k'>G5 파운데이션 <b>0</b>승</span></div>"
         "<p style='font-size:12px;margin-top:9px'>★세대 승수는 <b>G2가 3모델 중 best</b>를 뽑는 다중비교 이점이 있어 "
         "세대 본질성능이 아니다. G4 5승은 <b>통계 미검증</b>(§6·§9).</p></div>")

H.append("<nav><b>목차</b> "
         + " ".join(f"<a href='#s{i}'>{i}. {t}</a>" for i, t in enumerate(
             ["지표", "방법", "축① 세대", "축② 특징", "축③ 학습방식", "통계 판정",
              "독성 vs ADME", "자기정정", "한계", "결론"], 1)) + "</nav>")

# ── 1. 지표
H.append("<h2 id='s1'>1. 지표를 어떻게 읽는가</h2><div class='box'><ul>"
         "<li><b>AUROC/AUPRC</b>(분류·높을수록 좋음) · <b>MAE</b>(회귀·낮을수록 좋음) · "
         "<b>Spearman</b>(순위상관·높을수록 좋음).</li>"
         "<li><b>★E(배설) 3종과 VDss는 Spearman이 주지표</b> — 생리 지배·절대값 예측 한계 때문. "
         "다른 회귀(Caco2·Lipo·PPBR·용해도)는 MAE라 <b>방향이 반대</b>다. 승부 계산에 이 방향을 반영했다.</li>"
         "<li><b>부트스트랩 대응비교</b> — 같은 test 분자를 2000회 재표집해 두 모델의 지표 차 분포를 만들고, "
         "95% CI가 0을 넘지 않으면 '구분 가능', 0을 포함하면 '동률(구분 불가)'로 판정한다. "
         "<b>고정 test·5 seed의 SD는 일반화 불확실성이 아니라</b> 학습 변동이므로, 진짜 판정은 이 분자 부트스트랩으로 한다.</li></ul></div>")

# ── 2. 방법
H.append("<h2 id='s2'>2. 방법</h2>")
H.append("<div class='box'><b>분할·공정성(실측)</b><ul>"
         "<li>18개 전부 TDC <code>admet_group</code> 공식 고정 test(=scaffold seed=42) + "
         "<code>get_train_valid_split(seed=1..5)</code>. 독성 실험과 <b>완전히 같은 프로토콜</b>.</li>"
         "<li><b>train↔test 정확분자 중복 = 0 (18/18)</b>. chemprop 비호환 분자는 solubility 2개뿐 → 전 조합 공통 제외.</li>"
         "<li><b>★test 분자 동일성 실측</b> — 개수는 18/18 전 세대 완전 일치. 나아가 반감기·간세포청소율에서 "
         "<b>분자 단위 Jaccard = 1.0000</b> 확인(Uni-Mol이 conformer 실패 분자를 제외하지 않고 전량 예측). "
         "→ G4가 '더 쉬운 부분집합'에서 평가받았을 가능성 배제.</li></ul></div>")
H.append("<table><tr><th>세대</th><th>정의</th><th>모델</th></tr>"
         "<tr><td><b>G2</b></td><td>고전 ML</td><td>물리화학 서술자(RDKit 210)+XGBoost / +RandomForest / ECFP4(2048)+XGBoost</td></tr>"
         "<tr><td><b>G3</b></td><td>GNN</td><td>chemprop D-MPNN <b>정직 학습</b>(train만) + ADMET-AI(<b>★누수 기준선</b>·순위 제외)</td></tr>"
         "<tr><td><b>G4</b></td><td>3D</td><td>Uni-Mol (ETKDG conformer)</td></tr>"
         "<tr><td><b>G5</b></td><td>파운데이션</td><td>ChemBERTa-2 · MoLFormer</td></tr></table>"
         "<p><small>★<b>G1(구조알림)은 세대에서 제외</b> — toxicophore는 독성 전용이라 Caco-2·청소율에 부적용. "
         "대신 의약화학 규칙(GSE·CNS MPO·이온화)을 축②의 특징으로 넣었다. "
         "3축: ①세대(G2~G5) ②특징 ablation(G2 트리·누적 스택) ③학습방식(멀티태스크).</small></p>")

# ── 3. 축① 세대
H.append("<h2 id='s3'>3. 축① 세대 — 엔드포인트별</h2>")
for pil in ("A", "D", "M", "E"):
    eps = [e for e in EPS if EPS[e]["pillar"] == pil]
    if not eps:
        continue
    H.append(f"<h3>{pil} {PILLAR[pil]}</h3>")
    H.append("<table><tr><th>엔드포인트</th><th>주지표</th><th class='n'>G2 xgb·phys</th>"
             "<th class='n'>G2 rf·phys</th><th class='n'>G2 xgb·ecfp</th><th class='n'>G3 dmpnn</th>"
             "<th class='n'>G4 unimol</th><th class='n'>G5 cbert</th><th class='n'>G5 molf</th>"
             "<th class='n'>ADMET-AI</th><th>최고</th></tr>")
    for ep in eps:
        m = EPS[ep]["primary"]
        cells = []
        best = best_gen.get(ep, (None, None, None))
        for mdl in ["xgb_physchem", "rf_physchem", "xgb_ecfp", "dmpnn_ours", "unimol",
                    "chemberta", "molformer"]:
            v, sd = val(ep, mdl)
            cls = " class='best'" if mdl == best[1] else " class='n'"
            g4b = (f"<span class='badge b-part'>{g4n.get(ep,0)}/5</span>"
                   if mdl == "unimol" and g4n.get(ep, 5) < 5 else "")
            cells.append(f"<td{cls}>{fmt(v)}{g4b}</td>")
        av, _ = val(ep, "admetai")
        cells.append(f"<td class='n leak'>{fmt(av)}</td>")
        bmark = f"{best[0]}" if best[0] else "—"
        H.append(f"<tr><td>{EPS[ep]['label']}<br><small>{ep} · {NSIZE[ep]:,}분자</small></td>"
                 f"<td>{m}{'↑' if hb(ep) else '↓'}</td>" + "".join(cells)
                 + f"<td><b>{bmark}</b></td></tr>")
    H.append("</table>")
H.append(f"<div class='warn'><b>세대 승수: "
         + " · ".join(f"{k} {v}" for k, v in sorted(win.items(), key=lambda x: -x[1]))
         + "</b> (독성 18과제는 G2 17·G4 1·G3 0·G5 0). "
         "<b>★단 이 승수를 세대 우열로 읽으면 안 된다</b> — G2는 <b>3모델 중 best</b>를 뽑는 다중비교 이점이 있고, "
         "G4의 5승은 아래 §6 통계 판정을 <b>받지 못했다</b>(G4는 분자별 예측 미저장). "
         "특히 <b>E(배설)에서 G4가 명목상 앞선 것</b>은 관찰일 뿐이며, 그중 <b>반감기만 뚜렷하고 두 청소율은 "
         "G2와 구분 불가 동률</b>이다(§6·§8).</div>")

# ── 4. 축② 특징 (대칭 로스터)
H.append("<h2 id='s4'>4. 축② 특징 — 무엇을 넣어야 오르나</h2>")
H.append("<h3>4-A. 누적 스택 증분 (G2 XGBoost)</h3>")
inc = FA.pivot_table(index="pillar", columns="stack", values="delta_vs_prev", aggfunc="mean")
cols = [c for c in ["+phys", "+ion", "+3d", "+medchem"] if c in inc.columns]
H.append("<table><tr><th>기둥</th>" + "".join(f"<th class='n'>{c}</th>" for c in cols) + "</tr>")
for pil in ("A", "D", "M", "E"):
    if pil in inc.index:
        H.append(f"<tr><td>{pil} {PILLAR[pil]}</td>"
                 + "".join(f"<td class='n {'pos' if inc.loc[pil,c]>0 else 'neg'}'>{inc.loc[pil,c]:+.4f}</td>"
                           for c in cols) + "</tr>")
H.append("</table>")
maxc = FA.dropna(subset=["delta_vs_prev"]).loc[lambda d: d.groupby("endpoint").delta_vs_prev.idxmax()]
nphys = int((maxc["stack"] == "+phys").sum())
H.append(f"<div class='box'>18개 중 <b>{nphys}개에서 <code>+phys</code>(물리화학 서술자)가 최대 기여</b> — "
         "PPBR +2.25 · 용해도 +0.52 · 친유성 +0.135. 예상했던 <b>이온화·3D·의약화학 규칙은 미미하거나 음수</b>. "
         "<small>단 누적 스택이라 증분은 순서 의존적이다 — '+phys가 비법'이라기보다 "
         "'물리화학 서술자가 ECFP4보다 정보량이 많다'로 읽어야 한다.</small></div>")

H.append("<h3>4-B. ★대칭 로스터 재집계 — 특징 효과 격리 (적대 검증 반영)</h3>")
H.append("<div class='fix'>초기 요약의 <b>'물리화학 18/18 압승'은 틀렸다</b>. physchem엔 모델 2개(xgb+rf), "
         "ECFP엔 1개(xgb)만 줘서 <b>best-of-2</b>로 뽑은 비대칭 로스터의 산물이었다. "
         "빠져 있던 <b>rf_ecfp</b>를 추가해 <b>2×2 대칭(모델×특징)</b>으로 재집계했다.</div>")
nxgb = int(F2.xgb_physchem_wins.sum()); nrf = int(F2.rf_physchem_wins.sum())
H.append("<table><tr><th>비교 방식</th><th class='n'>물리화학 승</th><th class='n'>ECFP 승</th><th>해석</th></tr>"
         f"<tr><td>동일 XGB 통제(phys vs ecfp)</td><td class='n'>{nxgb}/18</td><td class='n'>{18-nxgb}/18</td>"
         "<td>가장 깨끗한 특징 비교</td></tr>"
         f"<tr><td>동일 RF 통제</td><td class='n'>{nrf}/18</td><td class='n'>{18-nrf}/18</td><td>—</td></tr>"
         "<tr class='na'><td>best-of-2(초기 '18/18')</td><td class='n'>18/18</td><td class='n'>0/18</td>"
         "<td>★비대칭 — 물성만 2모델</td></tr></table>")
ecfp_wins = F2[(~F2.xgb_physchem_wins) | (F2.rf_physchem_wins == False)]
H.append("<p><small>ECFP가 이긴 곳: "
         + " · ".join(f"<b>{r.endpoint}</b>({'XGB' if not r.xgb_physchem_wins else 'RF'})"
                      for _, r in ecfp_wins.iterrows())
         + f". <b>정직한 특징 우세는 17/18(XGB)·16/18(RF)</b>이지 18/18이 아니다. "
         "방향(물성이 대부분 우세)은 견고하나 '압승'은 과장. 출처 <code>results/feature_2x2.csv</code>.</small></p>")

# ── 5. 축③ 학습방식
H.append("<h2 id='s5'>5. 축③ 학습방식 — 멀티태스크가 이득인가</h2>")
H.append("<div class='crit'><b>★자체 발견한 누수로 초기 결론이 뒤집혔다.</b> CYP 3종(2C9·2D6·3A4)은 "
         "<b>같은 분자 라이브러리에 라벨만 다른</b> 데이터셋이다. 합집합 멀티태스크를 만들자 한 과제의 test 분자가 "
         "다른 과제의 train으로 들어가 <b>test의 88.9%가 학습에 노출</b>됐다. 초기 '멀티태스크 9/10 이득'은 "
         "이 누수의 산물이었다. → 누수 차단 게이트(합집합 train/valid에서 <b>어느 과제의 test에라도 등장하는 분자 전부 제외</b>) 후 재계산.</div>")
H.append("<table><tr><th>묶음</th><th class='n'>평균 Δ(멀티−단일)</th><th class='n'>이득</th><th>초기(누수)</th><th>판정</th></tr>")
for grp, g in LX.groupby("group"):
    ok = g[g.delta_multitask_minus_single.notna()]
    if ok.empty:
        continue
    mean = ok.delta_multitask_minus_single.mean()
    ng = int((ok.delta_multitask_minus_single > 0).sum())
    init = {"cyp_inhibition": "3/3 이득", "cyp_substrate": "2/3", "all_adme_cls": "9/10 이득"}.get(grp, "")
    verd = ("★이득 소멸" if grp == "cyp_inhibition" else "손해" if mean < 0 else "소표본 2개에 집중")
    H.append(f"<tr><td>{grp}</td><td class='n {'pos' if mean>0 else 'neg'}'>{mean:+.4f}</td>"
             f"<td class='n'>{ng}/{len(ok)}</td><td class='na'>{init}</td><td>{verd}</td></tr>")
H.append("</table>")
H.append("<div class='box'><b>가설 기각</b> — '닮은 과제(CYP)를 묶으면 이득'은 성립하지 않았다"
         "(CYP 억제 +0.0004·기질 −0.024). 남은 이득은 전체 묶음의 <b>소표본 2개</b>(생체이용률·HIA)에 집중된다. "
         "즉 '과제 유사도'보다 '데이터가 적은 과제만 덕을 본다'에 가깝다. "
         "과거 장기독성 멀티태스크 4/4 음의 전이와 함께 보면, <b>멀티태스크 이득은 조건적이고 대체로 미미하다</b>.</div>")

# ── 6. 통계 판정
H.append("<h2 id='s6'>6. 통계 판정 (부트스트랩 대응비교)</h2>")
H.append("<div class='crit'><b>★적용 범위 한계 — 먼저 읽을 것.</b> <b>G4(Uni-Mol)·G5(ChemBERTa·MoLFormer)는 "
         "분자별 예측을 저장하지 않았다</b> → 분자 부트스트랩·DeLong이 <b>원천 불가능</b>하다. "
         "아래 판정은 <b>예측이 있는 G2 3종·G3 사이</b>에 대해서만이다. "
         "<b>따라서 §3의 'E에서 G4 우세'와 G4의 5승은 이 표로 검증되지 않았다</b>(§9 한계).</div>")
H.append("<table><tr><th>엔드포인트</th><th class='n'>n분자</th><th>최고(G2·G3 중)</th>"
         "<th class='n'>구분가능/비교</th><th>판정</th></tr>")
for ep in EPS:
    if ep not in distinguishable:
        continue
    bm, nd, nc, vs = distinguishable[ep]
    verd = ("<span class='dist'>확실히 우세</span>" if nd == nc and nc > 0 else
            "<span class='tie'>대부분 동률</span>" if nd == 0 else f"부분({nd}/{nc})")
    H.append(f"<tr><td>{EPS[ep]['label']}<br><small>{ep}</small></td>"
             f"<td class='n'>{LK[ep]['n_test']}</td><td>{bm} ({GEN.get(bm,'')})</td>"
             f"<td class='n'>{nd}/{nc}</td><td>{verd}</td></tr>")
H.append("</table>")
H.append("<div class='box'><b>★확실한 세대 효과는 친유성 하나뿐</b> — <b>친유성에서 dmpnn(G3)이 "
         "G2 3종 전부를 부트스트랩 3/3으로 이긴다</b>(Δ −0.12~−0.25, 4,200분자). "
         "이것이 <b>이 연구에서 통계적으로 확실한 유일한 GNN 승리</b>다. "
         "그 외 physchem이 확실히 우세한 곳: Caco-2(3/3)·용해도·PPBR·CYP 억제 3종(부분). "
         "<b>구분 불가(동률)</b>: 반감기(G2끼리 0/3)·HIA·P-gp·CYP 기질 3종(CYP2C9 기질 0/3). "
         "즉 <b>소표본 엔드포인트에서는 어느 세대가 낫다고 말할 수 없다</b>.</div>")

# ── 7. 독성 vs ADME
H.append("<h2 id='s7'>7. ★독성 vs ADME 대비</h2>")
H.append("<table><tr><th>최고 세대</th><th class='n'>독성 18과제</th><th class='n'>ADME 18과제</th><th>비고</th></tr>"
         "<tr><td>G2 고전 ML</td><td class='n'>17</td><td class='n'>"
         + str(win.get("G2", 0)) + "</td><td>ADME도 여전히 최다</td></tr>"
         "<tr><td>G4 3D</td><td class='n'>1</td><td class='n'>" + str(win.get("G4", 0))
         + "</td><td>독성은 구분불가로 철회 · ADME는 <b>★통계 미검증</b></td></tr>"
         "<tr><td>G3 GNN(정직)</td><td class='n'>0</td><td class='n'>" + str(win.get("G3", 0))
         + "</td><td>ADME 친유성만 통계 확실</td></tr>"
         "<tr><td>G5 파운데이션</td><td class='n'>0</td><td class='n'>" + str(win.get("G5", 0))
         + "</td><td>두 영역 모두 0승</td></tr></table>")
H.append("<div class='warn'><b>해석</b> — 독성에서는 고전 ML이 압도적이었으나 <b>ADME에서는 세대 우열이 갈린다</b>. "
         "<b>★단 '두 승수 모두 비대칭(G2 다중비교)·미검증(G4) 요소가 있어 'ADME는 신세대가 낫다'로 단정하면 안 된다.</b> "
         "확실한 것은 친유성(G3)뿐이다.<br><br>"
         "<b>데이터 크기 패턴</b> — G3(GNN)는 <b>소표본에서 붕괴</b>한다: 생체이용률(640분자) AUROC "
         + f"{fmt(val('bioavailability_ma','dmpnn_ours')[0],4)}(≈무작위) · 반감기(667) Spearman "
         + f"{fmt(val('half_life_obach','dmpnn_ours')[0],4)} · CYP3A4 기질(670) "
         + f"{fmt(val('cyp3a4_substrate_carbonmangels','dmpnn_ours')[0],4)}. "
         "유일한 확실 승리 친유성은 <b>4,200분자</b>로 가장 큰 축에 든다. "
         "<b>G5는 두 영역 모두 0승</b> — 파운데이션 모델이 분자 물성·독성에서 자리를 못 잡는다는 일관된 신호.</div>")

# ── 8. 자기정정
H.append("<h2 id='s8'>8. ★자기정정 기록</h2>")
H.append("<div class='fix'>이 보고서가 초안에서 틀렸던 것들. 무엇이 틀렸고 무엇이 잡아냈는지 그대로 남긴다."
         "<table><tr><th>초안 주장</th><th>실제</th><th>어떻게 잡혔나</th></tr>"
         "<tr><td>축③ '멀티태스크 9/10 이득'</td><td>88.9% 누수 산물 → 가설 기각</td>"
         "<td><b>내가 직접 발견</b>(합집합 train에 test 섞임 의심)</td></tr>"
         "<tr><td>'물리화학 18/18 압승'</td><td>동일 XGB 17/18·동일 RF 16/18 "
         "(비대칭 로스터 산물)</td><td>적대검증 <b>6렌즈 중 5개</b>가 독립 지적·반박 0/3</td></tr>"
         "<tr><td>'E 3종 G4 석권'</td><td>반감기만 뚜렷 · 청소율 2종 G2와 동률"
         "(Δ0.0025≪SD)</td><td>적대검증 지적(반박 1/3)</td></tr>"
         "<tr><td>세대 승수를 단독 결론으로</td><td>G2 다중비교 이점·G4 미검증 병기 필요</td>"
         "<td>적대검증 seed-count 렌즈</td></tr></table>"
         "<small>★반대로, 적대검증이 무너뜨리려다 <b>실패한(기각된)</b> 것: 'E에서 G4가 낫다'는 방향 자체 — "
         "5건의 반박 시도가 모두 기각되어 <b>방향은 견고</b>하다(단 구분 가능성은 §6대로 미검증). "
         "가장 걱정했던 test 분자 정체성도 Jaccard 1.0으로 통과.</small></div>")

# ── 9. 한계
H.append("<h2 id='s9'>9. 한계 (정직 고지)</h2>")
H.append("<div class='warn'><ul>"
         "<li><b>★G4·G5 분자별 예측 미저장</b> → 이들이 관여한 비교는 통계 판정 불가. "
         "E의 G4 우세는 <b>5 seed 평균 관찰</b>이며, seed SD는 고정 test 위 학습 변동일 뿐 일반화 불확실성이 아니다. "
         "적대검증에서 이 우세를 무너뜨리려던 5건이 모두 기각돼 <b>방향은 견고하나 구분 가능성은 미검증</b>이다. "
         "<b>독성 연구에서는 챔피언을 동일 조건으로 재현 학습해 분자별 예측을 확보했으므로</b>(18/18 일치·최대 편차 0.0001), "
         "ADME에도 같은 절차(G4·G5 예측 저장 후 부트스트랩)를 적용하는 것이 후속 과제다.</li>"
         f"<li><b>VDss는 seed 4/5</b> — G4 seed 5가 TorchScript 오류(OOM 재시도 후에도 실패)로 미완. 셀에 <span class='badge b-part'>4/5</span> 표기.</li>"
         "<li><b>세대 승수의 다중비교 비대칭</b> — G2는 3모델 중 best를 뽑는다. §4-B에서 대칭 로스터로 특징 축만 교정했으나, 세대 승수 자체는 이 비대칭을 안고 있다.</li>"
         "<li><b>G2·G3 하이퍼파라미터 탐색 없음</b>(독성과 동일 조건). G1(구조알림)은 미실행(독성 전용).</li>"
         "<li><b>전향적 검증 없음</b> — 전부 TDC 벤치마크 안의 회고적 비교다.</li></ul></div>")

# ── 10. 결론
H.append("<h2 id='s10'>10. 결론과 후속</h2>")
H.append("<div class='box'><p><b>1.</b> '세대 ≠ 성능'은 ADME에서도 성립하지만 <b>독성보다 약하다</b> — "
         "독성은 G2 17/18로 압도적이었고, ADME는 G2 " + str(win.get("G2", 0)) + "승으로 여전히 최다지만 갈린다.</p>"
         "<p><b>2.</b> <b>물성 표현이 지문을 이기는 것은 견고(17/18)하나 '압승/18/18'은 아니다</b> — "
         "P-gp·CYP2D6 억제·CYP3A4 기질에서 ECFP가 이긴다.</p>"
         "<p><b>3.</b> <b>통계적으로 확실한 세대 효과는 친유성에서 G3(GNN) 하나뿐</b>. 나머지는 대부분 구분 불가거나(소표본) "
         "미검증(G4·G5).</p>"
         "<p><b>4.</b> <b>파운데이션 모델(G5)은 독성·물성 두 영역 모두 0승</b> — 분자 수준 예측에서 자리를 못 잡는다.</p></div>")
H.append("<div class='box'><b>후속 과제</b> — "
         "(1) <b>G4·G5 분자별 예측 확보 후 부트스트랩</b>(E 헤드라인 검증의 전제) "
         "(2) GNN 아키텍처 확장(Attentive FP 등) "
         "(3) 분할 난이도 축(random vs scaffold) "
         "(4) 라벨 노이즈 강건 학습 "
         "(5) 문헌 기반 특징(CPSA·예측 녹는점·SoM 주입).</div>")
H.append("<p class='src'>산출물: <code>results/adme_matrix.csv</code>(축①) · <code>feature_ablation.csv</code>·"
         "<code>feature_2x2.csv</code>(축②) · <code>learning_axis.csv</code>(축③) · "
         "<code>bootstrap_verdict.csv</code>(통계 판정) · <code>split_leakage.json</code> · "
         "<code>predictions/*.jsonl</code>(G2·G3만) · <code>progress.jsonl</code>(원장) · <code>notes.md</code>. "
         "★새 학습·재계산 0 — 확정값 이관·정정 반영만.</p>")

open(f"{RES}/report_adme_full.html", "w", encoding="utf-8").write("\n".join(H))
print(f"저장 → results/report_adme_full.html ({os.path.getsize(f'{RES}/report_adme_full.html')/1024:.0f} KB)")
print(f"  세대 승수 {dict(sorted(win.items()))} · 대칭 로스터 XGB {nxgb}/18·RF {nrf}/18")
print(f"  부트스트랩 구분가능 {sum(v[1] for v in distinguishable.values())}/{sum(v[2] for v in distinguishable.values())}쌍")
print(f"  +phys 최대기여 {nphys}/18 · VDss G4 seed {g4n.get('vdss_lombardo')}/5")
