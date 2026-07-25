#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step1234_reliability.py — [1~4단계] AD · 운영/예측구간 · 보정 · 정확 CI. env: admet · ★재학습 0.
1) AD: ★학습셋 기준 5-NN Tanimoto(ECFP4). 컷=학습셋 자기 LOO 분포 5%(OOD)·25%(경계) 백분위(근거 명시).
   ★유효성 검증 — 구간별 성능. OOD서 떨어지면 유효 / 안 떨어지면 ★"예측 못함" / OOD<20이면 "판정 불가".
2) 분류: 임계 0.5 vs ★valid MCC-최대 t*(test 튜닝 금지) · 민감도/특이도/PPV/NPV/혼동행렬 · 양성률.
   회귀: 잔차 분포·계통편향·절대오차 상위 · ★valid 잔차 분위수로 분할 컨포멀 90% 구간 → test 실제 커버리지.
3) 보정: 분류 ECE(10-bin)+양성구간 별도 · 회귀는 구간 커버리지로 대체(별도 ECE 금지).
4) CI: 부트스트랩 2000회. ★주 판정 = seed별 부트스트랩 후 종합(방법 비교이므로 학습변동 포함).
   '5seed 예측평균'도 병기하되 ★앙상블 평가임을 각주. 분류는 DeLong(챔피언 vs dmpnn_ours).
산출: results/adme_reliability.csv · reliability_detail.json
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                             confusion_matrix, mean_absolute_error)
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

ADME = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_adme_full"
G4V = "/home/nudge/Project/ADMET_integrated/2026-07-24/experiment_g4_verification"
NEW = "/home/nudge/Project/ADMET_integrated/2026-07-25/experiment_adme_reliability"
sys.path.insert(0, f"{ADME}/src")
import common as C
from common import EPS

PRED = f"{NEW}/predictions"
NBOOT, K = 2000, 5
RNG = np.random.default_rng(20260725)
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
CH = {r["endpoint"]: r for r in json.load(open(f"{NEW}/results/champions.json"))}
CONF_FAIL = json.load(open(f"{G4V}/logs/conformer_failed.json")) if os.path.exists(f"{G4V}/logs/conformer_failed.json") else {}
hb = lambda ep: EPS[ep]["primary"] != "MAE"


def load(ep, model, split):
    """seed 평균 예측 + seed별 예측."""
    f = f"{PRED}/{ep}__{model}__{split}.jsonl"
    if not os.path.exists(f):
        f2 = f"{G4V}/predictions/{ep}__{model}__seed1.jsonl"     # G4는 seed별 파일
        if not os.path.exists(f2):
            return None, None
        rows = []
        for p in sorted(glob.glob(f"{G4V}/predictions/{ep}__{model}__seed*.jsonl")):
            rows.append(pd.read_json(p, lines=True))
        d = pd.concat(rows)
        d = d.rename(columns={"y_pred": "v"}) if "y_pred" in d.columns else d.rename(columns={"y_prob": "v"})
    else:
        d = pd.read_json(f, lines=True)
        d = d.rename(columns={"y_pred": "v"}) if "y_pred" in d.columns else d.rename(columns={"y_prob": "v"})
    mean = d.groupby("smiles").agg(y=("y_true", "first"), v=("v", "mean")).reset_index()
    per = {s: g.set_index("smiles") for s, g in d.groupby("seed")}
    return mean, per


def metric(y, p, prim):
    if prim == "AUROC":
        return roc_auc_score(y, p)
    if prim == "AUPRC":
        return average_precision_score(y, p)
    if prim == "MAE":
        return mean_absolute_error(y, p)
    return spearmanr(y, p).correlation


# ── 1) AD ──────────────────────────────────────────────────────────
def fps(smis):
    out, ok = [], []
    for s in smis:
        m = Chem.MolFromSmiles(str(s))
        ok.append(m is not None)
        if m is not None:
            out.append(MORGAN.GetFingerprint(m))
    return out, np.array(ok)


def knn(qf, rf, k=K, self_ex=False):
    r = np.zeros(len(qf))
    for i, q in enumerate(qf):
        s = np.array(DataStructs.BulkTanimotoSimilarity(q, rf))
        if self_ex:
            s[np.argmax(s)] = -1
        r[i] = np.sort(s)[-k:].mean()
    return r


def ad_analysis(ep, mean_df):
    g = C.group(); tr = g.get(ep)["train_val"]
    trf, _ = fps(tr["Drug"].tolist())
    tef, ok = fps(mean_df.smiles.tolist())
    loo = knn(trf, trf, self_ex=True)
    cut_ood, cut_bord = float(np.percentile(loo, 5)), float(np.percentile(loo, 25))
    sim = knn(tef, trf)
    band = np.where(sim < cut_ood, "OOD", np.where(sim < cut_bord, "경계", "in-domain"))
    y = mean_df.y.to_numpy(float)[ok]; v = mean_df.v.to_numpy(float)[ok]
    prim = EPS[ep]["primary"]
    bands = {}
    for bn in ("in-domain", "경계", "OOD"):
        m = band == bn
        e = dict(n=int(m.sum()), frac=round(float(m.mean()), 4), perf=None)
        if m.sum() >= 20 and (EPS[ep]["task"] == "reg" or len(np.unique(y[m])) == 2):
            try:
                e["perf"] = round(float(metric(y[m], v[m], prim)), 4)
            except Exception:
                pass
        bands[bn] = e
    pin, pood = bands["in-domain"]["perf"], bands["OOD"]["perf"]
    if bands["OOD"]["n"] < 20 or pin is None or pood is None:
        verdict = "판정 불가(OOD 표본<20)"
    else:
        worse = (pin - pood) > 0.02 if hb(ep) else (pood - pin) > 0.02
        verdict = "AD 유효(OOD서 성능 하락)" if worse else "★AD가 성능을 예측 못함"
    return dict(cut_ood=round(cut_ood, 4), cut_border=round(cut_bord, 4),
                cutoff_basis="학습셋 자기 5-NN Tanimoto 분포의 5%·25% 백분위(★절대선 아님)",
                n_train_ref=len(tr), bands=bands, verdict=verdict, band_array=band, ok_mask=ok)


# ── 2) 운영지표(분류) / 예측구간(회귀) ───────────────────────────
def op_cls(y, p, t):
    yh = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    f = lambda a, b: (float(a / b) if b else None)
    return dict(threshold=round(float(t), 4), TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
                sensitivity=(None if f(tp, tp + fn) is None else round(f(tp, tp + fn), 4)),
                specificity=(None if f(tn, tn + fp) is None else round(f(tn, tn + fp), 4)),
                PPV=(None if f(tp, tp + fp) is None else round(f(tp, tp + fp), 4)),
                NPV=(None if f(tn, tn + fn) is None else round(f(tn, tn + fn), 4)),
                MCC=round(float(matthews_corrcoef(y, yh)), 4))


def best_t(yv, pv):
    if len(np.unique(yv)) < 2:
        return 0.5, None
    ts = np.unique(np.round(pv, 3)); bt, bm = 0.5, -9
    for t in ts:
        m = matthews_corrcoef(yv, (pv >= t).astype(int))
        if m > bm:
            bm, bt = m, float(t)
    return round(bt, 4), round(float(bm), 4)


def ece_curve(y, p, nb=10):
    b = np.clip((p * nb).astype(int), 0, nb - 1)
    tot, rows = 0.0, []
    for i in range(nb):
        m = b == i
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        tot += m.mean() * abs(conf - acc)
        rows.append(dict(bin=i, n=int(m.sum()), conf=round(conf, 4), freq=round(acc, 4)))
    return round(float(tot), 4), rows


# ── 4) 부트스트랩 ────────────────────────────────────────────────
def boot_ci(y, p, prim, n=NBOOT):
    idx = np.arange(len(y)); vals = []
    for _ in range(n):
        s = RNG.choice(idx, len(idx), replace=True)
        if prim in ("AUROC", "AUPRC") and len(np.unique(y[s])) < 2:
            continue
        try:
            vals.append(metric(y[s], p[s], prim))
        except Exception:
            pass
    if len(vals) < 100:
        return None
    v = np.array(vals)
    return dict(mean=round(float(v.mean()), 4), lo=round(float(np.percentile(v, 2.5)), 4),
                hi=round(float(np.percentile(v, 97.5)), 4))


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong(p1, p2, y):
    order = np.argsort(-y); y = y[order]
    P = np.vstack([p1, p2])[:, order]
    m = int(y.sum()); n = len(y) - m
    if m < 2 or n < 2:
        return None
    pos, neg = P[:, :m], P[:, m:]
    tx = np.array([_midrank(pos[r]) for r in range(2)])
    ty = np.array([_midrank(neg[r]) for r in range(2)])
    tz = np.array([_midrank(P[r]) for r in range(2)])
    auc = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return None
    z = (auc[0] - auc[1]) / np.sqrt(var)
    return dict(auc_champ=round(float(auc[0]), 4), auc_other=round(float(auc[1]), 4),
                delta=round(float(auc[0] - auc[1]), 4), z=round(float(z), 3),
                p_value=float(f"{2*stats.norm.sf(abs(z)):.3g}"),
                significant=bool(2 * stats.norm.sf(abs(z)) < 0.05))


# ══════════════════ 실행 ══════════════════
def run_all():
  global rows, detail
  rows, detail = [], {}
  for ep, ch in CH.items():
    model = ch["champion"]
    mean_te, per_te = load(ep, model, "test")
    if mean_te is None:
        print(f"  {ep:<32} 예측 없음 — N/A(예측 미생성)"); continue
    prim, task = EPS[ep]["primary"], EPS[ep]["task"]
    y = mean_te.y.to_numpy(float); v = mean_te.v.to_numpy(float)
    D = dict(endpoint=ep, label=EPS[ep]["label"], pillar=EPS[ep]["pillar"], task=task,
             metric=prim, higher_better=hb(ep), champion=model, n_test=len(y))

    # 1) AD
    ad = ad_analysis(ep, mean_te)
    D["AD"] = {k: ad[k] for k in ("cut_ood", "cut_border", "cutoff_basis", "n_train_ref", "bands", "verdict")}

    # 4) 부트스트랩 — ★주 판정=seed별 후 종합
    per_ci = []
    for sd, g in (per_te or {}).items():
        gg = g.reset_index()
        c = boot_ci(gg.y_true.to_numpy(float), gg.v.to_numpy(float), prim)
        if c:
            per_ci.append(c)
    if per_ci:
        D["ci_primary"] = dict(method="seed별 부트스트랩 후 종합(★주 판정·학습변동 포함)",
                               mean=round(float(np.mean([c["mean"] for c in per_ci])), 4),
                               lo=round(float(np.mean([c["lo"] for c in per_ci])), 4),
                               hi=round(float(np.mean([c["hi"] for c in per_ci])), 4), n_seed=len(per_ci))
    D["ci_ensemble"] = boot_ci(y, v, prim)
    if D["ci_ensemble"]:
        D["ci_ensemble"]["method"] = "5seed 예측평균 부트스트랩(★5-모델 앙상블 평가·변동 큰 모델에 유리)"

    if task == "cls":
        # 2) 운영지표
        mean_va, _ = load(ep, model, "valid")
        t_star, mcc_v = (0.5, None)
        if mean_va is not None:
            t_star, mcc_v = best_t(mean_va.y.to_numpy(float), mean_va.v.to_numpy(float))
        D["operating"] = dict(pos_rate=round(float(y.mean()), 4),
                              t_fixed=op_cls(y, v, 0.5), t_star=op_cls(y, v, t_star),
                              t_star_source=f"valid MCC 최대(MCC={mcc_v}·n_valid={0 if mean_va is None else len(mean_va)})")
        # 3) 보정
        e, curve = ece_curve(y, v)
        pos = y == 1
        D["calibration"] = dict(ECE=e, curve=curve, mean_conf=round(float(v.mean()), 4),
                                obs_rate=round(float(y.mean()), 4),
                                direction=("과신" if v.mean() > y.mean() else "과소"),
                                mean_conf_on_positives=(round(float(v[pos].mean()), 4) if pos.any() else None),
                                note="보정은 후처리 → AUROC 등 랭킹지표 불변")
        # DeLong: 챔피언 vs dmpnn_ours
        mo, _ = load(ep, "dmpnn_ours", "test")
        if mo is None:
            old = f"{ADME}/predictions/{ep}__G3_dmpnn__test.jsonl"
            if os.path.exists(old):
                d3 = pd.read_json(old, lines=True)
                mo = d3.groupby("smiles").agg(y=("y_true", "first"), v=("y_prob", "mean")).reset_index()
        if mo is not None:
            mg = mean_te.merge(mo[["smiles", "v"]], on="smiles", suffixes=("", "_o"))
            if len(mg) >= 30:
                D["delong_vs_dmpnn"] = delong(mg.v.to_numpy(float), mg.v_o.to_numpy(float),
                                              mg.y.to_numpy(float))
    else:
        # 2) 회귀 — 잔차 + 컨포멀 예측구간
        res = y - v
        mean_va, _ = load(ep, model, "valid")
        q90 = cov = None
        if mean_va is not None:
            rv = np.abs(mean_va.y.to_numpy(float) - mean_va.v.to_numpy(float))
            q90 = float(np.quantile(rv, 0.90))
            cov = float((np.abs(res) <= q90).mean())
        slope = float(np.polyfit(v, res, 1)[0]) if len(v) > 5 else None
        top = mean_te.assign(abs_err=np.abs(res)).nlargest(5, "abs_err")
        D["regression"] = dict(
            residual_mean=round(float(res.mean()), 4), residual_sd=round(float(res.std()), 4),
            residual_q=[round(float(np.percentile(res, q)), 4) for q in (5, 25, 50, 75, 95)],
            systematic_bias_slope=(None if slope is None else round(slope, 4)),
            bias_note="예측값 대비 잔차 회귀 기울기. 0에서 멀수록 계통 편향(음수=큰 값 과소예측)",
            PI90_halfwidth=(None if q90 is None else round(q90, 4)),
            PI90_coverage=(None if cov is None else round(cov, 4)),
            PI_note="valid 절대잔차 90% 분위수로 만든 분할 컨포멀 구간 → test 실제 커버리지(★test 튜닝 없음)",
            worst_errors=[dict(smiles=r.smiles[:60], y_true=round(float(r.y), 3),
                               y_pred=round(float(r.v), 3), abs_err=round(float(r.abs_err), 3))
                          for _, r in top.iterrows()])
    detail[ep] = D
    # CSV 행
    ci = D.get("ci_primary") or D.get("ci_ensemble") or {}
    row = dict(endpoint=ep, label=EPS[ep]["label"], pillar=EPS[ep]["pillar"], task=task,
               metric=prim, direction=("↑" if hb(ep) else "↓"), champion=model, n_test=len(y),
               AD_in=ad["bands"]["in-domain"]["frac"], AD_border=ad["bands"]["경계"]["frac"],
               AD_ood=ad["bands"]["OOD"]["frac"], AD_verdict=ad["verdict"],
               value=ci.get("mean"), ci_lo=ci.get("lo"), ci_hi=ci.get("hi"),
               ci_method=ci.get("method", ""), source="predictions/*.jsonl(재현·추론만)")
    if task == "cls":
        o = D["operating"]
        row.update(t_star=o["t_star"]["threshold"], sens_05=o["t_fixed"]["sensitivity"],
                   sens_star=o["t_star"]["sensitivity"], FN_05=o["t_fixed"]["FN"],
                   FN_star=o["t_star"]["FN"], NPV_star=o["t_star"]["NPV"],
                   pos_rate=o["pos_rate"], ECE=D["calibration"]["ECE"],
                   delong_p=(D.get("delong_vs_dmpnn") or {}).get("p_value"))
    else:
        r = D["regression"]
        row.update(PI90_halfwidth=r["PI90_halfwidth"], PI90_coverage=r["PI90_coverage"],
                   residual_sd=r["residual_sd"], bias_slope=r["systematic_bias_slope"])
    rows.append(row)
    print(f"  {ep:<32}{task}  AD {ad['verdict'][:18]:<20}"
          + (f"t*={D['operating']['t_star']['threshold']:.3f} sens {D['operating']['t_fixed']['sensitivity']}→"
             f"{D['operating']['t_star']['sensitivity']} ECE {D['calibration']['ECE']}"
             if task == "cls" else
             f"PI90 커버리지 {D['regression']['PI90_coverage']} (목표 0.90) 편향 {D['regression']['systematic_bias_slope']}"))

  pd.DataFrame(rows).to_csv(f"{NEW}/results/adme_reliability.csv", index=False)
  json.dump(detail, open(f"{NEW}/results/reliability_detail.json", "w", encoding="utf-8"),
            ensure_ascii=False, indent=1, default=str)
  nv = sum(1 for d in detail.values() if "유효" in d["AD"]["verdict"])
  nn = sum(1 for d in detail.values() if "예측 못함" in d["AD"]["verdict"])
  nu = len(detail) - nv - nn
  print(f"\n★AD: 유효 {nv} · 예측못함 {nn} · 판정불가 {nu} (독성은 11/3/4)")
  print(f"저장 → results/adme_reliability.csv({len(rows)}행) · reliability_detail.json")


if __name__ == "__main__":
    run_all()
