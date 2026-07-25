#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reliability.py — 배포 신뢰도 4종. env: admet · ★새 학습 0(저장된 분자별 예측만 사용).
1) AD  : ★학습셋 기준 5-NN Tanimoto(ECFP4). 컷오프는 ★학습셋 자기 LOO 분포의 백분위(절대선 아님).
         구간별 성능을 재서 ★AD가 신뢰도를 실제로 예측하는지 검증(못 하면 그대로 보고).
2) 운영: 임계 0.5 + ★valid에서 MCC 최대인 t*(test 튜닝 금지). 민감도·특이도·PPV·NPV·혼동행렬·FN.
3) 보정: ECE(10-bin)+신뢰도 곡선. 불균형이라 양성구간 별도. 회귀는 예측구간 커버리지로 대체.
4) CI  : 부트스트랩 2000회(분자 재표집) + ★DeLong 대응비교(챔피언 vs D-MPNN / vs ADMET-AI[누수]).
산출: results/reliability.csv · results/reliability.json
"""
import glob, json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                             confusion_matrix, mean_absolute_error)
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

B = "/home/nudge/Project/ADMET_integrated/2026-07-22"
ROOT = f"{B}/experiment_deploy_reliability"
PRED, RES = f"{ROOT}/predictions", f"{ROOT}/results"
SPL = f"{B}/experiment_gen_expansion_g1g3/splits"
TDC_DATA = "/home/nudge/Project/ADMET_structure/2026-06-27/experiment_tox_benchmark/src/tdc_data"
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
NBOOT, K = 2000, 5
RNG = np.random.default_rng(20260722)
FLAG_EPS = {"dili": "cls", "herg": "cls", "ames": "cls", "ld50_zhu": "reg"}


# ───────── DeLong (Sun & Xu fast algorithm) ─────────
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, float); T2[J] = T
    return T2


def delong_var(preds, y):
    """preds: [m, n] (m 모델), y: 0/1. 반환 (auc[m], cov[m,m])."""
    order = np.argsort(-y)
    y = y[order]; preds = preds[:, order]
    m = int(y.sum()); n = len(y) - m
    pos, neg = preds[:, :m], preds[:, m:]
    k = preds.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds[r]) for r in range(k)])
    auc = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s01 = np.cov(v01); s10 = np.cov(v10)
    if k == 1:
        s01 = np.array([[float(s01)]]); s10 = np.array([[float(s10)]])
    return auc, s01 / m + s10 / n


def delong_test(p1, p2, y):
    auc, cov = delong_var(np.vstack([p1, p2]), y)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(auc[0]), float(auc[1]), None, None
    z = (auc[0] - auc[1]) / np.sqrt(var)
    return float(auc[0]), float(auc[1]), float(z), float(2 * stats.norm.sf(abs(z)))


# ───────── 유틸 ─────────
def fp(smis):
    out, ok = [], []
    for s in smis:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            ok.append(False); continue
        out.append(GEN.GetFingerprint(m)); ok.append(True)
    return out, np.array(ok)


def knn_sim(qfps, ref_fps, k=K, exclude_self=False):
    """각 query의 ref 대비 상위 k Tanimoto 평균."""
    res = np.zeros(len(qfps))
    for i, q in enumerate(qfps):
        s = np.array(DataStructs.BulkTanimotoSimilarity(q, ref_fps))
        if exclude_self:
            s[np.argmax(s)] = -1                     # 자기 자신 제외
        res[i] = np.sort(s)[-k:].mean()
    return res


def load_pred(ep, task, model, split):
    f = f"{PRED}/{ep}__{task.replace('/', '_')}__{model}__{split}.jsonl"
    if not os.path.exists(f):
        return None
    d = pd.read_json(f, lines=True)
    g = d.groupby("smiles").agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"),
                                n_seed=("seed", "nunique")).reset_index()
    return g


def ece(y, p, nbin=10):
    b = np.clip((p * nbin).astype(int), 0, nbin - 1)
    tot, rows = 0.0, []
    for i in range(nbin):
        m = b == i
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        tot += m.mean() * abs(conf - acc)
        rows.append(dict(bin=i, lo=round(i / nbin, 2), hi=round((i + 1) / nbin, 2),
                         n=int(m.sum()), conf=round(conf, 4), freq=round(acc, 4)))
    return float(tot), rows


def op_metrics(y, p, t):
    yh = (p >= t).astype(int)
    tn, fp_, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp_) if (tn + fp_) else None
    ppv = tp / (tp + fp_) if (tp + fp_) else None
    npv = tn / (tn + fn) if (tn + fn) else None
    return dict(threshold=round(float(t), 4), TP=int(tp), FP=int(fp_), FN=int(fn), TN=int(tn),
                sensitivity=None if sens is None else round(float(sens), 4),
                specificity=None if spec is None else round(float(spec), 4),
                PPV=None if ppv is None else round(float(ppv), 4),
                NPV=None if npv is None else round(float(npv), 4),
                MCC=round(float(matthews_corrcoef(y, yh)), 4))


def boot_ci(y, p, fn, n=NBOOT):
    idx = np.arange(len(y)); vals = []
    for _ in range(n):
        s = RNG.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        try:
            vals.append(fn(y[s], p[s]))
        except Exception:
            pass
    if len(vals) < 100:
        return None, None, None
    v = np.array(vals)
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ───────── 학습셋(AD 기준) 로딩 ─────────
def train_smiles(ep, task):
    if ep in FLAG_EPS:
        from tdc.benchmark_group import admet_group
        g = admet_group(path=TDC_DATA)
        return [str(s) for s in g.get(ep)["train_val"]["Drug"]], \
               "train_val(5 seed의 train 파티션 합집합 ≈ 배포 앙상블이 본 전체)"
    d = pd.read_csv(f"{SPL}/{ep}_train.csv")
    col = "Y" if "Y" in d.columns else task
    y = pd.to_numeric(d[col], errors="coerce")
    return [str(s) for s in d["Drug"][y.notna()]], "고정 train 파티션(해당 과제 라벨 있는 분자만)"


CHK = json.load(open(f"{RES}/reproduction_check.json"))["checks"]
rows, detail = [], {}
for c in CHK:
    ep, task, model = c["endpoint"], c["task"], c["model"]
    is_reg = (ep == "ld50_zhu")
    te = load_pred(ep, task, model, "test")
    va = load_pred(ep, task, model, "valid")
    if te is None:
        continue
    y, p = te.y_true.to_numpy(float), te.y_prob.to_numpy(float)
    key = f"{ep}|{task}"
    D = dict(scope=c["scope"], endpoint=ep, task=task, model=model, n_test=len(y),
             reproduced=True, reported=c["reported"], reproduced_value=c["reproduced"])

    # ── 1) AD ──
    tr_smi, tr_note = train_smiles(ep, task)
    tr_fps, _ = fp(tr_smi)
    te_fps, ok = fp(te.smiles.tolist())
    loo = knn_sim(tr_fps, tr_fps, exclude_self=True)          # 학습셋 자기 분포
    cut_ood, cut_bord = float(np.percentile(loo, 5)), float(np.percentile(loo, 25))
    sim = knn_sim(te_fps, tr_fps)
    yv, pv = y[ok], p[ok]
    band = np.where(sim < cut_ood, "OOD", np.where(sim < cut_bord, "경계", "in-domain"))
    ad = dict(cut_ood=round(cut_ood, 4), cut_borderline=round(cut_bord, 4),
              cutoff_basis="학습셋 자기 5-NN Tanimoto 분포의 5%·25% 백분위(★절대선 아님)",
              train_ref=tr_note, n_train=len(tr_smi), sim_mean=round(float(sim.mean()), 4),
              bands={})
    for b in ("in-domain", "경계", "OOD"):
        m = band == b
        e = dict(n=int(m.sum()), frac=round(float(m.mean()), 4))
        if m.sum() >= 20 and (is_reg or len(np.unique(yv[m])) == 2):
            e["perf"] = (round(float(mean_absolute_error(yv[m], pv[m])), 4) if is_reg
                         else round(float(roc_auc_score(yv[m], pv[m])), 4))
        else:
            e["perf"] = None
            e["note"] = "표본/클래스 부족으로 산출 불가"
        ad["bands"][b] = e
    pin, pood = ad["bands"]["in-domain"]["perf"], ad["bands"]["OOD"]["perf"]
    if pin is None or pood is None:
        ad["verdict"] = "판정 불가(구간 표본 부족)"
    elif (pin - pood) > 0.02 if not is_reg else (pood - pin) > 0.02:
        ad["verdict"] = "★AD 유효(OOD에서 실제로 성능 하락)"
    else:
        ad["verdict"] = "★AD가 신뢰도를 예측하지 못함(OOD에서 성능이 떨어지지 않음)"
    D["AD"] = ad

    if is_reg:
        # ── 회귀: 잔차 + 예측구간 커버리지(valid 잔차 90% 분위 → test 커버리지) ──
        res = y - p
        q = float(np.quantile(np.abs(va.y_true - va.y_prob), 0.90)) if va is not None else None
        cov = None if q is None else float((np.abs(res) <= q).mean())
        mae, lo, hi = boot_ci(y, p, lambda a, b: mean_absolute_error(a, b))
        D["regression"] = dict(MAE=round(float(mean_absolute_error(y, p)), 4),
                               MAE_CI95=[None if lo is None else round(lo, 4),
                                         None if hi is None else round(hi, 4)],
                               residual_mean=round(float(res.mean()), 4),
                               residual_sd=round(float(res.std()), 4),
                               PI90_halfwidth=None if q is None else round(q, 4),
                               PI90_coverage=None if cov is None else round(cov, 4),
                               PI_note="valid 잔차의 90% 분위로 구간을 만들고 test에서 실제 커버리지 측정"
                                       "(등각예측 방식·test 튜닝 없음)")
    else:
        # ── 2) 운영지표 ──
        tstar, best = 0.5, -9
        if va is not None:
            yva, pva = va.y_true.to_numpy(float), va.y_prob.to_numpy(float)
            if len(np.unique(yva)) == 2:
                for t in np.unique(np.round(pva, 3)):
                    mc = matthews_corrcoef(yva, (pva >= t).astype(int))
                    if mc > best:
                        best, tstar = mc, float(t)
        D["operating"] = dict(t_fixed=op_metrics(y, p, 0.5), t_star=op_metrics(y, p, tstar),
                              t_star_source=f"valid에서 MCC 최대(MCC={best:.4f}, n_valid={0 if va is None else len(va)})",
                              pos_rate=round(float(y.mean()), 4))
        # ── 3) 보정 ──
        e, curve = ece(y, p)
        mpos = y == 1
        D["calibration"] = dict(ECE=round(e, 4), curve=curve,
                                mean_conf=round(float(p.mean()), 4), obs_rate=round(float(y.mean()), 4),
                                direction=("과신(예측확률>실제빈도)" if p.mean() > y.mean() else "과소(예측확률<실제빈도)"),
                                mean_conf_on_positives=round(float(p[mpos].mean()), 4) if mpos.any() else None,
                                note="보정은 후처리라 AUROC 등 랭킹 지표는 불변")
        # ── 4) CI ──
        a, alo, ahi = boot_ci(y, p, roc_auc_score)
        pr, plo, phi = boot_ci(y, p, average_precision_score)
        D["ci"] = dict(AUROC=round(float(roc_auc_score(y, p)), 4),
                       AUROC_CI95=[None if alo is None else round(alo, 4), None if ahi is None else round(ahi, 4)],
                       AUPRC=round(float(average_precision_score(y, p)), 4),
                       AUPRC_CI95=[None if plo is None else round(plo, 4), None if phi is None else round(phi, 4)],
                       n_boot=NBOOT)
        # ── DeLong 대응비교 ──
        cmp_ = {}
        for other, lab, leak in [("dmpnn_ours", "G3 정직 D-MPNN", False), ("admetai", "ADMET-AI", True)]:
            o = load_pred(ep, task, other, "test")
            if o is None:
                cmp_[other] = dict(status="예측 없음"); continue
            mg = te.merge(o[["smiles", "y_prob"]], on="smiles", suffixes=("", "_o"))
            if len(mg) < 30 or len(np.unique(mg.y_true)) < 2:
                cmp_[other] = dict(status=f"정렬 {len(mg)}분자 — 비교 불가"); continue
            a1, a2, z, pv_ = delong_test(mg.y_prob.to_numpy(float), mg.y_prob_o.to_numpy(float),
                                         mg.y_true.to_numpy(float))
            cmp_[other] = dict(label=lab, n_aligned=int(len(mg)), auc_champion=round(a1, 4),
                               auc_other=round(a2, 4), delta=round(a1 - a2, 4),
                               z=None if z is None else round(z, 3),
                               p_value=None if pv_ is None else float(f"{pv_:.3g}"),
                               significant=(None if pv_ is None else bool(pv_ < 0.05)),
                               leak_flag=("★누수 값과의 비교 — 정확 CI라도 실력 비교가 아님" if leak else "누수 없음"))
        D["delong"] = cmp_

    detail[key] = D
    r = dict(scope=c["scope"], endpoint=ep, task=task, model=model, n_test=len(y),
             reported=c["reported"], reproduced=c["reproduced"],
             AD_in=ad["bands"]["in-domain"]["frac"], AD_border=ad["bands"]["경계"]["frac"],
             AD_ood=ad["bands"]["OOD"]["frac"],
             AD_perf_in=ad["bands"]["in-domain"]["perf"], AD_perf_ood=ad["bands"]["OOD"]["perf"],
             AD_verdict=ad["verdict"])
    if is_reg:
        r.update(metric="MAE", value=D["regression"]["MAE"], ci_lo=D["regression"]["MAE_CI95"][0],
                 ci_hi=D["regression"]["MAE_CI95"][1], PI90_coverage=D["regression"]["PI90_coverage"])
    else:
        r.update(metric="AUROC", value=D["ci"]["AUROC"], ci_lo=D["ci"]["AUROC_CI95"][0],
                 ci_hi=D["ci"]["AUROC_CI95"][1], ECE=D["calibration"]["ECE"],
                 sens_05=D["operating"]["t_fixed"]["sensitivity"], spec_05=D["operating"]["t_fixed"]["specificity"],
                 NPV_05=D["operating"]["t_fixed"]["NPV"], FN_05=D["operating"]["t_fixed"]["FN"],
                 t_star=D["operating"]["t_star"]["threshold"], sens_star=D["operating"]["t_star"]["sensitivity"],
                 FN_star=D["operating"]["t_star"]["FN"],
                 delong_vs_dmpnn_p=(D["delong"].get("dmpnn_ours") or {}).get("p_value"),
                 delong_vs_admetai_p=(D["delong"].get("admetai") or {}).get("p_value"))
    r["source"] = "predictions/*.jsonl (재현 학습본)"
    rows.append(r)
    print(f"[{ep}/{task}] AD in {ad['bands']['in-domain']['frac']:.2f}·OOD {ad['bands']['OOD']['frac']:.2f} "
          f"→ {ad['verdict'][:22]} | " +
          (f"MAE {D['regression']['MAE']:.4f}" if is_reg else
           f"AUROC {D['ci']['AUROC']:.4f} [{D['ci']['AUROC_CI95'][0]:.3f},{D['ci']['AUROC_CI95'][1]:.3f}] "
           f"ECE {D['calibration']['ECE']:.3f} FN@0.5 {D['operating']['t_fixed']['FN']}"), flush=True)

pd.DataFrame(rows).to_csv(f"{RES}/reliability.csv", index=False)
json.dump(detail, open(f"{RES}/reliability.json", "w"), ensure_ascii=False, indent=1)
print(f"\n저장 → reliability.csv({len(rows)}행) · reliability.json")
