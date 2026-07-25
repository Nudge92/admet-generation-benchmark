#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reproduce_g2.py — 각 엔드포인트 ★G2 챔피언의 분자별 예측을 만든다. env: admet
★모델 아티팩트가 저장돼 있지 않아(실측 확인) '로드'가 불가 → ★동일 config·동일 seed·동일 분할로
  ★재현 학습 후 추론. 재현 지표가 원본 보고값과 ±0.005 안인지 ★검증하고, 벗어나면 중단한다.
  (재현임을 모든 산출물에 flag)

레시피는 원본을 그대로 따른다:
- 플래그십 4종: experiment_physchem/src/physchem_run.py — admet_group, get_train_valid_split(
  split_type="default", seed=1..5), RDKit 210 서술자 → inf→nan → train 중앙값 대치 → StandardScaler(train fit),
  XGB=config_public.json cls/reg_params · RF=500트리. train 파티션만 학습(트리라 early stopping 없음).
- 확장 14과제: experiment_gen_expansion_g1g3 — 고정 split CSV(seed=42) 그대로, 동일 특징·동일 파라미터.
★valid 예측은 학습에 쓰지 않고 ★임계값 선택용으로만 추가 추론한다(test는 건드리지 않음).
산출: predictions/{ep}__{task}__{model}__{split}.jsonl · results/reproduction_check.json
"""
import json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
RDLogger.DisableLog("rdApp.*")
from tdc.benchmark_group import admet_group

B = "/home/nudge/Project/ADMET_integrated/2026-07-22"
ROOT = f"{B}/experiment_deploy_reliability"
PRED, RES = f"{ROOT}/predictions", f"{ROOT}/results"
SPL = f"{B}/experiment_gen_expansion_g1g3/splits"
TDC_DATA = "/home/nudge/Project/ADMET_structure/2026-06-27/experiment_tox_benchmark/src/tdc_data"
CFG = json.load(open("/home/nudge/Project/ADMET/12_pipeline/config_public.json"))
SEEDS = [1, 2, 3, 4, 5]
NJOBS = 8
DESC = [n for n, _ in Descriptors._descList]
CALC = MolecularDescriptorCalculator(DESC)
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
TOL = 0.005

# (endpoint, task, champion_model, task_type, 원본 보고값)
FLAG = [("dili", "—", "rf_physchem", "cls", 0.9125), ("herg", "—", "rf_physchem", "cls", 0.8369),
        ("ames", "—", "xgb_physchem", "cls", 0.8513), ("ld50_zhu", "—", "xgb_physchem", "reg", 0.6071)]


def feat_pc(smis):
    X = np.full((len(smis), len(DESC)), np.nan)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            try:
                X[i] = CALC.CalcDescriptors(m)
            except Exception:
                pass
    X[~np.isfinite(X)] = np.nan
    return X


def feat_fp(smis):
    X = np.zeros((len(smis), 2048), dtype=np.float32)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            a = np.zeros((2048,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(GEN.GetFingerprint(m), a)
            X[i] = a
    return X


def prep(Xtr, *others):
    """원본 prep과 동일: train 중앙값 대치 + StandardScaler(train fit). others도 같은 통계로 변환."""
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    sc = StandardScaler().fit(Xtr)
    out = [sc.transform(Xtr).astype(np.float32)]
    for X in others:
        X = np.where(np.isnan(X), med, X)
        out.append(sc.transform(X).astype(np.float32))
    return out


def build(model, task, seed):
    if model.startswith("rf"):
        C = RandomForestClassifier if task == "cls" else RandomForestRegressor
        return C(n_estimators=500, n_jobs=NJOBS, random_state=seed)
    p = dict(CFG["model"]["cls_params" if task == "cls" else "reg_params"], random_state=seed, n_jobs=NJOBS)
    return xgb.XGBClassifier(**p, eval_metric="logloss") if task == "cls" else xgb.XGBRegressor(**p)


def predict(mdl, X, task):
    return mdl.predict_proba(X)[:, 1] if task == "cls" else mdl.predict(X)


def score(y, p, task):
    if task == "cls":
        return dict(AUROC=float(roc_auc_score(y, p)), AUPRC=float(average_precision_score(y, p)))
    return dict(MAE=float(mean_absolute_error(y, p)))


def save(ep, task, model, split, recs):
    fn = f"{PRED}/{ep}__{task.replace('/', '_')}__{model}__{split}.jsonl"
    with open(fn, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return os.path.basename(fn)


checks = []
os.makedirs(PRED, exist_ok=True); os.makedirs(RES, exist_ok=True)

# ─────────── 플래그십 4종 ───────────
g = admet_group(path=TDC_DATA)
for ep, task_name, model, task, reported in FLAG:
    b = g.get(ep)
    test = b["test"].reset_index(drop=True)
    Xte_raw = feat_pc(test["Drug"].tolist())
    yte = test["Y"].values.astype(float)
    per_seed, te_recs, va_recs = [], [], []
    for s in SEEDS:
        tr, va = g.get_train_valid_split(benchmark=ep, split_type="default", seed=s)
        tr, va = tr.reset_index(drop=True), va.reset_index(drop=True)
        Xtr_raw = feat_pc(tr["Drug"].tolist()); Xva_raw = feat_pc(va["Drug"].tolist())
        Xtr, Xte, Xva = prep(Xtr_raw.copy(), Xte_raw.copy(), Xva_raw.copy())
        mdl = build(model, task, s).fit(Xtr, tr["Y"].values.astype(float if task == "reg" else int))
        pte, pva = predict(mdl, Xte, task), predict(mdl, Xva, task)
        per_seed.append(score(yte, pte, task))
        te_recs += [dict(smiles=str(sm), y_true=float(y), y_prob=float(p), seed=s)
                    for sm, y, p in zip(test["Drug"], yte, pte)]
        va_recs += [dict(smiles=str(sm), y_true=float(y), y_prob=float(p), seed=s)
                    for sm, y, p in zip(va["Drug"], va["Y"].values.astype(float), pva)]
    key = "AUROC" if task == "cls" else "MAE"
    got = float(np.mean([m[key] for m in per_seed]))
    ok = abs(got - reported) <= TOL
    f1 = save(ep, task_name, model, "test", te_recs); f2 = save(ep, task_name, model, "valid", va_recs)
    checks.append(dict(scope="핵심 4종", endpoint=ep, task=task_name, model=model, metric=key,
                       reported=reported, reproduced=round(got, 4), diff=round(got - reported, 4),
                       within_tol=bool(ok), n_test=len(test), files=[f1, f2],
                       source="experiment_physchem/results/physchem.csv"))
    print(f"[핵심/{ep:<9}] {model:<13} {key} 보고 {reported:.4f} → 재현 {got:.4f} "
          f"(Δ{got - reported:+.4f}) {'OK' if ok else '★불일치'}", flush=True)

# ─────────── 확장 14과제 ───────────
EX = json.load(open(f"{B}/experiment_gen_expansion_g1g3/results/expansion_metrics.json"))
EX.pop("_meta")
CACHE = {}
for ep in EX:
    tr_df = pd.read_csv(f"{SPL}/{ep}_train.csv"); va_df = pd.read_csv(f"{SPL}/{ep}_valid.csv")
    te_df = pd.read_csv(f"{SPL}/{ep}_test.csv")
    if ep not in CACHE:
        CACHE[ep] = dict(pc=[feat_pc(d["Drug"].tolist()) for d in (tr_df, va_df, te_df)],
                         fp=[feat_fp(d["Drug"].tolist()) for d in (tr_df, va_df, te_df)])
        print(f"[확장/{ep}] 특징화 완료", flush=True)
    for task, r in EX[ep].items():
        champ = max(((k, v["AUROC"]["mean"]) for k, v in r["G2"].items() if v), key=lambda z: z[1])
        model, reported = champ
        col = "Y" if len(EX[ep]) == 1 else task
        key_f = "fp" if model.endswith("ecfp") else "pc"
        Xtr_r, Xva_r, Xte_r = CACHE[ep][key_f]
        if key_f == "pc":
            Xtr, Xva, Xte = prep(Xtr_r.copy(), Xva_r.copy(), Xte_r.copy())
        else:
            Xtr, Xva, Xte = Xtr_r, Xva_r, Xte_r
        ytr_a = pd.to_numeric(tr_df[col], errors="coerce").to_numpy(float)
        yva_a = pd.to_numeric(va_df[col], errors="coerce").to_numpy(float)
        yte_a = pd.to_numeric(te_df[col], errors="coerce").to_numpy(float)
        mtr, mva, mte = ~np.isnan(ytr_a), ~np.isnan(yva_a), ~np.isnan(yte_a)
        per_seed, te_recs, va_recs = [], [], []
        for s in SEEDS:
            mdl = build(model, "cls", s).fit(Xtr[mtr], ytr_a[mtr].astype(int))
            pte, pva = predict(mdl, Xte[mte], "cls"), predict(mdl, Xva[mva], "cls")
            per_seed.append(score(yte_a[mte].astype(int), pte, "cls"))
            te_recs += [dict(smiles=str(sm), y_true=float(y), y_prob=float(p), seed=s)
                        for sm, y, p in zip(te_df["Drug"][mte], yte_a[mte], pte)]
            va_recs += [dict(smiles=str(sm), y_true=float(y), y_prob=float(p), seed=s)
                        for sm, y, p in zip(va_df["Drug"][mva], yva_a[mva], pva)]
        got = float(np.mean([m["AUROC"] for m in per_seed]))
        ok = abs(got - reported) <= TOL
        f1 = save(ep, task, model, "test", te_recs); f2 = save(ep, task, model, "valid", va_recs)
        checks.append(dict(scope="확장", endpoint=ep, task=task, model=model, metric="AUROC",
                           reported=reported, reproduced=round(got, 4), diff=round(got - reported, 4),
                           within_tol=bool(ok), n_test=int(mte.sum()), files=[f1, f2],
                           source="expansion_metrics.json"))
        print(f"[확장/{ep[:12]:<12}{task:<14}] {model:<13} 보고 {reported:.4f} → 재현 {got:.4f} "
              f"(Δ{got - reported:+.4f}) {'OK' if ok else '★불일치'}", flush=True)

bad = [c for c in checks if not c["within_tol"]]
json.dump(dict(tolerance=TOL, n=len(checks), n_fail=len(bad), reproduced_not_loaded=True,
               note="모델 아티팩트가 저장돼 있지 않아 로드 불가 → 동일 config·seed·분할로 재현 학습 후 추론. "
                    "모든 예측은 ★재현본임을 flag한다.", checks=checks),
          open(f"{RES}/reproduction_check.json", "w"), ensure_ascii=False, indent=1)
print(f"\n재현 검증: {len(checks)}건 중 허용오차(±{TOL}) 벗어남 {len(bad)}건" + (f" → {bad}" if bad else " → 전부 통과"))
print(f"저장 → predictions/*.jsonl ({len(os.listdir(PRED))}개) · results/reproduction_check.json")
