#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_g2.py — G2(고전 ML) 3종을 확장 3종 엔드포인트에 학습. env: admet · CPU.
★플래그십과 동일 3종·동일 하이퍼파라미터:
  - 물리화학 서술자(RDKit 210) + XGBoost   (config_public.json cls_params)
  - 물리화학 서술자(RDKit 210) + RandomForest(500 트리)
  - ECFP4(r=2, 2048bit) + XGBoost          (동일 cls_params)
★분할: prep_splits.py가 고정한 파일만 읽는다(test 동일성 구조적 보장). 분할 seed=42 고정.
★seed 1..5는 ★모델 seed만 변주(플래그십 4종은 train/valid 파티션까지 변주했다는 점이 다름 → notes.md 명시).
★결측 라벨(Tox21)은 학습·평가에서 제외(마스킹). train은 valid를 쓰지 않음(트리 모델·early stopping 없음).
★NaN/실패는 0점 금지 → status로 기록. resume: g2_raw.jsonl.
"""
import json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
RDLogger.DisableLog("rdApp.*")

ROOT = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_gen_expansion_g1g3"
SPL, RES = f"{ROOT}/splits", f"{ROOT}/results"
RAW = f"{RES}/g2_raw.jsonl"
CFG = json.load(open("/home/nudge/Project/ADMET/12_pipeline/config_public.json"))
XGB_P = dict(CFG["model"]["cls_params"])
SEEDS = [1, 2, 3, 4, 5]
NJOBS = 8                                        # ★과다구독 방지(16코어)
DESC = [n for n, _ in Descriptors._descList]     # 210
CALC = MolecularDescriptorCalculator(DESC)
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def feat_physchem(smis):
    X = np.full((len(smis), len(DESC)), np.nan)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            try:
                X[i] = CALC.CalcDescriptors(m)
            except Exception:
                pass
    return X


def feat_ecfp(smis):
    X = np.zeros((len(smis), 2048), dtype=np.float32)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            a = np.zeros((2048,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(GEN.GetFingerprint(m), a)
            X[i] = a
    return X


def clean(Xtr, Xte):
    """★플래그십 physchem_run.py 의 featurize+prep 와 ★동일: inf→nan → train 중앙값 대치 →
    StandardScaler(train fit). ★StandardScaler가 없으면 RDKit Ipc 같은 초대형 서술자가
    float32 캐스팅에서 inf가 되어 XGBoost/RF가 실패한다(실제로 겪음)."""
    Xtr = np.where(np.isfinite(Xtr), Xtr, np.nan)
    Xte = np.where(np.isfinite(Xte), Xte, np.nan)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)


meta = json.load(open(f"{RES}/split_meta.json"))
done = set()
if os.path.exists(RAW):
    for line in open(RAW):
        r = json.loads(line)
        done.add((r["model"], r["endpoint"], r["task"], r["seed"]))

CACHE = {}
for ep, m in meta.items():
    tr = pd.read_csv(f"{SPL}/{ep}_train.csv"); te = pd.read_csv(f"{SPL}/{ep}_test.csv")
    if ep not in CACHE:
        CACHE[ep] = dict(pc=clean(feat_physchem(tr["Drug"]), feat_physchem(te["Drug"])),
                         fp=(feat_ecfp(tr["Drug"]), feat_ecfp(te["Drug"])))
        print(f"[{ep}] 특징화 완료 physchem{CACHE[ep]['pc'][0].shape} ecfp{CACHE[ep]['fp'][0].shape}", flush=True)
    for lab in m["labels"]:
        col = "Y" if m["n_labels"] == 1 else lab
        ytr_all = pd.to_numeric(tr[col], errors="coerce").to_numpy(float)
        yte_all = pd.to_numeric(te[col], errors="coerce").to_numpy(float)
        mtr, mte = ~np.isnan(ytr_all), ~np.isnan(yte_all)
        ytr, yte = ytr_all[mtr].astype(int), yte_all[mte].astype(int)
        if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
            continue
        for mk, key in [("xgb_physchem", "pc"), ("rf_physchem", "pc"), ("xgb_ecfp", "fp")]:
            Xtr_f, Xte_f = CACHE[ep][key]
            Xtr, Xte = Xtr_f[mtr], Xte_f[mte]
            for s in SEEDS:
                if (mk, ep, lab, s) in done:
                    continue
                try:
                    if mk == "rf_physchem":
                        mdl = RandomForestClassifier(n_estimators=500, n_jobs=NJOBS, random_state=s)
                    else:
                        mdl = xgb.XGBClassifier(**XGB_P, random_state=s, n_jobs=NJOBS,
                                                eval_metric="logloss")
                    mdl.fit(Xtr, ytr)
                    pte = mdl.predict_proba(Xte)[:, 1]
                    ptr = mdl.predict_proba(Xtr)[:, 1]
                    if np.isnan(pte).any():
                        rec = dict(model=mk, endpoint=ep, task=lab, seed=s, status="failed_nan")
                    else:
                        rec = dict(model=mk, endpoint=ep, task=lab, seed=s, status="ok",
                                   n_train=int(mtr.sum()), n_test=int(mte.sum()),
                                   pos_rate_test=round(float(yte.mean()), 4),
                                   test=dict(AUROC=round(float(roc_auc_score(yte, pte)), 4),
                                             AUPRC=round(float(average_precision_score(yte, pte)), 4)),
                                   train=dict(AUROC=round(float(roc_auc_score(ytr, ptr)), 4),
                                              AUPRC=round(float(average_precision_score(ytr, ptr)), 4)))
                except Exception as e:
                    rec = dict(model=mk, endpoint=ep, task=lab, seed=s, status="failed",
                               err=str(e)[:300])
                with open(RAW, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [{ep}/{lab}] 3모델×5seed 완료", flush=True)
print("\nG2 완료 → results/g2_raw.jsonl")
