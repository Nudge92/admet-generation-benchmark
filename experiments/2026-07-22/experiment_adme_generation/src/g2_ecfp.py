#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[1단계] 우리 모델(ECFP+XGBoost) TDC 독성 벤치마크 (env: admet).
★재구현 금지: EnsembleProfiler의 ECFP 특징화 + config XGBoost 하이퍼파라미터를 ★그대로 재사용.
★누수 금지: TDC admet_group ★공식 scaffold split. 고정 test는 마지막 한 번만 예측.
★5 seed: get_train_valid_split(seed=1..5) → 각 seed의 train 파티션으로 학습(데이터 변동 포함) → 고정 test 예측.
지표: 분류 AUROC+AUPRC, 회귀 MAE. 앙상블(5seed 평균예측)도 함께.
출력: results/ourml_metrics.json, data/test_<ep>.csv (smiles,y,our_pred — ADMET-AI/누수점검용)
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import spearmanr
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from rdkit import Chem, DataStructs, RDLogger
RDLogger.DisableLog("rdApp.*")
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import adme_common

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
os.makedirs(DATA, exist_ok=True); os.makedirs(RES, exist_ok=True)

# ---- 우리 모델 모듈 재사용: ECFP 특징화 + config 파라미터 ----
sys.path.insert(0, "/home/nudge/Project/ADMET/15_5_drugbank_profile")   # 심볼릭→2026-06-19
from profiler import EnsembleProfiler
PROF = EnsembleProfiler()                 # _gen(MorganGenerator r2/2048) + cfg 로드
CFG = PROF.cfg
N_BITS = PROF.n_bits
GEN = PROF._gen

ENDPOINTS = {"caco2_wang": "reg", "bbb_martins": "cls", "cyp2c9_veith": "cls", "solubility_aqsoldb": "reg"}
SEEDS = [1, 2, 3, 4, 5]


def ecfp(smiles_list):
    """리스트→(X[n,2048], valid_mask). 우리 모델과 동일 ECFP."""
    X = np.zeros((len(smiles_list), N_BITS), dtype=np.float32)
    mask = np.zeros(len(smiles_list), dtype=bool)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        arr = np.zeros((N_BITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(GEN.GetFingerprint(m), arr)
        X[i] = arr; mask[i] = True
    return X, mask


def ikey(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToInchiKey(m)[:14] if m else None


def main():
    g = adme_common.get_group(os.path.join(HERE, "tdc_data"))
    out = {}
    for name, task in ENDPOINTS.items():
        b = g.get(name)
        test = b["test"].reset_index(drop=True)
        train_val = b["train_val"].reset_index(drop=True)
        Xte, mte = ecfp(test["Drug"].tolist())
        yte = test["Y"].values.astype(float)
        assert mte.all(), f"{name}: test SMILES 파싱 실패 {(~mte).sum()}"

        # ---- 누수 점검: 고정 test 분자가 train_val 안에 있나 (exact molecule) ----
        tv_keys = set(filter(None, (ikey(s) for s in train_val["Drug"])))
        te_keys = [ikey(s) for s in test["Drug"]]
        overlap = sum(1 for k in te_keys if k and k in tv_keys)

        per_seed = []          # 각 seed별 지표
        preds_seed = []        # 각 seed의 test 예측 (앙상블용)
        cls_params = dict(CFG["model"]["cls_params"])
        reg_params = dict(CFG["model"]["reg_params"])
        for s in SEEDS:
            tr, va = g.get_train_valid_split(benchmark=name, split_type="default", seed=s)
            Xtr, mtr = ecfp(tr["Drug"].tolist())
            ytr = tr["Y"].values.astype(float)[mtr]; Xtr = Xtr[mtr]
            if task == "cls":
                model = xgb.XGBClassifier(**cls_params, random_state=s, n_jobs=-1,
                                          eval_metric="logloss")
                model.fit(Xtr, ytr.astype(int))
                p = model.predict_proba(Xte)[:, 1]
                m = dict(AUROC=round(float(roc_auc_score(yte, p)), 4),
                         AUPRC=round(float(average_precision_score(yte, p)), 4))
            else:
                model = xgb.XGBRegressor(**reg_params, random_state=s, n_jobs=-1)
                model.fit(Xtr, ytr)
                p = model.predict(Xte)
                m = dict(MAE=round(float(mean_absolute_error(yte, p)), 4),
                         Spearman=round(float(spearmanr(yte, p).correlation), 4))
            per_seed.append(m); preds_seed.append(p)
            print(f"  {name:10s} seed{s} {m}  (train={len(ytr)})", flush=True)

        # ---- 5seed 평균±SD ----
        keys = list(per_seed[0].keys())
        agg = {k: dict(mean=round(float(np.mean([d[k] for d in per_seed])), 4),
                       std=round(float(np.std([d[k] for d in per_seed])), 4),
                       seeds=[d[k] for d in per_seed]) for k in keys}
        # ---- 앙상블 예측(5seed 평균) ----
        ens = np.mean(preds_seed, axis=0)
        if task == "cls":
            ens_m = dict(AUROC=round(float(roc_auc_score(yte, ens)), 4),
                         AUPRC=round(float(average_precision_score(yte, ens)), 4))
        else:
            ens_m = dict(MAE=round(float(mean_absolute_error(yte, ens)), 4),
                         Spearman=round(float(spearmanr(yte, ens).correlation), 4))

        out[name] = dict(task=task, n_train_val=len(train_val), n_test=len(test),
                         pos_frac=round(float(train_val["Y"].mean()), 4) if task == "cls" else None,
                         leak_overlap=overlap, per_seed=agg, ensemble=ens_m)
        print(f"=> {name}: 5seed {agg} | ensemble {ens_m} | leak_overlap={overlap}\n")

        # ---- test 예측 저장 (ADMET-AI 비교 + 보관) ----
        import csv
        with open(os.path.join(DATA, f"test_{name}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["Drug", "Y", "our_pred"])
            for smi, y, pr in zip(test["Drug"], yte, ens):
                w.writerow([smi, y, round(float(pr), 5)])

    json.dump(out, open(os.path.join(RES, "ourml_metrics.json"), "w"),
              ensure_ascii=False, indent=2)
    print("저장: results/ourml_metrics.json, data/test_*.csv")


if __name__ == "__main__":
    main()
