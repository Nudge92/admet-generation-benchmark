#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
물리화학 descriptor + 트리 (env: admet, CPU). XGBoost(물리화학) + RandomForest(물리화학) × 4 엔드포인트 × 5 seed.
★공정: 기존과 동일 TDC scaffold split. train 파티션 학습 → 고정 test 1회. 5 seed.
★특징: RDKit 2D descriptor 210개. inf→nan, ★train 중앙값으로 대체(개수 기록). StandardScaler(train fit).
★XGB는 ECFP와 ★동일 하이퍼파라미터(특징만 교체=공정 비교). 누수=descriptor·scaler·median 전부 train만.
출력: results/physchem_raw.jsonl, results/feat_importance.json, results/impute_stats.json
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
RDLogger.DisableLog("rdApp.*")
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import adme_common

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results"); os.makedirs(RES, exist_ok=True)
TDC_DATA = "/home/nudge/Project/ADMET_structure/2026-06-27/experiment_tox_benchmark/src/tdc_data"
CFG = json.load(open("/home/nudge/Project/ADMET/12_pipeline/config_public.json"))
RAW = os.path.join(RES, "physchem_raw.jsonl")

DESC_NAMES = [n for n, _ in Descriptors._descList]      # 210
CALC = MolecularDescriptorCalculator(DESC_NAMES)
ENDPOINTS = {"caco2_wang": "reg", "bbb_martins": "cls", "cyp2c9_veith": "cls", "solubility_aqsoldb": "reg"}
SEEDS = [1, 2, 3, 4, 5]


def featurize(smiles):
    """SMILES 리스트 → (X[n,210], n_badcells). inf→nan. 파싱 실패행은 전부 nan."""
    X = np.full((len(smiles), len(DESC_NAMES)), np.nan, dtype=np.float64)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        try:
            X[i] = CALC.CalcDescriptors(m)
        except Exception:
            pass
    X[~np.isfinite(X)] = np.nan                          # inf/-inf → nan
    return X


def prep(Xtr, Xte):
    """train 중앙값 대체 + StandardScaler(train fit). 반환: 변환된 X + 대체셀수."""
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)           # 전체 nan 컬럼 → 0
    n_imp = int(np.isnan(Xtr).sum() + np.isnan(Xte).sum())
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    sc = StandardScaler().fit(Xtr)
    # ★float32 안전 클리핑: solubility test에 Ipc=6.08e158 인 분자가 있어(train 최대 9.9e60)
    #   train 통계로 표준화하면 5.1e99 가 되고 트리 모델이 float32로 캐스팅하며 inf가 된다.
    #   ±1e6 시그마 밖은 의미가 없으므로 잘라낸다. 실측상 caco2/bbb/cyp2c9의 표준화 최대는
    #   각각 25 / 1.6e4 / 3.0e4 라 ★이 클리핑은 그 세 엔드포인트에 no-op(값 불변)이다.
    CLIP = 1e6
    return (np.clip(sc.transform(Xtr), -CLIP, CLIP),
            np.clip(sc.transform(Xte), -CLIP, CLIP), n_imp)


def metrics(y, p, task):
    p = np.asarray(p).reshape(-1)
    if not np.all(np.isfinite(p)):
        return None
    if task == "cls":
        return dict(AUROC=float(roc_auc_score(y, p)), AUPRC=float(average_precision_score(y, p)))
    from scipy.stats import spearmanr
    return dict(MAE=float(mean_absolute_error(y, p)),
                Spearman=float(spearmanr(y, p).correlation))


def fit_predict(model, task, Xtr, ytr, Xte, seed):
    if model == "xgb":
        p = dict(CFG["model"]["cls_params" if task == "cls" else "reg_params"],
                 random_state=seed, n_jobs=-1)
        if task == "cls":
            m = xgb.XGBClassifier(**p, eval_metric="logloss"); m.fit(Xtr, ytr.astype(int))
            pred = m.predict_proba(Xte)[:, 1]; pred_tr = m.predict_proba(Xtr)[:, 1]
        else:
            m = xgb.XGBRegressor(**p); m.fit(Xtr, ytr)
            pred = m.predict(Xte); pred_tr = m.predict(Xtr)
    else:  # rf
        if task == "cls":
            m = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed); m.fit(Xtr, ytr.astype(int))
            pred = m.predict_proba(Xte)[:, 1]; pred_tr = m.predict_proba(Xtr)[:, 1]
        else:
            m = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=seed); m.fit(Xtr, ytr)
            pred = m.predict(Xte); pred_tr = m.predict(Xtr)
    return pred, pred_tr, m.feature_importances_


def done_keys():
    ks = set()
    if os.path.exists(RAW):
        for l in open(RAW):
            l = l.strip()
            if l:
                d = json.loads(l); ks.add((d["model"], d["endpoint"], d["seed"]))
    return ks


def main():
    g = adme_common.get_group(TDC_DATA)
    skip = done_keys()
    feat_imp = {}            # (model,ep) -> list of importance arrays
    impute = {}
    for ep, task in ENDPOINTS.items():
        b = g.get(ep); test = b["test"].reset_index(drop=True)
        Xte_raw = featurize(test["Drug"].tolist())
        yte = test["Y"].values.astype(float)
        for seed in SEEDS:
            tr, va = g.get_train_valid_split(benchmark=ep, split_type="default", seed=seed)
            tr = tr.reset_index(drop=True)
            Xtr_raw = featurize(tr["Drug"].tolist())
            ytr = tr["Y"].values.astype(float)
            Xtr, Xte, n_imp = prep(Xtr_raw.copy(), Xte_raw.copy())
            impute[f"{ep}_s{seed}"] = dict(n_imputed=n_imp,
                                           total_cells=int(Xtr_raw.size + Xte_raw.size),
                                           pct=round(100 * n_imp / (Xtr_raw.size + Xte_raw.size), 3))
            for model in ["xgb", "rf"]:
                if (f"{model}_physchem", ep, seed) in skip:
                    continue
                t0 = time.time()
                pred, pred_tr, imp = fit_predict(model, task, Xtr, ytr, Xte, seed)
                te = metrics(yte, pred, task); trm = metrics(ytr, pred_tr, task)
                feat_imp.setdefault((model, ep), []).append(imp)
                rec = dict(model=f"{model}_physchem", endpoint=ep, task=task, seed=seed,
                           status="ok", n_train=len(tr), n_test=len(test),
                           n_imputed=n_imp, train=trm, test=te, sec=round(time.time() - t0, 1))
                with open(RAW, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"  [{model}_physchem/{ep}/s{seed}] test={ {k: round(v,4) for k,v in te.items()} } "
                      f"(train { {k: round(v,4) for k,v in trm.items()} }) imp_cells={n_imp} {rec['sec']}s", flush=True)

    # feature importance 상위 10 (모델×엔드포인트, seed 평균)
    fi_out = {}
    for (model, ep), arrs in feat_imp.items():
        mean_imp = np.mean(arrs, axis=0)
        top = np.argsort(mean_imp)[::-1][:10]
        fi_out[f"{model}_{ep}"] = [[DESC_NAMES[i], round(float(mean_imp[i]), 4)] for i in top]
    json.dump(fi_out, open(os.path.join(RES, "feat_importance.json"), "w"), ensure_ascii=False, indent=2)
    json.dump(impute, open(os.path.join(RES, "impute_stats.json"), "w"), ensure_ascii=False, indent=2)
    print("저장: physchem_raw.jsonl, feat_importance.json, impute_stats.json")
    print(f"총 descriptor {len(DESC_NAMES)}개. 대체셀 비율 예시:",
          {k: v["pct"] for k, v in list(impute.items())[:3]})


if __name__ == "__main__":
    main()
