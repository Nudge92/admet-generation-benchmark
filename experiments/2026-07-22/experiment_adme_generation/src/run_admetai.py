#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[1단계] ADMET-AI(GNN) 를 ★같은 고정 test★ 에 적용 (env: ADMET_AI).
data/test_<ep>.csv (bench_ourml이 저장한 TDC 공식 test) 의 SMILES로 예측 → 같은 지표.
★중요 한계(정직): ADMET-AI 공개모델은 TDC 전체 데이터로 사전학습 → 이 test 분자를 ★이미 봤을 가능성★
  (= ADMET-AI 점수는 낙관/누수). 공정 비교 기준은 SOTA 리더보드(scaffold split). 리포트에 명시.
출력: results/admetai_metrics.json
"""
import os, sys, json, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from admet_ai import ADMETModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")

EP2COL = {"caco2_wang": "Caco2_Wang", "bbb_martins": "BBB_Martins",
          "cyp2c9_veith": "CYP2C9_Veith", "solubility_aqsoldb": "Solubility_AqSolDB"}
TASK = {"caco2_wang": "reg", "bbb_martins": "cls", "cyp2c9_veith": "cls", "solubility_aqsoldb": "reg"}


def main():
    model = ADMETModel()
    out = {}
    for name, col in EP2COL.items():
        rows = list(csv.DictReader(open(os.path.join(DATA, f"test_{name}.csv"))))
        smis = [r["Drug"] for r in rows]
        y = np.array([float(r["Y"]) for r in rows])
        pred = model.predict(smiles=smis).reset_index(drop=True)
        p = pred[col].values.astype(float)
        if TASK[name] == "cls":
            out[name] = dict(task="cls",
                             AUROC=round(float(roc_auc_score(y, p)), 4),
                             AUPRC=round(float(average_precision_score(y, p)), 4),
                             n_test=len(y))
        else:
            out[name] = dict(task="reg",
                             MAE=round(float(mean_absolute_error(y, p)), 4),
                             n_test=len(y))
        print(f"  ADMET-AI {name:10s} {out[name]}", flush=True)
    out["_caveat"] = ("ADMET-AI 공개모델은 TDC 전체로 사전학습 → 이 test 분자 일부를 학습에 포함했을 수 "
                      "있음(낙관/누수 가능). 우리 모델은 train_val로만 학습(누수0). 공정기준=SOTA 리더보드.")
    json.dump(out, open(os.path.join(RES, "admetai_metrics.json"), "w"),
              ensure_ascii=False, indent=2)
    print("저장: results/admetai_metrics.json")


if __name__ == "__main__":
    main()
