#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_admetai.py — ★추론만(학습 아님). env: ADMET_AI.
확장 3종의 ★고정 test 분자(prep_splits.py 산출)에 ADMET-AI 적용 → 분자별 예측 + 과제별 지표.
★커버 확인: 해당 엔드포인트 컬럼이 ADMET-AI 출력에 없으면 '미커버'로 정직히 기록(억지 매핑 금지).
★누수: ADMET-AI는 TDC 전체로 사전학습 → 이 test 분자를 이미 봤을 가능성. 패키지 메타 크기도 함께 기록.
산출: results/admetai_preds.jsonl · results/admetai_metrics.json
"""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from admet_ai import ADMETModel

ROOT = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_gen_expansion_g1g3"
SPL, RES = f"{ROOT}/splits", f"{ROOT}/results"
AI_META = "/home/nudge/miniforge3/envs/ADMET_AI/lib/python3.11/site-packages/admet_ai/resources/data/admet.csv"
COLMAP = {"Carcinogens_Lagunin": {"Y": "Carcinogens_Lagunin"}, "ClinTox": {"Y": "ClinTox"}}

meta = json.load(open(f"{RES}/split_meta.json"))
ai_size = pd.read_csv(AI_META).set_index("id")["size"].to_dict()
model = ADMETModel()
recs, out = [], {}

for ep, m in meta.items():
    te = pd.read_csv(f"{SPL}/{ep}_test.csv")
    smis = [str(s) for s in te["Drug"]]
    pred = model.predict(smiles=smis).reset_index(drop=True)
    cmap = COLMAP.get(ep) or {t: t for t in m["labels"]}       # Tox21은 과제명 그대로
    res = {}
    for lab, aicol in cmap.items():
        col = "Y" if m["n_labels"] == 1 else lab
        if aicol not in pred.columns:
            res[lab] = dict(status="미커버", note=f"ADMET-AI 출력에 '{aicol}' 컬럼 없음")
            continue
        y = pd.to_numeric(te[col], errors="coerce").to_numpy(float)
        p = pd.to_numeric(pred[aicol], errors="coerce").to_numpy(float)
        msk = ~np.isnan(y) & ~np.isnan(p)
        for s, yy, pp in zip(np.array(smis)[msk], y[msk], p[msk]):
            recs.append(dict(endpoint=ep, task=lab, smiles=str(s), y=float(yy), admetai_pred=float(pp)))
        if msk.sum() < 20 or len(np.unique(y[msk])) < 2:
            res[lab] = dict(status="평가불가", n=int(msk.sum())); continue
        res[lab] = dict(status="ok", n=int(msk.sum()),
                        AUROC=round(float(roc_auc_score(y[msk], p[msk])), 4),
                        AUPRC=round(float(average_precision_score(y[msk], p[msk])), 4),
                        pos_rate=round(float(y[msk].mean()), 4),
                        admetai_train_size=ai_size.get(aicol))
    out[ep] = res
    okn = [v for v in res.values() if v.get("status") == "ok"]
    print(f"[{ep}] 커버 {len(okn)}/{len(res)} · 평균 AUROC "
          f"{np.mean([v['AUROC'] for v in okn]):.4f}" if okn else f"[{ep}] 커버 0", flush=True)

with open(f"{RES}/admetai_preds.jsonl", "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
out["_caveat"] = ("ADMET-AI 공개모델은 TDC 전체로 사전학습 → 이 test 분자를 학습에 포함했을 가능성이 매우 높음"
                  "(패키지 메타의 학습 size가 해당 TDC 데이터셋 전체 크기와 일치). ★누수 의심 값.")
json.dump(out, open(f"{RES}/admetai_metrics.json", "w"), ensure_ascii=False, indent=1)
print(f"\n저장 → admetai_preds.jsonl({len(recs)}행) · admetai_metrics.json")
