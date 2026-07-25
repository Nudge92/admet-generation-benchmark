#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[2차] Uni-Mol (3D, unimol_tools 0.1.6, model_size 84m) 파인튜닝 (env: adme-bench, GPU).
★1차와 동일 TDC scaffold split·5 seed. seed별 train 파티션으로 fit(MolTrain 내부 val로 early stopping)→고정 test 예측.
★3D conformer = unimol_tools 내부 RDKit ETKDG(분자당 고정). batch 8(8GB OOM 회피). fp16(use_amp) — Uni-Mol은 안정.
★회귀는 target_normalize='auto'(내부 표준화·역변환) → 1차 회귀버그 회피.
★NaN/실패 = status=failed (0점 아님). resume: jsonl에 있으면 skip.
"""
import os, sys, json, time, shutil, warnings, logging
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for n in logging.root.manager.loggerDict:
    logging.getLogger(n).setLevel(logging.ERROR)
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import adme_common

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results"); os.makedirs(RES, exist_ok=True)
TDC_DATA = "/home/nudge/Project/ADMET_structure/2026-06-27/experiment_tox_benchmark/src/tdc_data"
RAW = os.path.join(RES, "finetune2_raw.jsonl")
WORK = "/tmp/unimol_work"
ENDPOINTS = {"caco2_wang": "reg", "hia_hou": "cls", "bioavailability_ma": "cls", "pgp_broccatelli": "cls", "lipophilicity_astrazeneca": "reg", "solubility_aqsoldb": "reg", "bbb_martins": "cls", "ppbr_az": "reg", "vdss_lombardo": "reg", "cyp2c9_veith": "cls", "cyp2d6_veith": "cls", "cyp3a4_veith": "cls", "cyp2c9_substrate_carbonmangels": "cls", "cyp2d6_substrate_carbonmangels": "cls", "cyp3a4_substrate_carbonmangels": "cls", "half_life_obach": "reg", "clearance_hepatocyte_az": "reg", "clearance_microsome_az": "reg"}
SEEDS = [1, 2, 3, 4, 5]
EPOCHS = 20; BATCH = 8; LR = 1e-4


def done_keys():
    ks = set()
    if os.path.exists(RAW):
        for l in open(RAW, encoding="utf-8"):
            l = l.strip()
            if l:
                d = json.loads(l)
                # ★status=ok 만 완료로 인정 — 실패(OOM 등)는 재시도 대상으로 남긴다
                if d.get("status") == "ok":
                    ks.add((d["model"], d["endpoint"], d["seed"]))
    return ks


def write_rec(rec):
    # ★encoding 필수 — reason 에 한글이 들어가면 로케일에 따라 UnicodeEncodeError 로 죽는다(실제로 겪음).
    with open(RAW, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def metrics(y, p, task):
    p = np.asarray(p).reshape(-1)
    if not np.all(np.isfinite(p)):
        return None
    if task == "cls":
        return dict(AUROC=float(roc_auc_score(y, p)), AUPRC=float(average_precision_score(y, p)))
    return dict(MAE=float(mean_absolute_error(y, p)),
                Spearman=float(spearmanr(y, p).correlation))


def run_one(ep, task, seed, g):
    from unimol_tools import MolTrain, MolPredict
    t0 = time.time()
    b = g.get(ep); test = b["test"].reset_index(drop=True)
    tr, va = g.get_train_valid_split(benchmark=ep, split_type="default", seed=seed)
    wd = os.path.join(WORK, f"{ep}_s{seed}")
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd, exist_ok=True)
    trcsv = os.path.join(wd, "train.csv"); tecsv = os.path.join(wd, "test.csv")
    exp = os.path.join(wd, "exp")
    tr.rename(columns={"Drug": "SMILES", "Y": "TARGET"})[["SMILES", "TARGET"]].to_csv(trcsv, index=False)
    test.rename(columns={"Drug": "SMILES", "Y": "TARGET"})[["SMILES", "TARGET"]].to_csv(tecsv, index=False)
    yte = test["Y"].values.astype(float)
    def _train(bs):
        clf = MolTrain(task=("classification" if task == "cls" else "regression"),
                       epochs=EPOCHS, batch_size=bs, learning_rate=LR,
                       early_stopping=(5 if task == "cls" else 3),   # ★회귀 patience↓
                       metrics=("auroc" if task == "cls" else "mae"),
                       kfold=1, split="scaffold", save_path=exp,
                       smiles_col="SMILES", target_cols="TARGET",
                       model_size="84m", use_cuda=True, use_amp=True,
                       target_normalize="auto", seed=seed)
        clf.fit(trcsv)
        mp = MolPredict(load_model=exp)
        return mp.predict(tecsv), mp.predict(trcsv)       # (test, train) — train은 과적합 점검용

    used_bs = BATCH
    try:
        pred, pred_tr = _train(BATCH)
    except Exception as e:
        # ★GPU OOM 이면 batch 를 절반으로 줄여 딱 1회 재시도(사용자 규격). 그래도 실패면 그 조합만 skip·기록.
        oom = ("out of memory" in str(e).lower() or "OutOfMemoryError" in type(e).__name__)
        if not oom:
            rec = dict(model="unimol", endpoint=ep, task=task, seed=seed, status="failed",
                       reason=str(e)[:200], sec=round(time.time() - t0, 1))
            write_rec(rec); print(f"  [unimol/{ep}/s{seed}] FAILED: {str(e)[:120]}", flush=True)
            shutil.rmtree(wd, ignore_errors=True); return rec
        try:
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass
        used_bs = max(2, BATCH // 2)
        print(f"  [unimol/{ep}/s{seed}] ★OOM → batch {BATCH}→{used_bs} 로 1회 재시도", flush=True)
        shutil.rmtree(exp, ignore_errors=True)
        try:
            pred, pred_tr = _train(used_bs)
        except Exception as e2:
            rec = dict(model="unimol", endpoint=ep, task=task, seed=seed, status="failed",
                       reason=f"OOM 재시도(batch={used_bs})도 실패: {str(e2)[:160]}",
                       oom_retry=True, sec=round(time.time() - t0, 1))
            write_rec(rec); print(f"  [unimol/{ep}/s{seed}] FAILED(재시도 후): {str(e2)[:100]}", flush=True)
            shutil.rmtree(wd, ignore_errors=True); return rec
    m = metrics(yte, pred, task)
    ytr = tr["Y"].values.astype(float)
    m_tr = metrics(ytr, pred_tr, task)
    if m is None:
        rec = dict(model="unimol", endpoint=ep, task=task, seed=seed, status="failed",
                   reason="예측 NaN", sec=round(time.time() - t0, 1))
        write_rec(rec); print(f"  [unimol/{ep}/s{seed}] FAILED NaN", flush=True)
        shutil.rmtree(wd, ignore_errors=True); return rec
    rec = dict(model="unimol", endpoint=ep, task=task, seed=seed, status="ok",
               n_train=len(tr), n_test=len(test), train=m_tr, test=m, batch_size=used_bs,
               oom_retry=(used_bs != BATCH), sec=round(time.time() - t0, 1))
    write_rec(rec)
    print(f"  [unimol/{ep}/s{seed}] test={ {k: round(v,4) for k,v in m.items()} } "
          f"train={ {k: round(v,4) for k,v in (m_tr or {}).items()} } {rec['sec']}s", flush=True)
    shutil.rmtree(wd, ignore_errors=True)
    return rec


def main():
    g = adme_common.get_group(TDC_DATA)
    skip = done_keys()
    print(f"이미 기록 {len(skip)}건 skip", flush=True)
    for ep, task in ENDPOINTS.items():
        for seed in SEEDS:
            if ("unimol", ep, seed) in skip:
                continue
            run_one(ep, task, seed, g)
    print("Uni-Mol 완료. results/finetune2_raw.jsonl", flush=True)


if __name__ == "__main__":
    main()
