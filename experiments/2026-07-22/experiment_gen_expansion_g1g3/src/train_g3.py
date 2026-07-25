#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_g3.py — G3 정직한 Chemprop D-MPNN을 확장 3종에 학습. env: admet + chemprop 바이너리(ADMET_AI).
★플래그십(experiment_g3_dmpnn_seed42)과 ★동일 config:
  epochs 50 · -b 50 · -n 0 · --pytorch-seed <seed> · metrics roc · 분류 --class-balance · GPU
★순수 D-MPNN(외부 특징 없음) · train만 학습 · valid로만 early stopping · test는 마지막 1회 예측.
★Tox21은 ★멀티태스크(12 타깃 동시) — 결측 라벨은 CSV의 빈칸으로 두면 chemprop이 마스킹.
  (--class-balance는 멀티태스크에서 지원 안 될 수 있음 → 실패 시 자동으로 빼고 재시도하고 그 사실을 기록)
★분할은 prep_splits.py가 고정한 파일만 사용.
★실패는 0점 금지 → status 기록. resume: g3_raw.jsonl.
"""
import json, os, shutil, subprocess, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_gen_expansion_g1g3"
SPL, RES, WORK = f"{ROOT}/splits", f"{ROOT}/results", f"{ROOT}/work"
RAW = f"{RES}/g3_raw.jsonl"
CHEMPROP = "/home/nudge/miniforge3/envs/ADMET_AI/bin/chemprop"
SEEDS = [1, 2, 3, 4, 5]
EPOCHS = 50


def prep_csv(df, cols, path):
    d = df[["Drug"] + cols].rename(columns={"Drug": "smiles"})
    d.to_csv(path, index=False)


def predict(inp, ckpt, out):
    r = subprocess.run([CHEMPROP, "predict", "-i", inp, "-s", "smiles", "--model-paths", ckpt,
                        "-o", out, "-n", "0", "-q"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"predict 실패: {r.stderr[-400:]}")
    return pd.read_csv(out)


def per_task(dfp, df_true, cols):
    """예측 csv에서 과제별 AUROC/AUPRC (결측 제외)."""
    pcols = [c for c in dfp.columns if c != "smiles"]
    out = {}
    for i, t in enumerate(cols):
        y = pd.to_numeric(df_true[t], errors="coerce").to_numpy(float)
        p = pd.to_numeric(dfp[pcols[i]], errors="coerce").to_numpy(float)
        m = ~np.isnan(y) & ~np.isnan(p)
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            out[t] = None; continue
        out[t] = dict(AUROC=round(float(roc_auc_score(y[m], p[m])), 4),
                      AUPRC=round(float(average_precision_score(y[m], p[m])), 4),
                      n=int(m.sum()))
    return out


meta = json.load(open(f"{RES}/split_meta.json"))
done = set()
if os.path.exists(RAW):
    for line in open(RAW):
        r = json.loads(line)
        done.add((r["endpoint"], r["seed"]))

for ep, m in meta.items():
    cols = ["Y"] if m["n_labels"] == 1 else m["labels"]
    tr = pd.read_csv(f"{SPL}/{ep}_train.csv"); va = pd.read_csv(f"{SPL}/{ep}_valid.csv")
    te = pd.read_csv(f"{SPL}/{ep}_test.csv")
    for s in SEEDS:
        if (ep, s) in done:
            print(f"  [{ep}/s{s}] skip", flush=True); continue
        wd = f"{WORK}/{ep}_s{s}"
        shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd, exist_ok=True)
        ftr, fva, fte = f"{wd}/train.csv", f"{wd}/val.csv", f"{wd}/test.csv"
        prep_csv(tr, cols, ftr); prep_csv(va, cols, fva); prep_csv(te, cols, fte)
        base = [CHEMPROP, "train", "-i", ftr, fva, fte, "-s", "smiles",
                "--target-columns", *cols, "-t", "classification",
                "--epochs", str(EPOCHS), "--accelerator", "gpu", "--devices", "1",
                "-n", "0", "-b", "50", "--metrics", "roc", "--pytorch-seed", str(s),
                "-o", f"{wd}/model", "-q"]
        t0 = time.time()
        used_cb = True
        r = subprocess.run(base + ["--class-balance"], capture_output=True, text=True)
        if r.returncode != 0:                       # 멀티태스크에서 미지원일 수 있음 → 빼고 재시도
            r = subprocess.run(base, capture_output=True, text=True)
            used_cb = False
        if r.returncode != 0:
            rec = dict(endpoint=ep, seed=s, status="failed_train", err=r.stderr[-700:])
            with open(RAW, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  [{ep}/s{s}] ★failed_train", flush=True); continue
        pts = sorted(os.path.join(rt, fn) for rt, _, fs in os.walk(f"{wd}/model") for fn in fs
                     if fn.endswith(".pt") and "best" in fn.lower()) or \
              sorted(os.path.join(rt, fn) for rt, _, fs in os.walk(f"{wd}/model") for fn in fs
                     if fn.endswith(".pt"))
        if not pts:
            rec = dict(endpoint=ep, seed=s, status="failed_nockpt")
        else:
            try:
                pte = predict(fte, pts[0], f"{wd}/pred_test.csv")
                ptr = predict(ftr, pts[0], f"{wd}/pred_train.csv")
                rec = dict(endpoint=ep, seed=s, status="ok", class_balance=used_cb,
                           n_train=len(tr), n_valid=len(va), n_test=len(te),
                           test=per_task(pte, te, cols), train=per_task(ptr, tr, cols),
                           sec=round(time.time() - t0, 1))
            except Exception as e:
                rec = dict(endpoint=ep, seed=s, status="failed_predict", err=str(e)[:400])
        shutil.rmtree(f"{wd}/model", ignore_errors=True)
        with open(RAW, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if rec["status"] == "ok":
            sm = {k: (v["AUROC"] if v else None) for k, v in rec["test"].items()}
            avg = np.mean([v for v in sm.values() if v is not None])
            print(f"  [{ep}/s{s}] AUROC 평균 {avg:.4f} (class_balance={used_cb}) {rec['sec']}s", flush=True)
        else:
            print(f"  [{ep}/s{s}] ★{rec['status']}", flush=True)
print("\nG3 완료 → results/g3_raw.jsonl")
