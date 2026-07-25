#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_dmpnn.py — ★G3 빈칸 메우기: 우리 자체 Chemprop D-MPNN을 seed=42 본 분할에 정직하게 학습.
env: admet (tdc/rdkit/sklearn) + chemprop 바이너리는 ADMET_AI env를 subprocess로 호출(교차 env).

★분할: 새로 만들지 않는다. 2026-06-27 실험들과 ★완전히 같은 호출을 재현한다 —
    g = admet_group(...); b = g.get(ep) → b["test"] (공식 고정 test)
    tr, va = g.get_train_valid_split(benchmark=ep, split_type="default", seed=s)   s=1..5
  그리고 본 보고서 test 분자 집합과 ★100% 일치(Jaccard=1.0)인지 실측 → 아니면 즉시 중단.
★순수 G3: RDKit 서술자 등 외부 특징 ★미사용(순수 D-MPNN).
★test는 마지막 1회 예측만. early stopping은 valid로만(chemprop이 -i의 2번째 파일을 val로 사용).
★설정: seed=1 런(T_toxicity/surface/src/models.py:g3_chemprop)과 동일 —
    epochs 50 · -b 50 · -n 0 · --class-balance(분류) · --pytorch-seed <seed> · metrics roc|mae
★NaN/실패는 0점이 아니라 status="failed"로 기록.
★resume: results/dmpnn_raw.jsonl 에 (endpoint,seed)별 1줄 append, 이미 있으면 skip.
"""
import os, sys, json, csv, shutil, subprocess, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from tdc.benchmark_group import admet_group

ROOT = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_g3_dmpnn_seed42"
RES, WORK = f"{ROOT}/results", f"{ROOT}/work"
REPORT_TEST = "/home/nudge/Project/ADMET_structure/2026-06-27/experiment_tox_benchmark/data"
TDC_DATA = "/home/nudge/Project/ADMET_structure/2026-06-27/experiment_tox_benchmark/src/tdc_data"
CHEMPROP = "/home/nudge/miniforge3/envs/ADMET_AI/bin/chemprop"
EPS = {"dili": "cls", "herg": "cls", "ames": "cls", "ld50_zhu": "reg"}
SEEDS = [1, 2, 3, 4, 5]
EPOCHS = 50
RAW = f"{RES}/dmpnn_raw.jsonl"


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m else None


def ikey(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToInchiKey(m)[:14] if m else None


def metrics(y, p, task):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if task == "cls":
        return dict(AUROC=round(float(roc_auc_score(y, p)), 4),
                    AUPRC=round(float(average_precision_score(y, p)), 4))
    return dict(MAE=round(float(mean_absolute_error(y, p)), 4))


def write_csv(df, path):
    df[["Drug", "Y"]].rename(columns={"Drug": "smiles", "Y": "y"}).to_csv(path, index=False)


def chemprop_predict(test_csv, ckpt, out_csv):
    r = subprocess.run([CHEMPROP, "predict", "-i", test_csv, "-s", "smiles",
                        "--model-paths", ckpt, "-o", out_csv, "-n", "0", "-q"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"predict 실패: {r.stderr[-400:]}")
    pdf = pd.read_csv(out_csv)
    col = [c for c in pdf.columns if c.startswith("pred")] or [c for c in pdf.columns if c != "smiles"]
    return pdf[col[0]].to_numpy(float)


def run_one(ep, task, seed, g):
    wd = f"{WORK}/{ep}_s{seed}"
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(wd, exist_ok=True)
    b = g.get(ep)
    test = b["test"].reset_index(drop=True)
    tr, va = g.get_train_valid_split(benchmark=ep, split_type="default", seed=seed)
    tr = tr.reset_index(drop=True); va = va.reset_index(drop=True)
    ftr, fva, fte = f"{wd}/train.csv", f"{wd}/val.csv", f"{wd}/test.csv"
    write_csv(tr, ftr); write_csv(va, fva); write_csv(test, fte)

    ttype = "classification" if task == "cls" else "regression"
    cmd = [CHEMPROP, "train", "-i", ftr, fva, fte, "-s", "smiles", "--target-columns", "y",
           "-t", ttype, "--epochs", str(EPOCHS), "--accelerator", "gpu", "--devices", "1",
           "-n", "0", "-b", "50", "--metrics", ("roc" if task == "cls" else "mae"),
           "--pytorch-seed", str(seed), "-o", f"{wd}/model", "-q"]
    if task == "cls":
        cmd.append("--class-balance")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return dict(endpoint=ep, seed=seed, status="failed_train", err=r.stderr[-600:])
    pts = sorted(os.path.join(rt, fn) for rt, _, fs in os.walk(f"{wd}/model") for fn in fs
                 if fn.endswith(".pt") and "best" in fn.lower()) or \
          sorted(os.path.join(rt, fn) for rt, _, fs in os.walk(f"{wd}/model") for fn in fs if fn.endswith(".pt"))
    if not pts:
        return dict(endpoint=ep, seed=seed, status="failed_nockpt")

    yte = test["Y"].to_numpy(float)
    pte = chemprop_predict(fte, pts[0], f"{wd}/pred_test.csv")
    ptr = chemprop_predict(ftr, pts[0], f"{wd}/pred_train.csv")
    if np.isnan(pte).any() or np.isnan(ptr).any():
        return dict(endpoint=ep, seed=seed, status="failed_nan")
    if np.mean(np.abs(pte - yte) < 1e-9) > 0.99:          # 라벨 복사 오염 가드
        return dict(endpoint=ep, seed=seed, status="failed_label_copy")

    rec = dict(endpoint=ep, task=task, seed=seed, status="ok",
               n_train=len(tr), n_valid=len(va), n_test=len(test),
               test=metrics(yte, pte, task),
               train=metrics(tr["Y"].to_numpy(float), ptr, task),
               sec=round(time.time() - t0, 1), ckpt=os.path.basename(pts[0]))
    shutil.rmtree(f"{wd}/model", ignore_errors=True)      # 체크포인트 정리(디스크)
    return rec


def main():
    os.makedirs(RES, exist_ok=True)
    g = admet_group(path=TDC_DATA)

    # ── ★분할 동일성 검증 (급소) ──────────────────────────────────
    ident = {}
    print("=" * 92)
    print("★분할 동일성 검증 — admet_group 공식 test vs 본 보고서 test (Jaccard=1.0 이어야 진행)")
    print("=" * 92)
    for ep in EPS:
        b = g.get(ep)
        mine = {canon(s) for s in b["test"]["Drug"]} - {None}
        rep = {canon(r["Drug"]) for r in csv.DictReader(open(f"{REPORT_TEST}/test_{ep}.csv"))} - {None}
        j = len(mine & rep) / len(mine | rep)
        tv = b["train_val"]
        tvk = set(filter(None, (ikey(s) for s in tv["Drug"])))
        tek = [ikey(s) for s in b["test"]["Drug"]]
        ov = sum(1 for k in tek if k and k in tvk)
        ident[ep] = dict(n_test_official=len(b["test"]), n_unique_canon=len(mine),
                         n_report=len(rep), intersection=len(mine & rep), jaccard=round(j, 6),
                         train_test_exact_overlap=ov, n_train_val=len(tv))
        print(f"  {ep:<10} official n={len(b['test']):<5} 고유 {len(mine):<5} 보고서 {len(rep):<5} "
              f"교집합 {len(mine&rep):<5} Jaccard={j:.4f}  train∩test 중복={ov}  "
              f"{'OK' if j == 1.0 else '★불일치'}")
    json.dump(ident, open(f"{RES}/leakage.json", "w"), ensure_ascii=False, indent=1)
    bad = [e for e, v in ident.items() if v["jaccard"] != 1.0]
    if bad:
        print(f"\n★중단 — test 동일성 100% 아님: {bad}")
        sys.exit(2)
    print("\n→ 4/4 엔드포인트 test 집합 100% 일치. 학습 진행.\n")

    done = set()
    if os.path.exists(RAW):
        for line in open(RAW):
            d = json.loads(line)
            done.add((d["endpoint"], d["seed"]))
    for ep, task in EPS.items():
        for s in SEEDS:
            if (ep, s) in done:
                print(f"  [{ep}/s{s}] skip(이미 있음)", flush=True); continue
            rec = run_one(ep, task, s, g)
            with open(RAW, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["status"] == "ok":
                print(f"  [{ep}/s{s}] test={rec['test']} (train {rec['train']}) {rec['sec']}s", flush=True)
            else:
                print(f"  [{ep}/s{s}] ★{rec['status']} — 0점 아님, 실패로 기록", flush=True)
    print("\n학습 완료 → results/dmpnn_raw.jsonl")


if __name__ == "__main__":
    main()
