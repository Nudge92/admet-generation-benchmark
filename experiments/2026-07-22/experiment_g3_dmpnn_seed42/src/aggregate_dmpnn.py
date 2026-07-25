#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aggregate_dmpnn.py — dmpnn_raw.jsonl → dmpnn_metrics.json · overfit.json · leakage.json(정정판).
★leakage.json 정정: train_dmpnn.py가 쓴 'train_test_exact_overlap'은 실제로 InChIKey14(골격) 중복이었다.
  여기서 ★정확분자(canonical SMILES) 중복과 ★골격(InChIKey14) 중복을 ★따로 계산해 이름을 바로잡는다
  (본 보고서 leakage.json의 exact_canonical_overlap / inchikey14_overlap 정의와 일치시킴).
★실패(status != ok)는 0점으로 넣지 않고 failed 목록에만 기록.
"""
import json, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from tdc.benchmark_group import admet_group

ROOT = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_g3_dmpnn_seed42"
RES = f"{ROOT}/results"
REPORT_TEST = "/home/nudge/Project/ADMET_structure/2026-06-27/experiment_tox_benchmark/data"
TDC_DATA = "/home/nudge/Project/ADMET_structure/2026-06-27/experiment_tox_benchmark/src/tdc_data"
EPS = {"dili": "cls", "herg": "cls", "ames": "cls", "ld50_zhu": "reg"}


def canon(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None


def ik_full(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToInchiKey(m) if m else None


def ik14(s):
    k = ik_full(s); return k[:14] if k else None


recs = [json.loads(l) for l in open(f"{RES}/dmpnn_raw.jsonl")]
ok = [r for r in recs if r["status"] == "ok"]
failed = [dict(endpoint=r["endpoint"], seed=r["seed"], status=r["status"]) for r in recs if r["status"] != "ok"]

met, overfit = {}, []
for ep, task in EPS.items():
    rs = [r for r in ok if r["endpoint"] == ep]
    if not rs:
        met[ep] = dict(task=task, status="failed_all", n_ok=0); continue
    keys = list(rs[0]["test"].keys())
    agg = {}
    for k in keys:
        te = [r["test"][k] for r in rs]; tr = [r["train"][k] for r in rs]
        agg[k] = dict(test_mean=round(float(np.mean(te)), 4), test_std=round(float(np.std(te)), 4),
                      test_seeds=te,
                      train_mean=round(float(np.mean(tr)), 4), train_std=round(float(np.std(tr)), 4))
        gap = (np.mean(tr) - np.mean(te)) if k != "MAE" else (np.mean(te) - np.mean(tr))
        overfit.append(dict(model="dmpnn_ours", endpoint=ep, metric=k,
                            train=round(float(np.mean(tr)), 4), test=round(float(np.mean(te)), 4),
                            gap=round(float(gap), 4), n_seed=len(rs)))
    met[ep] = dict(task=task, n_ok=len(rs), n_train=rs[0]["n_train"], n_valid=rs[0]["n_valid"],
                   n_test=rs[0]["n_test"], seconds_mean=round(float(np.mean([r["sec"] for r in rs])), 1),
                   **agg)

met["_config"] = dict(
    model="Chemprop D-MPNN (v2 CLI, 순수 — RDKit 등 외부 특징 미사용)",
    config_source="T_toxicity/surface/src/models.py:g3_chemprop 와 동일",
    epochs=50, batch_size=50, num_workers=0, class_balance="분류만 --class-balance",
    metric_for_early_stop="roc(분류)/mae(회귀) · valid로만 · test는 마지막 1회 예측",
    seeds=[1, 2, 3, 4, 5], split="TDC admet_group 공식 (test 고정) + get_train_valid_split(split_type='default', seed=s)",
    accelerator="gpu (RTX 4060 Ti)", chemprop_bin="/home/nudge/miniforge3/envs/ADMET_AI/bin/chemprop")
met["_failed"] = failed
json.dump(met, open(f"{RES}/dmpnn_metrics.json", "w"), ensure_ascii=False, indent=1)
json.dump(overfit, open(f"{RES}/overfit.json", "w"), ensure_ascii=False, indent=1)

# ── leakage.json 정정판 ────────────────────────────────────────────
g = admet_group(path=TDC_DATA)
lk = {}
for ep in EPS:
    b = g.get(ep)
    te_smi = list(b["test"]["Drug"]); tv_smi = list(b["train_val"]["Drug"])
    te_c = {canon(s) for s in te_smi} - {None}
    rep_c = {canon(r["Drug"]) for r in csv.DictReader(open(f"{REPORT_TEST}/test_{ep}.csv"))} - {None}
    tv_c = {canon(s) for s in tv_smi} - {None}
    tv_f = set(filter(None, (ik_full(s) for s in tv_smi)))
    tv_14 = set(filter(None, (ik14(s) for s in tv_smi)))
    lk[ep] = dict(
        n_test=len(te_smi), n_train_val=len(tv_smi),
        exact_canonical_overlap=sum(1 for s in te_smi if canon(s) in tv_c),
        full_inchikey_overlap=sum(1 for s in te_smi if ik_full(s) in tv_f),
        inchikey14_overlap=sum(1 for s in te_smi if ik14(s) in tv_14),
        test_set_identity_vs_report=dict(n_ours=len(te_c), n_report=len(rep_c),
                                         intersection=len(te_c & rep_c),
                                         jaccard=round(len(te_c & rep_c) / len(te_c | rep_c), 6)))
json.dump(lk, open(f"{RES}/leakage.json", "w"), ensure_ascii=False, indent=1)

print("=" * 100)
print("★우리 자체 D-MPNN (정직 학습 · seed=42 본 분할) — 5 seed 평균±SD")
print("=" * 100)
print(f"{'엔드포인트':<11}{'n_tr/va/te':>18}{'주지표':>9}{'test':>18}{'train':>10}{'과적합 gap':>11}{'ok':>4}")
for ep, task in EPS.items():
    m = met[ep]
    if m.get("status") == "failed_all":
        print(f"{ep:<11}{'—':>18}{'—':>9}{'전 seed 실패':>18}"); continue
    k = "AUROC" if task == "cls" else "MAE"
    a = m[k]
    o = [x for x in overfit if x["endpoint"] == ep and x["metric"] == k][0]
    nstr = "{}/{}/{}".format(m["n_train"], m["n_valid"], m["n_test"])
    vstr = "{:.4f}±{:.4f}".format(a["test_mean"], a["test_std"])
    print(f"{ep:<11}{nstr:>18}{k:>9}{vstr:>18}{a['train_mean']:>10.4f}{o['gap']:>11.4f}{m['n_ok']:>4}")
print(f"\n실패 {len(failed)}건: {failed if failed else '없음'}")
print("\n★분할·누수 정정 실측")
for ep, v in lk.items():
    t = v["test_set_identity_vs_report"]
    print(f"  {ep:<10} 정확분자 중복 {v['exact_canonical_overlap']}  전체InChIKey {v['full_inchikey_overlap']}  "
          f"InChIKey14 {v['inchikey14_overlap']}  |  본 보고서 test와 Jaccard {t['jaccard']}")
print("\n저장 → dmpnn_metrics.json · overfit.json · leakage.json")
