#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
harvest_preds.py — ★이미 저장돼 있는 분자별 예측을 같은 형식으로 모은다(재계산·재학습 0).
- G3 정직 D-MPNN: 두 실험의 work/{ep}_s{seed}/pred_test.csv (+ 같은 폴더 test.csv의 정답)
- ADMET-AI(누수): 두 실험의 results/admetai_preds.jsonl
DeLong 대응비교용. 산출: predictions/{ep}__{task}__{model}__test.jsonl · results/harvest.json
"""
import glob, json, os, re
import pandas as pd

B = "/home/nudge/Project/ADMET_integrated/2026-07-22"
PRED = f"{B}/experiment_deploy_reliability/predictions"
RES = f"{B}/experiment_deploy_reliability/results"
os.makedirs(PRED, exist_ok=True)
WORKS = {"flagship": f"{B}/experiment_g3_dmpnn_seed42/work",
         "expansion": f"{B}/experiment_gen_expansion_g1g3/work"}
AI = {"flagship": f"{B}/experiment_generation_matrix/results/admetai_preds.jsonl",
      "expansion": f"{B}/experiment_gen_expansion_g1g3/results/admetai_preds.jsonl"}

out = {}
# ── G3 D-MPNN ─────────────────────────────────────────────────────
buf = {}
for scope, wd in WORKS.items():
    for d in sorted(glob.glob(f"{wd}/*_s[1-5]")):
        m = re.match(r"(.+)_s(\d)$", os.path.basename(d))
        if not m or not os.path.exists(f"{d}/pred_test.csv"):
            continue
        ep, seed = m.group(1), int(m.group(2))
        p = pd.read_csv(f"{d}/pred_test.csv"); t = pd.read_csv(f"{d}/test.csv")
        cols = [c for c in t.columns if c != "smiles"]
        for c in cols:
            task = "—" if c == "y" else c
            y = pd.to_numeric(t[c], errors="coerce"); pr = pd.to_numeric(p[c], errors="coerce")
            msk = y.notna() & pr.notna()
            buf.setdefault((ep, task), []).extend(
                dict(smiles=str(s), y_true=float(a), y_prob=float(b), seed=seed)
                for s, a, b in zip(t["smiles"][msk], y[msk], pr[msk]))
for (ep, task), recs in buf.items():
    fn = f"{PRED}/{ep}__{task.replace('/', '_')}__dmpnn_ours__test.jsonl"
    with open(fn, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out[f"dmpnn_ours:{ep}:{task}"] = len(recs)

# ── ADMET-AI ──────────────────────────────────────────────────────
for scope, path in AI.items():
    if not os.path.exists(path):
        continue
    d = pd.read_json(path, lines=True)
    if "task" not in d.columns:
        d["task"] = "—"
    d["task"] = d["task"].replace({"Y": "—"})
    for (ep, task), sub in d.groupby(["endpoint", "task"]):
        fn = f"{PRED}/{ep}__{str(task).replace('/', '_')}__admetai__test.jsonl"
        with open(fn, "w") as f:
            for _, r in sub.iterrows():
                f.write(json.dumps(dict(smiles=str(r.smiles), y_true=float(r.y),
                                        y_prob=float(r.admetai_pred), seed=0),
                                   ensure_ascii=False) + "\n")
        out[f"admetai:{ep}:{task}"] = len(sub)

json.dump(dict(n_files=len(out), rows=out,
               note="전부 기존 저장 예측을 형식만 통일해 옮긴 것 — 재계산·재학습 없음. "
                    "ADMET-AI는 단일 배포 앙상블이라 seed=0으로 기록."),
          open(f"{RES}/harvest.json", "w"), ensure_ascii=False, indent=1)
print(f"수집 {len(out)}개 예측셋")
for k, v in sorted(out.items()):
    print(f"  {k:<45}{v:>7}행")
