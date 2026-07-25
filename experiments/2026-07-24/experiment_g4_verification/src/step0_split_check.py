#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step0_split_check.py — [0단계] 분할 고정 확인. env: admet.
★ADME 실험(2026-07-22)이 쓴 ★동일 분할을 그대로 읽는다(새로 만들지 않음).
확인: (a) train∩test 정확분자 중복 0 (b) test 분자 수가 adme_matrix.csv 기록과 일치.
불일치면 즉시 중단.
산출: results/step0_split_check.json
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

ADME = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_adme_full"
NEW = "/home/nudge/Project/ADMET_integrated/2026-07-24/experiment_g4_verification"
sys.path.insert(0, f"{ADME}/src")
import common as C                      # ★ADME 실험의 분할 접근을 그대로 재사용

EPS = ["half_life_obach", "solubility_aqsoldb"]
canon = lambda s: (lambda m: Chem.MolToSmiles(m) if m else None)(Chem.MolFromSmiles(str(s)))

MAT = pd.read_csv(f"{ADME}/results/adme_matrix.csv")
LK = json.load(open(f"{ADME}/results/split_leakage.json"))

out, bad = {}, []
g = C.group()
print("=" * 92)
print("[0단계] 분할 고정 확인 — ADME 실험(2026-07-22)과 ★동일 분할을 그대로 사용")
print("=" * 92)
for ep in EPS:
    b = g.get(ep)
    test, tv = b["test"], b["train_val"]
    te_c = [canon(s) for s in test["Drug"]]
    tv_c = {canon(s) for s in tv["Drug"]} - {None}
    overlap = sum(1 for c in te_c if c and c in tv_c)
    n_test = len(test)
    # adme_matrix 기록과 대조(그 엔드포인트 어떤 행이든 n_test는 동일해야)
    rec_n = int(LK[ep]["n_test"])
    ok = (overlap == 0) and (n_test == rec_n)
    if not ok:
        bad.append(ep)
    out[ep] = dict(n_test=n_test, n_test_recorded=rec_n, match=bool(n_test == rec_n),
                   n_train_val=len(tv), exact_overlap_trainval_test=overlap,
                   n_test_unique_canonical=len({c for c in te_c if c}),
                   primary=C.EPS[ep]["primary"], task=C.EPS[ep]["task"])
    print(f"  {ep:<22} test {n_test:>5} (기록 {rec_n:>5}) {'일치' if n_test==rec_n else '★불일치'} · "
          f"train∩test 정확분자 중복 {overlap} {'OK' if overlap==0 else '★누수'} · "
          f"주지표 {C.EPS[ep]['primary']}")

json.dump(out, open(f"{NEW}/results/step0_split_check.json", "w"), ensure_ascii=False, indent=1)
if bad:
    print(f"\n★중단 — 분할 불일치: {bad}")
    sys.exit(2)
print("\n→ 2/2 통과. 동일 분할 확인. 다음 단계 진행 가능.")
print(f"저장 → results/step0_split_check.json")
