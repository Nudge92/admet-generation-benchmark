#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prep_splits.py — 확장 3종(발암성·ClinTox·Tox21)의 분할을 ★한 번만 만들어 파일로 고정.
이후 G1/G2/G3/ADMET-AI 모든 모델은 ★이 파일만 읽는다 → 엔드포인트 내 test 동일성이 구조적으로 보장됨.
env: admet

분할 규율(플래그십 계승):
- 발암성·ClinTox: TDC 공식 Tox(name).get_split(method="scaffold", seed=42)
- Tox21: 12과제가 서로 다른 분자 부분집합이라 '과제별 공식 분할'로는 멀티태스크 D-MPNN과
  과제별 G2를 ★같은 test에서 비교할 수 없다. 그래서 12과제 ★합집합 분자셋을 만들고
  TDC 내부 함수 create_scaffold_split(seed=42, frac 0.7/0.1/0.2)을 ★한 번 적용한다.
  → 방법론(Bemis-Murcko scaffold)은 동일하나 ★TDC 과제별 공식 분할과는 다르다(notes.md에 명시).
  결측 라벨은 NaN으로 두고 학습·평가에서 마스킹.
★재현성: 같은 분할을 두 번 생성해 완전 일치 확인.
★누수: train↔test 정확분자(canonical) 중복 = 0 확인.
산출: splits/<ep>_{train,valid,test}.csv · results/leakage.json · results/split_meta.json
"""
import json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list
from tdc.utils.split import create_scaffold_split

ROOT = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_gen_expansion_g1g3"
SPL, RES, TDCP = f"{ROOT}/splits", f"{ROOT}/results", f"{ROOT}/tdc_data"
os.makedirs(SPL, exist_ok=True); os.makedirs(RES, exist_ok=True)
SEED = 42
FRAC = [0.7, 0.1, 0.2]
SINGLE = ["Carcinogens_Lagunin", "ClinTox"]


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m else None


def build_tox21():
    """12과제 합집합 분자셋 + 12 라벨열(결측 NaN)."""
    labels = retrieve_label_name_list("Tox21")
    frames = {}
    for t in labels:
        d = Tox(name="Tox21", label_name=t, path=TDCP).get_data()[["Drug", "Y"]].copy()
        d["c"] = d["Drug"].map(canon)
        d = d.dropna(subset=["c"]).drop_duplicates("c")
        frames[t] = d.set_index("c")
    allc = sorted(set().union(*[set(f.index) for f in frames.values()]))
    rep = {}                                   # 대표 SMILES(첫 등장)
    for t in labels:
        for c, s in frames[t]["Drug"].items():
            rep.setdefault(c, s)
    df = pd.DataFrame({"Drug": [rep[c] for c in allc]}, index=allc)
    for t in labels:
        df[t] = frames[t]["Y"].reindex(allc).values
    return df.reset_index(drop=True), labels


def split_of(ep):
    if ep in SINGLE:
        sp = Tox(name=ep, path=TDCP).get_split(method="scaffold", seed=SEED)
        return {k: sp[k].reset_index(drop=True)[["Drug", "Y"]] for k in ("train", "valid", "test")}, ["Y"]
    df, labels = build_tox21()
    sp = create_scaffold_split(df, SEED, FRAC, "Drug")
    return {k: sp[k].reset_index(drop=True) for k in ("train", "valid", "test")}, labels


def chemprop_bad(smiles_list):
    """★G3(chemprop)이 도는 ADMET_AI env의 RDKit이 거부하는 SMILES(두 env RDKit 버전 불일치 대비).
    ★플래그십 T_toxicity/surface/src/featurize.py:chemprop_bad_smiles 와 동일 방식 —
    모든 세대가 ★같은 분자셋★을 쓰도록 ★분할 이후에 제외한다(분할 자체는 건드리지 않음)."""
    import subprocess, tempfile, os as _os
    py = "/home/nudge/miniforge3/envs/ADMET_AI/bin/python"
    if not _os.path.exists(py):
        return set()
    uniq = sorted(set(map(str, smiles_list)))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(uniq)); path = f.name
    code = ("import sys;from rdkit import Chem,RDLogger;RDLogger.DisableLog('rdApp.*');"
            "[print(l.strip()) for l in open(sys.argv[1]) if l.strip() and Chem.MolFromSmiles(l.strip()) is None]")
    try:
        r = subprocess.run([py, "-c", code, path], capture_output=True, text=True, timeout=600)
        return set(r.stdout.split("\n")) - {""}
    except Exception:
        return set()
    finally:
        _os.unlink(path)


meta, leak = {}, {}
for ep in SINGLE + ["Tox21"]:
    sp, labels = split_of(ep)
    sp2, _ = split_of(ep)                                     # ★재현성 확인
    same = all(list(sp[k]["Drug"]) == list(sp2[k]["Drug"]) for k in sp)

    # ★chemprop 호환 필터 — 분할 후 제외(모든 세대가 동일 분자셋을 보도록)
    bad = chemprop_bad([s for k in sp for s in sp[k]["Drug"]])
    n_drop = {}
    for k in sp:
        before = len(sp[k])
        sp[k] = sp[k][~sp[k]["Drug"].astype(str).isin(bad)].reset_index(drop=True)
        n_drop[k] = before - len(sp[k])
    for k, d in sp.items():
        d.to_csv(f"{SPL}/{ep}_{k}.csv", index=False)

    tr_c = {canon(s) for s in sp["train"]["Drug"]} | {canon(s) for s in sp["valid"]["Drug"]}
    te_c = [canon(s) for s in sp["test"]["Drug"]]
    ov = sum(1 for c in te_c if c and c in tr_c)
    leak[ep] = dict(n_train=len(sp["train"]), n_valid=len(sp["valid"]), n_test=len(sp["test"]),
                    chemprop_incompatible_dropped=n_drop, n_bad_smiles_total=len(bad),
                    exact_canonical_overlap_trainval_test=ov,
                    n_test_unique_canonical=len({c for c in te_c if c}),
                    deterministic_regeneration=bool(same))

    per = {}
    for t in labels:
        col = "Y" if t == "Y" else t
        r = {}
        for k in ("train", "valid", "test"):
            y = pd.to_numeric(sp[k][col], errors="coerce")
            r[k] = dict(n=int(y.notna().sum()), pos=int((y == 1).sum()),
                        pos_rate=(round(float(y[y.notna()].mean()), 4) if y.notna().any() else None))
        per[t] = r
    meta[ep] = dict(split="TDC scaffold (Bemis-Murcko) seed=42 · frac 0.7/0.1/0.2",
                    source=("Tox(name).get_split(method='scaffold', seed=42)" if ep in SINGLE
                            else "12과제 합집합 + tdc.utils.split.create_scaffold_split(seed=42)"),
                    n_labels=len(labels), labels=labels, per_label=per,
                    n_train=len(sp["train"]), n_valid=len(sp["valid"]), n_test=len(sp["test"]))

    print(f"[{ep}] train/valid/test = {len(sp['train'])}/{len(sp['valid'])}/{len(sp['test'])} "
          f"· 라벨 {len(labels)} · 재현 {'OK' if same else '★실패'} · train∩test 중복 {ov} "
          f"· chemprop 비호환 제외 {sum(n_drop.values())}")
    if ep == "Tox21":
        print("   과제별 test 라벨 수: " +
              ", ".join(f"{t}={per[t]['test']['n']}({per[t]['test']['pos']}+)" for t in labels[:6]))
        print("                        " +
              ", ".join(f"{t}={per[t]['test']['n']}({per[t]['test']['pos']}+)" for t in labels[6:]))

json.dump(meta, open(f"{RES}/split_meta.json", "w"), ensure_ascii=False, indent=1)
json.dump(leak, open(f"{RES}/leakage.json", "w"), ensure_ascii=False, indent=1)
bad = [e for e, v in leak.items() if v["exact_canonical_overlap_trainval_test"] > 0 or not v["deterministic_regeneration"]]
print(f"\n{'★문제 있음: ' + str(bad) if bad else '→ 3/3 재현 OK · 정확분자 중복 0'}")
print(f"저장 → splits/*.csv · results/split_meta.json · results/leakage.json")
