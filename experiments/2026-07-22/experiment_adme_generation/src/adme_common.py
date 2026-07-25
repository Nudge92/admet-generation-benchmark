#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
adme_common.py — ★모든 세대가 동일 분자셋을 보도록 하는 공용 래퍼.
두 env(admet / ADMET_AI)의 RDKit 버전 차이로 chemprop이 거부하는 분자가 있다(solubility 2개).
플래그십 `chemprop_bad_smiles` 방식 계승 — ★분할은 건드리지 않고, admet_group이 돌려준 뒤에 제외한다.
사용: g = adme_common.get_group(TDC_DATA)  ← admet_group(path=...) 자리에 그대로 교체
"""
import json, os
from tdc.benchmark_group import admet_group

_HERE = os.path.dirname(os.path.abspath(__file__))
_BADF = os.path.join(os.path.dirname(_HERE), "data", "chemprop_incompatible.json")
BAD = json.load(open(_BADF)) if os.path.exists(_BADF) else {}


def drop_bad(df, ep):
    b = set(BAD.get(ep, []))
    if not b or not hasattr(df, "columns") or "Drug" not in df.columns:
        return df
    return df[~df["Drug"].astype(str).isin(b)].reset_index(drop=True)


class FilteredGroup:
    """admet_group과 동일 인터페이스 + chemprop 비호환 분자 제외."""

    def __init__(self, path):
        self._g = admet_group(path=path)

    def get(self, name):
        b = self._g.get(name)
        return {k: drop_bad(v, name) for k, v in b.items()}

    def get_train_valid_split(self, benchmark=None, split_type="default", seed=1, **kw):
        tr, va = self._g.get_train_valid_split(benchmark=benchmark, split_type=split_type, seed=seed, **kw)
        return drop_bad(tr, benchmark), drop_bad(va, benchmark)

    def __getattr__(self, k):
        return getattr(self._g, k)


def get_group(path):
    return FilteredGroup(path)
