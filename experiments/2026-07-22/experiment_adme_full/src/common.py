#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
common.py — ADME 전면 벤치마크 공용 모듈. env: admet
- 18개 엔드포인트 레지스트리(기둥·task·주지표)
- TDC admet_group 분할 접근 + ★chemprop 비호환 분자 전 조합 공통 제외
- 특징 빌더(축② ablation용 누적 스택)
- 지표·예측 저장·진행상태(progress.jsonl)·실패기록(failures.jsonl)
★모든 조합은 독립 — 한 조합 실패가 전체를 멈추지 않는다.
"""
import json, os, sys, traceback, warnings
from datetime import datetime
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
RDLogger.DisableLog("rdApp.*")

ROOT = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_adme_full"
RES, PRED, LOGS, DATA, CKPT = (f"{ROOT}/results", f"{ROOT}/predictions", f"{ROOT}/logs",
                               f"{ROOT}/data", f"{ROOT}/checkpoints")
for d in (RES, PRED, LOGS, DATA, CKPT, f"{ROOT}/work"):
    os.makedirs(d, exist_ok=True)
TDC_DATA = "/home/nudge/Project/ADMET_structure/2026-06-27/experiment_tox_benchmark/src/tdc_data"
CFG = json.load(open("/home/nudge/Project/ADMET/12_pipeline/config_public.json"))
SEEDS = [1, 2, 3, 4, 5]
NJOBS = 8
PROGRESS, FAILURES = f"{RES}/progress.jsonl", f"{LOGS}/failures.jsonl"

# ── 18개 엔드포인트 (pillar, task, 주지표) ────────────────────────
EPS = {
    # A 흡수
    "caco2_wang":                     dict(pillar="A", task="reg", primary="MAE",      label="Caco-2 투과"),
    "hia_hou":                        dict(pillar="A", task="cls", primary="AUROC",    label="HIA 흡수"),
    "bioavailability_ma":             dict(pillar="A", task="cls", primary="AUROC",    label="경구 생체이용률"),
    "pgp_broccatelli":                dict(pillar="A", task="cls", primary="AUROC",    label="P-gp 기질"),
    "lipophilicity_astrazeneca":      dict(pillar="A", task="reg", primary="MAE",      label="친유성 logD"),
    "solubility_aqsoldb":             dict(pillar="A", task="reg", primary="MAE",      label="수용해도"),
    # D 분포
    "bbb_martins":                    dict(pillar="D", task="cls", primary="AUROC",    label="BBB 투과"),
    "ppbr_az":                        dict(pillar="D", task="reg", primary="MAE",      label="혈장단백결합"),
    "vdss_lombardo":                  dict(pillar="D", task="reg", primary="Spearman", label="분포용적 VDss"),
    # M 대사
    "cyp2c9_veith":                   dict(pillar="M", task="cls", primary="AUROC",    label="CYP2C9 억제"),
    "cyp2d6_veith":                   dict(pillar="M", task="cls", primary="AUROC",    label="CYP2D6 억제"),
    "cyp3a4_veith":                   dict(pillar="M", task="cls", primary="AUROC",    label="CYP3A4 억제"),
    "cyp2c9_substrate_carbonmangels": dict(pillar="M", task="cls", primary="AUROC",    label="CYP2C9 기질"),
    "cyp2d6_substrate_carbonmangels": dict(pillar="M", task="cls", primary="AUROC",    label="CYP2D6 기질"),
    "cyp3a4_substrate_carbonmangels": dict(pillar="M", task="cls", primary="AUROC",    label="CYP3A4 기질"),
    # E 배설
    "half_life_obach":                dict(pillar="E", task="reg", primary="Spearman", label="반감기"),
    "clearance_hepatocyte_az":        dict(pillar="E", task="reg", primary="Spearman", label="간세포 청소율"),
    "clearance_microsome_az":         dict(pillar="E", task="reg", primary="Spearman", label="마이크로솜 청소율"),
}
ADMETAI_COL = {
    "caco2_wang": "Caco2_Wang", "hia_hou": "HIA_Hou", "bioavailability_ma": "Bioavailability_Ma",
    "pgp_broccatelli": "Pgp_Broccatelli", "lipophilicity_astrazeneca": "Lipophilicity_AstraZeneca",
    "solubility_aqsoldb": "Solubility_AqSolDB", "bbb_martins": "BBB_Martins", "ppbr_az": "PPBR_AZ",
    "vdss_lombardo": "VDss_Lombardo", "cyp2c9_veith": "CYP2C9_Veith", "cyp2d6_veith": "CYP2D6_Veith",
    "cyp3a4_veith": "CYP3A4_Veith",
    "cyp2c9_substrate_carbonmangels": "CYP2C9_Substrate_CarbonMangels",
    "cyp2d6_substrate_carbonmangels": "CYP2D6_Substrate_CarbonMangels",
    "cyp3a4_substrate_carbonmangels": "CYP3A4_Substrate_CarbonMangels",
    "half_life_obach": "Half_Life_Obach", "clearance_hepatocyte_az": "Clearance_Hepatocyte_AZ",
    "clearance_microsome_az": "Clearance_Microsome_AZ",
}

# ── 분할 (chemprop 비호환 제외 래퍼) ──────────────────────────────
_BADF = f"{DATA}/chemprop_incompatible.json"
BAD = json.load(open(_BADF)) if os.path.exists(_BADF) else {}


def _drop(df, ep):
    b = set(BAD.get(ep, []))
    if not b or not hasattr(df, "columns") or "Drug" not in df.columns:
        return df.reset_index(drop=True) if hasattr(df, "reset_index") else df
    return df[~df["Drug"].astype(str).isin(b)].reset_index(drop=True)


class Group:
    def __init__(self):
        from tdc.benchmark_group import admet_group
        self._g = admet_group(path=TDC_DATA)

    def get(self, ep):
        return {k: _drop(v, ep) for k, v in self._g.get(ep).items()}

    def split(self, ep, seed):
        tr, va = self._g.get_train_valid_split(benchmark=ep, split_type="default", seed=seed)
        return _drop(tr, ep), _drop(va, ep)


_G = None


def group():
    global _G
    if _G is None:
        _G = Group()
    return _G


# ── 특징 빌더 (축② 누적 스택) ────────────────────────────────────
DESC210 = [n for n, _ in Descriptors._descList]
_CALC = MolecularDescriptorCalculator(DESC210)
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
PHYS6 = ["MolLogP", "TPSA", "MolWt", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"]


def _mols(smis):
    return [Chem.MolFromSmiles(str(s)) for s in smis]


def f_ecfp(smis):
    X = np.zeros((len(smis), 2048), dtype=np.float32)
    for i, m in enumerate(_mols(smis)):
        if m is not None:
            a = np.zeros((2048,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(_MORGAN.GetFingerprint(m), a)
            X[i] = a
    return X, [f"ecfp_{i}" for i in range(2048)]


def f_phys(smis):
    X = np.full((len(smis), len(DESC210)), np.nan)
    for i, m in enumerate(_mols(smis)):
        if m is not None:
            try:
                X[i] = _CALC.CalcDescriptors(m)
            except Exception:
                pass
    X[~np.isfinite(X)] = np.nan
    return X, list(DESC210)


def f_ion(smis):
    """이온화 대용 특징: 산/염기 작용기 수 + pH7.4 하전 근사 + logD 근사."""
    ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
    SULF = Chem.MolFromSmarts("[SX4](=O)(=O)[OX2H1,OX1-]")
    TET = Chem.MolFromSmarts("[NX4+]")
    BASE = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(N[C,S]=[O,S,N]);!$(N=*);!$(Na)]")
    rows = []
    for m in _mols(smis):
        if m is None:
            rows.append([0] * 7); continue
        na = len(m.GetSubstructMatches(ACID)) + len(m.GetSubstructMatches(SULF))
        nb = len(m.GetSubstructMatches(BASE))
        nq = len(m.GetSubstructMatches(TET))
        logp = Crippen.MolLogP(m)
        # pH7.4에서 산은 대부분 음전하·염기는 상당수 양전하라는 단순 근사
        net = nq + 0.5 * nb - 1.0 * na
        logd = logp - (1.0 if na else 0.0) - (0.5 if nb else 0.0)
        rows.append([na, nb, nq, net, logp, logd, float(na > 0 and nb > 0)])
    return np.array(rows, float), ["n_acid", "n_base", "n_quat", "net_charge74",
                                   "logP", "logD74_approx", "zwitterion"]


D3_NAMES = ["NPR1", "NPR2", "Asphericity", "Eccentricity", "InertialShapeFactor",
            "RadiusOfGyration", "SpherocityIndex", "PBF"]
_D3_PATH = f"{DATA}/desc3d_cache.jsonl"
_D3 = {}
if os.path.exists(_D3_PATH):
    for _l in open(_D3_PATH):
        try:
            _d = json.loads(_l)
            _D3[_d["s"]] = _d["v"]
        except Exception:
            pass


def _calc3d_one(smi):
    """ETKDG 1 conformer + MMFF 최적화 -> 3D 서술자 8종. 실패 시 NaN."""
    from rdkit.Chem import AllChem, Descriptors3D
    v = [float("nan")] * len(D3_NAMES)
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return v
    try:
        mh = Chem.AddHs(m)
        if AllChem.EmbedMolecule(mh, randomSeed=42, maxAttempts=50) == 0:
            AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
            v = [float(Descriptors3D.NPR1(mh)), float(Descriptors3D.NPR2(mh)),
                 float(Descriptors3D.Asphericity(mh)), float(Descriptors3D.Eccentricity(mh)),
                 float(Descriptors3D.InertialShapeFactor(mh)),
                 float(Descriptors3D.RadiusOfGyration(mh)),
                 float(Descriptors3D.SpherocityIndex(mh)), float(rdMolDescriptors.CalcPBF(mh))]
    except Exception:
        pass
    return v


def f_3d(smis):
    """3D 서술자 - ★SMILES 키 디스크 캐시.
    conformer 생성이 비싸서 (엔드포인트 x seed x 분할)마다 다시 계산하면 밤을 다 쓴다.
    분자당 딱 한 번만 계산하고 desc3d_cache.jsonl 에 누적한다(재시작에도 유지)."""
    uniq = [x for x in dict.fromkeys(str(v) for v in smis) if x not in _D3]
    if uniq:
        with open(_D3_PATH, "a", encoding="utf-8") as fh:
            for k, sm in enumerate(uniq):
                _D3[sm] = _calc3d_one(sm)
                fh.write(json.dumps({"s": sm, "v": _D3[sm]}) + "\n")
                if (k + 1) % 2000 == 0:
                    fh.flush()
                    log(f"    3D conformer {k+1}/{len(uniq)} (캐시 총 {len(_D3)})")
    X = np.array([_D3.get(str(v), [np.nan] * len(D3_NAMES)) for v in smis], float)
    X[~np.isfinite(X)] = np.nan
    return X, list(D3_NAMES)


def f_medchem(smis):
    """의약화학 규칙 특징(G1 자리): GSE 용해도 추정·CNS MPO·Lipinski/Veber 위반·efflux 대용."""
    rows = []
    for m in _mols(smis):
        if m is None:
            rows.append([np.nan] * 8); continue
        logp = Crippen.MolLogP(m); tpsa = rdMolDescriptors.CalcTPSA(m)
        mw = Descriptors.MolWt(m); hbd = rdMolDescriptors.CalcNumHBD(m)
        hba = rdMolDescriptors.CalcNumHBA(m); rot = rdMolDescriptors.CalcNumRotatableBonds(m)
        arom = sum(1 for a in m.GetAtoms() if a.GetIsAromatic())
        gse = 0.5 - 0.01 * (150 - 25) - logp                      # GSE(녹는점 150℃ 대용)
        # CNS MPO 근사(6요소 각 0~1)
        d = lambda x, lo, hi: float(np.clip((hi - x) / (hi - lo), 0, 1))
        mpo = (d(logp, 3, 5) + d(tpsa, 120, 40) * 0 + float(40 <= tpsa <= 90)
               + d(mw, 360, 500) + d(hbd, 0.5, 3.5) + float(logp - 0 < 4))
        lip = float(mw > 500) + float(logp > 5) + float(hbd > 5) + float(hba > 10)
        veb = float(rot > 10) + float(tpsa > 140)
        rows.append([gse, mpo, lip, veb, tpsa, arom, rot, tpsa / (mw + 1e-9)])
    X = np.array(rows, float); X[~np.isfinite(X)] = np.nan
    return X, ["GSE_logS", "CNS_MPO_approx", "lipinski_viol", "veber_viol",
               "TPSA", "n_aromatic_atoms", "n_rotb", "TPSA_per_MW"]


FEATURES = {"ecfp": f_ecfp, "phys": f_phys, "ion": f_ion, "d3": f_3d, "medchem": f_medchem}
# 축② 누적 스택 (공통)
STACKS = [("ecfp", ["ecfp"]),
          ("+phys", ["ecfp", "phys"]),
          ("+ion", ["ecfp", "phys", "ion"]),
          ("+3d", ["ecfp", "phys", "ion", "d3"]),
          ("+medchem", ["ecfp", "phys", "ion", "d3", "medchem"])]

_FCACHE = {}


def build_X(ep, which, smis, tag):
    key = (ep, tag, tuple(which))
    if key in _FCACHE:
        return _FCACHE[key]
    Xs, names = [], []
    for w in which:
        k2 = (ep, tag, w)
        if k2 not in _FCACHE:
            _FCACHE[k2] = FEATURES[w](smis)
        X, nm = _FCACHE[k2]
        Xs.append(X); names += nm
    out = (np.hstack(Xs), names)
    _FCACHE[key] = out
    return out


def prep(Xtr, *others, clip=1e6):
    """train 중앙값 대치 + StandardScaler(train fit) + ★float32 안전 클리핑(독성 실험 계승)."""
    from sklearn.preprocessing import StandardScaler
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    sc = StandardScaler().fit(Xtr)
    out = [np.clip(sc.transform(Xtr), -clip, clip).astype(np.float32)]
    for X in others:
        X = np.where(np.isnan(X), med, X)
        out.append(np.clip(sc.transform(X), -clip, clip).astype(np.float32))
    return out


# ── 지표 ─────────────────────────────────────────────────────────
def score(y, p, task):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if task == "cls":
        return dict(AUROC=round(float(roc_auc_score(y, p)), 4),
                    AUPRC=round(float(average_precision_score(y, p)), 4),
                    pos_rate=round(float(y.mean()), 4))
    return dict(MAE=round(float(mean_absolute_error(y, p)), 4),
                Spearman=round(float(spearmanr(y, p).correlation), 4))


# ── 진행 상태·실패 기록 ──────────────────────────────────────────
def done_keys():
    s = set()
    if os.path.exists(PROGRESS):
        for line in open(PROGRESS, encoding="utf-8"):
            try:
                s.add(json.loads(line)["key"])
            except Exception:
                pass
    return s


def record(key, **kw):
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(key=key, ts=datetime.now().isoformat(timespec="seconds"), **kw),
                           ensure_ascii=False) + "\n")


def fail(key, err, **kw):
    with open(FAILURES, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(key=key, ts=datetime.now().isoformat(timespec="seconds"),
                                error=str(err)[:600], trace=traceback.format_exc()[-900:], **kw),
                           ensure_ascii=False) + "\n")


def save_pred(name, recs):
    p = f"{PRED}/{name}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return os.path.basename(p)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
