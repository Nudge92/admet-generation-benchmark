#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""f_3d 를 ★SMILES 키 디스크 캐시 버전으로 교체(1회용 패치)."""
import re
from pathlib import Path

P = Path(__file__).with_name("common.py")
s = P.read_text(encoding="utf-8")
i = s.index("def f_3d(smis):")
j = s.index("def f_medchem(smis):")

NEW = '''D3_NAMES = ["NPR1", "NPR2", "Asphericity", "Eccentricity", "InertialShapeFactor",
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
        with open(_D3_PATH, "a") as fh:
            for k, sm in enumerate(uniq):
                _D3[sm] = _calc3d_one(sm)
                fh.write(json.dumps({"s": sm, "v": _D3[sm]}) + "\\n")
                if (k + 1) % 2000 == 0:
                    fh.flush()
                    log(f"    3D conformer {k+1}/{len(uniq)} (캐시 총 {len(_D3)})")
    X = np.array([_D3.get(str(v), [np.nan] * len(D3_NAMES)) for v in smis], float)
    X[~np.isfinite(X)] = np.nan
    return X, list(D3_NAMES)


'''
P.write_text(s[:i] + NEW + s[j:], encoding="utf-8")
print("f_3d 디스크 캐시 적용 완료")
