
import json, os, sys
import pandas as pd, numpy as np
from admet_ai import ADMETModel
COL = {"caco2_wang": "Caco2_Wang", "hia_hou": "HIA_Hou", "bioavailability_ma": "Bioavailability_Ma", "pgp_broccatelli": "Pgp_Broccatelli", "lipophilicity_astrazeneca": "Lipophilicity_AstraZeneca", "solubility_aqsoldb": "Solubility_AqSolDB", "bbb_martins": "BBB_Martins", "ppbr_az": "PPBR_AZ", "vdss_lombardo": "VDss_Lombardo", "cyp2c9_veith": "CYP2C9_Veith", "cyp2d6_veith": "CYP2D6_Veith", "cyp3a4_veith": "CYP3A4_Veith", "cyp2c9_substrate_carbonmangels": "CYP2C9_Substrate_CarbonMangels", "cyp2d6_substrate_carbonmangels": "CYP2D6_Substrate_CarbonMangels", "cyp3a4_substrate_carbonmangels": "CYP3A4_Substrate_CarbonMangels", "half_life_obach": "Half_Life_Obach", "clearance_hepatocyte_az": "Clearance_Hepatocyte_AZ", "clearance_microsome_az": "Clearance_Microsome_AZ"}
D = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_adme_full/data/test"; OUT = "/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_adme_full/predictions"
model = ADMETModel(); res = {}
for ep, col in COL.items():
    p = os.path.join(D, ep + ".csv")
    if not os.path.exists(p): continue
    try:
        d = pd.read_csv(p); pred = model.predict(smiles=[str(s) for s in d.Drug]).reset_index(drop=True)
        if col not in pred.columns:
            res[ep] = dict(status="미커버"); continue
        with open(os.path.join(OUT, ep + "__G3_admetai__test.jsonl"), "w") as f:
            for s, y, v in zip(d.Drug, d.Y, pred[col].values):
                f.write(json.dumps(dict(smiles=str(s), y_true=float(y), y_prob=float(v), seed=0))+"\n")
        res[ep] = dict(status="ok", n=len(d))
    except Exception as e:
        res[ep] = dict(status="failed", error=str(e)[:300])
json.dump(res, open("/home/nudge/Project/ADMET_integrated/2026-07-22/experiment_adme_full/results/admetai_status.json","w"), ensure_ascii=False, indent=1)
print(json.dumps({k:v.get("status") for k,v in res.items()}, ensure_ascii=False))
