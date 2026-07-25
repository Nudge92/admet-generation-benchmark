#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[1차] SMILES 트랜스포머 파인튜닝 (env: adme-bench, GPU). ChemBERTa-2 + MoLFormer × 4 엔드포인트 × 5 seed.
★공정: 벤치마크와 동일 TDC scaffold split. ★누수: test는 마지막 1회. valid로만 early stopping.
★과적합 방지: 작은 lr·weight decay·dropout·early stopping. train/valid 지표 같이 기록.
★resume: results/finetune_raw.jsonl 에 (model,ep,seed)별 1줄 append, 이미 있으면 skip.

[2026-06-27 수정]
- ★MoLFormer NaN 대응: 모델별 lr·정밀도 분리. MoLFormer = lr 1e-5 + ★fp32(AMP off). ChemBERTa = 2e-5 + AMP.
- ★회귀 MAE 버그 수정: 예측을 ★표준화 역변환(p*sd+mu) 후 원척도로 MAE. (이전엔 표준화예측-원척도y 비교 → 잘못).
- ★NaN 방어: loss NaN이면 그 step 건너뛰기 / 예측에 NaN 있으면 그 seed ★실패로 기록(0점 아님), skip.
- grad clipping(max_norm=1.0) 유지.
"""
import os, sys, json, time, copy, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import adme_common

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)
TDC_DATA = "/home/nudge/Project/ADMET_structure/2026-06-27/experiment_tox_benchmark/src/tdc_data"
RAW = os.path.join(RES, "finetune_raw.jsonl")
DEV = "cuda"

# (hf_name, batch, trust_remote_code, lr, use_amp[fp16])
MODELS = {
    "chemberta": ("DeepChem/ChemBERTa-77M-MLM", 64, False, 2e-5, True),
    "molformer": ("ibm/MoLFormer-XL-both-10pct", 32, True, 1e-5, False),   # ★lr↓ + fp32
}
ENDPOINTS = {"caco2_wang": "reg", "hia_hou": "cls", "bioavailability_ma": "cls", "pgp_broccatelli": "cls", "lipophilicity_astrazeneca": "reg", "solubility_aqsoldb": "reg", "bbb_martins": "cls", "ppbr_az": "reg", "vdss_lombardo": "reg", "cyp2c9_veith": "cls", "cyp2d6_veith": "cls", "cyp3a4_veith": "cls", "cyp2c9_substrate_carbonmangels": "cls", "cyp2d6_substrate_carbonmangels": "cls", "cyp3a4_substrate_carbonmangels": "cls", "half_life_obach": "reg", "clearance_hepatocyte_az": "reg", "clearance_microsome_az": "reg"}
SEEDS = [1, 2, 3, 4, 5]
MAX_LEN = 128; MAX_EPOCHS = 30; PATIENCE = 5; WD = 0.01

# ★MoLFormer 원격코드 리비전 고정 — 최신 스냅샷(a14249e5)은 transformers.masking_utils 를 요구하는데
#   이 env(transformers 4.50.3)에는 없어 로드가 실패한다. 플래그십이 실제로 쓰던 구버전으로 고정한다.
REVISION = {"ibm/MoLFormer-XL-both-10pct": "7b12d946c181a37f6012b9dc3b002275de070314"}


class MolNet(nn.Module):
    def __init__(self, backbone_name, trust):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone_name, trust_remote_code=trust,
            **({"revision": REVISION[backbone_name]} if backbone_name in REVISION else {}),
            **({"deterministic_eval": True} if trust else {}))
        h = self.backbone.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(h, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1e-6)   # masked mean
        return self.head(self.drop(pooled)).squeeze(-1)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def tok_split(tok, smiles):
    e = tok(list(smiles), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


def predict_logits(model, ids, mask, use_amp, bs=128):
    model.eval(); outs = []
    with torch.no_grad():
        for i in range(0, len(ids), bs):
            if use_amp:
                with torch.cuda.amp.autocast():
                    lg = model(ids[i:i+bs].to(DEV), mask[i:i+bs].to(DEV))
            else:
                lg = model(ids[i:i+bs].to(DEV), mask[i:i+bs].to(DEV))
            outs.append(lg.float().cpu())
    return torch.cat(outs).numpy()


def metrics(logits, y, task, mu, sd):
    """NaN 있으면 None. 회귀는 ★예측 역표준화(p*sd+mu) 후 원척도 MAE."""
    if logits is None or not np.all(np.isfinite(logits)):
        return None
    if task == "cls":
        prob = 1.0 / (1.0 + np.exp(-logits))
        return dict(AUROC=float(roc_auc_score(y, prob)),
                    AUPRC=float(average_precision_score(y, prob)))
    pred = logits * sd + mu                       # ★역표준화
    return dict(MAE=float(mean_absolute_error(y, pred)),
                Spearman=float(spearmanr(y, pred).correlation))


def done_keys():
    if not os.path.exists(RAW):
        return set()
    ks = set()
    for line in open(RAW):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line); ks.add((d["model"], d["endpoint"], d["seed"]))
    return ks


def write_rec(rec):
    with open(RAW, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_one(mkey, name, bs, trust, lr, use_amp, ep, task, seed, g):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(
        name, trust_remote_code=trust,
        **({"revision": REVISION[name]} if name in REVISION else {}))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.sep_token
    b = g.get(ep); test = b["test"].reset_index(drop=True)
    tr, va = g.get_train_valid_split(benchmark=ep, split_type="default", seed=seed)
    tr = tr.reset_index(drop=True); va = va.reset_index(drop=True)
    ytr = tr["Y"].values.astype(np.float32); yva = va["Y"].values.astype(np.float32)
    yte = test["Y"].values.astype(np.float32)
    if task == "reg":
        mu, sd = float(ytr.mean()), float(ytr.std() + 1e-8)
        ztr = (ytr - mu) / sd
    else:
        mu, sd = 0.0, 1.0; ztr = ytr

    itr, mtr = tok_split(tok, tr["Drug"]); iva, mva = tok_split(tok, va["Drug"])
    ite, mte = tok_split(tok, test["Drug"])
    ds = TensorDataset(itr, mtr, torch.tensor(ztr))
    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, generator=gen)

    model = MolNet(name, trust).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    total = len(dl) * MAX_EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    lossf = nn.BCEWithLogitsLoss() if task == "cls" else nn.MSELoss()
    val_key = "AUROC" if task == "cls" else "MAE"
    better = (lambda n, o: n > o) if task == "cls" else (lambda n, o: n < o)

    best_val = -1e9 if task == "cls" else 1e9
    best_state, best_ep, no_imp, nan_steps = None, -1, 0, 0
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        for ids, msk, yb in dl:
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logit = model(ids.to(DEV), msk.to(DEV)); loss = lossf(logit, yb.to(DEV))
                if not torch.isfinite(loss):
                    nan_steps += 1; continue                       # ★NaN step skip
                scaler.scale(loss).backward()
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                logit = model(ids.to(DEV), msk.to(DEV)); loss = lossf(logit, yb.to(DEV))
                if not torch.isfinite(loss):
                    nan_steps += 1; continue                       # ★NaN step skip
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
        vm = metrics(predict_logits(model, iva, mva, use_amp), yva, task, mu, sd)
        if vm is None:                                              # valid 예측 NaN → 이 epoch 무시
            no_imp += 1
        else:
            cur = vm[val_key]
            if better(cur, best_val):
                best_val = cur; best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                best_ep = epoch; no_imp = 0
            else:
                no_imp += 1
        if no_imp >= PATIENCE:
            break

    # best 없음 = 학습 내내 NaN/발산 → ★실패 기록(0점 아님)
    if best_state is None:
        rec = dict(model=mkey, endpoint=ep, task=task, seed=seed, status="failed_nan",
                   reason=f"valid 예측 NaN/발산(전 epoch). nan_steps={nan_steps}, lr={lr}, amp={use_amp}",
                   sec=round(time.time() - t0, 1))
        write_rec(rec); del model, opt; torch.cuda.empty_cache()
        print(f"  [{mkey}/{ep}/s{seed}] ★FAILED(NaN) — skip, 0점 아님", flush=True)
        return rec

    model.load_state_dict(best_state)
    tr_m = metrics(predict_logits(model, itr, mtr, use_amp), ytr, task, mu, sd)
    va_m = metrics(predict_logits(model, iva, mva, use_amp), yva, task, mu, sd)
    te_logit = predict_logits(model, ite, mte, use_amp)
    te_m = metrics(te_logit, yte, task, mu, sd)
    if te_m is None:                                               # test 예측 NaN → 실패
        rec = dict(model=mkey, endpoint=ep, task=task, seed=seed, status="failed_nan",
                   reason="test 예측 NaN", sec=round(time.time() - t0, 1))
        write_rec(rec); del model, opt; torch.cuda.empty_cache()
        print(f"  [{mkey}/{ep}/s{seed}] ★FAILED(test NaN) — skip", flush=True)
        return rec

    rec = dict(model=mkey, endpoint=ep, task=task, seed=seed, status="ok",
               n_train=len(tr), n_valid=len(va), n_test=len(test),
               best_epoch=best_ep, nan_steps=nan_steps,
               train=tr_m, valid=va_m, test=te_m, sec=round(time.time() - t0, 1))
    write_rec(rec); del model, opt; torch.cuda.empty_cache()
    print(f"  [{mkey}/{ep}/s{seed}] best_ep={best_ep} test={te_m} (train {tr_m}) "
          f"nan_steps={nan_steps} {rec['sec']}s", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="chemberta,molformer")
    ap.add_argument("--only", default="", help="ep:seed 한 건만 (smoke). 예: molformer/dili/1")
    args = ap.parse_args()
    g = adme_common.get_group(TDC_DATA)
    skip = done_keys()
    print(f"이미 완료/기록 {len(skip)}건 skip")
    for mkey in args.models.split(","):
        name, bs, trust, lr, amp = MODELS[mkey]
        for ep, task in ENDPOINTS.items():
            for seed in SEEDS:
                if (mkey, ep, seed) in skip:
                    continue
                run_one(mkey, name, bs, trust, lr, amp, ep, task, seed, g)
    print("완료. results/finetune_raw.jsonl")


if __name__ == "__main__":
    main()
