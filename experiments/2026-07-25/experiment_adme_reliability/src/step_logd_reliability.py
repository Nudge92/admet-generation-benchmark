#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step_logd_reliability.py — logD 한 칸만 신뢰도 계산 후 ★기존 CSV·detail에 외과적 병합.
★다른 17개 절대 손대지 않음: 파일에서 읽어와 그대로 두고 logD 행/엔트리만 삽입.
★캐논 로직 재사용: step1234_reliability의 load/ad_analysis/boot_ci/metric을 그대로 호출
  (회귀 분기를 동일하게 재현 — 잔차·컨포멀 예측구간·seed별 부트스트랩 주판정+앙상블 참고).
logD는 순번 4번이라 전체 재실행 시 RNG가 뒤 13개 CI를 바꾼다 → 단독 계산 필수.
재현성: logD 부트스트랩용 RNG를 초기 seed(20260725)로 리셋(단독 재현 가능·주석 명시).
산출: results/adme_reliability.csv(logD 행 삽입) · reliability_detail.json(logD 엔트리 삽입)
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

NEW = "/home/nudge/Project/ADMET_integrated/2026-07-25/experiment_adme_reliability"
sys.path.insert(0, f"{NEW}/src")
import step1234_reliability as R
from step1234_reliability import EPS

EP = "lipophilicity_astrazeneca"
MODEL = "dmpnn_ours"

# 0) 재현 게이트 재확인 (±0.005) — 벗어나면 중단
repro = json.load(open(f"{NEW}/results/logd_g3_repro.json"))
if not repro.get("within_tol"):
    print("★중단 — 재현 MAE 불일치:", repro); sys.exit(2)
print(f"[0] 재현 확인 — 보고 {repro['reported']} → 재현 {repro['reproduced']} (Δ{repro['diff']:+.4f}) · "
      f"seed {repro['n_seed']}/5 · chemprop 비호환 {repro['n_chemprop_bad']}")

# ★부트스트랩 재현성: logD 단독 계산이므로 모듈 RNG를 초기 상태로 리셋
R.RNG = np.random.default_rng(20260725)

# ── logD 회귀 신뢰도 (step1234 회귀 분기와 동일) ──
mean_te, per_te = R.load(EP, MODEL, "test")
if mean_te is None:
    print("★중단 — logD test 예측 없음"); sys.exit(2)
prim, task = EPS[EP]["primary"], EPS[EP]["task"]
assert task == "reg" and prim == "MAE", (task, prim)
y = mean_te.y.to_numpy(float); v = mean_te.v.to_numpy(float)
D = dict(endpoint=EP, label=EPS[EP]["label"], pillar=EPS[EP]["pillar"], task=task,
         metric=prim, higher_better=R.hb(EP), champion=MODEL, n_test=len(y))

# 1) AD
ad = R.ad_analysis(EP, mean_te)
D["AD"] = {k: ad[k] for k in ("cut_ood", "cut_border", "cutoff_basis", "n_train_ref", "bands", "verdict")}

# 4) 부트스트랩 — 주판정=seed별 후 종합
per_ci = []
for sd, g in (per_te or {}).items():
    gg = g.reset_index()
    c = R.boot_ci(gg.y_true.to_numpy(float), gg.v.to_numpy(float), prim)
    if c:
        per_ci.append(c)
if per_ci:
    D["ci_primary"] = dict(method="seed별 부트스트랩 후 종합(★주 판정·학습변동 포함)",
                           mean=round(float(np.mean([c["mean"] for c in per_ci])), 4),
                           lo=round(float(np.mean([c["lo"] for c in per_ci])), 4),
                           hi=round(float(np.mean([c["hi"] for c in per_ci])), 4), n_seed=len(per_ci))
D["ci_ensemble"] = R.boot_ci(y, v, prim)
if D["ci_ensemble"]:
    D["ci_ensemble"]["method"] = "5seed 예측평균 부트스트랩(★5-모델 앙상블 평가·변동 큰 모델에 유리)"

# 2) 회귀 — 잔차 + 컨포멀 예측구간(valid 분위수 → test 커버리지)
res = y - v
mean_va, _ = R.load(EP, MODEL, "valid")
q90 = cov = None
if mean_va is not None:
    rv = np.abs(mean_va.y.to_numpy(float) - mean_va.v.to_numpy(float))
    q90 = float(np.quantile(rv, 0.90))
    cov = float((np.abs(res) <= q90).mean())
slope = float(np.polyfit(v, res, 1)[0]) if len(v) > 5 else None
top = mean_te.assign(abs_err=np.abs(res)).nlargest(5, "abs_err")
D["regression"] = dict(
    residual_mean=round(float(res.mean()), 4), residual_sd=round(float(res.std()), 4),
    residual_q=[round(float(np.percentile(res, q)), 4) for q in (5, 25, 50, 75, 95)],
    systematic_bias_slope=(None if slope is None else round(slope, 4)),
    bias_note="예측값 대비 잔차 회귀 기울기. 0에서 멀수록 계통 편향(음수=큰 값 과소예측)",
    PI90_halfwidth=(None if q90 is None else round(q90, 4)),
    PI90_coverage=(None if cov is None else round(cov, 4)),
    PI_note="valid 절대잔차 90% 분위수로 만든 분할 컨포멀 구간 → test 실제 커버리지(★test 튜닝 없음)",
    worst_errors=[dict(smiles=r.smiles[:60], y_true=round(float(r.y), 3),
                       y_pred=round(float(r.v), 3), abs_err=round(float(r.abs_err), 3))
                  for _, r in top.iterrows()])

# CSV 행 (step1234 회귀 분기와 동일)
ci = D.get("ci_primary") or D.get("ci_ensemble") or {}
row = dict(endpoint=EP, label=EPS[EP]["label"], pillar=EPS[EP]["pillar"], task=task,
           metric=prim, direction=("↑" if R.hb(EP) else "↓"), champion=MODEL, n_test=len(y),
           AD_in=ad["bands"]["in-domain"]["frac"], AD_border=ad["bands"]["경계"]["frac"],
           AD_ood=ad["bands"]["OOD"]["frac"], AD_verdict=ad["verdict"],
           value=ci.get("mean"), ci_lo=ci.get("lo"), ci_hi=ci.get("hi"),
           ci_method=ci.get("method", ""),
           source="predictions/lipophilicity_astrazeneca__dmpnn_ours__*.jsonl(chemprop 재현·추론만)")
r_ = D["regression"]
row.update(PI90_halfwidth=r_["PI90_halfwidth"], PI90_coverage=r_["PI90_coverage"],
           residual_sd=r_["residual_sd"], bias_slope=r_["systematic_bias_slope"])

print(f"[logD] AD {ad['verdict']} · in/경계/OOD "
      f"{ad['bands']['in-domain']['frac']:.3f}/{ad['bands']['경계']['frac']:.3f}/{ad['bands']['OOD']['frac']:.3f} "
      f"(OOD n={ad['bands']['OOD']['n']})")
print(f"      잔차 SD {r_['residual_sd']:.3f} · 계통편향 {r_['systematic_bias_slope']} · "
      f"PI90 폭 ±{r_['PI90_halfwidth']} 커버리지 {r_['PI90_coverage']} (목표 0.90)")
print(f"      CI 주판정 {ci.get('mean')} [{ci.get('lo')}, {ci.get('hi')}] · 앙상블 참고 "
      f"{(D['ci_ensemble'] or {}).get('mean')} [{(D['ci_ensemble'] or {}).get('lo')}, {(D['ci_ensemble'] or {}).get('hi')}]")

# ══ 병합: 챔피언 순서대로 재구성(logD를 4번 자리에 삽입, 나머지 17개는 원본 그대로) ══
CH_ORDER = [c["endpoint"] for c in json.load(open(f"{NEW}/results/champions.json"))]
DET = json.load(open(f"{NEW}/results/reliability_detail.json"))
CSVdf = pd.read_csv(f"{NEW}/results/adme_reliability.csv")

# 안전장치: 병합 전 나머지 17개 스냅샷(사후 대조용)
before_other = {k: json.dumps(DET[k], ensure_ascii=False, sort_keys=True, default=str)
                for k in DET if k != EP}

# detail: 순서 보존 삽입
new_det = {}
for ep in CH_ORDER:
    if ep == EP:
        new_det[EP] = D
    elif ep in DET:
        new_det[ep] = DET[ep]
json.dump(new_det, open(f"{NEW}/results/reliability_detail.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=str)

# CSV: logD 행 삽입 후 챔피언 순서로 정렬
CSVdf = CSVdf[CSVdf.endpoint != EP]                        # 혹시 있을 자리표시 제거
CSVdf = pd.concat([CSVdf, pd.DataFrame([row])], ignore_index=True)
order = {ep: i for i, ep in enumerate(CH_ORDER)}
CSVdf["__o"] = CSVdf.endpoint.map(order)
CSVdf = CSVdf.sort_values("__o").drop(columns="__o").reset_index(drop=True)
CSVdf.to_csv(f"{NEW}/results/adme_reliability.csv", index=False)

# 사후 검증: 나머지 17개 detail 불변 확인
after = json.load(open(f"{NEW}/results/reliability_detail.json"))
changed = [k for k in before_other
           if json.dumps(after.get(k), ensure_ascii=False, sort_keys=True, default=str) != before_other[k]]
print(f"\n★병합 검증 — detail {len(after)}개(=18) · logD 삽입 위치 {list(after).index(EP)}(=4) · "
      f"나머지 17개 불변: {'OK' if not changed else '★변경됨 '+str(changed)}")
print(f"★CSV {len(CSVdf)}행 · logD 행 위치 {CSVdf.index[CSVdf.endpoint==EP].tolist()}")
if changed:
    sys.exit(2)
