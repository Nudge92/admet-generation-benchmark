# G3 빈칸 메우기 — 우리 자체 Chemprop D-MPNN을 seed=42 본 분할에 학습

작업일 2026-07-22 · `Project/ADMET_integrated/2026-07-22/experiment_g3_dmpnn_seed42/`

## 왜 했나

세대 매트릭스 보고서(`experiment_generation_matrix`)의 **G3 자리가 ADMET-AI(누수)로만 채워져 있었다.** 우리 자체 D-MPNN은 seed=1(다른 분할)에만 있어 직접 비교가 불가능했다. 그래서 **같은 분할에 정직하게 학습**해 (a) 순수 G3 실력을 측정하고 (b) ADMET-AI가 같은 D-MPNN이라는 점을 이용해 **누수 크기를 정량화**했다.

> ★이번 작업은 예외적으로 **새 학습 있음** — 딱 G3 1종(4 엔드포인트 × 5 seed = 20 런).

## 분할 — 새로 만들지 않았다

seed=42 실험들이 쓴 **바로 그 호출을 재현**했다(`experiment_tox_benchmark/src/bench_ourml.py`, `experiment_finetune/src/finetune.py`와 동일):

```python
g = admet_group(path=TDC_DATA)
b = g.get(ep)                                                    # 공식 고정 test
tr, va = g.get_train_valid_split(benchmark=ep, split_type="default", seed=s)   # s = 1..5
```

★**test 집합 동일성 실측 = Jaccard 1.000000 (4/4 엔드포인트)** — 100% 아니면 학습 전에 중단하도록 스크립트에 게이트를 넣었고, 통과했다.

| | Jaccard | 정확분자(canonical) 중복 | 전체 InChIKey | InChIKey14(골격) |
|---|---|---|---|---|
| dili | 1.0 | **0** | 0 | 0 |
| herg | 1.0 | **0** | 1 | 6 |
| ames | 1.0 | **0** | 1 | 3 |
| ld50_zhu | 1.0 | **0** | 0 | 0 |

원본 `experiment_tox_benchmark/results/leakage.json`의 값과 완전히 일치(재현 확인). train↔test **정확분자 중복 0**.

> ★자기수정: `train_dmpnn.py`가 처음 찍은 `train_test_exact_overlap`은 실제로 **InChIKey14(골격) 중복**이었다(herg 6·ames 3). `aggregate_dmpnn.py`에서 정확분자/전체InChIKey/골격 셋을 **따로** 계산해 이름을 바로잡았다.

## 설정 — seed=1 런과 ★거의 동일(에폭만 다름)

`T_toxicity/surface/src/models.py:g3_chemprop`의 설정을 재사용했다. ★단 **에폭이 다르다** — seed=1 런은 `run.py:27 G3_EPOCHS=30`, 이번은 **50**. 따라서 "동일 config"라는 표현은 틀렸고, 두 런의 차이를 **분할 효과로만 귀속할 수 없다**(에폭·분할이 함께 다름).

- Chemprop v2 CLI (`ADMET_AI` env 바이너리를 교차 env subprocess로 호출)
- `--epochs 50` · `-b 50` · `-n 0` · `--pytorch-seed <seed>` · metrics `roc`(분류)/`mae`(회귀)
- 분류만 `--class-balance` · GPU(RTX 4060 Ti)
- **순수 D-MPNN** — RDKit 서술자 등 외부 특징 미사용
- `-i train val test` 3파일 = chemprop이 각각 train/val/test로 사용. **early stopping은 val만**, test는 마지막 1회 예측
- ★**이 분할에 맞춘 하이퍼파라미터 탐색은 하지 않았다.** G2(XGBoost)도 튜닝하지 않았으므로 조건은 대등하나, 양쪽 모두 튜닝 여지는 남는다.

소요: DILI/hERG 각 ~14초, AMES/LD50 각 ~44초 · 총 20런 약 9분 · **실패 0건**.

## 결과 — 우리 D-MPNN (5 seed 평균±SD)

| 엔드포인트 | n_tr/va/te | 주지표 | test | train | 과적합 gap |
|---|---|---|---|---|---|
| dili | 325/54/96 | AUROC | **0.9039 ±0.0119** | 0.8690 | **−0.0349** |
| herg | 457/66/132 | AUROC | **0.7972 ±0.0101** | 0.8751 | 0.0779 |
| ames | 5093/728/1457 | AUROC | **0.8333 ±0.0089** | 0.9024 | 0.0692 |
| ld50_zhu | 5168/739/1478 | MAE | **0.6398 ±0.0212** | 0.4199 | 0.2199 |

★DILI는 **gap이 음수** — train보다 test가 좋다. 과적합이 없다. Uni-Mol의 train AUROC 1.000(gap 0.13~0.20)과 대조적으로 **D-MPNN이 이 데이터 크기에서 가장 얌전한 모델**이다.

## ★ADMET-AI와의 격차 — 누수의 **상한**(크기가 아니다)

★★**정정(적대검증)**: 두 시스템은 '같은 모델'이 아니다. 패키지 실측 — ADMET-AI는 **5-모델 앙상블 × 31과제(회귀 10과제) 멀티태스크·튜닝된 배포본**, 우리는 **단일모델·단일과제·미튜닝**(5 seed는 예측 앙상블이 아니라 지표 평균). 따라서 아래 격차에는 **누수 + 앙상블 + 멀티태스크 전이 + 튜닝**이 교락돼 있으며 **누수의 상한**으로만 읽어야 한다.

| 엔드포인트 | 우리 D-MPNN (정직) | ADMET-AI (누수) | 프리미엄 |
|---|---|---|---|
| DILI AUROC | 0.9039 | 0.9557 | **+0.0518** |
| hERG AUROC | 0.7972 | 0.9113 | **+0.1141** |
| AMES AUROC | 0.8333 | 0.9303 | **+0.0970** |
| LD50 MAE | 0.6398 | 0.3361 | **0.3037 좋아짐** |

**분류 3종 평균 0.088 AUROC(=상한).** 누수 크기의 **가장 깨끗한 추정은 증거 ②(+0.049~0.074)**다 — 거기서는 ADMET-AI가 자기 자신과 비교되므로 앙상블·멀티태스크·튜닝이 양쪽에서 상쇄되고 평가 분자만 달라진다. 증거 ③은 크기 확정이 아니라 **방향의 독립 확인**으로만 쓴다.

## 결론 — 세대 매트릭스에 미친 영향

- **G3(정직)은 4/4 어디서도 1위가 아니다** (7모델 중 DILI 3위 · hERG 5위 · AMES 4위 · LD50 4위). → **"GNN이 고전 ML을 이긴다"는 통념이 이 4개 독성 엔드포인트에서 성립하지 않았다.**
- 기대와 다르게 나오면 그대로 적는다는 원칙대로, **seed=1에서 관측된 방향(hERG G3 0.746 < G2 0.828)이 seed=42에서도 재현**됐다(0.797 < 0.837).
- 세대 간 격차를 **1위 vs 2위**로 재판정하면(초안의 range 판정은 오류) **4/4 모두 구분 불가**: DILI 0.009 · hERG 0.020 · AMES 0.016 · LD50 0.013 — 전부 각 기준선 미만. 초안의 'LD50 G4만 확실'은 **철회**.
- 같은 D-MPNN이 분할이 바뀌면 hERG 0.746→0.797로 움직인다 → **분할을 섞으면 안 되는 이유의 크기**를 보여준다.

## 한계

- 하이퍼파라미터 탐색 없음(위 참조). "D-MPNN의 최대 성능"이 아니라 "동일 조건에서의 성능"이다.
- 5 seed는 **train/valid 파티션 변동 + 모델 초기화 변동**을 함께 담는다. test는 고정이므로 이 SD는 **일반화 불확실성이 아니다**(본 보고서 §6과 동일한 한계).
- 회귀(LD50)의 과적합 gap 0.22는 크지만, 이는 MAE 단위 차이라 분류 AUROC gap과 직접 비교할 수 없다.

## 산출물

```
src/train_dmpnn.py       분할 동일성 게이트 + 20런 학습(resume 지원)
src/aggregate_dmpnn.py   집계 + 누수 정정 계산
results/dmpnn_metrics.json   엔드포인트별 test/train 평균±SD + _config + _failed
results/dmpnn_raw.jsonl      20런 원자료(seed별)
results/overfit.json         train vs test 격차
results/leakage.json         정확분자/InChIKey/골격 중복 + test 동일성 Jaccard
results/train.log            학습 로그
```
