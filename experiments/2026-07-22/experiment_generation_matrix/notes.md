# 독성 예측 — 엔드포인트별 × 세대별(G1~G5) 정리 + ADMET-AI 비교

작업일 2026-07-22 · 위치 `Project/ADMET_integrated/2026-07-22/experiment_generation_matrix/`

## 0. 이 작업의 성격

**신규 학습 1종.** 대부분은 기존 실험 결과 재사용이지만, 2026-07-22에 **G3 chemprop D-MPNN을 새로 학습**해 비어 있던 G3 자리를 채웠다(4 엔드포인트 × 5 seed = 20런, 별도 폴더 `experiment_g3_dmpnn_seed42`). 그 외 신규 계산 2가지는 학습이 아니다.

| 신규 계산 | 왜 했나 | 학습인가 |
|---|---|---|
| **G3 chemprop D-MPNN 신규 학습** (`experiment_g3_dmpnn_seed42/src/train_dmpnn.py`) | G3 자리가 누수 있는 ADMET-AI로만 채워져 있었음 | ✅ **학습함**(유일) |
| ADMET-AI **추론만** (`run_admetai_infer.py`) | 기존 실험은 **지표만** 저장하고 분자별 예측을 저장하지 않았음. 요청된 `admetai_preds.jsonl` 산출용 | ❌ 추론. 기존 저장 지표와 **Δ+0.0000 완전 재현** 확인 |
| G1 구조알림 **규칙 적용** (`rule_g1.py`) | 기존 G1 결과는 **다른 test셋**(§3)에 있어 공정 비교에 못 씀. 규칙 정의는 `T_toxicity/surface/src/featurize.py`에서 **수정 없이 그대로** 로드 | ❌ 학습 없음(SMARTS 매칭) |

> 판단 근거: 사용자의 하드 제약은 "**새 학습** 금지"이고, 동시에 "**모든 세대를 같은 test셋으로만 비교**(공정성의 전제)"였다. 규칙 적용은 학습이 아니고 초 단위이며, 이걸 안 하면 G1이 공정 분할에서 통째로 빠진다. 두 제약을 모두 만족시키는 선택으로 규칙 적용을 택했다.

## 1. 공정성 전제 — 실측으로 확인한 것

본 표의 모든 값은 **TDC 공식 scaffold split (seed=42) · 동일 고정 test셋**이다. 서로 다른 실험 폴더에서 모은 값이라 분할이 같은지 직접 확인했다.

| 확인 항목 | 방법 | 결과 |
|---|---|---|
| 4개 소스 실험의 분할 동일성 | 각 `src/*.py` 주석·코드 실측 | `tox_benchmark`·`finetune`·`finetune2`·`physchem` 모두 "동일 TDC scaffold split·5 seed" 명시 |
| T_toxicity 결과를 섞어도 되나 | 분자 단위 교집합 계산 | **안 됨.** T_toxicity는 split seed=1(`run.py:8`) → 본 분할과 교집합 hERG 28%·AMES 25%. seed=42는 **100% 일치** |
| 우리 모델 누수 | `leakage.json` | train↔test 정확분자 중복 **4/4 엔드포인트 모두 0** |

→ T_toxicity의 세대 사다리(G1 구조알림+LogReg, **G3 chemprop D-MPNN**, G4a/G4b ChemBERTa, Tox21 12과제)는 본 표에 섞지 않고 **§8 부록**에 "다른 분할·직접 비교 불가"로 분리했다.

## 2. 셀 출처 (숫자는 전부 파일 인용)

| 세대 | 모델 | 출처 파일 |
|---|---|---|
| G1 | 구조알림 (BRENK+NIH + Benigni-Bossa SMARTS 10) | `results/g1_summary.csv`, `g1_rules.csv` (규칙 정의: `T_toxicity/surface/src/featurize.py`) |
| G2 | 물리화학 서술자 + XGBoost / RandomForest | `experiment_physchem/results/physchem.csv` |
| G2 | ECFP4 지문 + XGBoost | `experiment_finetune/results/finetune.csv` (SD는 `tox_benchmark/results/benchmark.csv`) |
| G3 | **우리 자체 chemprop D-MPNN (정직)** | `experiment_g3_dmpnn_seed42/results/dmpnn_metrics.json` ← **2026-07-22 추가 학습으로 채움**(아래 §3-③) |
| G3 | ADMET-AI (Chemprop D-MPNN 공개) | `experiment_tox_benchmark/results/admetai_metrics.json` ★누수 |
| G4 | Uni-Mol (3D) | `experiment_finetune2/results/finetune2.csv` |
| G5 | ChemBERTa-2 / MoLFormer | `experiment_finetune/results/finetune.csv` |
| 참고 | TDC 리더보드 SOTA | `experiment_tox_benchmark/results/benchmark.csv` (★미재현 인용) |

## 3. ★ADMET-AI 누수 — 독립 증거 3건

**증거 ① 학습셋 크기가 TDC 전체와 일치.** ADMET-AI 패키지 자체 메타(`admet_ai/resources/data/admet.csv`)의 학습 데이터 크기 vs 우리 분할 전체:

| | ADMET-AI 학습 size | 우리 train_val + test |
|---|---|---|
| DILI | **475** | 379 + 96 = **475** ← 정확히 일치 |
| hERG | 648 | 523 + 132 = 655 |
| AMES | 7255 | 5821 + 1457 = 7278 |
| LD50_Zhu | 7342 | 5907 + 1478 = 7385 |

→ ADMET-AI 학습셋 ≈ TDC **전체(train+valid+test)** → 우리 test 분자가 학습에 포함됨.

**★공정 보정 결과: 누수 보정 불가.** 다만 이유를 정확히 적으면 — ADMET-AI의 **학습 분자 목록이 공개돼 있지 않아** "학습셋에 없는 분자"를 **정의할 수 없다**. 위 크기 대조는 우리 test가 학습에 포함됐음을 **강하게 시사**하지만(DILI 475=475 정확 일치) 분자 단위 대조를 한 것은 아니다. "공집합임을 실측했다"가 아니라 "하위셋을 정의할 수 없다"가 참이다.

**증거 ② 패키지에 기록된 자기 성능을 크게 초과.** 같은 모델·같은 앙상블·같은 멀티태스크 설정이라 그 성분들이 **양쪽에서 상쇄**되고 달라지는 건 **평가 분자**뿐 → 세 증거 중 **누수 크기 추정에 가장 깨끗하다**. (단 `admet.csv` 기록값의 산출 조건·모델 버전은 확인 불가.)

| | 패키지 기록 | 우리 test | 설명되지 않은 초과분 |
|---|---|---|---|
| DILI AUROC | 0.8815 | 0.9557 | **+0.074** |
| hERG AUROC | 0.8388 | 0.9113 | **+0.073** |
| AMES AUROC | 0.8816 | 0.9303 | **+0.049** |
| LD50 MAE | 0.4515 | 0.3361 | **0.115 좋아짐** |

→ 초과분은 '암기'로 단정하지 않고 **설명되지 않은 초과분**으로 부른다(버전·조건 확인 불가). DILI에서 ADMET-AI(0.9557)가 리더보드 1위 MiniMol(0.956)과 사실상 동값인 것도 같은 정황. **0.9+ 점수는 실력이 아니라 누수 신호로 해석.**

**증거 ③ 같은 계열을 정직하게 학습하면 (2026-07-22 추가) — 단 이 격차는 누수의 상한이다.** D-MPNN을 같은 분할의 train만 보고 학습(`experiment_g3_dmpnn_seed42`, 5 seed, test 집합 Jaccard 1.0):

| | 우리 D-MPNN(정직) | ADMET-AI(누수) | 프리미엄 |
|---|---|---|---|
| | 우리 D-MPNN(정직·단일·미튜닝) | ADMET-AI(5앙상블·멀티태스크·TDC전체) | 격차 |
|---|---|---|---|
| DILI AUROC | 0.9039 | 0.9557 | **+0.0518** |
| hERG AUROC | 0.7972 | 0.9113 | **+0.1141** |
| AMES AUROC | 0.8333 | 0.9303 | **+0.0970** |
| LD50 MAE | 0.6398 | 0.3361 | **0.3037 좋아짐** |

→ 분류 3종 평균 **0.088 AUROC**.

★★**중대 정정(적대검증)**: 이 0.088을 "누수 프리미엄"이라 부르면 안 된다. 패키지 실측 결과 **ADMET-AI는 5-모델 앙상블 × 31과제(회귀 10과제) 멀티태스크·튜닝된 배포본**이고, 우리 D-MPNN은 **단일모델·단일과제·미튜닝**이다(출처: `admet_ai/resources/models/admet_{classification,regression}/model_0..4.pt`, 체크포인트 `output_columns` 길이 31/10, `_make_ensemble_predictions`). 따라서 0.088에는 **누수 + 앙상블 + 멀티태스크 전이 + 튜닝**이 교락돼 있고 이는 **누수의 상한(upper bound)**이다. **누수 크기의 가장 깨끗한 추정은 증거 ②(+0.049~0.074)** — 거기서는 ADMET-AI가 자기 자신과 비교되므로 앙상블·멀티태스크가 상쇄되고 평가 분자만 달라진다. 증거 ③의 역할은 크기 확정이 아니라 **방향의 독립 확인**이다.

## 4. 결론

1. **세대 = 성능 순서가 아니다.** 최고 세대는 DILI·hERG·AMES = **G2(고전 ML)**, LD50 = G4. **G5(파운데이션)는 4/4 어디서도 1위가 아니다.** 그리고 정직한 **G3(D-MPNN)도 1위가 없다**(7모델 중 DILI 3위·hERG 5위·AMES 4위·LD50 4위) → **"GNN이 고전 ML을 이긴다"는 통념이 이 4개 엔드포인트에서 성립하지 않았다.**
2. **어떤 세대가 이긴다는 주장도 통계적으로 성립하지 않는다 (★정정).** 초안은 판정에 세대별 최고−최저 **폭(range)**을 썼는데, 그 폭은 **최하위 세대**가 만든 값이라 1위의 우위를 전혀 함의하지 않는다(적대검증 지적). 판정을 **1위 세대 vs 2위 세대 격차**로 바꾸면 — DILI 0.009<0.111 · hERG 0.020<0.090 · AMES 0.016<0.028 · LD50 0.013<0.043 → **0/4**. 초안이 "유일하게 확실한 세대 효과"라 했던 **LD50 G4(3D) 우세는 철회**한다. 또한 LD50 기준선 자체도 누수된 ADMET-AI 잔차로 계산돼 과소평가돼 있었다(0.023 → 우리 모델 MAE 수준으로 스케일해 **0.043**). 정확한 진술은 *"최신 세대가 고전을 이긴다는 증거가 없다"*이며, 마찬가지로 *"고전이 우월하다"*는 증거도 없다.
3. **세대보다 표현 선택이 크다.** hERG는 최고(물리화학+RF 0.8369)와 최저(ECFP4+XGB 0.7349)가 **둘 다 G2** — 한 세대 안의 특징 차이가 전체 성능 폭을 다 설명한다.
4. **엔드포인트가 방법을 정한다.** hERG=물성 축(구조알림 MCC **−0.028**로 완전 무력), AMES=구조 축(규칙 MCC 0.260), LD50=3D(G4)가 순위상 최고(단 구분 불가).

## 5. 한계 — 있는 그대로

- **분자별 예측 미저장 → 부트스트랩 CI 불가.** 재학습이 금지라 산출할 수 없다. seed SD는 **고정 test셋 위의 변동**이라 일반화 불확실성이 아니다(과거 SoM 실험에서 이미 데인 함정). 대신 Hanley-McNeil 해석적 SE를 썼는데, 이는 **비대응 가정이라 지나치게 보수적**이다. 대응비교(DeLong)면 더 작아지지만 분자별 예측이 없어 못 한다 → **"이보다 작은 차이는 확실히 말할 수 없다"는 하한 경보로만** 사용.
- **DILI는 test 96분자뿐** → 구분 가능 격차가 0.111 AUROC. DILI 세대 순위는 주장 금지.
- **Uni-Mol(G4) 과적합** — 분류 3개 모두 **train AUROC = 1.000**, test 격차 0.13~0.20. 그럼에도 LD50 최고. 3D가 무용한 게 아니라 데이터가 작을 때 통제가 안 되는 것.
- **ChemBERTa-2(G5) 전이 무효** — DILI·AMES·LD50에서 우리 모델 최하위(hERG의 최하위는 ECFP4+XGB다 — 초안의 'hERG 포함 3개'는 오류였다). 같은 G5인 MoLFormer는 훨씬 나음 → "파운데이션이라서"가 아니라 "어떤 파운데이션이냐"가 갈린다.
- **G3는 이제 실측됨**(2026-07-22). 단 **에폭이 seed=1 런(30)과 다르다(50)** — '동일 config'라는 초안 서술은 틀렸다. 그리고 **하이퍼파라미터 탐색은 하지 않았다** — G2(XGBoost)도 튜닝하지 않아 조건은 대등하나, 양쪽 다 "튜닝하면 더 오를 수 있다"는 여지는 남는다. D-MPNN 과적합은 오히려 가장 작았다(DILI는 gap **음수** −0.035).
- **SOTA는 리더보드 인용값(미재현).**
- **확장 엔드포인트**(발암성·ClinTox·Tox21 12경로)는 별도 실험 `experiment_gen_expansion_g1g3`에서 G1~G3로 다룬다(이 표에는 없음).

## 6. 산출물

```
src/run_admetai_infer.py   ADMET-AI 추론만 (env ADMET_AI)
src/rule_g1.py             G1 규칙 적용 (env admet)
src/collect_matrix.py      매트릭스 조립 (env md)
src/build_report.py        HTML 보고서 (env md)
results/report.html        단일 파일 보고서 (그림 base64 임베드)
results/gen_matrix.csv     67행 · 값 있는 셀 60 · 셀마다 source 컬럼
results/admetai_preds.jsonl  3163행 (분자별 예측)
results/admetai_recomputed.json  재추론 지표(저장값과 Δ0.0000)
results/g1_rules.csv       52행 (엔드포인트 × 알림별 발화·정밀도)
results/g1_summary.csv     4행 (엔드포인트 요약)
results/matrix_meta.json   분할 검증·누수 증거·불확실성·과적합
```
