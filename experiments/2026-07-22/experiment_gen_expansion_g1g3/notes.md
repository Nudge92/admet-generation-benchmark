# 독성 세대 사다리 확장 — 발암성 · ClinTox · Tox21 (G1~G3)

작업일 2026-07-22 · `Project/ADMET_integrated/2026-07-22/experiment_gen_expansion_g1g3/`
스코프 **G1~G3만** (G4·G5 파인튜닝 없음). 플래그십(핵심 4종 DILI/hERG/AMES/LD50) 방법론 계승.

## 1. 분할 — 한 번만 만들어 파일로 고정

`prep_splits.py`가 분할을 **한 번** 만들어 CSV로 저장하고, G1·G2·G3·ADMET-AI가 **모두 그 파일만** 읽는다 → 엔드포인트 내 test 동일성이 구조적으로 보장된다(사후 비교가 아니라 설계로 보장).

| 엔드포인트 | 분할 | train/valid/test | train∩test 정확분자 | chemprop 비호환 제외 | 재현 |
|---|---|---|---|---|---|
| 발암성(Carcinogens_Lagunin) | `Tox.get_split(scaffold, seed=42)` | 196/28/56 | **0** | 0 | OK |
| ClinTox | `Tox.get_split(scaffold, seed=42)` | 1034/147/297 | **0** | 0 | OK |
| Tox21 (12과제) | 합집합 + `create_scaffold_split(seed=42)` | 5474/783/1566 | **0** | 8 | OK |

- **재현성**: 같은 분할을 두 번 생성해 완전 일치 확인(3/3).
- **chemprop 호환 필터**: 두 env(`admet` vs `ADMET_AI`)의 RDKit 버전 차이로 chemprop이 거부하는 분자(`CC(=O)O[AlH3](O)O` 등 8개)를 **분할 이후** 전 세대에서 함께 제외했다(플래그십 `chemprop_bad_smiles` 방식 계승). 분할 자체는 건드리지 않았다.

### ★Tox21 분할에 대한 정직한 고지
Tox21 12과제는 **서로 다른 분자 부분집합**이라, '과제별 TDC 공식 분할'로는 멀티태스크 D-MPNN과 과제별 G2를 **같은 test에서** 비교할 수 없다. 그래서 12과제 **합집합** 분자셋에 TDC 내부 함수 `create_scaffold_split(seed=42, frac 0.7/0.1/0.2)`를 **한 번** 적용했다. 방법론(Bemis-Murcko)은 TDC와 같지만 **과제별 공식 분할과는 다르므로 이 Tox21 숫자를 TDC 리더보드와 직접 비교하면 안 된다.**

## 2. 세대별 설정

| 세대 | 모델 | 비고 |
|---|---|---|
| G1 | BRENK+NIH + Benigni-Bossa SMARTS 10 | 학습 없음 · 플래그십과 **같은 규칙셋**(`featurize.py` 무수정 로드) · 지표 형태 다름 |
| G2 | 물리화학(RDKit 210)+XGBoost / +RandomForest(500) / ECFP4(2048)+XGBoost | 플래그십과 동일 3종·동일 하이퍼파라미터 · 5 seed |
| G3 | chemprop D-MPNN (정직) | train만 학습·valid로만 early stopping·test 1회 · **Tox21은 12과제 멀티태스크**(결측 마스킹) · 순수(외부특징 없음) |
| G3 | ADMET-AI | **★누수 의심** · 고정 비교선으로만 |

- **5 seed의 의미가 플래그십과 다르다**: 플래그십 핵심 4종은 `get_train_valid_split(seed)`로 train/valid 파티션까지 변주했지만, 여기서는 **분할이 seed=42로 고정**이고 seed 1~5는 **모델 초기화만** 변주한다 → 여기의 SD는 일반화 불확실성이 아니라 **초기화 변동**일 뿐이다.
- 실패 **0건** (G2 210런 · G3 15런).

> ★사고 기록 1: G2 첫 실행에서 210건 중 95건이 실패했다. 원인은 RDKit `Ipc` 같은 초대형 서술자가 float64에선 유한하지만 **float32 캐스팅에서 inf**가 되는 것. 플래그십 `physchem_run.py`는 대치 후 **StandardScaler**를 적용해 이를 피하고 있었는데 내가 그 단계를 빠뜨렸다. 고친 뒤 — **성공분 115건도 전처리가 다르므로 전량 폐기하고 재학습**했다(섞으면 표가 오염된다).
>
> ★사고 기록 2: Tox21 G3가 5/5 실패. chemprop env의 RDKit이 알루미늄 화합물을 거부. 위 호환 필터로 해결하고 **분자셋이 바뀌었으므로 G1·G2·G3 전부 재실행**했다.

## 3. 결과

| 엔드포인트/과제 | n_test | 양성률 | G1 MCC | G2 최고 | G3 정직 | ADMET-AI | 격차(상한) | 구분선 |
|---|---|---|---|---|---|---|---|---|
| 발암성 | 56 | 0.196 | **+0.358** | **0.837** | 0.757 | 0.870 | +0.113 | 0.235 |
| ClinTox | 297 | 0.081 | +0.093 | **0.801** | 0.689 | 0.899 | +0.209 | 0.154 |
| Tox21 (12과제 평균) | ~1200 | 0.026~0.255 | −0.03~+0.05 | **G2가 12/12 최고** | — | 0.901 | +0.10~0.21 | 0.048~0.127 |

### 핵심 발견

1. **G2(고전 ML)가 14/14 과제 전부에서 정직한 G3(GNN)를 앞섰다.** 플래그십(4종에서 G2 3승)보다 더 강한 패턴.
2. ★★**이 프로젝트에서 처음으로 '구분 가능한 세대 효과'가 나왔다 — 방향은 통념과 반대.** 14과제 중 **3개**(NR-Aromatase 0.803>0.707 · SR-ARE 0.795>0.732 · SR-MMP 0.904>0.837)에서 1위·2위 세대 격차가 구분 가능선을 넘고, **셋 다 G2 > G3**다. 플래그십 핵심 4종은 0/4였는데, 표본이 큰 Tox21에서 비로소 신호가 잡혔고 그 신호는 **"고전이 낫다"**를 가리킨다.
3. **G1은 발암성에서만 작동한다**(MCC +0.358 = 의미 있는 신호). ClinTox는 약하고(+0.093), **Tox21은 12/12 전부 무력**(−0.03~+0.05). Benigni-Bossa는 원래 변이원·발암 알림이므로 당연한 방향이며, 플래그십에서 AMES·DILI엔 듣고 hERG엔 무력했던 패턴의 **확장 재현**이다.
4. **ADMET-AI와의 격차는 +0.10~0.21로 플래그십(0.088)보다 크다.** 단 아래 한계 참조 — 이는 **누수의 상한**이다.

## 4. ★ADMET-AI와의 격차는 '누수 크기'가 아니라 '상한'

패키지 실측 결과 두 시스템은 **같은 모델이 아니다**:

| | ADMET-AI | 우리 D-MPNN |
|---|---|---|
| 모델 수 | **5-모델 앙상블** | seed당 단일 모델(5 seed는 지표 평균) |
| 과제 수 | **분류 31과제 멀티태스크** | 발암성·ClinTox 단일 / Tox21 12과제 |
| 하이퍼파라미터 | 튜닝된 배포본 | 탐색 없음(epochs 50·batch 50) |
| 학습 분자 | TDC 전체(우리 test 포함) | train 파티션만 |

출처: `admet_ai/resources/models/admet_classification/model_0..4.pt`(5개) · 체크포인트 `output_columns` 길이 31 · `_make_ensemble_predictions`.

→ 격차에는 **누수 + 앙상블 + 멀티태스크 전이 + 튜닝**이 교락돼 있다. **누수의 크기가 아니라 상한(upper bound)**으로만 읽어야 한다. (플래그십에도 같은 정정을 적용했다 — 초안은 이를 100% 누수로 귀속했고 적대 검증에서 CRITICAL로 지적받았다.)

## 5. 한계 — 있는 그대로

- **발암성은 test 56분자·양성 11개** → 구분 가능선 0.235. **순위 주장 자체가 불가능**하다.
- Hanley-McNeil은 **비대응 가정**이라 같은 test 위 두 모델 비교엔 지나치게 보수적 → "이보다 작으면 확실히 말할 수 없다"는 **하한 경보**로만 사용.
- **G3에 하이퍼파라미터 탐색을 하지 않았다.** G2도 안 했으므로 조건은 대등하나, 3개 과제의 "G2 > G3" 결론은 **이 조건에서**의 결과다.
- **불균형 처리가 세대 간 비대칭**: G3는 `--class-balance`를 쓰고 G2는 안 쓴다(플래그십 설정 계승). 이번 데이터는 양성이 2.6~25%로 훨씬 희박해 이 비대칭의 영향이 플래그십보다 클 수 있다. 다만 AUROC/AUPRC는 순위 지표라 재가중이 기계적으로 점수를 올리지는 않는다.
- **Tox21 합집합 분할**은 TDC 과제별 공식 분할과 다르다(§1).
- **기존 실험은 부록으로만**: `experiment_clintox_benchmark`는 train/valid/test 개수가 1034/147/297로 **우리와 완전히 같지만 분자 교집합은 Jaccard 0.134**다 — 개수가 같다고 같은 분할이 아니다(플래그십에서도 겪은 함정). 섞지 않았다.

## 6. 산출물

```
src/prep_splits.py        분할 1회 생성·고정 + 재현성/누수/호환 검증
src/rule_g1.py            G1 규칙 적용(학습 없음)
src/train_g2.py           G2 3종 × 5 seed (210런)
src/train_g3.py           G3 정직 D-MPNN × 5 seed (15런, Tox21 멀티태스크)
src/run_admetai.py        ADMET-AI 추론만 + 커버 확인
src/aggregate_expansion.py  집계·구분가능선·과적합
src/build_report.py       단일 HTML
results/report_expansion.html   보고서
results/expansion_metrics.json  엔드포인트×과제×세대 전체 수치
results/admetai_preds.jsonl     15097행(분자별 예측)
results/g1_rules.csv(182) · g1_summary.csv(14) · overfit.json
results/leakage.json · split_meta.json · g2_raw.jsonl(210) · g3_raw.jsonl(15)
splits/*.csv              고정 분할 9개 파일
```
