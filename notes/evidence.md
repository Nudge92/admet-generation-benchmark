# 근거 색인 — 무엇을 보고 정정했나

`self_corrections.md`의 8건 각각이 **어느 파일의 어느 줄에서** 일어났는지 적는다.

주장만 있고 근거 경로가 없으면 검증할 수 없다. 이 문서는 그 간격을 메우기 위한 것이고,
저장소에 커밋된 실제 파일만 가리킨다.

> 경로는 저장소 루트 기준이다. `experiments/` 아래는 실험 당시의 원본 트리를 그대로 옮긴 것이다.

📊 아래 표를 그림으로 본 것: `docs/assets/selfcorrection/sc00_timeline.png`
   (각 정정이 어느 날짜 폴더의 어느 실험에서 일어났나)

---

## 한눈에

| # | 정정 | 1차 근거 (판단이 바뀐 지점) | 뒷받침 데이터 |
|:--:|---|---|---|
| ① | 누수 프리미엄 → 상한 | `experiments/2026-07-22/experiment_generation_matrix/notes.md:81` | `.../results/leakage.json` · `experiment_gen_expansion_g1g3/notes.md:53-66` |
| ② | range 판정 → 1위·2위 격차 | `experiments/2026-07-22/experiment_generation_matrix/notes.md:86` | `experiment_g3_dmpnn_seed42/notes.md:75` |
| ③ | LD50 G4 승 철회 | 위와 같은 줄 (②의 직접 결과) | `results/master_matrix.csv` · `experiments/2026-07-22/master_report/results/consistency.json` |
| ④ | Hanley-McNeil → DeLong | `experiments/2026-07-22/master_report/notes.md:47-48` | `results/finalize_check.json` · `master_report_before_final.html` |
| ⑤ | 멀티태스크 이득 → 기각 | `experiments/2026-07-22/experiment_adme_full/notes.md` (축③) | `.../results/learning_axis.csv` |
| ⑥ | 물성 18/18 → 17/18·16/18 | `experiments/2026-07-22/experiment_adme_full/src/rf_ecfp_symmetric.py` (머리말) | `results/feature_2x2.csv` · `.../results/feature_ablation.csv` |
| ⑦ | 배설 3종 → 반감기만 | `experiments/2026-07-22/experiment_adme_full/notes.md` (축①) | `results/adme_matrix.csv` |
| ⑧ | seed 처리 의존 | `experiments/2026-07-24/experiment_g4_verification/notes.md` (결론 3줄) | `results/g4_verification.json` · `experiment_adme_reliability/notes.md:75-76` |

---

## ① 누수 프리미엄 → 누수 상한

**판단이 바뀐 문장** — `experiment_generation_matrix/notes.md:81`

> ★★중대 정정(적대검증): 이 0.088을 "누수 프리미엄"이라 부르면 안 된다. 패키지 실측 결과
> ADMET-AI는 5-모델 앙상블 × 31과제 멀티태스크·튜닝된 배포본이고, 우리 D-MPNN은
> 단일모델·단일과제·미튜닝이다 … 따라서 0.088에는 누수 + 앙상블 + 멀티태스크 전이 + 튜닝이
> 교락돼 있고 이는 **누수의 상한**이다.

**어떻게 실측했나** — 배포 패키지를 직접 뜯었다. `admet_ai/resources/models/admet_{classification,regression}/model_0..4.pt` 파일 5개, 체크포인트의 `output_columns` 길이 31/10, `_make_ensemble_predictions` 호출.

**누수 판정 근거 3건** — `experiment_generation_matrix/notes.md:42` 이하

1. TDC 전체로 사전학습
2. 자체 논문 보고치를 우리 test에서 초과 → **누수 크기의 가장 깨끗한 추정 +0.049~0.074**
3. 정직 학습 동종 모델과의 격차 → 상한 (핵심 0.0876 · 확장 0.1491, `experiments/2026-07-22/master_report/results/consistency.json`)

**★한계까지 적어뒀다** — 같은 파일 `:55`

> 공정 보정 결과: 누수 보정 불가. ADMET-AI의 학습 분자 목록이 공개돼 있지 않아
> "학습셋에 없는 분자"를 정의할 수 없다. … "공집합임을 실측했다"가 아니라
> **"하위셋을 정의할 수 없다"가 참이다.**

📎 `experiments/2026-07-22/experiment_g3_dmpnn_seed42/results/leakage.json`
   — 엔드포인트별 exact/InChIKey/골격 중복 수 + test 동일성 Jaccard

---

## ② 세대 구분선의 측정 방식

**판단이 바뀐 문장** — `experiment_generation_matrix/notes.md:86`

> 초안은 판정에 세대별 최고−최저 **폭(range)**을 썼는데, 그 폭은 **최하위 세대**가 만든 값이라
> 1위의 우위를 전혀 함의하지 않는다(적대검증 지적).

**바꾼 뒤 결과** — 핵심 4종 전부 구분 불가.

| 엔드포인트 | 1위·2위 격차 | 구분 가능선 |
|---|---:|---:|
| DILI | 0.009 | 0.111 |
| hERG | 0.020 | 0.090 |
| AMES | 0.016 | 0.028 |
| LD50 | 0.013 | 0.043 |

**★기준선 자체도 정정했다** — LD50의 구분 가능선은 처음에 **누수된 ADMET-AI의 잔차**로 계산돼 0.023으로 과소평가돼 있었다. 우리 모델 MAE 수준으로 스케일해 **0.043**이 됐다. 즉 이 항목은 *측정 대상*과 *기준선*을 둘 다 고쳤다.

📎 `experiments/2026-07-22/experiment_g3_dmpnn_seed42/notes.md:75` (같은 정정의 독립 기록)

---

## ③ LD50 G4 승 철회

②의 직접 결과다. 별도 분석이 아니라 **기준을 바꾸자 자동으로 무너진 것**이다.

같은 줄이 정확한 진술도 남겨뒀다:

> 정확한 진술은 *"최신 세대가 고전을 이긴다는 증거가 없다"*이며,
> 마찬가지로 *"고전이 우월하다"*는 증거도 없다.

📊 `docs/assets/selfcorrection/sc03_ld50_withdrawn.png`
📎 `experiments/2026-07-22/master_report/results/consistency.json` — `best_gen_dist {G2:17, G4:1}` · `g3_first_place: 0`

---

## ④ Hanley-McNeil → DeLong

**판단이 바뀐 문장** — `master_report/notes.md:47`

> 구분 가능선은 Hanley-McNeil **비대응 가정**이라 보수적 → 하한 경보로만.
> ★최종화에서 **DeLong 정확 대응비교로 보강했다** — 이 한계는 §6에서 해소됐다.

**무엇이 이걸 가능하게 했나** — `master_report/notes.md:48`

> ~~분자별 예측 미저장으로 부트스트랩·DeLong 불가~~ → ★`experiment_deploy_reliability`에서
> 챔피언의 분자별 예측을 만들어 부트스트랩 CI·DeLong을 산출했고 §6·§11에 반영했다.

즉 ④는 "검정을 바꿨다"가 아니라 **"검정을 가능하게 만드는 데이터를 새로 만들었다"**가 먼저다.

**★정정 전 문서가 그대로 남아 있다**

- `experiments/2026-07-22/master_report/results/master_report_before_final.html` — DeLong 반영 전
- `experiments/2026-07-22/master_report/results/master_report_before_polish.html` — 그 이전
- `results/finalize_check.json` — 전후 숫자 대조 (`n_numbers_before/after` 464 · `numbers_identical: true`)

**★검증식 자체도 한 번 고쳤다** — `master_report/notes.md:144`

> 처음엔 "숫자 multiset 동일"로 검사했는데 (B)가 중복 행을 지우므로 **당연히 줄어드는** 것을
> 실패로 잡았다. "새 숫자 0 + 고유값 집합 동일 + 감소분이 §8 중복분과 정확히 일치"
> 세 조건으로 바꿔 확인했다.

📎 `experiments/2026-07-22/master_report/results/polish_check.json` — 표 재정렬 before/after 전문

---

## ⑤ 멀티태스크 이득 → 가설 기각 ★자체 발견

**판단이 바뀐 문장** — `experiment_adme_full/notes.md` (축③)

> ★자체 발견한 누수로 결론 뒤집힘. CYP 3종이 같은 분자 라이브러리(라벨만 다름)라
> **합집합 train에 test의 88.9%가 섞였다** → 누수 차단 후 재계산하니 "닮은 과제 이득" 가설
> **기각**(CYP 억제 +0.0004 · 기질 −0.024). 남은 이득은 소표본 2개에 집중.

**88.9%가 무엇인지 정확히** — *이득 중 누수가 차지한 비율이 아니라*, 멀티태스크 학습의
합집합 train에 섞여 들어간 **test 분자의 비율**이다. 누수의 원인이지 효과의 크기가 아니다.

**원자료** — `experiments/2026-07-22/experiment_adme_full/results/learning_axis.csv`
그룹(`all_adme_cls` / `cyp_inhibition` / `cyp_substrate`)별 멀티태스크 − 단일 델타.
`cyp_inhibition` 세 값 (−0.0061, +0.0108, −0.0034)의 평균이 **+0.0004**로 재계산된다.

📊 `docs/assets/selfcorrection/sc05_multitask.png`
📎 초기(누수) 관측 3그룹 — `experiments/2026-07-22/experiment_adme_full/src/build_final.py:234`
   (`cyp_inhibition` 3/3 · `cyp_substrate` 2/3 · `all_adme_cls` 9/10)

---

## ⑥ 물성 18/18 → 17/18 · 16/18

**판단이 바뀐 문장** — `experiment_adme_full/src/rf_ecfp_symmetric.py` 머리말

> 적대검증 확정 HIGH: "물리화학 18/18 ECFP 압승"은 physchem에 2모델(xgb+rf)·ECFP에 1모델만 준
> **best-of-2 산물**이었다. rf_ecfp(빠져 있던 4번째 조합)를 추가해 **2×2 대칭(모델×특징)으로
> 재집계** → 진짜 '특징 효과'를 격리한다. 기존 예측·기록은 건드리지 않고 별도 파일에만 쓴다.

**이 항목의 근거는 문서가 아니라 실행 코드다.** 빠져 있던 조합을 실제로 돌려서 2×2를 채웠다.

📊 `docs/assets/selfcorrection/sc06_asymmetric_roster.png`
📎 `results/feature_2x2.csv` (2×2 결과) · `.../results/rf_ecfp_raw.jsonl` (원자료)
📎 `.../results/feature_ablation.csv` — 축② 특징 기여도

---

## ⑦ 배설 3종 → 반감기만

**판단이 바뀐 문장** — `experiment_adme_full/notes.md` (축①)

> ★적대검증 정정: **반감기는 결정적, 두 청소율은 G2와 구분 불가 동률**(간세포 Δ0.0025 ≪ sd 0.05)이라
> '3종 석권'은 과장(**방향은 진짜 · 표현 과했음**).

"방향은 진짜, 표현이 과했다"는 자기평가가 원문에 그대로 있다.

★원문이 든 `sd 0.05`는 **G2가 아니라 G4(Uni-Mol)의 seed SD**(0.0514)다. G2 최고(`rf_physchem`)의 SD는 0.0167이고,
어느 쪽을 기준으로 잡아도 "동률" 판정은 바뀌지 않는다. `self_corrections.md` ⑦의 표에 둘 다 적어뒀다.

📊 `docs/assets/selfcorrection/sc07_excretion.png`
📎 `results/adme_matrix.csv`

---

## ⑧ seed 처리 의존 — 그리고 그 뒤 주 판정이 바뀌었다

**1단계 (07-24)** — `experiment_g4_verification/notes.md` 결론 3줄

> 1. 반감기 — 주 판정(A: 5seed 예측평균)에서 G4 우세 구분됨 … ★단 보수 판정(B: seed별)에서는
>    CI [−0.018, +0.325]로 구분 안 됨.
> 2. 수용해도 — … ★단 (B)에서는 구분 안 됨. 게다가 **G4가 3/5 seed**(1·2 OOM 실패).
> 3. **여전히 미검증** — VDss·간세포청소율·마이크로솜청소율 3개는 검증하지 않았다.
>    **이 결과를 그 3개로 일반화하면 안 된다.**

**★2단계 (07-25) — 주 판정이 A에서 B로 바뀌었다** — `experiment_adme_reliability/notes.md:75-76`

> ★주 판정 = **seed별 부트스트랩 후 종합**. 이유: 이 연구의 주장은 특정 배포물이 아니라
> **방법 비교**이므로 **학습 변동을 CI에 포함**해야 정직하다.
> '5seed 예측평균'도 병기하되 그것은 5-모델 앙상블 평가이며 **변동이 큰 모델(Uni-Mol)에
> 비대칭적으로 유리**하다 — G4 검증 실험에서 이 차이가 판정 자체를 뒤집었다.

따라서 `results/g4_verification.json`의 `primary_method: "A(5seed 예측평균…)"`은 **07-24 시점의 값이고,
하루 뒤 B로 교체됐다.** 파일을 수정하지 않고 남겨둔 이유는 그때의 판단을 보존하기 위해서다.
현재의 주 판정은 B이며, **두 판정 모두 보고한다.**

**부수 기록**
- seed 3/5 (OOM 실패): `experiments/2026-07-24/.../results/g4_raw.jsonl`
- conformer 생성 실패 분자와 사유: `experiments/2026-07-24/.../logs/conformer_failed.json`
  (`ETKDG embed failed` — 반감기 6/135, 수용해도 17/1995)
- 재현 검증: `.../results/step1_repro.json` (seed별 값 포함) · `step2_repro.json` · `step0_split_check.json`

📊 `docs/assets/selfcorrection/sc08_seed_flip.png`

---

## 8건에 안 들어간 기록들

자기정정으로 번호를 붙이지는 않았지만 같은 성격의 기록이다. 남겨둔다.

**재현 방식** — `experiments/2026-07-22/experiment_deploy_reliability/results/reproduction_check.json`

> 모델 아티팩트가 저장돼 있지 않아 로드 불가 → 동일 config·seed·분할로 **재현 학습 후 추론**.
> 모든 예측은 ★재현본임을 flag한다.

18개 전부 허용오차 0.005 안, `n_fail: 0`. logD는 별도로 `logd_g3_repro.json`(diff 0.0).

**"동일 config"가 틀린 표현이었다** — `experiment_g3_dmpnn_seed42/notes.md:36`

> ★단 **에폭이 다르다** — seed=1 런은 30, 이번은 50. 따라서 "동일 config"라는 표현은 틀렸고,
> 두 런의 차이를 **분할 효과로만 귀속할 수 없다**.

**변수명이 틀렸던 것** — 같은 파일 `:32`

> `train_dmpnn.py`가 처음 찍은 `train_test_exact_overlap`은 실제로 **InChIKey14(골격) 중복**이었다.
> 정확분자/전체InChIKey/골격 셋을 따로 계산해 이름을 바로잡았다.

**실행 사고 2건** — `experiment_gen_expansion_g1g3/notes.md:34,36`

> ★사고 1: G2 첫 실행에서 210건 중 95건 실패. RDKit `Ipc` 같은 초대형 서술자가 float64에선
> 유한하나 **float32 캐스팅에서 inf**가 됐다. 고친 뒤 — **성공분 115건도 전처리가 다르므로
> 전량 폐기하고 재학습**했다(섞으면 표가 오염된다).
>
> ★사고 2: Tox21 G3가 5/5 실패. chemprop env의 RDKit이 알루미늄 화합물을 거부.
> 필터로 해결하고 **분자셋이 바뀌었으므로 G1·G2·G3 전부 재실행**했다.

**지표 방향 단위 테스트** — `experiment_adme_reliability/notes.md:28`

> MAE↓ 4개, Spearman↑ 4개, AUROC↑ 10개가 섞여 있어 부호 실수 시 순위가 통째로 뒤집힌다.
> 코드에 테스트를 박아 검증했다(4/4 통과).

**logD 외과적 병합** — `experiment_adme_reliability/notes.md:26`

> logD는 챔피언 순번 4번이라 전체 재실행 시 RNG가 뒤 13개 CI를 바꿈 → 단독 계산 후 기존
> CSV·detail에 **외과적 병합**(나머지 17개 파일수준 변경 셀 **0** 확인).
