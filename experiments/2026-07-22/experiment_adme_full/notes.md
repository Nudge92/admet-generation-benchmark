# ADME 전면 벤치마크 (전수판) — 세대 × 특징 × 학습방식

작업 시작 2026-07-22 밤 · `Project/ADMET_integrated/2026-07-22/experiment_adme_full/`
독성 세대 분석(18과제)과 **동일 방법론**을 18개 ADME 엔드포인트에 3축으로 적용.

---

## ★아침 요약 (2026-07-24 완료)

**밤샘 무인 실행 완결 — 18개 ADME × 4세대(G2·G3·G4·G5) + ADMET-AI × 3축, 실패 격리 0 중단.**
완료 1114+조합. 오케스트레이터는 세션 종료로 두 번 죽었으나 매번 `progress.jsonl` 이어하기로 손실 없이 재개됨.

### 세 축 핵심 결과 (★잠정 — CI·구분가능선 미계산, 최종 판정은 부트스트랩·DeLong 후)
1. **축① 세대: 물성은 독성과 다르다.** 1위 세대 승수 **G2 9 · G4 5 · G3 4 · G5 0** (독성은 G2 17·G4 1·G3 0·G5 0). ★단 **G2는 3모델(xgb_physchem·rf_physchem·xgb_ecfp) 중 best**를 뽑는 다중비교 이점이 있어 승수는 세대 본질성능이 아님(부트스트랩 구분가능선 후 재판정 필요).
   ★**E(배설)에서 G4(Uni-Mol 3D)가 강하다** — 단 ★적대검증 정정: **반감기는 결정적, 두 청소율은 G2와 구분 불가 동률**(간세포 Δ0.0025≪sd 0.05)이라 '3종 석권'은 과장(방향은 진짜·표현 과했음). **물리화학 서술자 vs ECFP: ★동일 XGB 통제비교 17/18**(cyp3a4 기질만 ECFP 승 0.6456>0.6405)·**'18/18'은 physchem에 2모델(xgb+rf)·ECFP에 1모델만 준 비대칭 best-of-2 집계 산물**(rf_ecfp 부재)→②에서 2×2 대칭 재집계. **G5 파운데이션은 독성·물성 두 영역 모두 0승**.
2. **축② 특징: 비법은 압도적으로 '물성'.** 18개 중 16개에서 `+phys`(RDKit 210)가 최대 기여(PPBR +2.25·용해도 +0.52). 예상했던 이온화·3D·의약화학 규칙은 미미하거나 음수.
3. **축③ 멀티태스크: ★자체 발견한 누수로 결론 뒤집힘.** CYP 3종이 같은 분자 라이브러리(라벨만 다름)라 합집합 train에 test의 88.9%가 섞였다 → 누수 차단 후 재계산하니 "닮은 과제 이득" 가설 **기각**(CYP 억제 +0.0004·기질 −0.024). 남은 이득은 소표본 2개(생체이용률·HIA)에 집중.

### 실패·미완 (0으로 채우지 않음)
- G4 VDss seed 5: OOM 재시도(batch 8→4) 후에도 TorchScript 오류 → **VDss만 seed 4/5로 확정** 표기.
- bbb_martins G4: 초기 OOM 4건 → 재개 시 재시도로 **5/5 완주**(done_keys가 status==ok만 인정하도록 수정한 효과).

### 내가 만든 버그 2건(수정 완료)
- OOM 재시도의 실패 사유에 한글 → `write_rec`이 encoding 미지정으로 UnicodeEncodeError → G4 조기 종료. → 전 원장 쓰기에 encoding="utf-8".
- 3D 서술자가 조합마다 재계산될 뻔 → SMILES 키 디스크 캐시로 수십 시간 절약.

### 남은 일 (최종화)
부트스트랩·DeLong 세대 판정 → 축②·축③ 통합 → 독성 vs ADME 대비 → G4 실패 정직 표기 → report_adme_full.html.
적대 검증(축①·축②) 재개 필요 — 축③ 누수를 자체 발견했으나 나머지 축은 아직 독립 검증 미완.

---

|---|
| 실패 격리 | 조합마다 `try/except` → `logs/failures.jsonl`에 대상·에러·트레이스 기록하고 **다음 조합 진행**. 전체 중단 없음 |
| 즉시 저장 | 조합 완료 즉시 `results/progress.jsonl` append + `predictions/*.jsonl` |
| 이어하기 | 재실행 시 `progress.jsonl`의 완료 키를 읽어 **건너뜀** |
| 중간 리포트 | 각 단계 종료마다 `results/partial_report.html` 갱신 |
| 순서 | 가벼운 것부터 — 1 축①G2 → 2 축②특징 → 3 축③멀티태스크 → 4 축①G3 → 5 ADMET-AI → 7 축①G4 → 8 축①G5 → 9 리포트 |

## 2. 분할·공정성 (0단계 실측 완료)

- **TDC `admet_group` 공식 고정 test**(= scaffold seed=42) + `get_train_valid_split(split_type="default", seed=1..5)`. 독성 실험과 **완전히 같은 프로토콜**.
- **18/18 엔드포인트에서 train↔test 정확분자 중복 = 0** (`results/split_leakage.json`).
- **chemprop 비호환 분자**(두 env의 RDKit 버전 차이): `solubility_aqsoldb` **2개**뿐, 나머지 17개는 0. **분할 이후** 전 조합 공통 제외(`data/chemprop_incompatible.json`) — 독성 실험의 `chemprop_bad_smiles` 방식 계승.

## 3. 대상 — 18개 (기둥·task·주지표)

| 기둥 | 엔드포인트 | task | 주지표 |
|---|---|---|---|
| A 흡수(6) | Caco2 · HIA · Bioavailability · Pgp · Lipophilicity · Solubility | 회귀3·분류3 | MAE / AUROC |
| D 분포(3) | BBB · PPBR · VDss | 분류1·회귀2 | AUROC / MAE / **VDss는 Spearman** |
| M 대사(6) | CYP2C9·2D6·3A4 억제 + 2C9·2D6·3A4 기질 | 분류6 | AUROC |
| E 배설(3) | Half-life · Clearance(hepatocyte) · Clearance(microsome) | 회귀3 | **전부 Spearman** |

E와 VDss는 생리 지배·절대값 예측 한계 때문에 **Spearman을 주지표**로 삼았다.

## 4. 세 축

### 축① 세대 G2~G5 (G1 제외)
- G2: 물리화학 서술자+XGBoost / +RandomForest / ECFP4+XGBoost (독성과 동일 3종·동일 HP)
- G3: chemprop D-MPNN **정직**(train만·valid early stop·test 1회) + **ADMET-AI(★누수 기준선)**
- G4: Uni-Mol(3D) · G5: ChemBERTa-2 · MoLFormer
- **★G1(규칙)은 세대에서 뺐다** — toxicophore는 독성 전용이라 Caco-2·청소율에 부적용. 대신 의약화학 규칙(GSE·CNS MPO·이온화)을 **축②의 특징**으로 넣었다.

### 축② 특징 ablation (G2 XGBoost 고정 · 누적 스택)
`ecfp` → `+phys`(RDKit 210) → `+ion`(산/염기·pH7.4 하전·logD 근사) → `+3d`(ETKDG conformer PMI·구형성·PBF) → `+medchem`(GSE·CNS MPO·Lipinski/Veber·TPSA 비율)
각 단계의 **증분**을 기록해 "어느 특징이 얼마나 올리나"를 직접 제시.

### 축③ 학습방식 (멀티태스크)
- `cyp_inhibition` — CYP2C9·2D6·3A4 억제 3종 멀티태스크 D-MPNN (닮은 과제)
- `cyp_substrate` — CYP 기질 3종 멀티태스크 (작은 과제)
- `all_adme_cls` — 분류 10개 전부 멀티태스크 (잡탕 → 음의 전이 예상)
각각 **단일과제 D-MPNN(축①)과 대조**. 과거 발견(장기독성 멀티태스크 4/4 음의 전이)과 연결해 *"멀티태스크는 과제 유사도에 달렸다"*를 검증.

## 5. 코드 재사용 (재작성 안 함)

| 재사용 원본 | 용도 |
|---|---|
| `experiment_physchem/src/physchem_run.py` | G2 물리화학 레시피(210 서술자·대치·StandardScaler) |
| `experiment_tox_benchmark/src/bench_ourml.py` | G2 ECFP·분할 프로토콜 |
| `experiment_g3_dmpnn_seed42/src/train_dmpnn.py` | G3 정직 chemprop 설정 |
| `experiment_finetune/src/finetune.py` | G5 ChemBERTa-2·MoLFormer |
| `experiment_finetune2/src/unimol_run.py` | G4 Uni-Mol |

## 6. 앞선 4종 예비 실행에서 이미 겪고 고친 함정 (본 실행에 반영됨)

1. **`Ipc` float32 오버플로** — solubility test에 `Ipc = 6.08e158`인 분자가 있어(train 최대 9.9e60) 표준화값이 5.1e99 → 트리 모델이 float32 캐스팅에서 inf. → `prep()`에 **±1e6 클리핑** 추가. 실측상 caco2/bbb/cyp2c9의 표준화 최대는 25/1.6e4/3.0e4라 **다른 엔드포인트엔 no-op**.
2. **MoLFormer 원격코드 드리프트** — HF 캐시의 최신 스냅샷(`a14249e5`)이 `transformers.masking_utils`를 요구하는데 이 env(4.50.3)엔 없어 로드 실패. → 플래그십이 실제로 쓰던 **구버전 리비전 `7b12d946` 고정**.
3. **chemprop 비호환 분자** — env 간 RDKit 버전 차이. → 분할 후 전 조합 공통 제외.

## 7. 정직성 규율 (독성 실험 계승)

- **honest G3 필수** — train만 학습. ADMET-AI(누수)와 절대 혼동하지 않음.
- **ADMET-AI는 고정 비교선으로만** — 세대 우열 판정에 쓰지 않고 **★누수 플래그** 유지. 격차는 "누수 크기"가 아니라 **앙상블·멀티태스크·튜닝이 교락된 상한**.
- **실패 조합은 0이 아니라 N/A(사유)** — 표에서 빈칸(—)으로 두고 `logs/failures.jsonl`에 이유 기록.
- **소표본 주의** — 기질 3종(664~667)·Half-life(665)·HIA(578)·Bioavailability(640)는 구분선이 커 순위 주장 자제.
- **가설대로 안 나와도 그대로** — 신세대가 이기거나 멀티태스크가 손해여도 보고. E가 안 올라도 "천장이 낮다"는 정직한 음의 결과.

## 8. 산출물

```
src/common.py         엔드포인트 레지스트리·분할·특징빌더·지표·진행상태
src/run_all.py        밤샘 오케스트레이터(실패격리·이어하기·중간리포트)
src/build_report.py   3축 표·CSV·HTML (부분 결과에도 동작)
src/g4_unimol.py · g5_finetune.py   G4·G5 러너(18개 조준)
results/progress.jsonl        완료 조합 원장
results/adme_matrix.csv       축① 세대
results/feature_ablation.csv  축② 특징 증분
results/learning_axis.csv     축③ 멀티태스크
results/partial_report.html   중간 리포트(실행 중 갱신)
results/report_adme_full.html 최종 리포트
results/split_leakage.json    분할·누수 실측
predictions/*.jsonl           분자별 예측(재사용 자산)
logs/failures.jsonl           실패 기록
```

---

## ★최종 보고서 (2026-07-24 · report_adme_full.html)

**새 학습·재계산 0 — 확정값 이관 + 적대검증 정정 반영만.** `src/build_final.py`.

### 반영한 정정 4건 (적대검증 확정)
1. **"물리화학 18/18 압승" 삭제** → 대칭 로스터 재집계값: **동일 XGB 17/18 · 동일 RF 16/18**. ECFP 승: P-gp·CYP2D6 억제·CYP3A4 기질. '18/18'은 physchem 2모델 vs ECFP 1모델 best-of-2 산물임을 명시(rf_ecfp 추가로 2×2 대칭화). → `feature_2x2.csv`
2. **"E 3종 G4 석권" 삭제** → "반감기만 뚜렷, 청소율 2종은 G2와 구분 불가 동률"로 분해.
3. **세대 승수(G2 9·G4 5·G3 4·G5 0)에 비대칭 경고 병기** — G2는 3모델 중 best(다중비교 이점)·G4 5승은 통계 미검증.
4. **축③ 누수 기각 반영** — 88.9% 누수 산물, "닮은 과제 이득" 기각.

### 부트스트랩 통계 판정(§6)
- 예측 있는 G2 3종·G3만 대응비교 가능(2000회). **54쌍 중 구분가능 23쌍**.
- ★**확실한 세대 효과는 친유성 dmpnn(G3) 3/3 완승 하나뿐**(4200분자). physchem 확실 우세: Caco2 3/3·용해도·PPBR·CYP억제 3종. 동률: 반감기(0/3)·HIA·P-gp·CYP기질 3종.
- ★**G4·G5는 분자별 예측 미저장 → 부트스트랩·DeLong 원천 불가** = E 헤드라인·G4 5승 미검증(§9). 방향은 견고(적대검증 5건 기각)하나 구분가능성 미검증.

### 위생
- 그림 0(표 중심)·nan/None 0·섹션 10·목차 10. "18/18 압승"·"석권"은 §8 자기정정 인용 자리에만(주장 아님).
