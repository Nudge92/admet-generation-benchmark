# 통합 마스터 보고서 — 36과제 ADMET 세대 벤치마크 + 배포 신뢰도 + 배포 가이드

작업일 2026-07-25 · `Project/ADMET_integrated/2026-07-25/master_integrated/`
독성 18과제 + ADME 18개 = **36과제**를 단일 최상위 문서로 통합. ★**새 학습·새 계산 0건** — 기존 4+1개 보고서의 확정값을 원본 데이터 파일에서 **프로그램으로 읽어 이관·조립만**(`src/build_master.py`).

---

## 사전 정정 (완료)
- `adme_reliability_report.html` §3 "AD 유효 8" → **"AD 유효 9"** — logD가 AD 유효로 추가돼서. ★이미 logD 채우기 작업 때 `build_report.py`의 nv를 CSV 동적 집계로 바꿔 재생성되어 반영됨(하드코딩 "유효 8" 0건 실측). 다른 곳 손대지 않음.

## ★이관 매핑 (어느 절 ← 어느 원본 데이터)

| 마스터 절 | 원본 보고서 | 읽은 데이터 파일(확정값) |
|---|---|---|
| §0 배너·§2 독성 세대 | master_report.html | `master_matrix.csv`(kind='우리 학습'만)·`consistency.json`·`finalize_check.json` |
| §2 독성 판정(승수·DeLong·누수상한) | master_report.html | `consistency.json`(best_gen_dist·resolvable·leak_ub)·`finalize_check.json`(delong·ad) |
| §3 ADME 축① 세대 | report_adme_full.html | `adme_matrix.csv` |
| §3 ADME 축② 특징 | report_adme_full.html | `feature_2x2.csv`(대칭 2×2) |
| §3 ADME 축③ 학습방식 | report_adme_full.html | `learning_axis.csv`·`split_leakage.json` |
| §3 ADME 통계판정 | report_adme_full.html | `bootstrap_verdict.json` |
| §5 G4 검증 | g4_verification.html | `g4_verification.json`(bootstrap_A/B) |
| §6 배포신뢰도(독성) | reliability_report.html | `reliability.csv` |
| §6·§7 배포신뢰도(ADME) | adme_reliability_report.html | `adme_reliability.csv`·`reliability_detail.json`·`champions.json`·`logd_g3_repro.json` |
| §7 배포가이드 | 독성 §11 + ADME §7 합본 | `reliability.csv` + `adme_reliability.csv` |
| §8 자기정정 | 4개 보고서 + 메모리 | 서술 이관(각 정정의 출처 표기) |

## ★핵심 확정값 (이관 결과)
- **독성 18**: 최고세대 **G2 17·G4 1(LD50)·정직 G3 0·G5 0**(consistency.json). G3가 높아 보이는 건 ADMET-AI **누수 기준선**(kind='누수 기준선')이며 세대 승수에서 제외 — 정직 chemprop G3는 전부 G2 미만. DeLong **10/17 유의·전부 G2 우세**. 누수 상한(참고선) 핵심 0.0876·확장 0.1491. AD 11/3/4.
- **ADME 18**: 세대 승수 **G2 9·G4 5·G3 4·G5 0**(adme_matrix). 물성 vs 지문 대칭 **17/18(XGB)·16/18(RF)**(feature_2x2). 멀티태스크 **88.9% 누수 → 가설 기각**. 부트스트랩 통계 확실한 세대효과 = **logD G3 3/3 하나뿐**. AD 9/3/6.
- **G4 검증**: 반감기·용해도 **(B)seed별 주판정 CI가 0 포함 → 미확립**(반감기[-0.0184,0.3247]·용해도[-0.0017,0.0479]). (A)예측평균은 구분되나 앙상블 비대칭(용해도 Δ0.0234→0.0424).
- **배포**: 임계 0.5 붕괴 — 독성 NR-PPAR-gamma 민감도 0.000·ADME CYP2C9 기질 0.026. t* 만능 아님(ADME 4개 악화). LD50 과소커버 0.843·나머지 회귀 과대커버.

## ★원본 불일치 기록 (1건)
- **G4 검증 주판정법**: `g4_verification.json`은 자체적으로 `primary_method=A`로 "구분됨"이라 판정. 이 마스터는 배포 신뢰도(§6) 통일 원칙(방법 비교엔 학습 변동 포함 = (B)seed별)에 따라 **(B)를 주판정으로 채택 → "미확립"**으로 표기. ★두 수치(A·B) 모두 원본 값 그대로 병기하고, 채택 사유를 §5에 명시. 값 조작 아님(프레이밍 통일).

## 정직성 체크
- 재계산 0: 모든 표는 원본 CSV/JSON을 `build_master.py`가 읽어 렌더(승수·DeLong·CI 재계산 없음).
- 누수 기준선(ADMET-AI)·리더보드(TDC SOTA)는 세대 승수에서 제외(참고선 표기).
- "18/18 압승"·"석권" 표현 미사용 — 대칭 17/16 병기·G4 미검증 병기.
- Tox21 12경로 전부 명시(뭉개지 않음). G4·G5 미실행은 스코프 결정으로 표기.
- 빈칸 0 채우기 없음(누출 검사: None/nan/N/A/빈셀 0건 실측).
- 독성=분자별 예측 "메움"(재현 18/18) / ADME=logD만 메우고 G4·G5 "한계로 남김" — 구분해 서술.
- HTML(브라우저 렌더)·Nanum Gothic 폰트·matplotlib 그림 없음 → □ 위험 없음.

## 산출물
```
src/build_master.py                   원본 5소스 읽어 조립(재계산 0)
results/master_integrated_report.html 단일 문서·목차·36과제 전부 (40 KB)
notes.md                              이관 매핑(이 파일)
```
★기존 4+1개 원본 보고서는 **삭제하지 않음** — 이 문서가 최상위 인덱스, 원본은 상세로 보존.
