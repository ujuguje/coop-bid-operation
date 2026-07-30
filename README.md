# Cooperative Bid and Operation Strategy in Two-Settlement Energy Market

**논문**: Cooperative Bid and Operation Strategy in Two-Settlement Energy Market through
Dual-Agent Imitation Learning on Joint Optimized Policy
(*Applied Energy*, **accepted 2026-07**, Ms. Ref. APEN-D-26-05691R1)

이 폴더는 억셉된 최종 상태 기준으로 **코드·데이터·결과를 역할별로 재구성한 정리본**입니다.
(원본 작업 폴더 `APEN_Major_Revision/`은 백업으로 보존. 모든 경로는 상대경로로 수정되어
이 폴더를 어디로 옮겨도 그대로 동작합니다.)

---

## 1. 폴더 구조 (데이터 흐름 순서)

```
APEN_Final/
├─ data/
│   ├─ raw/                          ① 원시 데이터 (CAISO 시장 데이터)
│   │    ├─ FinalData_FMM_LMP.csv         15분 FMM LMP 가격 원본
│   │    ├─ Interval_LMP_15min_avg.csv    5분 LMP → 15분 평균
│   │    └─ Output_Solar_NetLoad_15min.csv 태양광 발전·넷로드 15분
│   └─ processed/                    ② 후처리 데이터
│        ├─ rl_inputs/                    RL 환경 입력 (LSTM 예측 기반)
│        │    LMP_Bid_LSTM, LMP_Ope_LSTM, LMP_Set, Solar_Bid_LSTM, Solar_Ope
│        ├─ expert_actions/               OJPD 전문가 시연 (완전예견 MILP 최적해)
│        │    Offline_Expert_Action_joint_deg{D}_tol{T}.csv × 12설정
│        │    = optimization_results에서 파생한 2컬럼 요약:
│        │      Bid_Action = B_Cha−B_Dis, Ope_Action = (O_Cha−O_Dis)−Bid_Action.
│        │      학습 코드가 직접 읽는 파일은 이쪽.
│        ├─ optimization_results/         MILP 전체 변수 해 (17컬럼: 매매량·SoC·수익분해 등)
│        │    expert_actions의 원천 기록. 현재 코드는 읽지 않음(아카이브용;
│        │    재생성에는 설정당 수십 분 소요되므로 보존)
│        └─ legacy_inputs/                구(persistence) 예측 입력 — LSTM 교체 전 버전.
│                                         gen_sac_inputs_LSTM.py의 행 정렬 기준으로만 사용
├─ code/                             ③ 코드 (전체 파이프라인)
│   ├─ paths.py                          ★ 모든 경로 정의 (이 파일만 보면 경로 규칙 파악 가능)
│   ├─ Forecast/                         LSTM/Transformer/SARIMA 예측모델 학습 + RL 입력 생성
│   ├─ optimization/                     OJPD MILP(run_optimization) + MPC 벤치마크 4종
│   ├─ envs/                             입찰·운영 2단계 시장 RL 환경
│   ├─ models/, algorithms/              네트워크 / SAC·BC 알고리즘 + eval_utils(평가 백엔드)
│   ├─ data/settings.py                  데이터 로딩·분할·배터리 파라미터 (rl_inputs를 읽음)
│   ├─ run_training.py                   Single-SAC / Inde-DASAC 학습 러너
│   ├─ run_coop_dasac.py              Coop-DASAC (Case C) 러너
│   └─ run_coop_dabc.py               Coop-DABC (Case D, 제안기법) 러너
├─ results/                          ④ 학습·실험 결과물
│   ├─ baseline/                         본선 체크포인트 (10시드, deg5/tol5) + MPC 캐시
│   ├─ sensitivity/                      민감도 스윕 체크포인트 (5시드) + MPC 분해 캐시
│   ├─ coop_dasac/  coop_dabc/       협력기법 체크포인트 (본선 10시드 + 스윕)
│   ├─ forecast/                         예측모델 가중치·예측치·벤치마크 비교
│   └─ ood_analysis/                     OOD(Table 5) 비교표·그림 + MPC 캐시
├─ outputs/                          ⑤ 논문 최종 산출물
│   ├─ Fig9_*.png, fig7_*                노트북이 생성하는 논문 그림
│   ├─ Statistical_Significance.csv      Table 4 유의성 검정 최종치
│   └─ graphical_abstract/               그래픽 초록 차트·pptx
├─ analysis/                         ⑥ 분석 노트북 (논문 표·그림 재현)
└─ submission/                       ⑦ 제출물 (원고·그림·답변서)
    ├─ 1st_submission/                   최초 제출본
    └─ revised_final/                    Major Revision 최종 제출본 (억셉된 버전)
```

**데이터 흐름**: `raw` → (Forecast/LSTM) → `processed/rl_inputs` → (MILP) →
`processed/expert_actions` → (RL 학습) → `results` → (analysis 노트북) → `outputs`

## 2. 논문 표·그림 재현 (노트북 B1–B5)

`analysis/`에서 노트북을 열고 위에서부터 실행. **5개 전부 실행 검증 완료**
(B1·B2 = 원고 값과 100% 일치, B3 = ±1 반올림 이내).

| 노트북 | 논문 산출물 | 실행 시간 |
|---|---|---|
| `B1_Table4_Main_Results` | **Table 4** (수익 분해 + Welch/one-sample t + separation) | ~1–3분 |
| `B2_Sensitivity_TableB1_B2` | **Table B.1 / B.2** (민감도, 5시드) | ~3–5분 |
| `B3_OOD_Table5` | **Table 5** (OOD, 10시드) | ~5–15분 |
| `B4_Computational_Cost` | **Table B.3** (학습·추론 시간) | ~5–10분 (GPU) |
| `B5_Paper_Figures` | **Fig 7 · 9 · B.1 · B.2 + 그래픽초록 차트** → `outputs/` | ~2–4분 |

보조 노트북: `A1`(Appendix C 예측모델 비교), `A4`(민감도 재학습), `A8`(베이스라인 10시드 학습).

### 헷갈리기 쉬운 핵심 사실

1. **최종 수치의 기준 = eval 모드(dropout OFF) 재평가.** 학습 로그의 검증 수치에는 SAC 계열이
   train 모드로 평가된 버그가 있었음. B1–B3(= `eval_utils.py`)가 정답 기준.
2. **논문 기법 구현 위치**: Coop-DASAC = `algorithms/coop_dasac.py`,
   Coop-DABC(제안) = `algorithms/coop_dabc.py` (QMIX hypernet `w_final.abs()` 단조성 수정판).
3. **체크포인트 파일명 규칙**: `actor_{tag}_deg{D}_tol{T}_seed{S}.pth`
   (tag: single_sac / inde_sac / coop_dasac / coop_dabc — 논문 명칭 그대로).
   구버전 내부 태그(`coop_sac_qfix`, `coop_bc_mono`)는 2026-07-21 일괄 개명됨;
   백업 폴더(`APEN_Major_Revision/`)에는 옛 태그가 남아 있음.
   `baseline/`·`sensitivity/`의 `coop_sac`·`coop_bc`(접미사 없음)는 단조성 수정 전
   구버전 변형의 잔존 체크포인트로, 논문 표에는 쓰이지 않음.
4. **시드 수**: 본선(Table 4·5) 10시드, 민감도(B.1·B.2) 5시드. MPC·OJPD는 결정론적.
   Single-SAC은 tol 스윕 미학습 (원고에 명시).
5. **파라미터 규칙**: 기본 β_deg = 5 $/MWh, ε_tol = 5 MW. 환경 생성 전
   `Cap_Pcs_Ope = tol + 2` 필수 — `eval_utils.set_cap_ope(tol)`가 처리.
6. **유의성 검정**: 시드가 표본. 확률적 벤치마크 = Welch two-sample,
   결정론적 MPC = one-sample. 모두 단측 p<0.001 + 완전 분리.

## 3. 재학습·무거운 캐시 재생성

| 대상 | 방법 |
|---|---|
| Single-SAC / Inde-DASAC | `A8` 노트북(본선) / `A4` 노트북(스윕) — 내부적으로 `code/run_training.py` |
| Coop-DASAC | `python code/run_coop_dasac.py --tol 5 --deg 5 --seed 0` |
| Coop-DABC (제안기법) | `python code/run_coop_dabc.py --tol 5 --deg 5 --seed 0` |
| OJPD 전문가 시연 생성 | `python code/optimization/run_optimization.py --mode joint` → `data/processed/expert_actions/` |
| LSTM 예측 + RL 입력 재생성 | `A1` 노트북(학습) → `python code/Forecast/gen_sac_inputs_LSTM.py` → `data/processed/rl_inputs/` |
| MPC 캐시 (설정당 30–40분) | `code/optimization/mpc_components_{tol,deg}.py` · `mpc_partB.py` · `mpc_sensitivity.py` |

## 4. 환경

```bash
pip install -r requirements.txt
```

GPU(CUDA) 권장. MILP는 cvxpy 기본 솔버로 동작. 노트북은 저장소 안 어느 위치에서든
상대경로로 동작하지만, **`analysis/` 폴더에서 여는 것을 기준**으로 작성됨.

## 5. 정리 이력

- **2026-07-09**: 임시 스크립트 삭제, ver2_qfix 최종 알고리즘을 code/로 병합,
  분석 스크립트를 노트북 B1–B5로 재구성 (원본 `APEN_Major_Revision/ANALYSIS_GUIDE.md` 참고).
- **2026-07-21 (이 폴더)**: 억셉 확정 후 정리본 생성 — 데이터를 raw/processed로 분리,
  결과물·최종산출물·제출물 분리, 모든 절대경로(구 PC `C:\Users\WJ\...`) 제거 및
  상대경로화(`code/paths.py`), `__pycache__`·탐색 아카이브 제외.
  내부 개발 태그를 논문 명칭으로 일괄 개명
  (`coop_sac_qfix → coop_dasac`, `coop_bc_mono → coop_dabc`; 폴더·체크포인트 236개·코드·노트북 일괄).
  원본 폴더는 `APEN_Major_Revision/`에 그대로 백업.
