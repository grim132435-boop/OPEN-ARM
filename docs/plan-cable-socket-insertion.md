# OpenArm 케이블–소켓 삽입: Sim 폐루프 검증 계획

대상 스택: DGX Spark · Nemotron 3 (로컬) · NemoClaw/OpenShell · Isaac Sim 6.0 · Isaac Lab 3.0 · Isaac Lab-Arena 0.3 · GR00T (비교 기준선)

## 0. 목표와 판정 기준

| 항목 | 값 |
|---|---|
| 과제 | OpenArm이 케이블 커넥터를 집어 소켓에 삽입 |
| 정책 | ACT (텔레옵 데이터로 학습, LeRobot 포맷) |
| 기준선 | GR00T N1.7 post-train (동일 데이터) |
| 성공 판정 | 커넥터 원점이 소켓 목표 pose 대비 위치 오차 ≤ 2 mm, 각도 오차 ≤ 3°, 삽입 깊이 ≥ 목표의 95%, 접촉력 ≤ 15 N를 0.5 s 이상 유지 |
| 합격선 | 기본 환경 성공률 ≥ 80 % (50 에피소드), 변형 스윕 평균 ≥ 60 % |
| 폐루프 | 에이전트가 평가 → 실패 분석 → 환경/정책 설정 수정 → 재평가를 자동 반복 |

## 1. 하드웨어/소프트웨어 전제 (중요)

- **DGX Spark는 arm64(GB10)** 입니다. Isaac Sim pip 패키지는 x86_64 전용이므로 **공식 DGX Spark 플레이북의 소스 빌드 경로**를 따라야 합니다 (`NVIDIA/dgx-spark-playbooks/nvidia/isaac`). 빌드 실패 시 대안은 Isaac Lab 3.0의 **Kit-less(Newton) 모드**이며, 이 경우 카메라 렌더링·USD 임포터 등 일부 기능이 빠집니다.
- Python **3.12** 필수 (Isaac Sim 6.0.x/6.1 · Isaac Lab 3.0 요구; Lab 3.0 문서는 현재 Sim 6.1 과 짝지어져 있으므로 설치 시 문서 기준 버전을 따를 것). `uv venv --python 3.12`.
- 메모리: 128 GB 통합 메모리를 Isaac Sim, Arena 병렬 env, 정책 서버, LLM이 나눠 씁니다. LLM은 **Nemotron 3 Nano 30B** (Ollama) 로 시작하고, 메모리 여유를 측정한 뒤 Super 급을 검토합니다.
- NemoClaw의 로컬 추론(Ollama/vLLM)은 공식 문서상 *experimental* 입니다. 실패하면 OpenShell 없이 에이전트를 직접 실행하는 fallback 경로(§5)를 둡니다.

## 2. 단계별 계획

### Phase A — 환경 구축 (Spark)
1. `scripts/check_env.sh` 로 OS/Python/CUDA/디스크 확인.
2. Isaac Sim 6.0 소스 빌드 (플레이북) → `isaacsim` 실행 확인.
3. Isaac Lab 3.0 설치 (`./isaaclab.sh --install`), OpenArm 확장(`enactic/openarm_isaac_lab`) 을 `source/extensions/` 에 클론·설치. 이 확장은 Isaac Lab 2.3 기준으로 작성됐으므로 3.0 API 변경(특히 `isaaclab.envs` 네이밍) 패치가 필요할 수 있습니다.
4. Isaac Lab-Arena `release/0.3.0` 브랜치 설치.
5. Ollama + `nemotron-3-nano:30b` 서빙, NemoClaw onboard (로컬 provider).

### Phase B — 과제 환경 (Arena)
1. 자산: 케이블 커넥터·소켓 USD (SimReady 규격, 충돌 메시는 convex decomposition). 초기에는 단순 원통 핀/홀로 시작해 물리 안정성을 확보한 뒤 실제 형상으로 교체.
2. `sim/arena/openarm_cable_insert_environment.py` (`--external_environment_class_path` 로 로드) + `cable_insert_task.py` (TaskBase). 파라미터는 `openarm_cable_insert_task.yaml`, 판정은 `success_criteria.yaml`.
3. 변형 스윕: 소켓 위치 ±15 mm, 회전 ±10°, 조명, 카메라 외부 파라미터 ±5 mm/±2°, 커넥터 질량 ±20 %.
4. Arena 0.3 의 *agentic environment generation* 은 실험 기능이므로 **수동 스펙을 기본**으로 하고, 에이전트 생성은 변형 확장용으로만 사용.

### Phase C — 정책 서버
1. `sim/policy/act_policy_server.py`: LeRobot ACT `pretrained_model` 디렉터리를 pre/post-processor 와 함께 로드, 배치(num_envs) 관측 → `(B, chunk, dof)` 청크를 ZeroMQ 로 서빙. Arena 측 `act_remote_policy.py` 는 `PolicyBase` + `ActionChunkScheduler` 로 청크를 재생. GR00T 기준선은 Isaac-GR00T 표준 서버(5555).
2. 관측 매핑: sim 카메라 3대 → 실기와 동일한 이름/해상도/MJPEG 압축 여부, 관절 순서 동일. **sim-real 관측 불일치가 가장 흔한 실패 원인**이므로 `scripts/validate_scaffold.py` 가 매핑 테이블을 검사합니다.

### Phase D — 폐루프 에이전트
1. 에이전트(NemoClaw 샌드박스 내 Nemotron)가 `scripts/run_closed_loop.sh` 를 호출해 평가 실행.
2. 결과(JSON: 성공률, 서브태스크 진행률, 실패 모드별 카운트, 영상 경로) 를 읽고 `agent/prompts/closed_loop_verifier.md` 규칙에 따라 다음 액션 결정:
   - 성공률 < 합격선 → 실패 모드 분류(정렬 실패/과도한 힘/그립 미스/타임아웃) → 허용된 수정(스윕 범위, 물리 솔버 설정, 커넥터 스폰 범위, action_horizon, 체크포인트 선택; seed 고정) 적용 후 재평가. 컨트롤러 게인·에피소드 길이·판정 기준은 수정 불가.
   - 합격선 통과 → 리포트 생성, 사람 승인 요청.
3. 에이전트가 **수정할 수 없는 것**: `sim/arena/success_criteria.yaml`(판정 기준·합격선), 정책 가중치, 에피소드 길이. 샌드박스 `deny_write` 와 `validate_scaffold.py` 의 SHA-256 핀으로 이중 강제합니다.
4. 선택: Cosmos Reason 으로 실패 에피소드 영상에 구조화 피드백 생성 → 에이전트 입력으로 사용. 단, VLM 점수는 보조 신호이며 성공 판정에는 쓰지 않음.

### Phase E — 승인·배포
1. 리포트(`reports/<date>/summary.md`) 사람 검토.
2. 실기 전 체크: udev 고정 이름, 허브 분리(제어/카메라), 힘 제한.

## 3. 리스크

| 리스크 | 대응 |
|---|---|
| Spark에서 Isaac Sim 6.0 빌드 실패 | Kit-less Newton 모드로 관절·접촉 검증만 먼저 수행; 렌더링 필요 검증은 x86 워크스테이션 |
| openarm_isaac_lab 이 Lab 3.0 과 비호환 | 포크 후 API 패치, 상위에 PR |
| 삽입 접촉 물리 불안정 (관통/튕김) | PhysX solver iteration ↑, dt 1/240, 접촉 오프셋 조정, 초기엔 단순 형상 |
| NemoClaw 로컬 추론 실패 | OpenShell 없이 Ollama 직접 호출 fallback |
| 에이전트가 판정 기준을 완화 | `success_criteria.yaml` 분리 + 샌드박스 deny_write + validate 해시 핀 (3중) |

## 4. 검증 상태

`scripts/validate_scaffold.py` 는 **GPU 없이** YAML/Python 구문, 관측 매핑 일관성, 판정 기준 SHA-256 핀, 포트/샌드박스 정합성, chunk/horizon 범위를 검사합니다.

독립 검토 에이전트가 Isaac Lab-Arena `release/0.3.0` 과 LeRobot v0.6.2 소스를 직접 클론해 대조한 결과를 반영했습니다 (policy_runner 플래그, `PolicyBase`/`ActionChunkScheduler` 시그니처, `episode_results_rank*.jsonl` 스키마, LeRobot pre/post-processor). 2차 검토 통과 항목은 커밋 메시지에 기록합니다.

**아직 검증되지 않은 것 (Spark 필요):** `sim/arena/README.md` 에 나열된 미작성 모듈(임바디먼트·자산·변형·메트릭), Isaac Lab 3.0 manager cfg 클래스 명, 실제 삽입 물리 안정성.

## 5. 참고

- Isaac Lab-Arena: https://github.com/isaac-sim/IsaacLab-Arena (release/0.3.0)
- GR00T sim evaluation (policy server/client): https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/simulation-workflow/sim-evaluation.html
- DGX Spark Isaac playbook: https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/isaac/README.md
- NemoClaw: https://docs.nvidia.com/nemoclaw/
- OpenArm Isaac Lab: https://github.com/enactic/openarm_isaac_lab
- Isaac Lab 3.0 설치: https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/setup/installation/index.html
