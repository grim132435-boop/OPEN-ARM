# sim/demo — 양팔 케이블 핸드오버 시연 (스크립트 모션)

> **Scripted motion-planning demo — not a learned policy; cable pose from simulator ground truth; rigid cable proxy**
> (영상 캡션과 시작 배너에 항상 표시된다.)

`handover_demo.py` 는 Isaac Lab 3.0.0 (af1bab4) + Isaac Sim 6.0.1 용 단독 스크립트다.
오른쪽에 떨어진 케이블을 오른팔이 집고 → 몸 앞(handover.center)에서 왼팔에 넘기고 → 왼팔이 왼쪽 박스에 놓는다.
장면 치수는 모두 `sim/workbench/workbench.yaml` 에서 읽는다 (로봇 USD 외에는 프리미티브만, Nucleus 불필요).

## 실행

```bash
export OPENARM_ASSETS=~/src/openarm_isaac_lab        # enactic/openarm_isaac_lab (USD 위치)
cd ~/IsaacLab
# 1) 운동학 점검 (영상 없음, 수 분): USD 로드, 관절 이름·한계, 어깨 축 대조, 웨이포인트별 IK 가능 여부
./isaaclab.sh -p ~/OPEN-ARM/sim/demo/handover_demo.py --headless --dry_run_kinematics --episodes 5
# 2) 시연 + 녹화
./isaaclab.sh -p ~/OPEN-ARM/sim/demo/handover_demo.py --headless --enable_cameras --episodes 5
# 3) 순수 물리 파지 비교
./isaaclab.sh -p ~/OPEN-ARM/sim/demo/handover_demo.py --headless --enable_cameras --episodes 5 --attach off
```

산출물 (`--out_dir`, 기본 `~/openarm-out/demo/<시각>/`, 커밋 금지): `demo_oblique.mp4` (`--top_view` 시 `demo_top.mp4`),
`episodes.jsonl` (에피소드별 성공·실패 사유·상태별 TCP 오차·관절 여유·양손 최소 거리), `run_meta.json`, 드라이런 시 `dry_run.jsonl`.
mp4 인코더(imageio-ffmpeg / cv2)가 없으면 PNG 프레임을 저장하고 ffmpeg 명령을 출력한다.

## 주요 플래그

| 플래그 | 기본 | 내용 |
|---|---|---|
| `--openarm_usd` | `$OPENARM_ASSETS/.../openarm_bimanual.usd` | 로봇 USD |
| `--episodes`, `--seed` | 5, 0 | 에피소드 i 는 seed+i (READY 자세는 고정 시드 12345 → `--seed i --episodes 1` 로 같은 에피소드 재현) |
| `--attach on/off` | on | on: 닫는 순간 파지점이 손가락 사이(TCP 기준 ±10/20/25 mm)에 있으면 케이블을 그 손에 강체로 붙임(데모 보조), 아니면 `attach refused: grasp miss` 기록. off: 마찰 파지만 |
| `--partition on/off` | off | 가운데 칸막이 (`partition.sim_demo: false` 와 동일). on 이어도 경로 계획은 칸막이를 모름 |
| `--yaw_min_deg/--yaw_max_deg` | −45/45 | 케이블 몸통 방향(커넥터→끝) 무작위 yaw 범위 |
| `--grasp_pitch_deg` | 0 | 오른손 파지 피치 1순위 후보 |
| `--place_mode corner/center` | corner | 놓기 위치 (아래 참고) |
| `--place_elev_deg`, `--place_azim_deg` | 60, 자동 | 놓을 때 케이블 기울기·방향 |
| `--no_resample` | 꺼짐 | IK 불가능한 낙하 자세를 다시 뽑지 않음 (시연·드라이런 모두 같은 샘플러 사용) |
| `--record on/off`, `--video_res 720/1080`, `--top_view`, `--fps` | on, 720, -, 30 | 녹화 (`--enable_cameras` 또는 `ENABLE_CAMERAS=1` 필요). `--video` 가 아닌 이유: AppLauncher 가 `args.video` 를 bool 로 읽어 자체 뷰포트 녹화를 켬 |
| `--dry_run_kinematics` (+`--no_teleport_check`) | - | 제어기 동작 없이 계획만 + 관절 순간이동으로 모델 대조. 야코비안·링크 자세를 읽으려고 목표=현재 상태로 물리 스텝 몇 번은 돈다(그동안 케이블은 자유낙하). `--no_teleport_check` 는 대조용 스텝만 생략 |
| `--speed` | 1.0 | 동작 속도 배율 |

## 동작 방식 (요약)

- 시작 시 PhysX 가 주는 손(`openarm_{side}_hand`) 야코비안·자세로 팔별 PoE 운동학 모델을 만든다 (URDF 불필요, USD 그대로).
  TCP = hand + 0.0835 m (hand z).
- 웨이포인트마다 이 모델에서 다중 시드 DLS IK (LeRobot 실기 관절 한계 적용) → 큰 자세 변경은 관절 공간 직선, 짧은 이동은 직교 직선.
- 실행: 팔마다 `DifferentialIKController` (pose, dls) 가 계획된 TCP 자세를 매 스텝 추종, 여유 자유도는 계획 관절 경로 쪽으로 null-space 보정.
- LeRobot 한계(deg, 오른팔 j1 ±75, j2 −9..90, j3 ±85, j4 0..135, j5 ±85, j6 ±40, j7 ±80; 왼팔 j2 −90..9)를
  `write_joint_position_limit_to_sim_index` 로 시뮬에 넣고 시작 시 `robot.data.joint_pos_limits` 를 출력한다.

## 가짜인 것 (발표 시 명시)

- 학습 정책 아님: 상태머신 + IK. 케이블 자세는 시뮬 정답값 (인식 없음).
- 케이블 = 강체 1개 (0.30 × 0.0055 × 0.0055 m 막대 + 커넥터 상자, 0.017 kg). 휘지 않는다.
- `--attach on`(기본): 손을 닫으면(파지점이 손가락 사이에 있을 때만) 케이블을 손에 매 스텝 순간이동시켜 붙인다.
- 손가락 패드 간격 모델: finger.stl 기준 TCP 높이 패드 간격 ≈ 2·(q − 2.8 mm) → q=0 이면 패드가 ~5.5 mm 겹침(무엇이든 꽉 쥠).
  시작 시 USD 의 손가락 링크 원점 간격을 재서 q=0 패드 간격을 출력하고, 케이블 지름/커넥터 폭보다 크면 경고한다
  (경고가 뜨면 `--attach off` 로는 그 손이 케이블을 쥘 수 없음).
- 로봇: 고게인 PD (강성 400 / 감쇠 80, 참조값 80/4), 로봇 링크 중력 off, 자기충돌 off. 손목 D405 충돌체 없음.
- IK 로 집을 수 없는 낙하 자세는 다시 뽑는다 (횟수는 `drop_rejected_infeasible` 로 기록).

## 알려진 한계

- LeRobot 한계에서 수직(top-down) 자세는 상판 위로 높이 올라갈수록 불가능해진다. 그래서 접근/들기/박스 위 높이를
  0.08/0.12/0.10 m 에서 필요하면 낮춘다(사용값 로그). 수직 파지 자세의 관절 여유는 0–5° 인 경우가 많다.
- 오른손 파지 피치는 ~0° 여야 한다 (피치가 있으면 핸드오버에서 필요한 손 yaw 가 한계 밖). 대신 케이블 축 롤 ±15/30° 를
  허용(attach on 전용)해 도달 범위를 넓혔다. 오프라인 검증(URDF+LeRobot, 명목 30 시드): attach on 28/30, off 15/30 전 웨이포인트 IK 가능 — **IK 만** 본 수치이고 마찰 파지 유지 여부는 검증 안 됨.
- 핸드오버: 오른손가락 위로, 왼손가락 아래로 닫힘(`handover.fingers`). 케이블이 x 방향으로 놓이므로 두 TCP 는
  **x 로 0.10 m (= tcp_gap, 케이블 따라)** 떨어진다 (reach 도구의 y 간격 가정과 다름). 케이블 끝과 몸통 기둥 간격 ~3 cm.
- 0.30 m 강체는 0.14 × 0.16 m 박스에 눕지 않는다 → 커넥터를 아래로 ~60° 기울여 가까운 모서리에 내려놓고 대각선으로
  넘어지게 한다 (`--place_mode corner`). 성공 판정: 케이블 중심이 박스 내부 AABB(5 mm 안쪽) 안, 벽 높이 아래.
- `--attach off` 파지는 손끝이 커넥터 위 4 mm 만 겹쳐 약하고, 왼손은 지름 5.5 mm 케이블을 q=0 전체 조임으로 잡는다(물리 미검증). LeRobot 0점·부호 = URDF 가정(실기 미확인).
- 이 스크립트는 GPU 없이 작성됐다 (계획부는 numpy 로 오프라인 검증, Isaac API 는 af1bab4 소스 대조). 첫 실행은 반드시 드라이런.
- GUI(비 headless) 실행 시 녹화를 안 해도 캡처 주기마다 렌더해 창이 갱신된다.
- 경로 계획은 CPU numpy: 에피소드당 10–20 s 정지(영상에는 안 보임).

## 멈춤 조건 (docs/plan-bimanual-handover-demo.md §3 과 동일 기준)

- 시작 전 검사: USD / workbench.yaml 경로가 틀리면 Kit 를 띄우기 전에 종료(exit 2). 어떤 종료든 `simulation_app.close()` 는 항상 호출.
- **P3 드라이런**: USD 로드 실패, `openarm_{side}_hand`/관절 이름 없음, 모델 대 시뮬 TCP 오차 > 2 mm (teleport check 경고),
  또는 한 팔이라도 5 시드 중 과반 IK 불가 → 토요일 밤까지 해결 안 되면 **L1 폴백**(케이블을 손에 붙인 운동학 재생).
- **P4 시연**: 월요일 12:00 KST 에 `--attach on` 5 시드 중 성공 3 미만 → L1.
- 스크립트 자체: 상태별 IK 불가, 들기 후 케이블 미상승, 핸드오버 후 케이블이 왼손에 없음 → 해당 에피소드만 실패 기록 후 다음 시드 진행.
  창 닫힘/Ctrl+C → 영상·로그 정리 후 종료.
