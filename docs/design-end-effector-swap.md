# 설계: OpenArm 그리퍼 → ORCA Hand 교체 대비

## 목표
지금은 OpenArm v1.0 기본 평행 그리퍼(1 dof)로 케이블·소켓 삽입을 학습·검증하고, 나중에 ORCA Hand v2(17 dof)로 바꿀 때 **코드 수정 없이 설정 한 줄 + 재학습**으로 넘어가게 한다.

## 핵심 아이디어: 엔드이펙터 프로파일
```
sim/robot/end_effectors.yaml
  arms:           openarm_v1_right  (7 dof, CAN)
  end_effectors:  openarm_gripper    (1 dof, CAN)
                  orca_hand_v2_right (17 dof, orca_core/U2D2)
  profiles:       openarm_v1_right_gripper = arm + gripper  →  8 dof
                  openarm_v1_right_orca    = arm + orca     → 24 dof
```
`sim/robot/embodiment_profile.py` 가 프로파일을 읽어 **정책 관절 순서와 sim 관절 이름**을 만든다. 이 모듈을 쓰는 곳:

| 사용처 | 무엇을 가져가나 |
|---|---|
| `sim/policy/act_policy_server.py` | dof → 체크포인트 action 차원과 불일치 시 즉시 에러 |
| `sim/policy/act_remote_policy.py` | dof → Arena 행동 텐서 크기 |
| `scripts/validate_scaffold.py` | 모든 프로파일 해석 가능 여부, ORCA URDF·orca_core 매핑 검증 |
| (Spark) `sim/arena/openarm_embodiment.py` | `sim_joint_exprs` → 관측 순서, `sim_command_exprs` → 행동 대상 (평행 그리퍼는 1값 → 두 손가락) |
| (실기) 텔레옵·데이터 수집 | 같은 순서로 LeRobot `observation.state`/`action` 기록 |

정책 순서 규칙: **팔 7개가 항상 앞**, 엔드이펙터가 뒤. 그래서 팔 부분은 두 프로파일에서 동일하다 (테스트로 고정).

## ORCA 관절 매핑 — 왜 표가 필요한가
ORCA v2 URDF 의 관절 이름은 CAD 해시 (`M-AP_6ec59111_to_R-Carpals_8d1f1041`) 이고, `orca_core` 는 의미 이름 (`ring_abd`) 을 쓴다. 게다가 **중지와 약지가 둘 다 `M-` 접두사**라 이름만으론 구분이 안 된다.

결정 방법 (재현 가능, 테스트로 고정):
1. **체인 연결**: URDF 부모→자식 링크로 wrist → 각 손가락 abd → mcp → pip, 엄지 cmc → abd → mcp → dip 순서 확인
2. **측방향 순서**: 손가락 베이스(`*-AP_*_to_R-Carpals`) 원점 x 좌표 → index −0.027 < middle −0.006 < ring 0.015 < pinky 0.036 (중지·약지는 limit 이 같아 이걸로만 구분)
3. **관절 한계**: URDF limit(rad→deg) 이 `range_deg` 와 정확히 일치 (반전 허용 안 함)
4. **orca_core 교차 확인**: 정책 순서·범위가 `orca_core` `joint_ids`/`joint_roms` 와 일치 (`ORCA_CORE` 설정 시)

`tests/test_embodiment_profile.py` 는 중지↔약지(abd, mcp), 검지↔중지 pip, 엄지↔검지 mcp, 엄지 cmc↔abd 이름을 일부러 바꾸고 반전된 손목 범위를 넣었을 때 **모두 실패**하는 것을 확인한다.

## 교체 절차 (나중에 할 일)
1. **기구**: link7 ↔ ORCA forearm 마운트 어댑터 설계 → `end_effectors.yaml` 의 `asset.mount.xyz/rpy_deg` 채우기
2. **Sim 자산**: Isaac Sim URDF importer 로 ORCA v2 URDF → USD. OpenArm USD 에서 기존 그리퍼 링크(`remove_from_arm`) 제거 후 mount 에 fixed joint 로 결합
3. **설정**: `policy_config_act.yaml` 의 `embodiment_profile: openarm_v1_right_orca`
4. **텔레옵**: 손 리타게팅 필요 (평행 그리퍼처럼 트리거 하나로 안 됨). 데이터 글러브 또는 비전 기반 손 추적 → 17 관절
5. **데이터·학습**: 24 dof 로 새로 수집·학습. 기존 8 dof 체크포인트는 서버가 차원 불일치로 거부함 (의도된 안전장치)
6. **과제 조정**: 성공 판정은 그대로 (커넥터·소켓 pose 기반, 손 종류와 무관). `grasped` 판정도 커넥터 들림 높이 기반이라 변경 불필요
7. **하드웨어**: U2D2 2개(양손) → 제어 전용 USB 허브. `/dev/orca_right` udev 고정, FTDI latency 1 ms, 서보 전원은 별도 PSU

## 리스크
| 리스크 | 대응 |
|---|---|
| 텐던 구동이라 관절각 ≠ 모터각 | 정책은 관절 공간만 다룸. 변환은 `orca_core` 담당 (`real.driver: orca_core`) |
| sim 관절 부호와 실기 모터 부호가 다를 수 있음 | URDF limit 은 `joint_roms` 와 정확히 일치함을 확인했으나, 실기 모터 방향은 `orca_core` 의 `joint_to_motor_map` 음수 ID 로 처리됨. **Spark/실기에서 한 관절씩 움직여 sim 과 같은 방향인지 확인** 필요 |
| 17 dof 로 ACT 학습 데이터 요구량 증가 | 처음엔 손가락을 synergy(예: 파지 1–3 차원)로 묶어 행동 공간 축소 검토 |
| 손 무게로 팔 끝 부하 증가 | OpenArm j5–j7 토크 한계(7 Nm) 대비 ORCA 질량 확인 |
| Feetech vs Dynamixel 버전 차이 | `orca_core` 가 motor_type 자동 감지. 구매한 손의 모터 종류를 실물로 확인 |
