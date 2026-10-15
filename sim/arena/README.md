# sim/arena — Isaac Lab-Arena 0.3 연동

| 파일 | 상태 | 역할 |
|---|---|---|
| `success_criteria.yaml` | 읽기 전용 | 성공 판정·합격선. 해시가 `scripts/validate_scaffold.py` 에 고정 |
| `openarm_cable_insert_task.yaml` | 에이전트 수정 가능 (physics/variations/evaluation) | 장면·물리·스윕 파라미터 |
| `policy_config_act.yaml` | 에이전트 수정 가능 (checkpoint_dir/action_horizon) | ACT 서버/클라이언트 공용 설정 |
| `openarm_cable_insert_environment.py` | 구조 검증됨, 실행 미검증 | `--external_environment_class_path` 로 로드되는 Arena 환경 |
| `cable_insert_task.py` | 구조 검증됨, 실행 미검증 | TaskBase 구현 (success/force 종료 조건, 메트릭) |

## Spark 에서 작성해야 하는 모듈 (아직 없음)
이 파일들은 Isaac Sim 런타임 없이는 작성·검증이 불가능해 의도적으로 비워 두었습니다. 각 파일이 노출해야 하는 인터페이스는 `cable_insert_task.py` 의 import 에 고정돼 있습니다.

- `openarm_embodiment.py` — `@register_asset` 로 `openarm_joint` 임바디먼트 등록. `isaaclab_arena/embodiments/franka/franka.py` 를 템플릿으로, `enactic/openarm_isaac_lab` 의 USD/관절 설정 사용. 카메라 이름은 `policy_config_act.yaml` 의 `cam_high/cam_wrist/cam_side` 와 일치시킬 것.
- `openarm_assets.py` — 두 클래스. 모두 Arena `Object`(`isaaclab_arena/assets/object.py`) 서브클래스, `ObjectType.RIGID`. `env.scene[name]` 은 Isaac Lab `RigidObject` 이므로 **커스텀 `.data` 래퍼를 기대하지 않는다.**
  - `ConnectorAsset(usd_path, mass_kg)` — 커넥터. 접촉 센서는 `CableInsertTask.get_scene_cfg()` 가 `self.connector.get_contact_sensor_cfg()` 로 붙인다 (Arena 관용구: `tasks/sorting_task.py`). 별도 센서 자산 불필요.
  - `SocketAsset(usd_path, insertion_axis, insertion_depth_m)` — kinematic. 평범한 속성 `insertion_axis_local: (3,)`, `goal_offset_local: (3,)`, `goal_quat_local_xyzw: (4,)`, `insertion_depth_m` 를 노출 (소켓 프레임, xyzw).
- `variations.py` — `attach_variation_events(cfg, variations, connector, socket)`: Arena variation term 으로 소켓 pose/질량/카메라/조명 샘플링. 샘플 값은 Arena `variation_recorder` 를 통해 `episode_results_rank*.jsonl` 의 `variations` 로 기록된다.
- (삭제됨) `metrics.py` — 에피소드별 키(`final_pos_err_m, final_rot_err_deg, max_contact_force_n, insertion_depth_ratio, grasped`)는 `cable_insert_task.py::record_insertion` 이 `episode_recorder_terms` 로 jsonl 에 직접 기록한다. `subtasks_completed` 는 아직 기록하지 않는다 (서브태스크 진행률은 후속).
- `assets/connector_simple.usd`, `assets/socket_simple.usd` — SimReady 규격, convex decomposition 충돌 메시.
