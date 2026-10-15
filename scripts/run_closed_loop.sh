#!/usr/bin/env bash
# 폐루프 평가 1회 실행 (에이전트가 호출).  --mode baseline | sweep
# 출력: reports/<run_id>/result.json
# Arena release/0.3.0 policy_runner 플래그 기준 (isaaclab_arena/evaluation/policy_runner_cli.py 확인됨).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

MODE=baseline
while [ $# -gt 0 ]; do case "$1" in --mode) MODE=$2; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ "$MODE" = baseline ] || [ "$MODE" = sweep ] || { echo "mode must be baseline|sweep" >&2; exit 2; }

ARENA_DIR=${ARENA_DIR:-${HOME:-/root}/IsaacLab-Arena}
PY=${ARENA_PYTHON:-python3}          # Isaac Lab venv 의 python 을 ARENA_PYTHON 으로 지정
TASK=sim/arena/openarm_cable_insert_task.yaml
POLICY_CFG=sim/arena/policy_config_act.yaml
RUN_ID=$(date +%Y%m%d-%H%M%S)-$MODE
OUT=reports/$RUN_ID
mkdir -p "$OUT"

python3 scripts/validate_scaffold.py --quiet

yq(){ python3 -c 'import sys,yaml;d=yaml.safe_load(open(sys.argv[1]))
for k in sys.argv[2].split("."): d=d[k]
print(d)' "$@"; }
PORT=$(yq "$POLICY_CFG" policy.server.port)
NUM_ENVS=$(yq "$TASK" evaluation.num_envs)
SEED=$(yq "$TASK" evaluation.seed)
if [ "$MODE" = baseline ]; then NUM_EP=$(yq "$TASK" evaluation.num_episodes_baseline); VARS=none
else NUM_EP=$(yq "$TASK" evaluation.num_episodes_sweep); VARS=full; fi

# --- 정책 서버: 설정 해시가 바뀌면 재시작, 종료 시 정리 ---
CFG_HASH=$(sha256sum "$POLICY_CFG" | cut -c1-16)
PID_FILE=reports/.act_server.pid; HASH_FILE=reports/.act_server.hash
port_open(){ (exec 3<>/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; }
if port_open && [ -f "$HASH_FILE" ] && [ "$(cat "$HASH_FILE")" != "$CFG_HASH" ] && [ -f "$PID_FILE" ]; then
  kill "$(cat "$PID_FILE")" 2>/dev/null || true; sleep 1
fi
STARTED=0
trap '[ $STARTED = 1 ] && kill "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null || true' EXIT
if ! port_open; then
  nohup "$PY" sim/policy/act_policy_server.py --config "$POLICY_CFG" >"$OUT/policy_server.log" 2>&1 &
  echo $! >"$PID_FILE"; echo "$CFG_HASH" >"$HASH_FILE"; STARTED=1
  for _ in $(seq 1 60); do port_open && break; sleep 1; done
  port_open || { echo "policy server failed to open port $PORT (see $OUT/policy_server.log)" >&2; exit 3; }
fi

# --- Arena 평가 (pipefail 로 실패해도 summarize 는 실행해 server_error 를 기록) ---
set +e
"$PY" "$ARENA_DIR/isaaclab_arena/evaluation/policy_runner.py" \
  --headless --enable_cameras --num_envs "$NUM_ENVS" --seed "$SEED" \
  --policy_type sim.policy.act_remote_policy.ActRemoteClosedloopPolicy \
  --policy_config_yaml_path "$POLICY_CFG" \
  --remote_host 127.0.0.1 --remote_port "$PORT" \
  --num_episodes "$NUM_EP" \
  --output_base_dir "$OUT" --record_camera_video \
  --external_environment_class_path sim.arena.openarm_cable_insert_environment:OpenArmCableInsertEnvironment \
  openarm_cable_insert --variations "$VARS" \
  2>&1 | tee "$OUT/arena.log"
ARENA_RC=${PIPESTATUS[0]}
set -e

python3 scripts/summarize_result.py --run "$OUT" --criteria sim/arena/success_criteria.yaml \
  --control_hz "$(yq "$TASK" task.embodiment.control_hz)" --arena_rc "$ARENA_RC"
echo "$OUT/result.json"
