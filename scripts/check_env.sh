#!/usr/bin/env bash
# DGX Spark 사전 점검. 실패 항목은 docs/plan-cable-socket-insertion.md §1 참고.
set -u
ok(){ printf '  [OK]   %s\n' "$*"; }
warn(){ printf '  [WARN] %s\n' "$*"; }
fail(){ printf '  [FAIL] %s\n' "$*"; RC=1; }
RC=0

echo "== Platform"
ARCH=$(uname -m); [ "$ARCH" = "aarch64" ] && ok "arch $ARCH (DGX Spark)" || warn "arch $ARCH (not Spark; pip Isaac Sim path may apply)"
command -v nvidia-smi >/dev/null && ok "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)" || fail "nvidia-smi missing"

echo "== Python (conda envs from scripts/setup_conda_isaac.sh)"
APY=${ARENA_PYTHON:-}; PPY=${POLICY_PYTHON:-}
[ -n "$APY" ] || APY=$(conda run -n openarm-isaac python -c 'import sys;print(sys.executable)' 2>/dev/null || true)
[ -n "$PPY" ] || PPY=$(conda run -n openarm-policy python -c 'import sys;print(sys.executable)' 2>/dev/null || true)
if [ -n "$APY" ] && "$APY" -c 'import sys;assert sys.version_info[:2]==(3,12)' 2>/dev/null; then ok "isaac python $APY (3.12)"
else fail "openarm-isaac env python 3.12 not found (run scripts/setup_conda_isaac.sh or set ARENA_PYTHON)"; APY=false; fi
if [ -n "$PPY" ] && "$PPY" -c 'import lerobot' 2>/dev/null; then ok "policy python $PPY (lerobot ok)"
else warn "openarm-policy env not ready (lerobot); set POLICY_PYTHON"; fi

echo "== Isaac"
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
"$APY" -c 'import isaacsim' 2>/dev/null && ok "isaacsim importable" || warn "isaacsim not importable (source build pending?)"
"$APY" -c 'import isaaclab' 2>/dev/null && ok "isaaclab importable" || warn "isaaclab not importable"
"$APY" -c 'import isaaclab_arena' 2>/dev/null && ok "isaaclab_arena importable" || warn "isaaclab_arena not importable"
[ -n "${OPENARM_ASSETS:-}" ] && [ -d "$OPENARM_ASSETS" ] && ok "OPENARM_ASSETS=$OPENARM_ASSETS" || warn "OPENARM_ASSETS not set"
[ -n "${ORCA_DESCRIPTION:-}" ] && [ -d "$ORCA_DESCRIPTION" ] && ok "ORCA_DESCRIPTION set" || warn "ORCA_DESCRIPTION not set (ORCA mapping check skipped)"

echo "== Services"
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && ok "ollama up" || warn "ollama not reachable on 11434"
curl -fsS http://127.0.0.1:11434/api/tags 2>/dev/null | grep -q nemotron && ok "nemotron model pulled" || warn "nemotron model not found in ollama"
command -v nemoclaw >/dev/null && ok "nemoclaw $(nemoclaw --version 2>/dev/null | head -1)" || warn "nemoclaw CLI missing (fallback: direct ollama)"

echo "== Memory/Disk"
free -g | awk '/Mem:/{printf "  [INFO] mem total %sG, avail %sG\n",$2,$7}'
df -h . | awk 'NR==2{printf "  [INFO] disk avail %s\n",$4}'

exit $RC
