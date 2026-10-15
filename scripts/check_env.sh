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

echo "== Python"
if command -v python3.12 >/dev/null; then ok "python3.12 present"; else fail "python3.12 required by Isaac Sim 6.0 / Lab 3.0"; fi

echo "== Isaac"
python3.12 -c 'import isaacsim, sys; print(isaacsim.__version__ if hasattr(isaacsim,"__version__") else "?")' 2>/dev/null \
  && ok "isaacsim importable" || warn "isaacsim not importable (source build pending?)"
python3.12 -c 'import isaaclab' 2>/dev/null && ok "isaaclab importable" || warn "isaaclab not importable"
python3.12 -c 'import isaaclab_arena' 2>/dev/null && ok "isaaclab_arena importable" || warn "isaaclab_arena not importable"

echo "== Services"
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && ok "ollama up" || warn "ollama not reachable on 11434"
curl -fsS http://127.0.0.1:11434/api/tags 2>/dev/null | grep -q nemotron && ok "nemotron model pulled" || warn "nemotron model not found in ollama"
command -v nemoclaw >/dev/null && ok "nemoclaw $(nemoclaw --version 2>/dev/null | head -1)" || warn "nemoclaw CLI missing (fallback: direct ollama)"

echo "== Memory/Disk"
free -g | awk '/Mem:/{printf "  [INFO] mem total %sG, avail %sG\n",$2,$7}'
df -h . | awk 'NR==2{printf "  [INFO] disk avail %s\n",$4}'

exit $RC
