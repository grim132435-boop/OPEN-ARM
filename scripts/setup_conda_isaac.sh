#!/usr/bin/env bash
# Isaac Sim(소스 빌드) + Isaac Lab + Arena 를 conda 환경에 설치한다. DGX Spark(aarch64) / x86_64 공용.
#
#   scripts/setup_conda_isaac.sh                         # 권장: v6.0.1 새로 빌드 (Arena 0.3.0 호환)
#   scripts/setup_conda_isaac.sh --isaacsim-src ~/IsaacSim --reuse-build --allow-version-mismatch
#                                                        # 기존(예: 6.3) 빌드 재사용 — 호환 미보장
# 옵션:
#   --prefix DIR               소스 체크아웃 위치 (기본 ~/openarm-sim)
#   --env NAME                 conda env 이름 (기본 openarm-isaac)
#   --isaacsim-src DIR         기존 IsaacSim 소스 트리 사용 (없으면 versions.yaml ref 로 새로 받음)
#   --reuse-build              _build 와 휠이 있으면 빌드/휠 생성 생략
#   --allow-version-mismatch   Isaac Sim minor 가 versions.yaml compatible_minor 와 달라도 진행
#   --with-policy-env          openarm-policy env 도 생성 (env/conda/policy.yml)
#   --dry-run                  실행할 명령만 출력
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
V="$REPO/env/versions.yaml"

PREFIX=${HOME:-/root}/openarm-sim; ENV_NAME=openarm-isaac; SIM_SRC=""; REUSE=0; ALLOW_MISMATCH=0; POLICY_ENV=0; DRY=0
while [ $# -gt 0 ]; do case "$1" in
  --prefix) PREFIX=$2; shift 2;; --env) ENV_NAME=$2; shift 2;;
  --isaacsim-src) SIM_SRC=$2; shift 2;; --reuse-build) REUSE=1; shift;;
  --allow-version-mismatch) ALLOW_MISMATCH=1; shift;; --with-policy-env) POLICY_ENV=1; shift;;
  --dry-run) DRY=1; shift;; -h|--help) sed -n 2,16p "$0"; exit 0;;
  *) echo "unknown arg $1" >&2; exit 2;; esac; done

run(){ printf '+'; printf ' %q' "$@"; echo; [ $DRY = 1 ] || "$@"; }
in_dir(){ local d=$1; shift; printf '+ (cd %q &&' "$d"; printf ' %q' "$@"; echo ')'; [ $DRY = 1 ] || (cd "$d" && "$@"); }
step(){ printf '\n== %s\n' "$*"; }
ver(){ python3 -c 'import sys,yaml;d=yaml.safe_load(open(sys.argv[1]))
for k in sys.argv[2].split("."): d=d[k]
print(d)' "$V" "$1"; }
fetch_ref(){ # url ref dir [lfs]  — 얕은 단일 ref 체크아웃 (main 전체 클론/LFS 다운로드 없이)
  local url=$1 ref=$2 dir=$3 lfs=${4:-0}
  if [ ! -d "$dir/.git" ]; then run git init -q "$dir"; run git -C "$dir" remote add origin "$url"; fi
  [ "$lfs" = 1 ] && run git -C "$dir" lfs install --local
  run git -C "$dir" fetch --depth 1 origin "$ref"
  run git -C "$dir" checkout -q FETCH_HEAD
  [ "$lfs" = 1 ] && run git -C "$dir" lfs pull
  return 0
}

ARCH=$(uname -m)
step "0. 사전 점검 (arch=$ARCH)"
command -v conda >/dev/null || { echo "conda 가 필요합니다 (Miniforge 권장)" >&2; [ $DRY = 1 ] || exit 1; }
python3 -c 'import yaml' 2>/dev/null || { echo "호스트 python3 에 PyYAML 필요: pip install pyyaml" >&2; exit 1; }
git lfs version >/dev/null 2>&1 || { echo "git-lfs 필요: sudo apt install git-lfs (Isaac Sim 자산이 LFS)" >&2; [ $DRY = 1 ] || exit 1; }
GCC_WANT=$(ver isaacsim.gcc)
if ! gcc --version 2>/dev/null | head -1 | grep -q " ${GCC_WANT}\."; then
  echo "WARN: Isaac Sim 빌드는 GCC ${GCC_WANT} 필요. sudo apt install gcc-${GCC_WANT} g++-${GCC_WANT} 후" \
       "update-alternatives 로 전환하세요 (README: GCC 12+ 미지원)."
fi
[ "$ARCH" = aarch64 ] && echo "aarch64: libx11-dev xorg-dev 필요 (휠 없는 의존성 빌드용)"
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}   # 비대화식 import 시 EULA 프롬프트 방지 (사용자 동의 전제)
mkdir -p "$PREFIX"

step "1. Isaac Sim 소스"
if [ -n "$SIM_SRC" ]; then
  SIM_DIR=$(cd "$SIM_SRC" && pwd)
  if [ -d "$SIM_DIR/.git" ]; then run git -C "$SIM_DIR" lfs pull; fi
else SIM_DIR="$PREFIX/IsaacSim"; fetch_ref "$(ver isaacsim.repo)" "$(ver isaacsim.ref)" "$SIM_DIR" 1; fi
SIM_VER=$(cat "$SIM_DIR/VERSION" 2>/dev/null || true)
if [ -z "$SIM_VER" ]; then
  if [ $DRY = 1 ] && [ -z "$SIM_SRC" ]; then SIM_VER=$(ver isaacsim.ref); SIM_VER=${SIM_VER#v}; echo "(dry-run: VERSION 대신 ref 로 가정)"
  else echo "ERROR: $SIM_DIR/VERSION 없음 — Isaac Sim 소스 트리가 맞는지 확인" >&2; exit 4; fi
fi
WANT_MINOR=$(ver isaacsim.compatible_minor)
echo "Isaac Sim 소스 버전: $SIM_VER (Arena 0.3.0 호환: ${WANT_MINOR}.x)"
case "$SIM_VER" in
  "$WANT_MINOR".*) ;;
  *) if [ $ALLOW_MISMATCH = 1 ]; then
       echo "WARN: $SIM_VER ≠ ${WANT_MINOR}.x — Arena/Isaac Lab 핀과 다릅니다. API 불일치 가능 (docs/setup-conda.md §호환성)."
     else
       echo "ERROR: Isaac Sim $SIM_VER 는 Arena 0.3.0(Isaac Sim ${WANT_MINOR}.x) 과 검증되지 않았습니다." >&2
       echo "  새로 빌드: --isaacsim-src 없이 실행 / 그대로 진행: --allow-version-mismatch" >&2; exit 4
     fi;;
esac
SIM_VER_BASE=${SIM_VER%%-*}   # 6.0.1-rc.7 → 6.0.1

step "2. Isaac Sim 빌드"
PLAT=linux-$ARCH
if [ $REUSE = 1 ] && [ -x "$SIM_DIR/_build/$PLAT/release/python.sh" ]; then echo "기존 빌드 재사용: $SIM_DIR/_build/$PLAT/release"
else in_dir "$SIM_DIR" ./build.sh -r; fi

step "3. Isaac Sim pip 휠 생성 (README §PIP Packages)"
DIST="$SIM_DIR/_build/packages/dist"
if [ $REUSE = 1 ] && ls "$DIST"/*"$SIM_VER_BASE"*.whl >/dev/null 2>&1; then echo "기존 휠 재사용 ($SIM_VER_BASE)"
else
  if [ -d "$DIST" ]; then run find "$DIST" -maxdepth 1 -name '*.whl' -delete; fi   # 다른 빌드 휠 혼입 방지
  in_dir "$SIM_DIR" ./repo.sh python_package --create
  in_dir "$SIM_DIR" ./repo.sh comment_archive_deps
  in_dir "$SIM_DIR" ./repo.sh python_package --wheel
fi

step "4. conda env: $ENV_NAME"
if [ $DRY = 0 ] && conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then echo "이미 존재 — 재사용"
else run conda env create -n "$ENV_NAME" -f "$REPO/env/conda/isaac.yml"; fi
if [ $DRY = 1 ]; then CPY="<conda-python>"; PYV=$(ver python)
else
  CPY=$(conda run -n "$ENV_NAME" python -c 'import sys;print(sys.executable)')
  PYV=$(conda run -n "$ENV_NAME" python -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
fi
[ "$PYV" = "$(ver python)" ] || { echo "env python $PYV ≠ $(ver python)" >&2; exit 5; }
UVPIP=(conda run -n "$ENV_NAME" uv pip install --python "$CPY")

step "5. Isaac Sim 휠 → conda env"
shopt -s nullglob; WHEELS=("$DIST"/*"$SIM_VER_BASE"*.whl); shopt -u nullglob
if [ ${#WHEELS[@]} -eq 0 ]; then
  if [ $DRY = 1 ]; then WHEELS=("$DIST/isaacsim-*${SIM_VER_BASE}*.whl")
  else echo "휠 없음: $DIST/*${SIM_VER_BASE}*.whl" >&2; exit 6; fi
fi
run "${UVPIP[@]}" "${WHEELS[@]}"

# 이후 단계가 torch 스택을 바꾸지 않도록 Isaac Sim 휠이 설치한 버전을 고정
CONSTRAINTS="$PREFIX/isaac-constraints.txt"
if [ $DRY = 0 ]; then
  conda run -n "$ENV_NAME" uv pip freeze --python "$CPY" \
    | grep -E '^(torch|torchvision|torchaudio|numpy|warp-lang)==' >"$CONSTRAINTS" || true
else : >"$CONSTRAINTS"; fi
echo "constraints: $(tr '\n' ' ' <"$CONSTRAINTS")"

step "6. Isaac Lab (Arena 고정 커밋) — 멤버 패키지를 직접 editable 설치"
LAB_DIR="$PREFIX/IsaacLab"; fetch_ref "$(ver isaaclab.repo)" "$(ver isaaclab.ref)" "$LAB_DIR"
# 루트 isaaclab-dev 는 uv sync 전용 가상 프로젝트(package=false, torch 2.10 고정)라 pip 설치 대상이 아니다.
# Arena 0.3.0 의 isaaclab-from-source 그룹과 같은 멤버를 설치하되, isaaclab 에 [isaacsim] extra 는 붙이지 않는다
# (붙이면 pypi isaacsim 휠이 로컬 빌드를 덮어씀).
S="$LAB_DIR/source"
run "${UVPIP[@]}" --constraint "$CONSTRAINTS" \
  -e "$S/isaaclab" -e "$S/isaaclab_assets" -e "$S/isaaclab_contrib" -e "$S/isaaclab_experimental" \
  -e "$S/isaaclab_mimic" -e "$S/isaaclab_newton[all]" -e "$S/isaaclab_ov" -e "$S/isaaclab_ovphysx" \
  -e "$S/isaaclab_physx[newton]" -e "$S/isaaclab_ppisp" -e "$S/isaaclab_rl[rsl-rl]" -e "$S/isaaclab_tasks" \
  -e "$S/isaaclab_tasks_experimental" -e "$S/isaaclab_teleop" -e "$S/isaaclab_visualizers"

step "7. Isaac Lab-Arena $(ver arena.ref)"
ARENA_DIR="$PREFIX/IsaacLab-Arena"; fetch_ref "$(ver arena.repo)" "$(ver arena.ref)" "$ARENA_DIR"
run "${UVPIP[@]}" --constraint "$CONSTRAINTS" -e "$ARENA_DIR" pyzmq pyyaml

step "8. OpenArm 자산 + ORCA 설명/코어"
fetch_ref "$(ver openarm_isaac_lab.repo)" "$(ver openarm_isaac_lab.ref)" "$PREFIX/openarm_isaac_lab" 1
fetch_ref "$(ver orca.description_repo)" "$(ver orca.description_ref)" "$PREFIX/orcahand_description" 1
fetch_ref "$(ver orca.core_repo)" "$(ver orca.core_ref)" "$PREFIX/orca_core"

if [ $POLICY_ENV = 1 ]; then
  step "9. 정책 서버 env: openarm-policy"
  if [ $DRY = 1 ] || ! conda env list | awk '{print $1}' | grep -qx openarm-policy; then
    run conda env create -f "$REPO/env/conda/policy.yml"; fi
fi

step "10. 검증"
run conda run -n "$ENV_NAME" python -c 'import isaacsim, isaaclab, isaaclab_arena, torch; print("imports ok, torch", torch.__version__)'
run conda run -n "$ENV_NAME" env ORCA_DESCRIPTION="$PREFIX/orcahand_description" ORCA_CORE="$PREFIX/orca_core" \
  python "$REPO/scripts/validate_scaffold.py"
cat <<EOF

완료. 다음 환경변수를 ~/.bashrc 등에 추가하세요:
  export ARENA_DIR=$ARENA_DIR
  export ARENA_PYTHON=$CPY
  export POLICY_PYTHON=\$(conda run -n openarm-policy python -c 'import sys;print(sys.executable)')
  export OPENARM_ASSETS=$PREFIX/openarm_isaac_lab
  export ORCA_DESCRIPTION=$PREFIX/orcahand_description
  export ORCA_CORE=$PREFIX/orca_core
그 다음: scripts/check_env.sh
EOF
