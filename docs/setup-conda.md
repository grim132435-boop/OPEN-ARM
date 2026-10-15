# Conda 환경에서 Isaac Sim + Isaac Lab + Arena 새로 빌드

대상: DGX Spark (aarch64, DGX OS / Ubuntu 24.04) 또는 x86_64 워크스테이션.

## 한 줄 요약
```bash
scripts/setup_conda_isaac.sh --with-policy-env      # v6.0.1 새로 빌드 (권장)
```

## 구조: conda 환경 2개
```
openarm-isaac (python 3.12)                 openarm-policy (python 3.12)
 ├ isaacsim  ← 로컬 소스 빌드 휠              ├ lerobot 0.6.2 (+ torch)
 ├ isaaclab  ← Arena 고정 커밋 (editable)     └ sim/policy/act_policy_server.py
 ├ isaaclab_arena 0.3.0 (editable)                    ▲
 └ sim/policy/act_remote_policy.py ── ZeroMQ :5556 ──┘
```
정책 서버를 분리하는 이유: LeRobot 의 torch 버전과 Isaac Sim 휠이 고정하는 torch 가 다를 수 있습니다. 둘은 ZeroMQ 로만 통신하므로 같은 env 일 필요가 없습니다.

## 버전 호환성 (중요)
| 구성요소 | 고정 버전 | 근거 |
|---|---|---|
| Isaac Sim | **v6.0.1** (6.0.x) | Arena 0.3.0 의 IsaacLab 서브모듈(af1bab4)이 `isaacsim==6.0.1.0`, Arena `uv.lock` 이 `isaacsim-core 6.0.1.0` + `torch 2.11.0` 으로 해석 |
| Isaac Lab | `af1bab4` | Arena 0.3.0 의 `submodules/IsaacLab` 커밋 |
| Isaac Lab-Arena | `release/0.3.0` | |
| Python | 3.12 | Isaac Sim 6.x 요구 |

모든 핀은 `env/versions.yaml` 한 곳에 있습니다.

### 기존 6.3 빌드를 쓰고 싶다면
공개 저장소 기준 정식 태그는 v6.0.0 / v6.0.1 / v6.1.0 / v6.2.0, 프리릴리스 v7.0.0a1 이 있고 6.3 태그는 없습니다 (2026-10-15, main 은 6.2.0-rc). 로컬에 6.3 개발 빌드가 있다면:
```bash
scripts/setup_conda_isaac.sh --isaacsim-src ~/IsaacSim --reuse-build --allow-version-mismatch
```
- `--allow-version-mismatch` 없이는 의도적으로 중단합니다 (exit 4). Arena 0.3.0 은 Isaac Sim 6.0 API 로 검증되었고, 6.x minor 사이에 URDF/MJCF importer 3.0, asset structure 3.0 같은 변경이 있었습니다.
- 권장: 6.3 빌드는 그대로 두고 `--prefix ~/openarm-sim-60` 으로 6.0.1 을 **별도 디렉터리**에 새로 빌드. 두 빌드를 conda env 이름으로 구분 (`--env openarm-isaac-63`).

## 단계 (스크립트가 하는 일)
0. 사전 점검: conda, git-lfs, **GCC 11** (12+ 미지원), aarch64 는 `libx11-dev xorg-dev`. 비대화식 import 를 위해 `OMNI_KIT_ACCEPT_EULA=YES` 를 설정합니다 (EULA 에 동의한다는 뜻이니 확인 후 사용)
1. Isaac Sim 소스를 해당 ref 만 얕게 받고 **Git LFS pull** (자산이 LFS) — 또는 `--isaacsim-src` → `VERSION` 확인
2. `./build.sh -r` (release 만)
3. pip 휠 생성: `./repo.sh python_package --create` → `comment_archive_deps` → `--wheel` (`_build/packages/dist/*.whl`)
4. `conda env create -f env/conda/isaac.yml`
5. 이번 버전의 로컬 휠만 conda env 에 설치 (다른 빌드 휠 혼입 방지) → 설치된 torch/numpy 를 constraint 파일로 고정
6. Isaac Lab 을 Arena 고정 커밋으로 받아 **`source/*` 멤버 패키지를 직접** editable 설치 (Arena 의 `isaaclab-from-source` 그룹과 같은 목록). 루트 `isaaclab-dev` 는 `uv sync` 전용 가상 프로젝트(torch 2.10 고정)라 설치 대상이 아니고, `isaaclab[isaacsim]` extra 도 쓰지 않습니다 — 쓰면 pypi 휠이 로컬 빌드를 덮어씁니다
7. Arena 설치 (같은 constraint 로 `sbi` 등이 torch 를 바꾸지 못하게 함)
8. OpenArm USD 자산(`openarm_isaac_lab`), ORCA 설명 파일·`orca_core` (모두 커밋 고정)
9. (옵션) `openarm-policy` env
10. import 확인 + `validate_scaffold.py` (ORCA URDF 체인/측방향 순서, `orca_core` 순서·가동범위 포함)

## OpenArm 원본 저장소 주의
`enactic/openarm_isaac_lab` 은 Isaac Lab 2.3 / Python 3.11 / Isaac Sim 5.1 기준이고, 내부 import 가 `from source.openarm.openarm...` 형태라 저장소 루트를 CWD 로 요구합니다. 이 프로젝트는 **USD 자산과 관절 설정값만** 재사용하고 그 파이썬 패키지는 설치하지 않습니다 (`OPENARM_ASSETS` 환경변수로 경로 전달).

## 문제 해결
| 증상 | 원인/조치 |
|---|---|
| `build.sh` 컴파일러 버전 오류 | GCC 11 로 전환. 임시로 `--skip-compiler-version-check` (비권장) |
| aarch64 에서 휠 빌드 실패 | `sudo apt install libx11-dev xorg-dev` |
| `import isaaclab_arena` 시 Isaac API 오류 | Isaac Sim minor 불일치 가능성 → 6.0.1 로 새로 빌드 |
| torch 가 바뀜 | `$PREFIX/isaac-constraints.txt` 확인 후 해당 단계 재실행 |
