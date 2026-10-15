# OPEN-ARM

OpenArm v1.0 케이블·소켓 삽입: 텔레옵 → ACT 학습 → Isaac Lab-Arena 폐루프 검증 (로컬 LLM 에이전트).

| 문서 | 내용 |
|---|---|
| `docs/setup-conda.md` | conda 환경에서 Isaac Sim(소스) + Isaac Lab + Arena 빌드 |
| `docs/plan-cable-socket-insertion.md` | 과제·판정 기준·폐루프 계획 |
| `docs/design-end-effector-swap.md` | 그리퍼 → ORCA Hand 교체 설계 |
| `sim/arena/README.md` | Arena 연동 파일 상태와 Spark 에서 작성할 모듈 |

```bash
scripts/setup_conda_isaac.sh --with-policy-env   # 환경 구축
scripts/check_env.sh                             # 점검
python3 scripts/validate_scaffold.py             # GPU 없이 정적 검증
scripts/run_closed_loop.sh --mode baseline       # 폐루프 평가 1회
```
