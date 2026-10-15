"""Collapse Arena episode outputs into reports/<run>/result.json for the agent.

Arena release/0.3.0 writes ``<output_base_dir>/<timestamp>/episode_results_rank{N}.jsonl``
with keys (isaaclab_arena/recording/common_terms.py): env_id, episode_in_env, seed,
success, episode_length, language_instruction, timestamp, and optionally
``variations``. Our metrics (sim/arena/metrics.py) add: final_pos_err_m,
final_rot_err_deg, max_contact_force_n, insertion_depth_ratio, grasped,
subtasks_completed.  ``success`` is the ground truth; the extra keys only
decide the failure bucket.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

MODES = ("align_fail", "force_exceed", "grasp_miss", "timeout", "server_error", "unclassified")


def failure_mode(ep: dict, crit: dict, control_hz: int) -> str:
    if ep.get("error"):
        return "server_error"
    if ep.get("max_contact_force_n", 0.0) > crit["contact_force_n_max"]:
        return "force_exceed"
    timed_out = ep.get("episode_length", 0) >= int(crit["episode_timeout_s"] * control_hz)
    if "grasped" in ep and not ep["grasped"]:
        return "grasp_miss"
    if timed_out:
        return "timeout"
    if "final_pos_err_m" in ep or "final_rot_err_deg" in ep:
        return "align_fail"
    return "unclassified"


def load_episodes(run: Path) -> list[dict]:
    eps: list[dict] = []
    for p in sorted(run.rglob("episode_results_rank*.jsonl")):
        eps += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return eps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--criteria", type=Path, required=True)
    ap.add_argument("--control_hz", type=int, default=30)
    ap.add_argument("--arena_rc", type=int, default=0)
    a = ap.parse_args()
    crit = yaml.safe_load(a.criteria.read_text())
    eps = load_episodes(a.run)

    modes: Counter[str] = Counter()
    subtask: Counter[str] = Counter()
    per_var: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    succ = 0
    for ep in eps:
        ok = bool(ep.get("success"))
        succ += ok
        if not ok:
            modes[failure_mode(ep, crit, a.control_hz)] += 1
        for st in ep.get("subtasks_completed", []):
            subtask[st] += 1
        for k, v in (ep.get("variations") or {}).items():
            per_var[k][json.dumps(v)].append(ok)

    n = len(eps)
    result = {
        "run": a.run.name,
        "arena_exit_code": a.arena_rc,
        "episodes": n,
        "success_rate": (succ / n) if n else None,
        "failure_modes": {m: modes.get(m, 0) for m in MODES},
        "subtask_progress": {k: v / n for k, v in subtask.items()} if n else {},
        "success_by_variation": {
            k: {val: sum(oks) / len(oks) for val, oks in vals.items()} for k, vals in per_var.items()
        },
        "videos": sorted(str(p) for p in a.run.rglob("*.mp4")),
    }
    if a.arena_rc != 0 and n == 0:
        result["failure_modes"]["server_error"] = 1
        result["note"] = f"policy_runner exited {a.arena_rc} with no episodes; see arena.log / policy_server.log"
    (a.run / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
