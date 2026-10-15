"""GPU-free static validation of the closed-loop scaffold.

Checks: YAML parse, Python AST compile, real<->sim observation mapping consistency,
success-criteria integrity (sha256 pin), port/host agreement between policy
config and sandbox policy, agent-writable paths exclude criteria/docs,
chunk/horizon sanity, and referenced-file existence. Exit 1 on any failure.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "sim/arena/openarm_cable_insert_task.yaml"
CRITERIA = ROOT / "sim/arena/success_criteria.yaml"
POLICY = ROOT / "sim/arena/policy_config_act.yaml"
SANDBOX = ROOT / "agent/nemoclaw/policies/openarm-sandbox.yaml"
PY_FILES = [
    ROOT / "sim/robot/embodiment_profile.py",
    ROOT / "sim/policy/act_policy_server.py",
    ROOT / "sim/policy/act_remote_policy.py",
    ROOT / "sim/arena/openarm_cable_insert_environment.py",
    ROOT / "sim/arena/cable_insert_task.py",
    ROOT / "scripts/summarize_result.py",
]
# 사람이 success_criteria.yaml 을 바꿀 때만 갱신한다.
CRITERIA_SHA256 = "120e8aa125a64695ce6b6a980a3204a2e16dfa8bcbad98a797e70c0378b7ed44"
REQUIRED_CRITERIA = {
    "position_error_m_max", "orientation_error_deg_max", "insertion_depth_ratio_min",
    "contact_force_n_max", "hold_time_s_min", "episode_timeout_s",
    "pass_rate_baseline", "pass_rate_sweep_mean",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--print-criteria-hash", action="store_true")
    a = ap.parse_args()
    errs: list[str] = []
    say = (lambda *_: None) if a.quiet else print

    digest = hashlib.sha256(CRITERIA.read_bytes()).hexdigest()
    if a.print_criteria_hash:
        print(digest)
        return 0

    task = yaml.safe_load(TASK.read_text())
    crit = yaml.safe_load(CRITERIA.read_text())
    pol = yaml.safe_load(POLICY.read_text())
    sbx = yaml.safe_load(SANDBOX.read_text())
    say("yaml: ok")

    for f in PY_FILES:
        try:
            ast.parse(f.read_text(), filename=str(f))
        except SyntaxError as e:
            errs.append(f"syntax {f.name}:{e.lineno}: {e.msg}")
    say("python syntax: checked")

    if digest != CRITERIA_SHA256:
        errs.append(f"success_criteria.yaml hash {digest[:12]} != pinned {CRITERIA_SHA256[:12]} (human must re-pin)")
    missing = REQUIRED_CRITERIA - set(crit)
    if missing:
        errs.append(f"success_criteria missing {sorted(missing)}")
    if "success_criteria" in task:
        errs.append("task yaml must not carry success_criteria (lives in success_criteria.yaml)")

    m = pol["observation_mapping"]
    sys.path.insert(0, str(ROOT))
    from sim.robot.embodiment_profile import load_profile, load_spec

    try:
        prof = load_profile(pol["embodiment_profile"])
        say(f"embodiment profile: {prof.name} ({prof.dof} dof)")
    except (KeyError, ValueError) as e:
        errs.append(f"embodiment_profile: {e}")
        prof = None
    for name in load_spec()["profiles"]:
        try:
            p_ = load_profile(name)
            if p_.arm_dof != 7:
                errs.append(f"profile {name}: arm dof {p_.arm_dof} != 7")
        except (KeyError, ValueError) as e:
            errs.append(f"profile {name}: {e}")
    if "joint_order" in m["state"] or "joint_order" in m["action"]:
        errs.append("joint_order must come from embodiment_profile, not policy_config")
    orca_dir = os.environ.get("ORCA_DESCRIPTION")
    if orca_dir:
        from sim.robot.embodiment_profile import check_against_urdf

        orca = load_profile("openarm_v1_right_orca")
        for e in check_against_urdf(orca, Path(orca_dir) / orca.ee_spec["asset"]["urdf"]):
            errs.append(f"ORCA URDF: {e}")
        say("ORCA URDF mapping: checked")
    core_dir = os.environ.get("ORCA_CORE")
    if core_dir:
        from sim.robot.embodiment_profile import check_against_orca_core

        for e in check_against_orca_core(load_profile("openarm_v1_right_orca"), Path(core_dir)):
            errs.append(f"ORCA orca_core: {e}")
        say("ORCA orca_core order/ranges: checked")
    if m["state"]["sim"] != "robot_joint_pos":
        errs.append("state.sim must be robot_joint_pos (Arena observation['policy'] key)")
    names = [c["sim"] for c in m["cameras"]]
    if len(names) != len(set(names)):
        errs.append("duplicate sim camera names")
    res0 = m["cameras"][0]["resolution"]
    for c in m["cameras"]:
        if c["resolution"] != res0:
            errs.append(f"{c['sim']}: resolution differs from first camera")
    if task["task"]["embodiment"]["control"] != "joint_position":
        errs.append("embodiment.control must be joint_position for ACT")

    p = pol["policy"]
    if not (1 <= p["action_horizon"] <= p["chunk_size"]):
        errs.append("action_horizon must be within [1, chunk_size]")
    if p.get("checkpoint") or str(p["checkpoint_dir"]).endswith((".ckpt", ".pt", ".safetensors")):
        errs.append("checkpoint_dir must be a LeRobot pretrained_model directory, not a file")

    host, port = p["server"]["host"], p["server"]["port"]
    allowed = {(r["host"], r["port"]) for r in sbx["network"]["allow"]}
    if (host, port) not in allowed:
        errs.append(f"sandbox does not allow policy server {host}:{port}")
    if port == 5555:
        errs.append("ACT server port collides with GR00T default 5555")
    for w in sbx["filesystem"]["write"]:
        if any(s in w for s in ("docs", "checkpoints", "success_criteria")):
            errs.append(f"agent may write protected path {w}")
    if not any("success_criteria.yaml" in d for d in sbx["filesystem"].get("deny_write", [])):
        errs.append("sandbox deny_write must include success_criteria.yaml")

    ev = task["evaluation"]
    if ev["num_episodes_baseline"] < 50:
        errs.append("baseline episodes < 50 gives a noisy success rate")
    if ev["num_envs"] < 1:
        errs.append("num_envs must be >= 1")

    for e in errs:
        print("FAIL:", e, file=sys.stderr)
    if not errs:
        say("validate_scaffold: all checks passed")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
