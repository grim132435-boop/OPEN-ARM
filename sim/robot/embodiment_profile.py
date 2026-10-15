"""Resolve an arm + end-effector profile into policy-facing joint lists.

Pure Python (no Isaac imports) so the policy server, the Arena client, the
validator and tests all share one definition of action/state layout.

    from sim.robot.embodiment_profile import load_profile
    p = load_profile("openarm_v1_right_orca")
    p.dof            # 24 (7 arm + 17 hand)
    p.policy_joints  # ["j1", ..., "j7", "wrist", "thumb_cmc", ...]
    p.sim_joint_exprs    # one sim joint per policy dim (state source), same order
    p.sim_command_exprs  # where each command goes (regex for mimic fingers)
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SPEC = Path(__file__).with_name("end_effectors.yaml")


@dataclass(frozen=True)
class Profile:
    name: str
    arm: str
    end_effector: str
    policy_joints: list[str]
    sim_joint_exprs: list[str]      # 상태를 읽을 sim 관절 (정확히 1개씩, 정책 순서)
    sim_command_exprs: list[str]    # 명령을 보낼 sim 관절 표현식 (mimic 손가락은 regex 로 여러 관절)
    ee_dof: int
    ee_spec: dict = field(repr=False)
    arm_spec: dict = field(repr=False)

    @property
    def dof(self) -> int:
        return len(self.policy_joints)

    @property
    def arm_dof(self) -> int:
        return self.dof - self.ee_dof


def load_spec(path: Path = SPEC) -> dict:
    return yaml.safe_load(path.read_text())


def load_profile(name: str, path: Path = SPEC) -> Profile:
    spec = load_spec(path)
    if name not in spec["profiles"]:
        raise KeyError(f"unknown profile {name!r}; have {sorted(spec['profiles'])}")
    pr = spec["profiles"][name]
    arm, ee = spec["arms"][pr["arm"]], spec["end_effectors"][pr["end_effector"]]
    if len(ee["joints"]) != ee["dof"]:
        raise ValueError(f"{pr['end_effector']}: dof={ee['dof']} but {len(ee['joints'])} joints listed")
    joints = arm["joints"] + ee["joints"]
    names = [j["name"] for j in joints]
    if len(names) != len(set(names)):
        raise ValueError(f"{name}: duplicate policy joint names")
    return Profile(
        name=name,
        arm=pr["arm"],
        end_effector=pr["end_effector"],
        policy_joints=names,
        sim_joint_exprs=[j["sim"] for j in joints],
        sim_command_exprs=[j.get("sim_command_expr", j["sim"]) for j in joints],
        ee_dof=ee["dof"],
        ee_spec=ee,
        arm_spec=arm,
    )


def check_against_urdf(profile: Profile, urdf_path: Path, tol_deg: float = 0.6) -> list[str]:
    """Verify the EE joint mapping against the URDF.

    Checks: every sim joint is a revolute joint; limits equal range_deg (no sign flip
    accepted); declared parent->child chains match the URDF link tree; lateral order of
    finger bases. Together these catch swapped names that limits alone cannot.
    """
    root = ET.parse(urdf_path).getroot()
    rev = {j.get("name"): j for j in root.iter("joint") if j.get("type") == "revolute"}
    errs: list[str] = []
    ee_joints = profile.ee_spec["joints"]
    by_name = {j["name"]: j.get("sim") for j in ee_joints}
    if len(rev) != len(ee_joints):
        errs.append(f"URDF has {len(rev)} revolute joints, profile lists {len(ee_joints)}")
    for j in ee_joints:
        sim = j.get("sim")
        if sim not in rev:
            errs.append(f"{j['name']}: sim joint {sim!r} not in URDF")
            continue
        if "range_deg" in j:
            lim = rev[sim].find("limit")
            lo, hi = math.degrees(float(lim.get("lower"))), math.degrees(float(lim.get("upper")))
            want_lo, want_hi = j["range_deg"]
            if abs(lo - want_lo) > tol_deg or abs(hi - want_hi) > tol_deg:
                errs.append(f"{j['name']}: URDF limit [{lo:.1f},{hi:.1f}]° ≠ range_deg {j['range_deg']}")

    # parent/child chains: joint k's child link must be joint k+1's parent link
    def link(jname: str | None, tag: str) -> str | None:
        jt = rev.get(jname)
        return None if jt is None else jt.find(tag).get("link")

    for chain in profile.ee_spec.get("chains", []):
        for a, b in zip(chain, chain[1:]):
            ca, pb = link(by_name.get(a), "child"), link(by_name.get(b), "parent")
            if ca is None or pb is None or ca != pb:
                errs.append(f"chain {a}->{b}: child({a})={ca} ≠ parent({b})={pb}")

    order = profile.ee_spec.get("lateral_order")
    if order:
        ax = "xyz".index(order["axis"])
        vals = []
        for n in order["joints"]:
            jt = rev.get(by_name.get(n))
            if jt is None:
                break
            vals.append(float(jt.find("origin").get("xyz").split()[ax]))
        else:
            pairs = list(zip(vals, vals[1:]))
            ok = all(a < b for a, b in pairs) if order.get("increasing", True) else all(a > b for a, b in pairs)
            if not ok:
                errs.append(f"lateral order {order['joints']} along {order['axis']} violated: {[round(v, 4) for v in vals]}")
    return errs


def check_against_orca_core(profile: Profile, orca_core_dir: Path) -> list[str]:
    """Policy joint order and ranges must equal orca_core's joint_ids / joint_roms for the configured model."""
    cfg_name = profile.ee_spec["real"]["config"]
    cfg_path = orca_core_dir / "orca_core" / "models" / "v2" / cfg_name / "config.yaml"
    if not cfg_path.exists():
        return [f"orca_core config not found: {cfg_path}"]
    cfg = yaml.safe_load(cfg_path.read_text())
    errs: list[str] = []
    ee_names = [j["name"] for j in profile.ee_spec["joints"]]
    if ee_names != list(cfg["joint_ids"]):
        errs.append(f"policy EE order {ee_names} != orca_core joint_ids {cfg['joint_ids']}")
    for j in profile.ee_spec["joints"]:
        rom = cfg.get("joint_roms", {}).get(j["name"])
        if rom is not None and list(rom) != list(j.get("range_deg", [])):
            errs.append(f"{j['name']}: range_deg {j.get('range_deg')} != orca_core joint_roms {rom}")
    return errs
