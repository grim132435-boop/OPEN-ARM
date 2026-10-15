"""Profile tests. The URDF test needs ORCA_DESCRIPTION=<orcahand_description checkout>."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy  # noqa: E402
import dataclasses  # noqa: E402

from sim.robot.embodiment_profile import (  # noqa: E402
    check_against_orca_core,
    check_against_urdf,
    load_profile,
    load_spec,
)


def test_gripper_profile_is_8_dof():
    p = load_profile("openarm_v1_right_gripper")
    assert p.dof == 8 and p.arm_dof == 7 and p.policy_joints[-1] == "gripper"


def test_orca_profile_is_24_dof_and_orca_core_order():
    p = load_profile("openarm_v1_right_orca")
    assert p.dof == 24 and p.ee_dof == 17
    assert p.policy_joints[7:] == [
        "wrist", "thumb_cmc", "thumb_abd", "thumb_mcp", "thumb_dip",
        "index_abd", "index_mcp", "index_pip", "middle_abd", "middle_mcp", "middle_pip",
        "ring_abd", "ring_mcp", "ring_pip", "pinky_abd", "pinky_mcp", "pinky_pip",
    ]


def test_arm_prefix_identical_across_profiles():
    a, b = load_profile("openarm_v1_right_gripper"), load_profile("openarm_v1_right_orca")
    assert a.policy_joints[:7] == b.policy_joints[:7]
    assert a.sim_joint_exprs[:7] == b.sim_joint_exprs[:7]


def test_every_profile_resolves():
    for name in load_spec()["profiles"]:
        load_profile(name)


@pytest.mark.skipif(not os.environ.get("ORCA_DESCRIPTION"), reason="ORCA_DESCRIPTION not set")
def test_orca_mapping_matches_urdf():
    p = load_profile("openarm_v1_right_orca")
    urdf = Path(os.environ["ORCA_DESCRIPTION"]) / p.ee_spec["asset"]["urdf"]
    assert check_against_urdf(p, urdf) == []


def _swap(p, a, b):
    spec = copy.deepcopy(p.ee_spec)
    idx = {j["name"]: i for i, j in enumerate(spec["joints"])}
    ja, jb = spec["joints"][idx[a]], spec["joints"][idx[b]]
    ja["sim"], jb["sim"] = jb["sim"], ja["sim"]
    return dataclasses.replace(p, ee_spec=spec)


@pytest.mark.skipif(not os.environ.get("ORCA_DESCRIPTION"), reason="ORCA_DESCRIPTION not set")
def test_urdf_check_catches_swaps():
    p = load_profile("openarm_v1_right_orca")
    urdf = Path(os.environ["ORCA_DESCRIPTION"]) / p.ee_spec["asset"]["urdf"]
    for a, b in [("middle_abd", "ring_abd"), ("middle_mcp", "ring_mcp"), ("index_pip", "middle_pip"),
                 ("thumb_mcp", "index_mcp"), ("thumb_cmc", "thumb_abd")]:
        assert check_against_urdf(_swap(p, a, b), urdf), f"swap {a}<->{b} not detected"
    spec = copy.deepcopy(p.ee_spec)
    spec["joints"][0]["range_deg"] = [-35, 65]  # mirrored wrist range must fail
    assert check_against_urdf(dataclasses.replace(p, ee_spec=spec), urdf)


@pytest.mark.skipif(not os.environ.get("ORCA_CORE"), reason="ORCA_CORE not set")
def test_orca_core_order_and_roms():
    assert check_against_orca_core(load_profile("openarm_v1_right_orca"), Path(os.environ["ORCA_CORE"])) == []
