"""External Arena environment: OpenArm cable-socket insertion.

Load with (verified flag, Arena release/0.3.0 ``isaaclab_arena/cli``):
    --external_environment_class_path sim.arena.openarm_cable_insert_environment:OpenArmCableInsertEnvironment
    openarm_cable_insert --variations {none,full}

Pattern: isaaclab_arena_examples/external_environments/basic.py (ExampleEnvironmentBase.get_env).
Requires the ``openarm_joint`` embodiment asset (sim/arena/openarm_embodiment.py) to be imported
so it registers before the scene is built.

STATUS: structure verified against Arena source; physics/asset behaviour must be run on the Spark.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

import sim.arena.openarm_embodiment  # noqa: F401  (registers the embodiment asset)
from sim.arena.cable_insert_task import CableInsertTask

def _rpy_deg_to_xyzw(rpy_deg) -> tuple[float, float, float, float]:
    """Roll-pitch-yaw (deg, ZYX convention) -> quaternion (x, y, z, w) as Arena's Pose expects."""
    import math

    r, p, y = (math.radians(v) / 2.0 for v in rpy_deg)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


TASK_YAML = Path(__file__).with_name("openarm_cable_insert_task.yaml")
CRITERIA_YAML = Path(__file__).with_name("success_criteria.yaml")


class OpenArmCableInsertEnvironment(ExampleEnvironmentBase):
    name: str = "openarm_cable_insert"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose, PoseRange

        from sim.arena.openarm_assets import ConnectorAsset, SocketAsset

        spec = yaml.safe_load(TASK_YAML.read_text())
        crit = yaml.safe_load(CRITERIA_YAML.read_text())
        t = spec["task"]
        sc = t["scene"]

        background = self.asset_registry.get_asset_by_name(sc["background"])()
        light = self.asset_registry.get_asset_by_name("light")()
        embodiment = self.asset_registry.get_asset_by_name(t["embodiment"]["asset_name"])(
            enable_cameras=getattr(args_cli, "enable_cameras", True)
        )

        table_ref = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
        )
        table_ref.add_relation(IsAnchor())

        connector = ConnectorAsset(usd_path=sc["connector"]["usd"], mass_kg=sc["connector"]["mass_kg"])
        lo, hi = sc["connector"]["spawn_xyz_min"], sc["connector"]["spawn_xyz_max"]
        connector.set_initial_pose(
            PoseRange(position_xyz_min=tuple(lo), position_xyz_max=tuple(hi), rpy_min=(0, 0, 0), rpy_max=(0, 0, 0))
        )
        connector.add_relation(On(table_ref))

        socket = SocketAsset(usd_path=sc["socket"]["usd"], insertion_axis=sc["socket"]["insertion_axis"],
                             insertion_depth_m=sc["socket"]["insertion_depth_m"])
        socket.set_initial_pose(
            Pose(position_xyz=tuple(sc["socket"]["xyz"]), rotation_xyzw=_rpy_deg_to_xyzw(sc["socket"]["rpy_deg"]))
        )

        task = CableInsertTask(
            connector=connector,
            socket=socket,
            criteria=crit,
            subtasks=spec["subtasks"],
            variations=spec["variations"] if args_cli.variations == "full" else None,
            physics=spec["physics"],
            episode_length_s=crit["episode_timeout_s"],
            task_description=t["description"],
        )

        scene = Scene(assets=[background, table_ref, light, connector, socket])
        return IsaacLabArenaEnvironment(
            name=self.name, embodiment=embodiment, scene=scene, task=task,
            episode_recorder_terms=task.get_episode_recorder_terms(),
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--variations", choices=["none", "full"], default="none")
