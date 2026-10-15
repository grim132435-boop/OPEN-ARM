"""CableInsertTask — Arena TaskBase implementation for connector→socket insertion.

TaskBase (release/0.3.0, isaaclab_arena/tasks/task_base.py) abstract methods:
    get_scene_cfg, get_termination_cfg, get_events_cfg, get_mimic_env_cfg, get_metrics
Arena records ``success`` from ``termination_manager.get_term("success")``
(isaaclab_arena/recording/common_terms.py), so the success term here IS the
ground-truth label that scripts/summarize_result.py reads.

Per-episode keys for the failure-mode classifier (final_pos_err_m, final_rot_err_deg,
max_contact_force_n, insertion_depth_ratio, grasped) are written to the jsonl via
``episode_recorder_terms`` (isaaclab_arena/recording/episode_recorder_manager.py);
``get_metrics`` only feeds the aggregate MetricsManager report.
Scene contract: connector and socket are Arena ``Object``s (ObjectType.RIGID). The
connector ContactSensor is attached here via ``get_scene_cfg`` using
``Object.get_contact_sensor_cfg()`` (Arena idiom: isaaclab_arena/tasks/sorting_task.py).
Written against Isaac Lab 3.0 naming; needs a Spark run to validate.
"""

from __future__ import annotations

import math
from typing import Any

import torch

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply, quat_mul

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.configclass import make_configclass


# ---------------- geometry helpers (sim side) ----------------
def _goal_pose(env, socket_name: str, goal_offset_local: list[float], goal_quat_local: list[float]):
    """Goal pose of the connector origin when fully inserted, in world frame."""
    s = env.scene[socket_name].data
    off = torch.tensor(goal_offset_local, device=env.device).expand(env.num_envs, 3)
    q_loc = torch.tensor(goal_quat_local, device=env.device).expand(env.num_envs, 4)
    return s.root_pos_w + quat_apply(s.root_quat_w, off), quat_mul(s.root_quat_w, q_loc)


def _insertion_depth(env, connector_name: str, socket_name: str, axis_local: list[float]) -> torch.Tensor:
    c, s = env.scene[connector_name].data, env.scene[socket_name].data
    axis_w = quat_apply(s.root_quat_w, torch.tensor(axis_local, device=env.device).expand(env.num_envs, 3))
    return ((s.root_pos_w - c.root_pos_w) * axis_w).sum(-1)  # +ve as connector goes down the axis


def _contact_force(env, sensor_name: str) -> torch.Tensor:
    f = env.scene.sensors[sensor_name].data.net_forces_w  # (N, bodies, 3)
    return torch.linalg.norm(f, dim=-1).amax(dim=-1)


def _quat_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.acos(torch.clamp(torch.abs((q1 * q2).sum(-1)), max=1.0))


def _state(env) -> dict[str, torch.Tensor]:
    """Per-env scratch buffers stored on the env object (ManagerBasedRLEnv has no task_state)."""
    st = getattr(env, "_cable_insert_state", None)
    if st is None:
        z = lambda dt: torch.zeros(env.num_envs, dtype=dt, device=env.device)
        st = {"hold": z(torch.int32), "max_force": z(torch.float32), "grasped": z(torch.bool),
              "spawn_z": z(torch.float32), "fresh": torch.ones(env.num_envs, dtype=torch.bool, device=env.device)}
        setattr(env, "_cable_insert_state", st)
    return st


# ---------------- MDP terms ----------------
def connector_inserted(env, connector_cfg: SceneEntityCfg, socket_cfg: SceneEntityCfg, axis_local, goal_offset_local,
                       goal_quat_local, pos_tol: float, rot_tol_rad: float, depth_min: float, hold_steps: int,
                       sensor_name: str, lift_min_m: float) -> torch.Tensor:
    c = env.scene[connector_cfg.name].data
    gp, gq = _goal_pose(env, socket_cfg.name, goal_offset_local, goal_quat_local)
    pos_err = torch.linalg.norm(c.root_pos_w - gp, dim=-1)
    rot_err = _quat_angle(c.root_quat_w, gq)
    depth = _insertion_depth(env, connector_cfg.name, socket_cfg.name, axis_local)
    st = _state(env)
    # Capture spawn height on the first step after reset (event ordering across merged cfgs is not guaranteed).
    st["spawn_z"] = torch.where(st["fresh"], c.root_pos_w[:, 2], st["spawn_z"])
    st["fresh"][:] = False
    force = _contact_force(env, sensor_name)
    st["max_force"] = torch.maximum(st["max_force"], force)
    st["grasped"] |= (c.root_pos_w[:, 2] - st["spawn_z"]) > lift_min_m  # lifted off the table => grasped
    ok = (pos_err <= pos_tol) & (rot_err <= rot_tol_rad) & (depth >= depth_min)
    st["hold"] = torch.where(ok, st["hold"] + 1, torch.zeros_like(st["hold"]))
    return st["hold"] >= hold_steps


def contact_force_exceeded(env, sensor_name: str, force_max: float) -> torch.Tensor:
    return _contact_force(env, sensor_name) > force_max


def reset_task_state(env, env_ids: torch.Tensor):
    st = _state(env)
    for k in ("hold", "max_force", "grasped", "spawn_z"):
        st[k][env_ids] = 0
    st["fresh"][env_ids] = True


# ---------------- per-episode recorder (-> episode_results_rank*.jsonl) ----------------
def record_insertion(env, env_id: int, connector_name: str, socket_name: str, axis_local, goal_offset_local,
                     goal_quat_local, insertion_depth_m: float) -> dict[str, Any]:
    c = env.scene[connector_name].data
    gp, gq = _goal_pose(env, socket_name, goal_offset_local, goal_quat_local)
    depth = _insertion_depth(env, connector_name, socket_name, axis_local)
    st = _state(env)
    return {
        "final_pos_err_m": float(torch.linalg.norm(c.root_pos_w[env_id] - gp[env_id])),
        "final_rot_err_deg": float(torch.rad2deg(_quat_angle(c.root_quat_w[env_id : env_id + 1], gq[env_id : env_id + 1]))[0]),
        "insertion_depth_ratio": float(depth[env_id] / insertion_depth_m),
        "max_contact_force_n": float(st["max_force"][env_id]),
        "grasped": bool(st["grasped"][env_id]),
    }


# ---------------- Task ----------------
class CableInsertTask(TaskBase):
    CONTACT_SENSOR = "connector_contact"

    def __init__(self, connector, socket, criteria: dict[str, Any], subtasks: list[str],
                 variations: dict[str, list[float]] | None, physics: dict[str, float],
                 episode_length_s: float, task_description: str):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.connector, self.socket = connector, socket
        self.crit, self.subtasks, self.variations, self.physics = criteria, subtasks, variations, physics
        self.control_hz = 30
        self.hold_steps = max(1, int(round(criteria["hold_time_s_min"] * self.control_hz)))
        # Socket-frame geometry (SocketAsset exposes these plain attributes; no .data wrapper needed)
        self.geom = dict(
            axis_local=list(socket.insertion_axis_local),          # e.g. [0, 0, -1]
            goal_offset_local=list(socket.goal_offset_local),      # connector origin at full insertion
            goal_quat_local=list(socket.goal_quat_local_xyzw),     # connector orientation at full insertion
        )

    def get_scene_cfg(self) -> Any:
        # Task-owned contact sensor on the connector (opens the USD -> only valid once assets exist on the Spark).
        sensor_cfg = self.connector.get_contact_sensor_cfg(contact_against_object=None)
        SceneCfg = make_configclass("SceneCfg", [(self.CONTACT_SENSOR, ContactSensorCfg, sensor_cfg)])
        return SceneCfg()

    def get_termination_cfg(self) -> Any:
        c, g = self.crit, self.geom

        @configclass
        class TerminationsCfg:
            time_out = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)  # Arena does NOT add this
            success = TerminationTermCfg(
                func=connector_inserted,
                params=dict(
                    connector_cfg=SceneEntityCfg(self.connector.name),
                    socket_cfg=SceneEntityCfg(self.socket.name),
                    **g,
                    pos_tol=c["position_error_m_max"],
                    rot_tol_rad=math.radians(c["orientation_error_deg_max"]),
                    depth_min=c["insertion_depth_ratio_min"] * self.socket.insertion_depth_m,
                    hold_steps=self.hold_steps,
                    sensor_name=self.CONTACT_SENSOR,
                    lift_min_m=0.02,
                ),
            )
            force_exceed = TerminationTermCfg(
                func=contact_force_exceeded,
                params=dict(sensor_name=self.CONTACT_SENSOR, force_max=c["contact_force_n_max"]),
            )

        return TerminationsCfg()

    def get_events_cfg(self) -> Any:
        @configclass
        class EventsCfg:
            reset_task_state = EventTermCfg(func=reset_task_state, mode="reset")

        cfg = EventsCfg()
        if self.variations:
            from sim.arena.variations import attach_variation_events  # sim-side only

            attach_variation_events(cfg, self.variations, self.connector, self.socket)
        return cfg

    def get_episode_recorder_terms(self) -> dict[str, EpisodeRecorderTermCfg]:
        """Passed to IsaacLabArenaEnvironment(episode_recorder_terms=...) by the environment factory."""
        return {
            "insertion": EpisodeRecorderTermCfg(
                func=record_insertion,
                params=dict(connector_name=self.connector.name, socket_name=self.socket.name, **self.geom,
                            insertion_depth_m=self.socket.insertion_depth_m),
            )
        }

    def get_mimic_env_cfg(self, arm_mode) -> Any:
        return None  # no mimic data generation for this task

    def get_metrics(self) -> list[MetricBase]:
        return []  # aggregate report not needed; per-episode keys go through the recorder above
