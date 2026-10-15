"""Arena-side ACT remote closed-loop policy.

Verified against Isaac Lab-Arena release/0.3.0:
  - ``isaaclab_arena.policy.policy_base.PolicyBase[PolicyCfgT]`` with
    ``get_action(env, observation) -> torch.Tensor`` and ``reset(env_ids)``.
  - ``@register_policy``; the cfg dataclass fields become CLI flags of
    ``policy_runner.py`` (``--policy_config_yaml_path``, ``--remote_host`` ...).
  - ``ActionChunkScheduler``: ``action_horizon`` = length of the chunk the fetch
    fn returns (buffer size); ``action_chunk_length`` = steps executed before a
    refetch (must be <= action_horizon). Fetch fn returns (num_envs, action_horizon, dof).
  - observation layout: ``observation["camera_obs"][cam]`` (N,H,W,C) and
    ``observation["policy"]["robot_joint_pos"]`` (N, dof).

Template: isaaclab_arena_gr00t/policy/gr00t_remote_closedloop_policy.py.
Transport: ZeroMQ REQ/REP, pickle payloads, loopback only (see sandbox policy).
"""

from __future__ import annotations

import atexit
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.action_scheduling import ActionChunkScheduler
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


@dataclass
class ActRemoteClosedloopPolicyCfg(PolicyCfg):
    """CLI-exposed config (each field -> ``--<field>`` on policy_runner)."""

    policy_config_yaml_path: str = "sim/arena/policy_config_act.yaml"
    remote_host: str = "127.0.0.1"
    remote_port: int = 5556
    num_envs: int = 1
    policy_device: str = "cuda"


@register_policy
class ActRemoteClosedloopPolicy(PolicyBase[ActRemoteClosedloopPolicyCfg]):
    name = "act_remote_closedloop"

    def __init__(self, config: ActRemoteClosedloopPolicyCfg):
        super().__init__(config)
        import zmq  # local import: keep module importable for static checks

        self._zmq = zmq
        yml = yaml.safe_load(Path(config.policy_config_yaml_path).read_text())
        self.mapping = yml["observation_mapping"]
        self.cameras: list[str] = [c["sim"] for c in self.mapping["cameras"]]
        from sim.robot.embodiment_profile import load_profile

        self.profile = load_profile(yml["embodiment_profile"])
        self.action_dim = self.profile.dof
        srv = yml["policy"]["server"]
        self.timeout_ms = int(srv.get("timeout_ms", 2000))
        self.chunk_len = int(yml["policy"]["chunk_size"])
        self.horizon = int(yml["policy"]["action_horizon"])
        assert 1 <= self.horizon <= self.chunk_len, "action_horizon must be in [1, chunk_size]"

        self.num_envs = config.num_envs
        self.device = config.policy_device
        self._scheduler = ActionChunkScheduler(
            num_envs=self.num_envs,
            action_chunk_length=self.horizon,   # steps executed per fetch
            action_horizon=self.chunk_len,      # chunk length the server returns
            action_dim=self.action_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self._sock = None
        self._connect()
        self._request({"cmd": "ping"})
        atexit.register(self.close)  # policy_runner never calls policy.close()
        self.task_description: str | None = None

    # ---------------- transport ----------------
    def _connect(self) -> None:
        zmq = self._zmq
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = zmq.Context.instance().socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://{self.config.remote_host}:{self.config.remote_port}")

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """REQ/REP with recovery: a timeout leaves REQ in a bad state, so reconnect."""
        try:
            self._sock.send(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
            resp = pickle.loads(self._sock.recv())
        except self._zmq.Again as e:
            self._connect()
            raise RuntimeError(
                f"ACT server timeout ({self.timeout_ms} ms) at "
                f"{self.config.remote_host}:{self.config.remote_port}"
            ) from e
        if not resp.get("ok", False):
            raise RuntimeError(f"ACT server error: {resp.get('err')}")
        return resp

    # ---------------- PolicyBase ----------------
    def set_task_description(self, task_description: str | None) -> str:
        self.task_description = task_description or ""
        return self.task_description

    def get_action(self, env: Any, observation: dict[str, Any]) -> torch.Tensor:
        def fetch_chunk() -> torch.Tensor:
            return self._fetch_chunk(observation)

        return self._scheduler.get_action(fetch_chunk, hold_action=None)

    def _fetch_chunk(self, observation: dict[str, Any]) -> torch.Tensor:
        cams = observation["camera_obs"]
        obs_np = {
            cam: cams[cam].detach().to("cpu", torch.uint8).numpy() for cam in self.cameras
        }
        joint_pos = observation["policy"]["robot_joint_pos"]
        obs_np["joint_pos"] = joint_pos.detach().to("cpu", torch.float32).numpy()
        resp = self._request({"cmd": "act", "obs": obs_np})
        chunk = torch.as_tensor(np.asarray(resp["action"]), dtype=torch.float32, device=self.device)
        assert chunk.shape == (self.num_envs, self.chunk_len, self.action_dim), chunk.shape
        return chunk

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = None if env_ids is None else env_ids.detach().to("cpu").tolist()
        self._request({"cmd": "reset", "env_ids": ids})
        self._scheduler.reset(env_ids)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
