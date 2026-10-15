"""ACT policy server for Isaac Lab-Arena closed-loop evaluation.

Serves a LeRobot ACT checkpoint *directory* over ZeroMQ REQ/REP.
Verified against LeRobot v0.6.x:
  - ``ACTPolicy.from_pretrained(dir)``; normalization lives in
    ``make_pre_post_processors(policy.config, pretrained_path=dir)``.
  - ``ACTPolicy.predict_action_chunk(batch) -> (B, chunk_size, action_dim)``.
Batched over Arena envs (B = num_envs).

Run on the DGX Spark (openarm-policy conda env, from the repo root):
    PYTHONPATH=. python sim/policy/act_policy_server.py --config sim/arena/policy_config_act.yaml

Wire format (pickle, loopback only):
    {"cmd": "ping"}                                     -> {"ok": true}
    {"cmd": "reset", "env_ids": [..] | null}            -> {"ok": true}
    {"cmd": "act", "obs": {cam: (B,H,W,3) uint8, "joint_pos": (B,dof) f32}}
        -> {"ok": true, "action": (B, chunk_size, dof) f32, "latency_ms": float}

ACT keeps one chunk per forward pass with no cross-step state when we call
``predict_action_chunk`` directly, so per-env reset is a no-op here; the
Arena-side ActionChunkScheduler owns per-env chunk replay.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


class ActPolicy:
    def __init__(self, cfg: dict):
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.torch = torch
        self.mapping = cfg["observation_mapping"]
        from sim.robot.embodiment_profile import load_profile

        self.profile = load_profile(cfg["embodiment_profile"])
        self.jpeg_roundtrip = bool(cfg["policy"].get("jpeg_roundtrip", True))
        ckpt = Path(cfg["policy"]["checkpoint_dir"])
        if not (ckpt / "config.json").exists():
            raise FileNotFoundError(f"{ckpt} is not a LeRobot pretrained_model dir (no config.json)")
        self.model = ACTPolicy.from_pretrained(str(ckpt)).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.pre, self.post = make_pre_post_processors(self.model.config, pretrained_path=str(ckpt))
        self.chunk_size = int(self.model.config.chunk_size)
        want = int(cfg["policy"]["chunk_size"])
        if want != self.chunk_size:
            raise ValueError(f"policy_config chunk_size={want} but checkpoint chunk_size={self.chunk_size}")
        act_ft = self.model.config.action_feature  # PolicyFeature from output_features
        if act_ft is None:
            raise ValueError("checkpoint config has no action output feature")
        act_dim = int(act_ft.shape[0])
        if act_dim != self.profile.dof:
            raise ValueError(
                f"checkpoint action dim {act_dim} != profile {self.profile.name} dof {self.profile.dof} "
                "(end effector changed? retrain ACT on data recorded with this profile)"
            )

    def _jpeg(self, img: np.ndarray) -> np.ndarray:
        """Mimic the real camera pipeline (MJPG) so sim frames carry the same artifacts."""
        import cv2

        out = np.empty_like(img)
        for i in range(img.shape[0]):
            ok, buf = cv2.imencode(".jpg", img[i, :, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
            assert ok
            out[i] = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
        return out

    def _to_batch(self, obs: dict) -> dict:
        torch = self.torch
        batch = {}
        for cam in self.mapping["cameras"]:
            img = np.asarray(obs[cam["sim"]])  # (B,H,W,3) uint8
            h, w = cam["resolution"]
            if img.ndim != 4 or img.shape[1:3] != (h, w):
                raise ValueError(f"{cam['sim']}: expected (B,{h},{w},3), got {img.shape}")
            if self.jpeg_roundtrip:
                img = self._jpeg(img)
            batch[cam["real"]] = torch.from_numpy(img).permute(0, 3, 1, 2).float() / 255.0
        state = np.asarray(obs["joint_pos"], dtype=np.float32)
        if state.shape[1] != self.profile.dof:
            raise ValueError(f"joint_pos dof {state.shape[1]} != profile {self.profile.name} dof {self.profile.dof}")
        batch[self.mapping["state"]["real"]] = torch.from_numpy(state)
        return {k: v.to(self.device) for k, v in batch.items()}

    def act_chunk(self, obs: dict) -> np.ndarray:
        with self.torch.no_grad():
            batch = self.pre(self._to_batch(obs))
            chunk = self.model.predict_action_chunk(batch)  # (B, chunk, dof) normalized
            chunk = self.post(chunk)  # unnormalize
        return chunk.detach().cpu().numpy().astype(np.float32)

    def reset(self, env_ids) -> None:
        self.model.reset()  # clears queue/ensembler; harmless for predict_action_chunk


def serve(cfg: dict) -> None:
    import zmq

    srv = cfg["policy"]["server"]
    policy = ActPolicy(cfg)
    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://{srv['host']}:{srv['port']}")
    print(f"[act-server] ready on {srv['host']}:{srv['port']} chunk={policy.chunk_size}", flush=True)
    while True:
        t0 = time.perf_counter()
        try:
            req = pickle.loads(sock.recv())
            cmd = req.get("cmd")
            if cmd == "ping":
                resp = {"ok": True}
            elif cmd == "reset":
                policy.reset(req.get("env_ids"))
                resp = {"ok": True}
            elif cmd == "act":
                resp = {"ok": True, "action": policy.act_chunk(req["obs"])}
            else:
                resp = {"ok": False, "err": f"unknown cmd {cmd!r}"}
        except Exception as e:  # keep serving; client surfaces the error
            resp = {"ok": False, "err": repr(e)}
        resp["latency_ms"] = (time.perf_counter() - t0) * 1e3
        sock.send(pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    serve(load_config(p.parse_args(argv).config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
