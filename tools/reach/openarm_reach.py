"""OpenArm v1.0 reachability for the bimanual cable pick -> handover -> place demo.

Pure numpy (no ROS, no Isaac). Reads the OpenArm v1.0 URDF, solves damped-least-squares
IK for full 6-D tool poses, and reports, in the body_link0 frame (x forward, y robot-left,
z up):

  * right-arm top-down pick map over the drop zone (fraction of tool yaw angles reachable
    with >= MARGIN_DEG joint-limit margin),
  * left-arm top-down place map over the box zone,
  * the handover pose (both grippers horizontal, facing each other) and its joint margins.

Two joint-limit sets are evaluated:
  urdf    -- limits in openarm_description v1.urdf (generous, simulation)
  lerobot -- LeRobot 0.6.2 RIGHT/LEFT_DEFAULT_JOINTS_LIMITS (what the real follower clamps to)

ASSUMPTION (unverified, see docs/plan-bimanual-handover-demo.md §blind spots): LeRobot joint
angles use the same zero and sign as the URDF joints. Check on the real robot by commanding
+10..30 deg per joint and comparing against this FK before trusting the lerobot maps.

Usage:
  python tools/reach/openarm_reach.py --urdf <openarm_description>/assets/robot/openarm_v1.0/urdf/example/v1.urdf \
      --workbench sim/workbench/workbench.yaml --out outputs/reach
Writes reach.json and *.svg under --out. Self-collision and arm-arm collision are NOT checked.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

LEROBOT_LIMITS_DEG = {  # lerobot/robots/openarm_follower/config_openarm_follower.py (v0.6.2)
    "right": [(-75, 75), (-9, 90), (-85, 85), (0, 135), (-85, 85), (-40, 40), (-80, 80)],
    "left": [(-75, 75), (-90, 9), (-85, 85), (0, 135), (-85, 85), (-40, 40), (-80, 80)],
}
MARGIN_DEG = 10.0


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = map(lambda f: f, (math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axang(ax: np.ndarray, t: float) -> np.ndarray:
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * k @ k


class Arm:
    def __init__(self, urdf: Path, side: str, limits: str):
        root = ET.parse(urdf).getroot()
        by_child = {j.find("child").get("link"): j for j in root.iter("joint")}
        chain, link = [], f"openarm_{side}_hand_tcp"
        while link in by_child:
            chain.append(by_child[link])
            link = by_child[link].find("parent").get("link")
        chain.reverse()
        self.side = side
        self.origins, self.axes = [], []
        for j in chain:
            o = j.find("origin")
            m = np.eye(4)
            m[:3, :3] = _rpy(*map(float, o.get("rpy").split()))
            m[:3, 3] = list(map(float, o.get("xyz").split()))
            self.origins.append(m)
            self.axes.append(np.array(list(map(float, j.find("axis").get("xyz").split())))
                             if j.get("type") == "revolute" else None)
        urdf_lim = [(float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
                    for j in chain if j.get("type") == "revolute"]
        if limits == "urdf":
            self.lim = np.array(urdf_lim)
        else:
            self.lim = np.radians(np.array(LEROBOT_LIMITS_DEG[side], dtype=float))
        assert self.lim.shape == (7, 2)

    def fk(self, q: np.ndarray):
        m, k, pts, zs = np.eye(4), 0, [], []
        for o, ax in zip(self.origins, self.axes):
            m = m @ o
            if ax is not None:
                pts.append(m[:3, 3].copy())
                zs.append(m[:3, :3] @ ax)
                a = np.eye(4)
                a[:3, :3] = _axang(ax, q[k])
                m = m @ a
                k += 1
        return m, np.array(pts), np.array(zs)

    def ik(self, pos: np.ndarray, rot: np.ndarray, seeds: int = 8, iters: int = 120, rng=None):
        """6-D IK. Returns (ok, q, min_margin_deg). ok = pos err < 3 mm and rot err < 3 deg."""
        rng = rng or np.random.default_rng(0)
        best = (False, None, -1e9)
        for s in range(seeds):
            q = self.lim.mean(1) if s == 0 else rng.uniform(self.lim[:, 0], self.lim[:, 1])
            for _ in range(iters):
                m, p, z = self.fk(q)
                ep = pos - m[:3, 3]
                er = 0.5 * sum(np.cross(m[:3, i], rot[:, i]) for i in range(3))
                if np.linalg.norm(ep) < 1e-3 and np.linalg.norm(er) < 1e-3:
                    break
                jp = np.cross(z, m[:3, 3] - p).T
                jr = z.T
                jm = np.vstack([jp, 0.3 * jr])
                err = np.concatenate([ep, 0.3 * er])
                dq = jm.T @ np.linalg.solve(jm @ jm.T + 1e-4 * np.eye(6), err)
                q = np.clip(q + dq, self.lim[:, 0], self.lim[:, 1])
            m, _, _ = self.fk(q)
            pe = np.linalg.norm(pos - m[:3, 3])
            re = math.degrees(math.acos(np.clip((np.trace(m[:3, :3].T @ rot) - 1) / 2, -1, 1)))
            margin = float(np.degrees(np.min(np.minimum(q - self.lim[:, 0], self.lim[:, 1] - q))))
            ok = pe < 3e-3 and re < 3.0
            if ok and margin > best[2]:
                best = (True, q, margin)
            if best[0] and best[2] >= MARGIN_DEG:
                break
        return best


def topdown_rot(yaw: float) -> np.ndarray:
    """Tool z straight down, tool x at `yaw` in the table plane (gripper closing axis = tool y)."""
    x = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])


def horizontal_rot(approach_y: float) -> np.ndarray:
    """Tool z horizontal along +-y (approach), fingers closing vertically (tool y = world z)."""
    z = np.array([0.0, approach_y, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    return np.column_stack([np.cross(y, z), y, z])


def yaw_map(arm: Arm, xs, ys, z, yaws):
    frac = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            good = 0
            for yaw in yaws:
                ok, _, margin = arm.ik(np.array([x, y, z]), topdown_rot(yaw), seeds=4, iters=80)
                good += ok and margin >= MARGIN_DEG
            frac[i, j] = good / len(yaws)
    return frac


def svg_map(path: Path, title: str, xs, ys, frac, rects):
    """Top view: robot at bottom center, +x up the page, +y to the LEFT (robot's left)."""
    cell, pad = 22, 70
    w = pad * 2 + cell * len(ys)
    h = pad * 2 + cell * len(xs) + 40
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="11">',
           f'<rect width="{w}" height="{h}" fill="#ffffff"/>', f'<text x="{pad}" y="24" font-size="14" font-weight="bold">{title}</text>']

    def px(x, y):  # world -> svg (y to the left)
        col = (ys[-1] - y) / (ys[1] - ys[0])
        row = (xs[-1] - x) / (xs[1] - xs[0])
        return pad + col * cell, pad + 10 + row * cell

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            cx, cy = px(x, y)
            f = frac[i, j]
            shade = int(245 - 175 * f)
            col = f"rgb({shade},{min(255, shade + 30)},{shade})" if f > 0 else "#f3f3f3"
            out.append(f'<rect x="{cx - cell / 2:.1f}" y="{cy - cell / 2:.1f}" width="{cell - 1}" height="{cell - 1}" fill="{col}"/>')
    for name, (x0, x1, y0, y1), color in rects:
        ax, ay = px(x1, y1)
        bx, by = px(x0, y0)
        out.append(f'<rect x="{min(ax, bx):.1f}" y="{min(ay, by):.1f}" width="{abs(bx - ax):.1f}" height="{abs(by - ay):.1f}" '
                   f'fill="none" stroke="{color}" stroke-width="2"/>')
        out.append(f'<text x="{min(ax, bx) + 3:.1f}" y="{min(ay, by) - 4:.1f}" fill="{color}">{name}</text>')
    bx, by = px(0, 0)
    out.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="6" fill="#333"/><text x="{bx + 8:.1f}" y="{by + 4:.1f}">body (0,0)</text>')
    for j, y in enumerate(ys):
        if j % 4 == 0:
            cx, _ = px(xs[0], y)
            out.append(f'<text x="{cx - 10:.1f}" y="{pad + 10 + cell * len(xs) + 6:.1f}">{y:+.2f}</text>')
    for i, x in enumerate(xs):
        if i % 2 == 0:
            _, cy = px(x, ys[-1])
            out.append(f'<text x="{pad - 40}" y="{cy + 4:.1f}">x={x:.2f}</text>')
    out.append(f'<text x="{pad}" y="{h - 12}">y (m, robot left = page left) | shade = fraction of tool yaw reachable with ≥{MARGIN_DEG:.0f}° joint margin</text>')
    out.append("</svg>")
    path.write_text("\n".join(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--workbench", type=Path, default=Path("sim/workbench/workbench.yaml"))
    ap.add_argument("--out", type=Path, default=Path("outputs/reach"))
    ap.add_argument("--limits", choices=["lerobot", "urdf", "both"], default="both")
    ap.add_argument("--step", type=float, default=0.04)
    a = ap.parse_args()
    wb = yaml.safe_load(a.workbench.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    top = wb["table"]["top_z"]
    grasp_z = top + wb["cable"]["grasp_tcp_above_table"]
    place_z = top + wb["box"]["release_tcp_above_table"]
    dz, bz, ho = wb["drop_zone"], wb["box"], wb["handover"]
    xs = np.round(np.arange(0.06, 0.50 + 1e-9, a.step), 3)
    ys = np.round(np.arange(-0.62, 0.62 + 1e-9, a.step), 3)
    yaws = np.radians(np.arange(-90, 90, 30))
    report = {"workbench": str(a.workbench), "grasp_tcp_z": grasp_z, "place_tcp_z": place_z, "margin_deg": MARGIN_DEG}
    for lim in (["lerobot", "urdf"] if a.limits == "both" else [a.limits]):
        right, left = Arm(a.urdf, "right", lim), Arm(a.urdf, "left", lim)
        fr = yaw_map(right, xs, ys[ys <= 0.1], grasp_z, yaws)
        fl = yaw_map(left, xs, ys[ys >= -0.1], place_z, yaws)
        ysr, ysl = ys[ys <= 0.1], ys[ys >= -0.1]

        def zone_stats(frac, ysub, zone):
            m = np.array([[zone["x"][0] <= x <= zone["x"][1] and zone["y"][0] <= y <= zone["y"][1] for y in ysub] for x in xs])
            v = frac[m]
            return {"cells": int(m.sum()), "cells_any_yaw": int((v > 0).sum()), "mean_yaw_fraction": float(v.mean()) if v.size else 0.0,
                    "min_yaw_fraction": float(v.min()) if v.size else 0.0}

        rep = {"drop_zone_right": zone_stats(fr, ysr, dz), "box_zone_left": zone_stats(fl, ysl, bz)}
        c = np.array(ho["center"])
        gap = ho["tcp_gap"] / 2
        okr, qr, mr = right.ik(c + [0, -gap, 0], horizontal_rot(+1.0), seeds=16)
        okl, ql, ml = left.ik(c + [0, +gap, 0], horizontal_rot(-1.0), seeds=16)
        rep["handover"] = {"center": ho["center"], "right_ok": bool(okr), "left_ok": bool(okl),
                           "right_margin_deg": round(mr, 1), "left_margin_deg": round(ml, 1),
                           "right_q_deg": None if qr is None else np.degrees(qr).round(1).tolist(),
                           "left_q_deg": None if ql is None else np.degrees(ql).round(1).tolist()}
        sweep = []
        for hx in (0.22, 0.26, 0.30, 0.34):
            for hz in (top + 0.10, top + 0.15, top + 0.20, top + 0.25, top + 0.30):
                o1, _, m1 = right.ik(np.array([hx, -gap, hz]), horizontal_rot(+1.0), seeds=8)
                o2, _, m2 = left.ik(np.array([hx, gap, hz]), horizontal_rot(-1.0), seeds=8)
                sweep.append({"x": hx, "z": round(hz, 3), "z_above_table": round(hz - top, 3), "ok": bool(o1 and o2),
                              "min_margin_deg": round(min(m1, m2), 1) if (o1 and o2) else None})
        rep["handover_sweep"] = sweep
        report[lim] = rep
        rects = [("drop zone", (dz["x"][0], dz["x"][1], dz["y"][0], dz["y"][1]), "#c0392b"),
                 ("box", (bz["x"][0], bz["x"][1], bz["y"][0], bz["y"][1]), "#1f6fb2")]
        svg_map(a.out / f"pick_right_{lim}.svg", f"Right arm top-down pick, TCP z={grasp_z:.3f} m ({lim} limits)", xs, ysr, fr, rects)
        svg_map(a.out / f"place_left_{lim}.svg", f"Left arm top-down place, TCP z={place_z:.3f} m ({lim} limits)", xs, ysl, fl, rects)
    (a.out / "reach.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k in ("lerobot", "urdf")}, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
