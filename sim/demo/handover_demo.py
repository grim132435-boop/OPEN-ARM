#!/usr/bin/env python3
"""OpenArm v1.0 bimanual cable handover - scripted motion-planning demo (Isaac Lab 3.0.0 @ af1bab4, PhysX).

    Scripted motion-planning demo - not a learned policy; cable pose from simulator ground truth; rigid cable proxy.

Sequence per episode (one environment):
    READY -> R_APPROACH -> R_DESCEND -> R_CLOSE -> R_LIFT -> R_TO_HANDOVER (via waypoint) -> L_APPROACH -> L_CLOSE
    -> HOLD (0.5 s) -> R_OPEN -> R_RETREAT -> L_TO_BOX -> L_OPEN -> L_RETREAT -> DONE

How the motion is produced
    * At start-up the script reads each arm's geometric Jacobian and hand pose from PhysX once and builds an exact
      product-of-exponentials (PoE) model of the 7-DoF chain (no URDF needed, USD kinematics are used as-is).
    * Each waypoint is solved with a multi-seed damped-least-squares IK on that model under the real-robot LeRobot
      joint limits (which are also written into the simulator). Large re-orientations are planned in joint space
      (a straight line between two in-limit configurations stays in limits); short moves are Cartesian lines.
    * Execution: one isaaclab DifferentialIKController (pose, dls) per arm tracks the planned TCP pose every physics
      step; a null-space term pulls the redundant DoF toward the planned joint path.

Geometry decisions (all checked offline against openarm_description v1.urdf + LeRobot limits, see README):
    * Cable proxy: connector end faces the robot, body points away (so the 0.30 m body stays on the table).
    * Right pick: top-down on the connector, tool x along the cable. Grasp pitch must stay ~0 (a pitched grasp needs
      a hand yaw at the handover that the LeRobot limits do not allow); roll about the cable (+-15/30 deg, attach
      mode only) widens coverage. Pre-grasp/lift heights fall back from 0.08/0.12 m when out of reach.
    * Handover: right fingers close upward, left downward (workbench.yaml handover.fingers). The cable lies along x;
      right TCP and the left grasp point (100 mm along the cable = tcp_gap) are centred on handover.center.
    * Release: a rigid 0.30 m stick cannot lie flat in the 0.14 x 0.16 m box, so the left hand (gripper pointing
      forward, workbench release_orientation) holds it connector-down at ~60 deg near the near box corner and lets
      it topple along the box diagonal.
    * Drops whose pick->handover chain is IK-infeasible are re-drawn (count logged; --no_resample to disable).

Usage (on the Spark; Isaac Sim 6.0.1 + Isaac Lab af1bab4):
    export OPENARM_ASSETS=/path/to/openarm_isaac_lab            # enactic/openarm_isaac_lab checkout
    ./isaaclab.sh -p /path/to/OPEN-ARM/sim/demo/handover_demo.py --headless --dry_run_kinematics
    ./isaaclab.sh -p /path/to/OPEN-ARM/sim/demo/handover_demo.py --headless --enable_cameras --episodes 5
See sim/demo/README.md (Korean) for flags, what is faked, known limits and stop conditions.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import struct
import sys
import time
import traceback
import zlib

import numpy as np

CAPTION = "Scripted motion-planning demo — not a learned policy; cable pose from simulator ground truth; rigid cable proxy"

# ================================ PLANNER BEGIN (pure numpy; no Isaac imports) ================================
# Everything between the BEGIN/END markers only needs numpy, so it can be unit-tested without Isaac Sim.

DEG = math.pi / 180.0
# LeRobot 0.6.2 openarm_follower RIGHT/LEFT_DEFAULT_JOINTS_LIMITS (deg); same values as tools/reach/openarm_reach.py.
LEROBOT_LIMITS_DEG = {
    "right": [(-75, 75), (-9, 90), (-85, 85), (0, 135), (-85, 85), (-40, 40), (-80, 80)],
    "left": [(-75, 75), (-90, 9), (-85, 85), (0, 135), (-85, 85), (-40, 40), (-80, 80)],
}
# Joint-space mirror between the URDF arms (used only to seed the left-arm IK).
MIRROR_SIGN = np.array([-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
# URDF openarm_{side}_hand_tcp_joint: TCP = hand frame + 0.0835 m along hand z.
TCP_OFFSET = np.array([0.0, 0.0, 0.0835])
# URDF finger joints sit at y = +-0.005 m on link7 (finger-link ORIGIN gap = 2 * (0.005 + q)).
FINGER_ORIGIN_HALF_GAP = 0.005
# openarm_description v1.0 parallel_link/meshes/collision/finger.stl (scale 1e-3, y mirrored, offset (0, 0.05, -0.661)):
# near the fingertip / TCP height (finger z 0.075..0.095 m) the inner pad face is at +0.00776 m in the finger frame,
# i.e. it reaches 0.00276 m PAST the centre line at q = 0. Pad-to-pad gap at the TCP ~ 2 * (0.005 + q - 0.00776)
# = 2 * (q - 0.00276): the pads already overlap at q = 0 (fully closed ~ q = 0.0028), so q = 0 squeezes anything.
# Checked at start-up against the finger-body origins in the USD (printed); the pad offset itself is from the STL.
FINGER_PAD_INNER = 0.00776


def finger_pad_gap(q, origin_half_gap=FINGER_ORIGIN_HALF_GAP):
    """Approximate pad-to-pad gap [m] at TCP height for finger joint position q (negative = pads overlap)."""
    return 2.0 * (origin_half_gap + q - FINGER_PAD_INNER)


def finger_q_for_gap(gap, origin_half_gap=FINGER_ORIGIN_HALF_GAP):
    """Finger joint target giving a pad gap `gap` (clamped to the joint range 0..FINGER_OPEN)."""
    return float(min(FINGER_OPEN, max(0.0, gap / 2.0 - origin_half_gap + FINGER_PAD_INNER)))


FINGER_OPEN = 0.044
READY_SEED = 12345  # READY configuration sampling seed (fixed, independent of --seed)
FINGER_PRESHAPE = 0.012  # used when the grasp is rolled, so the lower finger clears the table
# Torso column of openarm body_link0 (collision STL bounds, openarm_description v1.0): x/y/z ranges [m].
TORSO_BOXES = [((-0.03, 0.03), (-0.03, 0.03), (0.0, 0.60)), ((-0.085, 0.065), (-0.08, 0.08), (0.60, 0.773))]


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def exp_so3(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = skew(np.asarray(w) / th)
    return np.eye(3) + math.sin(th) * k + (1.0 - math.cos(th)) * (k @ k)


def log_so3(R):
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if math.pi - th < 1e-3:  # near pi: axis from the symmetric part
        B = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(B)))
        ax = B[:, i] / math.sqrt(max(B[i, i], 1e-12))
        return ax / np.linalg.norm(ax) * th
    return th / (2.0 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_T(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def inv_T(T):
    R, p = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ p)


def apply_T(T, p):
    return T[:3, :3] @ np.asarray(p, dtype=float) + T[:3, 3]


def _unit(v):
    n = float(np.linalg.norm(v))
    return np.asarray(v, dtype=float) / n if n > 1e-12 else np.asarray(v, dtype=float)


def _any_perp(v):
    a = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return _unit(np.cross(v, a))


def frame_xz(x, z_hint):
    """Right-handed frame with tool x exactly along `x` and tool z as close as possible to `z_hint`."""
    x = _unit(x)
    z = np.asarray(z_hint, dtype=float) - np.dot(z_hint, x) * x
    z = _any_perp(x) if np.linalg.norm(z) < 1e-6 else _unit(z)
    return np.column_stack([x, np.cross(z, x), z])


def frame_zy(z, y_hint):
    """Right-handed frame with tool z exactly along `z` and tool y as close as possible to `y_hint`."""
    z = _unit(z)
    y = np.asarray(y_hint, dtype=float) - np.dot(y_hint, z) * z
    y = _any_perp(z) if np.linalg.norm(y) < 1e-6 else _unit(y)
    return np.column_stack([np.cross(y, z), y, z])


def topdown_frame(yaw):
    """Tool z straight down, tool x at `yaw` in the table plane, closing axis = tool y (as tools/reach)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.column_stack([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def interp_T(Ta, Tb, s):
    R = Ta[:3, :3] @ exp_so3(s * log_so3(Ta[:3, :3].T @ Tb[:3, :3]))
    return make_T(R, (1.0 - s) * Ta[:3, 3] + s * Tb[:3, 3])


def pose_error(Ta, Tb):
    """(position error [m], rotation error [deg]) between two poses."""
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), float(np.linalg.norm(log_so3(Ta[:3, :3].T @ Tb[:3, :3])) / DEG)


def mirror_T(T):
    """Mirror a pose through the robot's x-z plane (left <-> right) keeping a right-handed tool frame."""
    S = np.diag([1.0, -1.0, 1.0])
    return make_T(S @ T[:3, :3] @ S, S @ T[:3, 3])


def quat_xyzw_from_R(R):
    tr = np.trace(R)
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def R_from_quat_xyzw(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class IKResult:
    def __init__(self, ok, q, pos_err, rot_err_deg, margin_deg):
        self.ok, self.q, self.pos_err, self.rot_err_deg, self.margin_deg = ok, q, pos_err, rot_err_deg, margin_deg

    def summary(self):
        return {"ok": bool(self.ok), "pos_err_mm": round(self.pos_err * 1e3, 2), "rot_err_deg": round(self.rot_err_deg, 2),
                "margin_deg": round(self.margin_deg, 1)}


class PoEArm:
    """Product-of-exponentials model of one 7-DoF revolute arm, built from the simulator's own Jacobian.

    For a revolute joint i with unit axis w_i through point p_i, the linear Jacobian column at the hand origin is
    J_lin,i = w_i x (p_hand - p_i); therefore J_lin,i x w_i is the component of (p_hand - p_i) perpendicular to the
    axis and p_i = p_hand - J_lin,i x w_i lies on the axis. FK(q) = prod_i exp([w_i, p_i](q_i - q0_i)) * M_tcp.
    This is exact for a serial revolute chain, so the model reproduces whatever kinematics the USD contains.
    """

    def __init__(self, side, axes, points, M_tcp, q0, lim):
        self.side = side
        self.w = np.asarray(axes, dtype=float)
        self.p = np.asarray(points, dtype=float)
        self.M = np.asarray(M_tcp, dtype=float)
        self.q0 = np.asarray(q0, dtype=float)
        self.lim = np.asarray(lim, dtype=float)
        self.tcp_offset = TCP_OFFSET.copy()

    @classmethod
    def from_jacobian(cls, side, J_hand, T_hand, q0, lim, tcp_offset=TCP_OFFSET):
        J = np.asarray(J_hand, dtype=float)
        w = J[3:6, :].T.copy()
        norms = np.linalg.norm(w, axis=1)
        w = w / norms[:, None]
        v = J[0:3, :].T
        p_hand = T_hand[:3, 3]
        pts = np.array([p_hand - np.cross(v[i], w[i]) for i in range(w.shape[0])])
        arm = cls(side, w, pts, T_hand @ make_T(None, tcp_offset), q0, lim)
        arm.tcp_offset = np.asarray(tcp_offset, dtype=float)
        return arm, norms

    def _chain(self, q):
        T = np.eye(4)
        Ts = []
        dq = np.asarray(q, dtype=float) - self.q0
        for i in range(7):
            Ts.append(T)
            R = exp_so3(self.w[i] * dq[i])
            T = T @ make_T(R, (np.eye(3) - R) @ self.p[i])
        return Ts, T

    def fk(self, q):
        return self._chain(q)[1] @ self.M

    def hand_pose(self, q):
        return self.fk(q) @ make_T(None, -self.tcp_offset)

    def jac(self, q):
        Ts, Tn = self._chain(q)
        T = Tn @ self.M
        p = T[:3, 3]
        J = np.zeros((6, 7))
        for i in range(7):
            wi = Ts[i][:3, :3] @ self.w[i]
            pi = apply_T(Ts[i], self.p[i])
            J[:3, i] = np.cross(wi, p - pi)
            J[3:, i] = wi
        return J, T

    def margin_deg(self, q):
        q = np.asarray(q)
        return float(np.min(np.minimum(q - self.lim[:, 0], self.lim[:, 1] - q)) / DEG)

    def clip(self, q):
        return np.clip(q, self.lim[:, 0], self.lim[:, 1])

    def ik(self, T_goal, q_init, iters=150, lam=0.02, w_rot=0.6, max_step=0.2, pos_tol=1.5e-3, rot_tol_deg=1.5, center=0.1):
        """Damped least squares with joint clipping; the redundant DoF is used to move away from joint limits
        (null-space gradient toward mid-range, weight `center`)."""
        q = self.clip(np.asarray(q_init, dtype=float).copy())

        def step(q, g=None):
            J, T = self.jac(q)
            ep = T_goal[:3, 3] - T[:3, 3]
            er = log_so3(T_goal[:3, :3] @ T[:3, :3].T)
            Jw = np.vstack([J[:3], w_rot * J[3:]])
            Jp = Jw.T @ np.linalg.inv(Jw @ Jw.T + lam * lam * np.eye(6))
            dq = Jp @ np.concatenate([ep, w_rot * er])
            if g is not None:
                dq = dq + (np.eye(7) - Jp @ Jw) @ g
            m = float(np.max(np.abs(dq)))
            if m > max_step:
                dq *= max_step / m
            return dq, float(np.linalg.norm(ep)), float(np.linalg.norm(er))

        for _ in range(iters):
            dq, pe, re = step(q)
            if pe < 1e-4 and re < 1e-3:
                break
            q = self.clip(q + dq)
        pe, re = pose_error(self.fk(q), T_goal)
        ok = pe < pos_tol and re < rot_tol_deg
        if ok and center > 0.0:
            # refinement: use the redundant DoF to move away from joint limits while staying on the target
            mid, half = self.lim.mean(axis=1), 0.5 * (self.lim[:, 1] - self.lim[:, 0])
            m0 = self.margin_deg(q)
            for _ in range(40):
                dq, _, _ = step(q, center * (mid - q) / half)
                q2 = self.clip(q + dq)
                pe2, re2 = pose_error(self.fk(q2), T_goal)
                m2 = self.margin_deg(q2)
                if pe2 > 0.5 * pos_tol or re2 > 0.5 * rot_tol_deg or m2 < m0 + 1e-3:
                    break
                q, m0 = q2, m2
            pe, re = pose_error(self.fk(q), T_goal)
        return IKResult(pe < pos_tol and re < rot_tol_deg, q, pe, re, self.margin_deg(q))

    def ik_global(self, T_goal, q_hint, rng, n_random=16, margin_cap=20.0, w_dist=0.03, extra_seeds=()):
        """Multi-seed IK. Among converged solutions prefer joint-limit margin (capped) and closeness to q_hint."""
        mid = self.lim.mean(axis=1)
        seeds = [np.asarray(q_hint, dtype=float), mid] + [np.asarray(s, dtype=float) for s in extra_seeds]
        seeds += [rng.uniform(self.lim[:, 0], self.lim[:, 1]) for _ in range(n_random)]
        best, best_score, best_bad = None, -1e18, None
        for s in seeds:
            r = self.ik(T_goal, s)
            if r.ok:
                dist = float(np.mean(np.abs(r.q - q_hint)) / DEG)
                score = min(r.margin_deg, margin_cap) - w_dist * dist
                if score > best_score:
                    best, best_score = r, score
            elif best_bad is None or (r.pos_err + 1e-3 * r.rot_err_deg) < (best_bad.pos_err + 1e-3 * best_bad.rot_err_deg):
                best_bad = r
        return best if best is not None else best_bad


class Plan:
    """A joint-space reference path for one arm plus bookkeeping."""

    def __init__(self, ok, path, kind, T_goal, info=None):
        self.ok = bool(ok)
        self.path = np.asarray(path, dtype=float)
        self.kind = kind
        self.T_goal = T_goal
        self.info = info or {}

    @property
    def q_end(self):
        return self.path[-1]

    def min_margin(self, arm):
        return min(arm.margin_deg(q) for q in self.path)


def plan_line(arm, q_start, T_goal, step_pos=0.004, step_rot_deg=1.5):
    """Cartesian straight line (lerp + slerp) from FK(q_start) to T_goal, solved incrementally (continuous branch)."""
    T0 = arm.fk(q_start)
    dist, ang = pose_error(T0, T_goal)
    n = max(2, int(math.ceil(dist / step_pos)), int(math.ceil(ang / step_rot_deg)))
    q = np.asarray(q_start, dtype=float).copy()
    path = [q.copy()]
    worst = (0.0, 0.0)
    for k in range(1, n + 1):
        r = arm.ik(interp_T(T0, T_goal, k / n), q, iters=40 if k < n else 150, pos_tol=2e-3, rot_tol_deg=2.0)
        worst = (max(worst[0], r.pos_err), max(worst[1], r.rot_err_deg))
        if not r.ok:
            return Plan(False, path, "line", T_goal, {"fail_at": k / n, "pos_err_mm": r.pos_err * 1e3, "rot_err_deg": r.rot_err_deg,
                                                      "length_m": dist})
        q = r.q
        path.append(q.copy())
    return Plan(True, path, "line", T_goal, {"length_m": dist, "angle_deg": ang, "max_pos_err_mm": worst[0] * 1e3})


def joint_path(q_from, q_to, step_deg=1.5):
    n = max(2, int(math.ceil(float(np.max(np.abs(q_to - q_from))) / (step_deg * DEG))))
    return [q_from + (q_to - q_from) * k / n for k in range(n + 1)]


def plan_joint(arm, q_start, T_goal, rng, n_random=16, q_hint=None):
    """Global IK for T_goal, then a straight joint-space path (stays inside the box-shaped joint limits)."""
    hint = np.asarray(q_start if q_hint is None else q_hint, dtype=float)
    r = arm.ik_global(T_goal, hint, rng, n_random=n_random)
    if r is None or not r.ok:
        return Plan(False, [q_start], "joint", T_goal, {"ik": None if r is None else r.summary()})
    return Plan(True, joint_path(np.asarray(q_start, dtype=float), r.q), "joint", T_goal, {"ik": r.summary()})


def plan_via(arm, q_start, T_via, T_goal, rng, n_random=16):
    """Joint-space move through an intermediate Cartesian waypoint (via IK is hinted with the joint-space midpoint)."""
    g = arm.ik_global(T_goal, q_start, rng, n_random=n_random)
    if g is None or not g.ok:
        return Plan(False, [q_start], "via", T_goal, {"ik_goal": None if g is None else g.summary()})
    hint = 0.5 * (np.asarray(q_start) + g.q)
    v = arm.ik_global(T_via, hint, rng, n_random=n_random // 2, w_dist=0.3, extra_seeds=[hint])
    if v is None or not v.ok:
        return Plan(True, joint_path(np.asarray(q_start, dtype=float), g.q), "joint", T_goal,
                    {"ik_goal": g.summary(), "via": "skipped (via IK failed)"})
    path = joint_path(np.asarray(q_start, dtype=float), v.q) + joint_path(v.q, g.q)[1:]
    return Plan(True, path, "via", T_goal, {"ik_goal": g.summary(), "ik_via": v.summary()})


def plan_joint_clear(arm, q_start, T_goal, rng, z_min, carried_pts_tcp=None, n_random=16, raise_by=0.08):
    """plan_joint, but if the straight joint path brings the TCP/hand/carried points below z_min, go through a
    raised Cartesian via waypoint instead (keeps whichever path stays higher)."""
    p = plan_joint(arm, q_start, T_goal, rng, n_random=n_random)
    if not p.ok:
        return p
    z = path_min_z(arm, p.path, carried_pts_tcp)
    if z >= z_min:
        p.info["min_z"] = round(z, 4)
        return p
    T_via = interp_T(arm.fk(q_start), T_goal, 0.5)
    T_via[2, 3] = max(T_via[2, 3], max(arm.fk(q_start)[2, 3], T_goal[2, 3])) + raise_by
    v = plan_via(arm, q_start, T_via, T_goal, rng, n_random=n_random)
    if v.ok:
        zv = path_min_z(arm, v.path, carried_pts_tcp)
        if zv > z:
            v.info["min_z"] = round(zv, 4)
            return v
    p.info["min_z"] = round(z, 4)
    p.info["warning"] = "path dips below clearance height"
    return p


def concat_plans(*plans):
    path = [plans[0].path[0]]
    for p in plans:
        path += list(p.path[1:])
    return Plan(all(p.ok for p in plans), path, "+".join(p.kind for p in plans), plans[-1].T_goal,
                {"parts": [p.info for p in plans]})


def path_min_z(arm, path, carried_pts_tcp=None, stride=3):
    """Lowest z of the TCP, the hand origin and any carried points (given in the TCP frame) along a joint path."""
    zmin = 1e9
    for q in list(path[::stride]) + [path[-1]]:
        T = arm.fk(q)
        zmin = min(zmin, T[2, 3], apply_T(T, -arm.tcp_offset)[2])
        if carried_pts_tcp is not None:
            for p in carried_pts_tcp:
                zmin = min(zmin, apply_T(T, p)[2])
    return float(zmin)


def torso_clearance(points):
    """Smallest distance [m] from any point to the torso boxes (negative = inside)."""
    best = 1e9
    for p in points:
        for (bx, by, bz) in TORSO_BOXES:
            lo, hi = np.array([bx[0], by[0], bz[0]]), np.array([bx[1], by[1], bz[1]])
            d_out = np.linalg.norm(np.maximum(0.0, np.maximum(lo - p, p - hi)))
            d_in = min(np.min(p - lo), np.min(hi - p))
            best = min(best, d_out if d_out > 0 else -d_in)
    return float(best)


def path_torso_clearance(arm, path, carried_pts_tcp, stride=3):
    pts = []
    for q in list(path[::stride]) + [path[-1]]:
        T = arm.fk(q)
        pts += [apply_T(T, p) for p in carried_pts_tcp]
    return torso_clearance(pts)


class CableGeom:
    """Rigid cable proxy in its body frame: cable axis = local x, connector at the +x end.

    The body direction (connector -> tail) is local -x. The connector box sits flush with the cable bottom.
    """

    def __init__(self, length, diameter, conn_size, left_dist):
        self.L, self.d = float(length), float(diameter)
        self.conn_size = tuple(float(c) for c in conn_size)
        self.conn_center = np.array([self.L / 2 - self.conn_size[0] / 2, 0.0, (self.conn_size[2] - self.d) / 2])
        self.left_dist = float(left_dist)
        self.left_pt = np.array([self.conn_center[0] - self.left_dist, 0.0, 0.0])
        self.tail = np.array([-self.L / 2, 0.0, 0.0])
        self.head = np.array([self.L / 2, 0.0, 0.0])
        self.body_dir = np.array([-1.0, 0.0, 0.0])

    def rest_pose(self, center_xy, body_yaw, top_z):
        """Cable lying on the table; `body_yaw` = direction connector -> tail."""
        return make_T(rot_z(body_yaw + math.pi), [center_xy[0], center_xy[1], top_z + self.d / 2])

    def end_points(self):
        return [self.head, self.tail, self.conn_center, 0.5 * (self.head + self.tail)]


class Params:
    """All task numbers in one place (filled from workbench.yaml + CLI)."""

    def __init__(self, wb, **over):
        t, c, dz, bx, ho = wb["table"], wb["cable"], wb["drop_zone"], wb["box"], wb["handover"]
        self.top_z = float(t["top_z"])
        self.table_x = (float(t["front_edge_x"]), float(t["front_edge_x"]) + float(t["depth"]))
        self.table_w = float(t["width"])
        m = re.search(r"(\d+(?:\.\d+)?)\s*mm", str(c.get("left_grasp", "")))
        left_dist = float(m.group(1)) / 1000.0 if m else float(ho.get("tcp_gap", 0.10))
        self.cable = CableGeom(c["length"], c["outer_diameter"], c["connector"]["size"], left_dist)
        self.cable_mass = float(c.get("linear_mass", 0.03)) * float(c["length"]) + float(c["connector"].get("mass", 0.008))
        self.grasp_above = float(c["grasp_tcp_above_table"])
        self.dz_x, self.dz_y = tuple(dz["x"]), tuple(dz["y"])
        ex = dz.get("exclude_corner") or {}
        self.ex_xmin, self.ex_ymin = ex.get("x_min"), ex.get("y_min")
        self.drop_h = float(dz.get("drop_height_above_table", 0.10))
        self.settle_s = float(dz.get("settle_s", 1.5))
        self.box_x, self.box_y = tuple(bx["x"]), tuple(bx["y"])
        self.wall_h = float(bx["wall_height"])
        self.wall_t = 0.006
        self.release_above = float(bx.get("release_tcp_above_table", 0.12))
        self.release_orientation = str(bx.get("release_orientation", "forward_horizontal"))
        self.ho_center = np.array(ho["center"], dtype=float)
        self.tcp_gap = float(ho.get("tcp_gap", 0.10))
        fing = ho.get("fingers") or {}
        self.right_fingers_up = -1.0 if str(fing.get("right", "closing_axis_up")).endswith("down") else 1.0
        self.left_fingers_up = 1.0 if str(fing.get("left", "closing_axis_down")).endswith("up") else -1.0
        # motion parameters (CLI overrides)
        self.yaw_range = (-45.0 * DEG, 45.0 * DEG)
        # Right grasp candidates (pitch about the closing axis, roll about the cable axis). Pitch must stay near 0:
        # it tilts the cable inside the hand and the handover then needs a hand yaw that leaves the LeRobot limits.
        # Roll does not change the cable direction in the hand, so it only matters for the pick itself.
        self.grasp_cands = [(0.0, 0.0), (0.0, 15 * DEG), (0.0, -15 * DEG), (-10 * DEG, 0.0), (-10 * DEG, 15 * DEG),
                            (-10 * DEG, -15 * DEG), (0.0, 30 * DEG), (0.0, -30 * DEG), (10 * DEG, 0.0), (-10 * DEG, 30 * DEG),
                            (-10 * DEG, -30 * DEG)]
        self.allow_roll = True
        # Under the LeRobot limits top-down poses high above the table are often out of reach, so the 0.08 m
        # pre-grasp / lift / place-above offsets fall back to lower values (the value used is logged).
        self.approach_heights = [0.08, 0.06, 0.045, 0.035]
        self.lift_heights = [0.12, 0.09, 0.06, 0.045]
        self.place_above = [0.10, 0.08, 0.065]
        self.retreats = [0.10, 0.07, 0.05, 0.035]
        self.left_retreats = [0.06, 0.045, 0.03]
        self.via_raise = 0.06
        self.left_pre = 0.08
        self.place_mode = "corner"
        self.place_elevs = [60 * DEG, 75 * DEG, 45 * DEG]
        self.place_azim = None  # None = toward the far box corner
        self.place_inset = 0.025
        self.good_margin = 8.0
        self.extra_candidates = 2  # after the first feasible candidate, try this many more and keep the best
        self.max_drop_resample = 50
        for k, v in over.items():
            setattr(self, k, v)

    def box_inner(self):
        t = self.wall_t
        return (self.box_x[0] + t, self.box_x[1] - t), (self.box_y[0] + t, self.box_y[1] - t)


def sample_drop(rng, P):
    """Connector position uniform in the drop zone (minus the excluded corner), body yaw in P.yaw_range."""
    cg = P.cable
    for _ in range(10000):
        x, y = rng.uniform(*P.dz_x), rng.uniform(*P.dz_y)
        if P.ex_xmin is not None and P.ex_ymin is not None and x >= P.ex_xmin and y >= P.ex_ymin:
            continue
        yaw = rng.uniform(*P.yaw_range)
        d = np.array([math.cos(yaw), math.sin(yaw)])
        center = np.array([x, y]) + cg.conn_center[0] * d
        tail = center + (cg.L / 2) * d
        if tail[0] > P.table_x[1] - 0.01 or abs(tail[1]) > P.table_w / 2 - 0.01 or center[0] < P.table_x[0] + 0.02:
            continue
        return {"conn_xy": [float(x), float(y)], "center_xy": [float(center[0]), float(center[1])], "body_yaw": float(yaw)}
    raise RuntimeError("could not sample a drop pose; check drop_zone in workbench.yaml")


def right_grasp_pose(T_cable, P, pitch, roll, flip=False):
    """Top-down grasp of the connector: tool x along the cable (body direction), closing axis across it.
    `pitch` rotates about the closing axis (tool y), `roll` about the cable axis (tool x)."""
    cg = P.cable
    conn = apply_T(T_cable, cg.conn_center)
    d = T_cable[:3, :3] @ cg.body_dir
    yaw = math.atan2(d[1], d[0]) + (math.pi if flip else 0.0)
    R = topdown_frame(yaw) @ rot_y(pitch) @ rot_x(roll)
    z = P.top_z + P.grasp_above + (FINGER_ORIGIN_HALF_GAP + FINGER_PRESHAPE) * abs(math.sin(roll))
    return make_T(R, [conn[0], conn[1], z])


def right_handover_pose(T_cable_in_tcp, P, shift_x=0.0):
    """Right TCP pose at the handover: tool z ~ +y, fingers closing vertically, cable lying along world x.
    The right TCP and the left grasp point (left_dist along the cable) are centred on handover.center."""
    cg = P.cable
    Rh = frame_zy([0.0, 1.0, 0.0], [0.0, 0.0, P.right_fingers_up])
    d0 = Rh @ (T_cable_in_tcp[:3, :3] @ cg.body_dir)
    if math.hypot(d0[0], d0[1]) > 1e-3:  # yaw about world z so the cable lies exactly along x (square left grasp)
        target = math.pi if d0[0] < 0 else 0.0
        a = target - math.atan2(d0[1], d0[0])
        a = (a + math.pi) % (2 * math.pi) - math.pi
        Rh = rot_z(a) @ Rh
    lg = apply_T(T_cable_in_tcp, cg.left_pt)
    p = P.ho_center - 0.5 * (Rh @ lg) + np.array([shift_x, 0.0, 0.0])
    return make_T(Rh, p)


def handover_pose_clear(T_cable_in_tcp, P):
    """Handover pose, shifted forward (+x) in 2 cm steps if the carried cable would touch the torso column."""
    pts_t = [apply_T(T_cable_in_tcp, p) for p in P.cable.end_points()]
    for k in range(4):
        T = right_handover_pose(T_cable_in_tcp, P, shift_x=0.02 * k)
        clr = torso_clearance([apply_T(T, p) for p in pts_t])
        if clr >= 0.02:
            return T, clr
    return T, clr


def left_grasp_pose(T_cable, P):
    """Left TCP at the cable point `left_dist` from the connector; approach about -y, fingers closing vertically."""
    cg = P.cable
    p = apply_T(T_cable, cg.left_pt)
    a = T_cable[:3, :3][:, 0].copy()
    a[2] = 0.0
    h = _unit(a) if np.linalg.norm(a) > 1e-6 else np.array([1.0, 0.0, 0.0])
    z = np.array([0.0, -1.0, 0.0])
    z = _unit(z - np.dot(z, h) * h)
    T = make_T(frame_zy(z, [0.0, 0.0, P.left_fingers_up]), p)
    T_pre = T.copy()
    T_pre[:3, 3] = p - P.left_pre * z
    return T, T_pre


def place_candidates(T_cable_in_tcp, P):
    """Release poses for the left TCP. Rigid 0.30 m proxy vs 0.14 x 0.16 m box: hold the cable steeply, connector
    down, tail leaning toward the far box corner, so it topples diagonally into the box. Tool z ~ forward (+x)
    ('forward_horizontal' in workbench.yaml). Yields (label, T_goal)."""
    cg = P.cable
    (ix0, ix1), (iy0, iy1) = P.box_inner()
    u_t = T_cable_in_tcp[:3, :3] @ cg.body_dir  # tail direction in the tool frame
    conn_t = apply_T(T_cable_in_tcp, cg.conn_center)
    ctr_t = T_cable_in_tcp[:3, 3]
    A = frame_xz(u_t, [0.0, 0.0, 1.0])
    azim = P.place_azim if P.place_azim is not None else math.atan2(iy1 - iy0, ix1 - ix0)
    hints = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.5], [1.0, 0.0, -0.5], [1.0, 0.5, 0.0], [1.0, -0.5, 0.0]]
    if P.release_orientation == "topdown":
        hints = [[0.0, 0.0, -1.0]] + hints
    floor_top = P.top_z + P.wall_t
    for e in P.place_elevs:
        td = np.array([math.cos(e) * math.cos(azim), math.cos(e) * math.sin(azim), math.sin(e)])
        for h in hints:
            R = frame_xz(td, h) @ A.T
            if P.place_mode == "center":
                c = R @ ctr_t
                p = np.array([(ix0 + ix1) / 2 - c[0], (iy0 + iy1) / 2 - c[1], P.top_z + P.release_above])
            else:
                cx = ix0 + P.place_inset if td[0] >= 0 else ix1 - P.place_inset
                cy = iy0 + P.place_inset if td[1] >= 0 else iy1 - P.place_inset
                c = R @ conn_t
                p = np.array([cx - c[0], cy - c[1], P.top_z + P.release_above])
            conn_bottom = p[2] + (R @ conn_t)[2] - 0.5 * max(cg.conn_size)
            if conn_bottom < floor_top + 0.008:
                p[2] += floor_top + 0.008 - conn_bottom
            yield f"elev{e / DEG:.0f}_azim{azim / DEG:.0f}_hint{tuple(h)}", make_T(R, p)


def _sample_ready(arm, P, rng, ysign, n_samples):
    best = None
    for _ in range(n_samples):
        q = rng.uniform(arm.lim[:, 0], arm.lim[:, 1])
        m = arm.margin_deg(q)
        if best is not None and m <= best[1]:
            continue
        T = arm.fk(q)
        hand = apply_T(T, -arm.tcp_offset)
        p = T[:3, 3]
        if not (0.15 <= p[0] <= 0.32 and 0.16 <= ysign * p[1] <= 0.32 and P.top_z + 0.12 <= p[2] <= P.top_z + 0.25):
            continue
        if hand[2] < P.top_z + 0.12 or T[2, 2] > 0.0:  # hand above the table, fingers not pointing up
            continue
        best = (q, m)
    return best


def compute_ready(arms, P, rng, n_samples=6000):
    """READY: a right-arm configuration with large joint-limit margin whose hand is above the table in front of the
    right shoulder (every first move out of READY is a joint-space move anyway); left = mirror image (or, if the
    model is not mirror-symmetric, an independently sampled left configuration)."""
    best = _sample_ready(arms["right"], P, rng, -1.0, n_samples)
    if best is None:
        raise RuntimeError("no READY configuration found for the right arm")
    r_r = arms["right"].ik(arms["right"].fk(best[0]), best[0])
    T_r = arms["right"].fk(r_r.q)
    r_l = arms["left"].ik_global(mirror_T(T_r), r_r.q * MIRROR_SIGN, rng, n_random=12, extra_seeds=[r_r.q * MIRROR_SIGN])
    if r_l is not None and r_l.ok and r_l.margin_deg >= 10.0:
        q_l = r_l.q
    else:
        bl = _sample_ready(arms["left"], P, rng, 1.0, n_samples)
        if bl is None:
            raise RuntimeError("no READY configuration found for the left arm")
        q_l = bl[0]
    return {"right": r_r.q, "left": q_l, "T_right": T_r, "T_left": arms["left"].fk(q_l)}


# ---- per-state planners (shared by the dry run and the live run) ----

def plan_right_lift(arm, q, T_grasp, P):
    last = None
    for h in P.lift_heights:
        T = T_grasp.copy()
        T[2, 3] += h
        last = plan_line(arm, q, T)
        if last.ok:
            last.info["lift_m"] = h
            return last
    return last


def plan_right_handover(arm, q, T_cable_in_tcp, P, rng):
    T_h, clr = handover_pose_clear(T_cable_in_tcp, P)
    T_now = arm.fk(q)
    T_via = interp_T(T_now, T_h, 0.5)
    T_via[2, 3] += P.via_raise
    carried = [apply_T(T_cable_in_tcp, pt) for pt in P.cable.end_points()]
    plan = plan_via(arm, q, T_via, T_h, rng)
    if plan.ok:
        zmin = path_min_z(arm, plan.path, carried)
        tclr = path_torso_clearance(arm, plan.path, carried)
        if zmin < P.top_z + 0.03 or tclr < 0.01:  # sweep too low or through the torso: raise the via point once
            T_via[2, 3] += 0.08
            p2 = plan_via(arm, q, T_via, T_h, rng)
            if p2.ok:
                z2, t2 = path_min_z(arm, p2.path, carried), path_torso_clearance(arm, p2.path, carried)
                if min(z2 - P.top_z - 0.03, t2 - 0.01) > min(zmin - P.top_z - 0.03, tclr - 0.01):
                    plan, zmin, tclr = p2, z2, t2
        plan.info["min_z_above_table"] = round(zmin - P.top_z, 4)
        plan.info["torso_clearance_m"] = round(tclr, 4)
        plan.info["handover_x_shift_m"] = round(float(T_h[0, 3] - right_handover_pose(T_cable_in_tcp, P)[0, 3]), 3)
    return plan


def plan_right_pick(arm, q, T_cable, P, rng, allow_roll=True):
    """Pick candidates (pitch, roll) are scored by the whole right-arm chain they imply:
    joint move to pre-grasp, line descend, line lift, joint move (via waypoint) to the handover pose."""
    best, extra = None, None
    for pitch, roll in P.grasp_cands:
        if roll != 0.0 and not allow_roll:
            continue
        if extra is not None:
            extra -= 1
            if extra < 0:
                break
        Tg = right_grasp_pose(T_cable, P, pitch, roll)
        if not arm.ik_global(Tg, q, rng, n_random=6).ok:
            continue
        a = d = None
        for h in P.approach_heights:
            Tpre = Tg.copy()
            Tpre[2, 3] += h
            a = plan_joint_clear(arm, q, Tpre, rng, P.top_z + 0.025, n_random=8)
            if not a.ok:
                continue
            d = plan_line(arm, a.q_end, Tg)
            if d.ok:
                d.info["approach_h"] = h
                break
        if a is None or d is None or not (a.ok and d.ok):
            continue
        lift = plan_right_lift(arm, d.q_end, Tg, P)
        if not lift.ok:
            continue
        T_c_in_tcp = inv_T(arm.fk(d.q_end)) @ T_cable  # nominal: cable does not move in the hand
        ho = plan_right_handover(arm, lift.q_end, T_c_in_tcp, P, rng)
        if not ho.ok:
            continue
        score = min(a.min_margin(arm), d.min_margin(arm), lift.min_margin(arm), ho.min_margin(arm))
        if ho.info.get("torso_clearance_m", 1.0) < 0.01:
            score -= 10.0
        cand = {"ok": True, "approach": a, "descend": d, "lift": lift, "handover": ho, "T_grasp": Tg, "T_pre": Tpre,
                "pitch_deg": round(pitch / DEG, 1), "roll_deg": round(roll / DEG, 1), "margin": score,
                "approach_h": d.info.get("approach_h"), "lift_m": lift.info.get("lift_m")}
        if best is None or score > best["margin"]:
            best = cand
        if score >= P.good_margin:
            break
        if extra is None:
            extra = P.extra_candidates
    return best or {"ok": False}


def plan_left_pick(arm, q, T_cable, P, rng):
    T_g, T_pre = left_grasp_pose(T_cable, P)
    a = plan_joint_clear(arm, q, T_pre, rng, P.top_z + 0.05)
    if not a.ok:
        return {"ok": False, "approach": a, "T_grasp": T_g, "T_pre": T_pre}
    b = plan_line(arm, a.q_end, T_g)
    return {"ok": b.ok, "approach": a, "insert": b, "T_grasp": T_g, "T_pre": T_pre}


def plan_retreat(arm, q, q_home, backs, up, rng):
    """Back off along -tool z (fingers slide off the cable), trying shorter distances if needed, then go home."""
    T = arm.fk(q)
    a = None
    for back in backs:
        T2 = T.copy()
        T2[:3, 3] = T[:3, 3] - back * T[:3, 2] + np.array([0.0, 0.0, up * back / backs[0]])
        a = plan_line(arm, q, T2)
        if a.ok:
            a.info["back_m"] = back
            break
    if not a.ok:  # keep whatever part of the straight retreat was feasible
        a = Plan(len(a.path) > 1, a.path, "line(partial)", arm.fk(a.path[-1]), a.info)
    b = Plan(True, joint_path(a.q_end, np.asarray(q_home, dtype=float)), "joint", arm.fk(q_home))
    z_min = min(arm.fk(a.q_end)[2, 3], arm.fk(q_home)[2, 3]) - 0.03
    if path_min_z(arm, b.path) < z_min:
        b2 = plan_joint_clear(arm, a.q_end, arm.fk(q_home), rng, z_min)
        if b2.ok:
            b = b2
    return a, b


def plan_left_place(arm, q, T_cable_in_tcp, P, rng):
    """For each release pose (see place_candidates) and above-offset: joint move above it, vertical line down.
    The carried cable must stay above the box rim on the way in."""
    best, extra = None, None
    carried = [apply_T(T_cable_in_tcp, pt) for pt in P.cable.end_points()]
    rim = P.top_z + P.wall_h
    for label, T in place_candidates(T_cable_in_tcp, P):
        if extra is not None:
            extra -= 1
            if extra < 0:
                break
        if not arm.ik_global(T, q, rng, n_random=8).ok:
            continue
        for dz in P.place_above:
            T_above = T.copy()
            T_above[2, 3] += dz
            a = plan_joint_clear(arm, q, T_above, rng, rim + 0.005, carried, n_random=10)
            if not a.ok:
                continue
            d = plan_line(arm, a.q_end, T)
            if not d.ok:
                continue
            zmin = path_min_z(arm, a.path, carried)
            score = min(a.min_margin(arm), d.min_margin(arm)) - (10.0 if zmin < rim + 0.005 else 0.0)
            cand = {"ok": True, "to_above": a, "descend": d, "T_goal": T, "T_above": T_above, "label": f"{label}_above{dz:.3f}",
                    "margin": score, "min_z_above_table": zmin - P.top_z}
            if best is None or score > best["margin"]:
                best = cand
            break
        if best is not None and best["margin"] >= P.good_margin:
            break
        if best is not None and extra is None:
            extra = P.extra_candidates
    return best or {"ok": False}


def plan_episode_nominal(arms, P, T_cable0, q_ready, rng):
    """Plan the whole episode kinematically (cable rigidly carried in the nominal grasp). Used by the dry run."""
    out = []

    def rec(state, side, plan_or_ok, T_goal=None, extra=None):
        arm = arms[side]
        if isinstance(plan_or_ok, Plan):
            ok, q_end = plan_or_ok.ok, plan_or_ok.q_end
            T_goal = plan_or_ok.T_goal if T_goal is None else T_goal
            m = plan_or_ok.min_margin(arm)
        else:
            ok, q_end, m = bool(plan_or_ok), None, None
        row = {"state": state, "side": side, "ok": ok}
        if T_goal is not None:
            row["target_xyz"] = [round(float(v), 4) for v in T_goal[:3, 3]]
            if q_end is not None:
                pe, re = pose_error(arm.fk(q_end), T_goal)
                row.update(pos_err_mm=round(pe * 1e3, 2), rot_err_deg=round(re, 2))
        if q_end is not None:
            row["q_end_deg"] = [round(float(v) / DEG, 1) for v in q_end]
            row["end_margin_deg"] = round(arm.margin_deg(q_end), 1)
            row["min_margin_deg"] = round(m, 1)
        if extra:
            row.update(extra)
        out.append(row)
        return ok

    qr, ql = np.asarray(q_ready["right"]).copy(), np.asarray(q_ready["left"]).copy()
    rp = plan_right_pick(arms["right"], qr, T_cable0, P, rng, allow_roll=P.allow_roll)
    if not rp["ok"]:
        rec("R_APPROACH", "right", False, right_grasp_pose(T_cable0, P, 0.0, 0.0))
        return out, False
    rec("R_APPROACH", "right", rp["approach"], extra={"grasp_pitch_deg": rp["pitch_deg"], "grasp_roll_deg": rp["roll_deg"]})
    rec("R_DESCEND", "right", rp["descend"], extra={"approach_h": rp["approach_h"]})
    qr = rp["descend"].q_end
    T_c_in_r = inv_T(arms["right"].fk(qr)) @ T_cable0
    lift = rp["lift"]
    rec("R_LIFT", "right", lift, extra={"lift_m": lift.info.get("lift_m")})
    ho = rp["handover"]
    rec("R_TO_HANDOVER", "right", ho, extra={k: ho.info.get(k) for k in ("min_z_above_table", "torso_clearance_m",
                                                                           "handover_x_shift_m")})
    qr = ho.q_end
    T_cable = arms["right"].fk(qr) @ T_c_in_r
    lp = plan_left_pick(arms["left"], ql, T_cable, P, rng)
    if not lp["ok"]:
        rec("L_APPROACH", "left", lp.get("insert", lp["approach"]), lp["T_grasp"])
        return out, False
    rec("L_APPROACH", "left", concat_plans(lp["approach"], lp["insert"]))
    ql = lp["insert"].q_end
    d_hands = float(np.linalg.norm(arms["right"].hand_pose(qr)[:3, 3] - arms["left"].hand_pose(ql)[:3, 3]))
    d_tcp = float(np.linalg.norm(arms["right"].fk(qr)[:3, 3] - arms["left"].fk(ql)[:3, 3]))
    out[-1]["hand_origin_dist_m"], out[-1]["tcp_dist_m"] = round(d_hands, 3), round(d_tcp, 3)
    T_c_in_l = inv_T(arms["left"].fk(ql)) @ T_cable
    a, b = plan_retreat(arms["right"], qr, q_ready["right"], P.retreats, 0.0, rng)
    rec("R_RETREAT", "right", concat_plans(a, b))
    pl = plan_left_place(arms["left"], ql, T_c_in_l, P, rng)
    if not pl["ok"]:
        rec("L_TO_BOX", "left", False)
        return out, False
    rec("L_TO_BOX", "left", concat_plans(pl["to_above"], pl["descend"]),
        extra={"place": pl["label"], "min_z_above_table": round(pl["min_z_above_table"], 3)})
    ql = pl["descend"].q_end
    T_rel = arms["left"].fk(ql) @ T_c_in_l
    a, b = plan_retreat(arms["left"], ql, q_ready["left"], P.left_retreats, 0.06, rng)
    rec("L_RETREAT", "left", concat_plans(a, b), extra={"release_cable_xyz": [round(float(v), 3) for v in T_rel[:3, 3]]})
    return out, all(r["ok"] for r in out)


def sample_feasible_drop(rng, P, arms, q_ready, max_tries):
    """Uniform drop sample, re-drawn while the right arm has no feasible pick -> handover chain for the nominal
    resting pose (the rejected samples are returned so the coverage can be reported)."""
    rejected = []
    drop = None
    for _ in range(max(1, max_tries)):
        drop = sample_drop(rng, P)
        T_c = P.cable.rest_pose(drop["center_xy"], drop["body_yaw"], P.top_z)
        if plan_right_pick(arms["right"], q_ready["right"], T_c, P, rng, allow_roll=P.allow_roll)["ok"]:
            return drop, rejected
        rejected.append(drop)
    return drop, rejected

# ================================ PLANNER END ================================

# ----------------------------------------------------------------------------------------------------------------------
# CLI + app launch (template: scripts/tutorials/05_controllers/run_diff_ik.py)
# ----------------------------------------------------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="OpenArm bimanual cable handover - scripted motion-planning demo.")
parser.add_argument("--openarm_usd", type=str, default=None,
                    help="OpenArm bimanual USD. Default: $OPENARM_ASSETS/source/openarm/openarm/tasks/manager_based/"
                         "openarm_manipulation/usds/openarm_bimanual/openarm_bimanual.usd")
parser.add_argument("--workbench", type=str, default=os.path.join(_HERE, "..", "workbench", "workbench.yaml"))
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--seed", type=int, default=0, help="Episode i uses seed + i.")
parser.add_argument("--attach", choices=["on", "off"], default="on",
                    help="on: on close the cable is rigidly glued to the closing hand (demo crutch). off: pure friction grasp.")
parser.add_argument("--partition", choices=["on", "off"], default="off", help="Centre socket partition (workbench partition).")
parser.add_argument("--yaw_min_deg", type=float, default=-45.0, help="Cable body yaw range (connector->tail), about table normal.")
parser.add_argument("--yaw_max_deg", type=float, default=45.0)
parser.add_argument("--grasp_pitch_deg", type=float, default=0.0,
                    help="Preferred pitch of the right grasp about its closing axis (tried first; 0 = straight top-down).")
parser.add_argument("--no_resample", action="store_true",
                    help="Do not re-draw drop poses whose pick->handover chain is IK-infeasible (episode then fails).")
parser.add_argument("--place_mode", choices=["corner", "center"], default="corner",
                    help="corner: connector lands near the near box corner, cable topples diagonally. center: yaml release point.")
parser.add_argument("--place_elev_deg", type=float, default=60.0, help="Cable elevation at release (connector down).")
parser.add_argument("--place_azim_deg", type=float, default=None, help="Tail azimuth at release (default: toward far box corner).")
parser.add_argument("--speed", type=float, default=1.0, help="Motion speed scale.")
parser.add_argument("--out_dir", type=str, default=None, help="Default ~/openarm-out/demo/<timestamp>.")
# NOTE: deliberately NOT named --video: AppLauncher reads args.video with bool() (app_launcher.py:980), so a string
# 'off' would still switch its own viewport-video mode on. This option only controls this script's Camera recording.
parser.add_argument("--record", dest="record_video", choices=["on", "off"], default="on",
                    help="Record mp4 from Camera sensors (needs --enable_cameras or ENABLE_CAMERAS=1).")
parser.add_argument("--video_res", choices=["720", "1080"], default="720")
parser.add_argument("--top_view", action="store_true", help="Also record a top view.")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--dry_run_kinematics", action="store_true",
                    help="No controller motion: plan every waypoint on the extracted kinematic model, print IK feasibility, "
                         "and teleport the arms to each solution to cross-check the model against the USD. Still takes a few "
                         "physics steps (targets = current state) to read the PhysX Jacobian / link poses; the cable falls "
                         "freely during those steps.")
parser.add_argument("--no_teleport_check", action="store_true",
                    help="Dry run: skip the teleport cross-check (fewer physics steps; start-up settle steps remain).")
AppLauncher_import_error = None
try:
    from isaaclab.app import AppLauncher
except Exception as e:  # pragma: no cover - only without Isaac Lab
    AppLauncher_import_error = e
    AppLauncher = None
if AppLauncher is None:
    print(f"[ERROR] isaaclab is not importable ({AppLauncher_import_error}). Run with ./isaaclab.sh -p ...")
    sys.exit(2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _resolve_inputs():
    """Check the robot USD and workbench yaml BEFORE Kit boots (a typo should not cost a full Kit start-up)."""
    if args_cli.openarm_usd:
        usd = os.path.expanduser(args_cli.openarm_usd)
    else:
        root = os.environ.get("OPENARM_ASSETS")
        if not root:
            print("[ERROR] pass --openarm_usd or set OPENARM_ASSETS to the enactic/openarm_isaac_lab checkout")
            sys.exit(2)
        usd = os.path.join(os.path.expanduser(root), "source/openarm/openarm/tasks/manager_based/openarm_manipulation/"
                           "usds/openarm_bimanual/openarm_bimanual.usd")
    if not os.path.isfile(usd):
        print(f"[ERROR] OpenArm USD not found: {usd}")
        sys.exit(2)
    wb = os.path.expanduser(args_cli.workbench)
    if not os.path.isfile(wb):
        print(f"[ERROR] workbench yaml not found: {wb}")
        sys.exit(2)
    return usd, wb


USD_PATH, WORKBENCH_PATH = _resolve_inputs()
# AppLauncher enables cameras from --enable_cameras OR the ENABLE_CAMERAS=1 env var (app_launcher.py:_resolve_camera_settings).
_cams_requested = bool(getattr(args_cli, "enable_cameras", False)) or os.environ.get("ENABLE_CAMERAS", "0") == "1"
if args_cli.dry_run_kinematics:
    args_cli.record_video = "off"
if args_cli.record_video == "on" and not _cams_requested:
    print("[WARN] --record on needs --enable_cameras (or ENABLE_CAMERAS=1); video disabled for this run.")
    args_cli.record_video = "off"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
if args_cli.record_video == "on" and not getattr(app_launcher, "_enable_cameras", True):
    print("[WARN] AppLauncher did not enable cameras; video disabled for this run.")
    args_cli.record_video = "off"

"""Rest everything follows."""

import torch  # noqa: E402
import yaml  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
# Non-deprecated PhysX schema cfgs (sim_utils.RigidBodyPropertiesCfg & co. are deprecated aliases in af1bab4 that warn).
from isaaclab_physx.sim.schemas import (  # noqa: E402
    PhysxArticulationRootPropertiesCfg,
    PhysxCollisionPropertiesCfg,
    PhysxRigidBodyPropertiesCfg,
)
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_inv,
    skew_symmetric_matrix,
    subtract_frame_transforms,
)

PHYSICS_DT = 1.0 / 120.0
DLS_LAMBDA = 0.05
NULLSPACE_GAIN = 1.0
VMAX_JOINT = 1.0  # rad/s peak (well below the 2.175 rad/s sim limit)
V_CART = 0.12  # m/s peak along Cartesian lines
W_CART = 60.0 * DEG  # rad/s peak

BANNER = "\n".join([
    "=" * 110,
    CAPTION,
    "DEMO SETTINGS: HIGH-PD arm actuators (stiffness 400, damping 80; reference 80/4), gravity DISABLED on robot links,",
    "               self-collision off (as reference cfg), real-robot LeRobot joint limits written into the simulator.",
    "               --attach on = cable glued to the closing hand (demo crutch); --attach off = friction grasp only.",
    "=" * 110,
])


# ----------------------------------------------------------------------------------------------------------------------
# Scene (all primitives; only the robot USD is loaded from disk)
# ----------------------------------------------------------------------------------------------------------------------

def _static_box(path, size, center, color, friction=0.8):
    cfg = sim_utils.CuboidCfg(
        size=tuple(float(s) for s in size),
        collision_props=PhysxCollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.8),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction, restitution=0.0),
    )
    cfg.func(path, cfg, translation=tuple(float(c) for c in center))


def design_scene(P, wb, usd_path, partition_on, video):
    top = P.top_z
    sim_utils.create_prim("/World/Scene", "Xform")
    # floor (static slab, top surface at z = 0 = body_link0 bottom)
    _static_box("/World/Scene/Floor", (4.0, 4.0, 0.05), (0.0, 0.0, -0.025), (0.35, 0.35, 0.37))
    # lights
    sim_utils.create_prim("/World/Light", "Xform")
    cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.85, 0.85, 0.85))
    cfg.func("/World/Light/Dome", cfg)
    cfg = sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 0.98, 0.95), angle=1.0)
    cfg.func("/World/Light/Key", cfg, orientation=(0.2588, 0.0, 0.0, 0.9659))
    # table: top surface at top_z, x in [front_edge, front_edge + depth], y in +-width/2
    th = 0.03
    (x0, x1), w = P.table_x, P.table_w
    _static_box("/World/Scene/TableTop", (x1 - x0, w, th), ((x0 + x1) / 2, 0.0, top - th / 2), (0.12, 0.12, 0.13), 0.9)
    for i, (lx, ly) in enumerate([(x0 + 0.03, w / 2 - 0.03), (x0 + 0.03, -w / 2 + 0.03), (x1 - 0.03, w / 2 - 0.03), (x1 - 0.03, -w / 2 + 0.03)]):
        _static_box(f"/World/Scene/Leg{i}", (0.04, 0.04, top - th), (lx, ly, (top - th) / 2), (0.25, 0.25, 0.27))
    # open-top box: floor + 4 walls (static)
    t, hgt = P.wall_t, P.wall_h
    (bx0, bx1), (by0, by1) = P.box_x, P.box_y
    bc = ((bx0 + bx1) / 2, (by0 + by1) / 2)
    col = (0.15, 0.35, 0.75)
    sim_utils.create_prim("/World/Scene/Box", "Xform")
    _static_box("/World/Scene/Box/Floor", (bx1 - bx0, by1 - by0, t), (bc[0], bc[1], top + t / 2), col)
    _static_box("/World/Scene/Box/WallXmin", (t, by1 - by0, hgt), (bx0 + t / 2, bc[1], top + hgt / 2), col)
    _static_box("/World/Scene/Box/WallXmax", (t, by1 - by0, hgt), (bx1 - t / 2, bc[1], top + hgt / 2), col)
    _static_box("/World/Scene/Box/WallYmin", (bx1 - bx0 - 2 * t, t, hgt), (bc[0], by0 + t / 2, top + hgt / 2), col)
    _static_box("/World/Scene/Box/WallYmax", (bx1 - bx0 - 2 * t, t, hgt), (bc[0], by1 - t / 2, top + hgt / 2), col)
    # optional partition (real cell has it; sim demo default off)
    if partition_on:
        pt = wb.get("partition") or {}
        px = pt.get("x") or [0.10, 0.60]
        pth = float(pt.get("thickness") or 0.02)
        ph = pt.get("height_above_table")
        ph = 0.25 if ph is None else float(ph)
        _static_box("/World/Scene/Partition", (px[1] - px[0], pth, ph),
                    ((px[0] + px[1]) / 2, float(pt.get("y_center", 0.0)), top + ph / 2), (0.6, 0.6, 0.6))
        print(f"[INFO] partition ON: x={px}, thickness={pth}, height={ph} (height is a placeholder if yaml says null)")

    # robot (reference: openarm_isaac_lab .../assets/openarm_bimanual.py, HIGH-PD variant)
    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=PhysxRigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
            articulation_props=PhysxArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={"openarm_(left|right)_joint[1-7]": 0.0, "openarm_(left|right)_finger_joint.*": FINGER_OPEN},
        ),
        actuators={
            "openarm_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_(left|right)_joint[1-7]"],
                velocity_limit_sim={"openarm_(left|right)_joint[1-4]": 2.175, "openarm_(left|right)_joint[5-7]": 2.61},
                effort_limit_sim={"openarm_(left|right)_joint[1-2]": 40.0, "openarm_(left|right)_joint[3-4]": 27.0,
                                  "openarm_(left|right)_joint[5-7]": 7.0},
                stiffness=400.0,
                damping=80.0,
            ),
            "openarm_gripper": ImplicitActuatorCfg(
                joint_names_expr=["openarm_(left|right)_finger_joint.*"],
                velocity_limit_sim=0.2,
                effort_limit_sim=333.33,
                stiffness=2e3,
                damping=1e2,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    robot = Articulation(cfg=robot_cfg)

    # Cable proxy: ONE rigid body = thin cuboid (length x d x d) + connector cuboid as a second collider under the
    # same rigid-body prim (compound shape). Why one body: a 2-body articulation would need hand-authored USD joints
    # (no primitive spawner for that in af1bab4); a compound rigid body is plain USD physics, RigidObject accepts it
    # (exactly one RigidBodyAPI prim under the asset path), attach mode teleports a single pose, and the success
    # check reads one pose. Mass is set on the body (MassAPI) = linear_mass * length + connector mass.
    cg = P.cable
    cable_cfg = RigidObjectCfg(
        prim_path="/World/Cable",
        spawn=sim_utils.CuboidCfg(
            size=(cg.L, cg.d, cg.d),
            rigid_props=PhysxRigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5, solver_position_iteration_count=16, solver_velocity_iteration_count=2,
                angular_damping=0.05,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=P.cable_mass),
            collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.45, 0.05), roughness=0.6),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, -0.25, P.top_z + 0.05)),
    )
    cable = RigidObject(cfg=cable_cfg)
    conn_cfg = sim_utils.CuboidCfg(
        size=cg.conn_size,
        collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.08), roughness=0.4),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9, restitution=0.0),
    )
    conn_cfg.func("/World/Cable/Connector", conn_cfg, translation=tuple(float(v) for v in cg.conn_center))

    cams = {}
    if video:
        from isaaclab.sensors.camera import Camera, CameraCfg  # tutorial: scripts/tutorials/04_sensors/run_usd_camera.py
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        h, w = (1080, 1920) if args_cli.video_res == "1080" else (720, 1280)
        sim_utils.create_prim("/World/DemoCams", "Xform")
        names = ["oblique"] + (["top"] if args_cli.top_view else [])
        for name in names:
            cam_cfg = CameraCfg(
                prim_path=f"/World/DemoCams/{name}",
                update_period=0,
                height=h,
                width=w,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955,
                                                 clipping_range=(0.05, 50.0)),
                renderer_cfg=IsaacRtxRendererCfg(),
            )
            cams[name] = Camera(cfg=cam_cfg)
    return robot, cable, cams


# ----------------------------------------------------------------------------------------------------------------------
# Video: Camera sensor frames -> imageio/ffmpeg, cv2, or PNG frames (+ printed ffmpeg command)
# ----------------------------------------------------------------------------------------------------------------------

def _png_bytes(img):
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(
        b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b"")


class VideoSink:
    def __init__(self, path, fps):
        self.path, self.fps, self.n, self.mode, self.w = path, fps, 0, None, None
        try:
            import imageio.v2 as iio

            self.w = iio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8)
            self.mode = "imageio"
        except Exception as e1:
            try:
                import cv2  # noqa: F401

                self.mode, self._cv2 = "cv2", __import__("cv2")
            except Exception as e2:
                self.mode = "png"
                self.dir = os.path.splitext(path)[0] + "_frames"
                os.makedirs(self.dir, exist_ok=True)
                print(f"[WARN] no video writer (imageio: {e1}; cv2: {e2}); writing PNG frames to {self.dir}")

        print(f"[INFO] video writer for {os.path.basename(path)}: {self.mode}")

    def _to_png(self, why):
        self.mode, self.w = "png", None
        self.dir = os.path.splitext(self.path)[0] + "_frames"
        os.makedirs(self.dir, exist_ok=True)
        print(f"[WARN] {why}; writing PNG frames to {self.dir}")

    def add(self, frame):
        if self.mode == "cv2" and self.w is None:
            fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
            self.w = self._cv2.VideoWriter(self.path, fourcc, float(self.fps), (frame.shape[1], frame.shape[0]))
            if not self.w.isOpened():  # aarch64 opencv builds often lack an FFmpeg backend / mp4v encoder
                self._to_png("cv2.VideoWriter could not open an mp4v stream")
        if self.mode == "imageio":
            self.w.append_data(frame)
        elif self.mode == "cv2":
            self.w.write(frame[:, :, ::-1].copy())
        else:
            fn = os.path.join(self.dir, f"frame_{self.n:06d}.png")
            try:
                from PIL import Image

                Image.fromarray(frame).save(fn)
            except Exception:
                with open(fn, "wb") as f:
                    f.write(_png_bytes(np.ascontiguousarray(frame)))
        self.n += 1

    def close(self):
        if self.mode == "imageio" and self.w is not None:
            self.w.close()
        elif self.mode == "cv2" and self.w is not None:
            self.w.release()
        if self.mode == "png":
            print(f"[INFO] {self.n} PNG frames in {self.dir}. Encode with:\n  ffmpeg -framerate {self.fps} -i "
                  f"{self.dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p -crf 18 {self.path}")
        else:
            print(f"[INFO] wrote {self.n} frames -> {self.path} ({self.mode})")


class Overlay:
    def __init__(self, height):
        self.ok, self.font = False, None
        try:
            from PIL import Image, ImageDraw, ImageFont

            self.Image, self.ImageDraw = Image, ImageDraw
            size = max(14, height // 40)
            for fp in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
                try:
                    self.font = ImageFont.truetype(fp, size)
                    break
                except Exception:
                    pass
            if self.font is None:
                try:
                    self.font = ImageFont.load_default(size=size)
                except Exception:
                    self.font = ImageFont.load_default()
            self.size, self.ok = size, True
        except Exception as e:
            print(f"[WARN] PIL not available ({e}); frames are written without the caption overlay.")

    def __call__(self, frame, lines):
        if not self.ok:
            return frame
        img = self.Image.fromarray(frame)
        d = self.ImageDraw.Draw(img, "RGBA")
        lh = int(self.size * 1.35)
        d.rectangle([0, 0, img.width, lh * len(lines) + 10], fill=(0, 0, 0, 150))
        for i, line in enumerate(lines):
            try:
                d.text((10, 5 + i * lh), line, fill=(255, 255, 255, 255), font=self.font)
            except Exception:
                d.text((10, 5 + i * lh), line.encode("ascii", "replace").decode(), fill=(255, 255, 255, 255), font=self.font)
        return np.asarray(img)


# ----------------------------------------------------------------------------------------------------------------------
# Robot access helpers (pattern from run_diff_ik.py: .torch views, subtract_frame_transforms into the root frame)
# ----------------------------------------------------------------------------------------------------------------------

class ArmIO:
    def __init__(self, robot, side, device):
        self.side, self.robot, self.device = side, robot, device
        names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
        # find_* -> resolve_matching_names raises ValueError on an unmatched key; re-raise with the full name lists.
        try:
            self.joint_ids, jn = robot.find_joints(names, preserve_order=True)
        except ValueError as e:
            raise RuntimeError(f"{side}: expected joints {names} ({e}). Robot joints: {robot.joint_names}") from e
        try:
            self.finger_ids, self.finger_names = robot.find_joints(f"openarm_{side}_finger_joint.*")
        except ValueError:
            print(f"[WARN] {side}: no finger joints matched openarm_{side}_finger_joint.*; gripper commands disabled. "
                  f"Robot joints: {robot.joint_names}")
            self.finger_ids, self.finger_names = [], []
        try:
            body_ids, bn = robot.find_bodies(f"openarm_{side}_hand")
        except ValueError as e:
            raise RuntimeError(f"{side}: body openarm_{side}_hand not found ({e}). Robot bodies: {robot.body_names}") from e
        if len(body_ids) != 1:
            raise RuntimeError(f"{side}: openarm_{side}_hand matched {bn}. Robot bodies: {robot.body_names}")
        self.hand_id = body_ids[0]
        self.jac_body = self.hand_id - 1 if robot.is_fixed_base else self.hand_id
        self.jac_cols = [j + robot.num_base_dofs for j in self.joint_ids]
        lim = np.array(LEROBOT_LIMITS_DEG[side], dtype=float) * DEG
        self.lim = lim
        self.lim_t = torch.tensor(lim, dtype=torch.float32, device=device)
        self.offset_pos = torch.tensor([TCP_OFFSET.tolist()], dtype=torch.float32, device=device)
        self.offset_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls",
                                          ik_params={"lambda_val": DLS_LAMBDA})
        self.ctrl = DifferentialIKController(cfg, num_envs=1, device=device)
        self.model = None

    def q(self):
        return self.robot.data.joint_pos.torch[:, self.joint_ids]

    def hand_and_jacobian_b(self):
        """Hand pose and its 6x7 Jacobian, both in the robot root frame."""
        d = self.robot.data
        root = d.root_pose_w.torch
        hand_w = d.body_pose_w.torch[:, self.hand_id]
        pos_b, quat_b = subtract_frame_transforms(root[:, 0:3], root[:, 3:7], hand_w[:, 0:3], hand_w[:, 3:7])
        J = d.body_link_jacobian_w.torch[:, self.jac_body]  # (1, 6, num_dofs)
        J = J[:, :, self.jac_cols].clone()  # (1, 6, 7)
        Rb = matrix_from_quat(quat_inv(root[:, 3:7]))
        J[:, 0:3, :] = torch.bmm(Rb, J[:, 0:3, :])
        J[:, 3:6, :] = torch.bmm(Rb, J[:, 3:6, :])
        return pos_b, quat_b, J

    def tcp_and_jacobian_b(self):
        """TCP = hand + TCP_OFFSET (hand z). Same idea as DifferentialInverseKinematicsAction body_offset
        (isaaclab/envs/mdp/actions/task_space_actions.py) but with the offset rotated into the root frame."""
        pos_b, quat_b, J = self.hand_and_jacobian_b()
        tcp_pos, tcp_quat = combine_frame_transforms(pos_b, quat_b, self.offset_pos, self.offset_rot)
        r = quat_apply(quat_b, self.offset_pos)
        J = J.clone()
        J[:, 0:3, :] += torch.bmm(-skew_symmetric_matrix(r), J[:, 3:6, :])
        return tcp_pos, tcp_quat, J

    def tcp_T(self):
        p, q, _ = self.tcp_and_jacobian_b()
        return make_T(R_from_quat_xyzw(q[0].cpu().numpy()), p[0].cpu().numpy())

    def hand_T(self):
        p, q, _ = self.hand_and_jacobian_b()
        return make_T(R_from_quat_xyzw(q[0].cpu().numpy()), p[0].cpu().numpy())

    def build_model(self):
        _, _, J = self.hand_and_jacobian_b()
        T_hand = self.hand_T()
        q0 = self.q()[0].cpu().numpy().astype(float)
        self.model, norms = PoEArm.from_jacobian(self.side, J[0].cpu().numpy(), T_hand, q0, self.lim)
        return norms

    def command(self, T_ref, q_ref):
        """One control step: DifferentialIKController (pose, dls) toward T_ref + null-space pull to q_ref."""
        tcp_pos, tcp_quat, J = self.tcp_and_jacobian_b()
        q = self.q()
        cmd = torch.tensor([list(T_ref[:3, 3]) + list(quat_xyzw_from_R(T_ref[:3, :3]))], dtype=torch.float32, device=self.device)
        self.ctrl.set_command(cmd)
        q_des = self.ctrl.compute(tcp_pos, tcp_quat, J, q)
        Jt = J.transpose(1, 2)
        Jpinv = Jt @ torch.inverse(J @ Jt + (DLS_LAMBDA ** 2) * torch.eye(6, device=self.device))
        N = torch.eye(7, device=self.device) - Jpinv @ J
        qr = torch.tensor(q_ref[None], dtype=torch.float32, device=self.device)
        q_des = q_des + NULLSPACE_GAIN * (N @ (qr - q).unsqueeze(-1)).squeeze(-1)
        q_des = torch.max(torch.min(q_des, self.lim_t[:, 1] - 1e-3), self.lim_t[:, 0] + 1e-3)
        self.robot.set_joint_position_target_index(target=q_des, joint_ids=self.joint_ids)

    def grip(self, width_q):
        if len(self.finger_ids):
            t = torch.full((1, len(self.finger_ids)), float(width_q), dtype=torch.float32, device=self.device)
            self.robot.set_joint_position_target_index(target=t, joint_ids=self.finger_ids)


def _min_jerk(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


class Motion:
    """A sequence of planned joint paths, each played with its own min-jerk timing (stops at the junctions).
    Line segments are timed by Cartesian speed, joint/via segments by joint speed. The TCP reference sent to the
    DifferentialIKController is FK(q_ref) on the PoE model."""

    def __init__(self, arm_model, plans, speed=1.0):
        plans = plans if isinstance(plans, (list, tuple)) else [plans]
        self.model = arm_model
        self.segs = []
        for plan in plans:
            path = np.asarray(plan.path, dtype=float)
            if len(path) > 1:
                L = float(np.max(np.sum(np.abs(np.diff(path, axis=0)), axis=0)))
                dist, ang = pose_error(arm_model.fk(path[0]), arm_model.fk(path[-1]))
            else:
                L, dist, ang = 0.0, 0.0, 0.0
            if plan.kind.startswith("line"):
                dur = max(0.6, 1.875 * dist / V_CART, 1.875 * ang * DEG / W_CART)
            else:
                dur = max(0.8, 1.875 * L / VMAX_JOINT)
            self.segs.append((path, dur / max(speed, 1e-3)))
        self.path = np.concatenate([s[0] for s in self.segs], axis=0)
        self.T_goal = plans[-1].T_goal
        self.duration = sum(s[1] for s in self.segs)

    def q_at(self, t):
        for path, dur in self.segs:
            if t <= dur or (path is self.segs[-1][0]):
                if len(path) == 1:
                    return path[0]
                s = _min_jerk(t / dur) * (len(path) - 1)
                i = min(int(s), len(path) - 2)
                f = s - i
                return (1.0 - f) * path[i] + f * path[i + 1]
            t -= dur
        return self.segs[-1][0][-1]


# ----------------------------------------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------------------------------------

class Demo:
    def __init__(self, sim, robot, cable, cams, P, out_dir):
        self.sim, self.robot, self.cable, self.cams, self.P, self.out_dir = sim, robot, cable, cams, P, out_dir
        self.dev = sim.device
        self.arms = {s: ArmIO(robot, s, self.dev) for s in ("right", "left")}
        self.models = None
        self.q_ready = None
        self.hold = {"right": None, "left": None}  # current joint reference per arm
        self.grip_q = {"right": FINGER_OPEN, "left": FINGER_OPEN}
        self.attached = None  # (side, T_hand_cable)
        self.finger_half_gap = FINGER_ORIGIN_HALF_GAP  # replaced by the USD measurement in setup()
        self.step_count = 0
        self.capture_every = max(1, int(round(1.0 / (PHYSICS_DT * args_cli.fps))))
        self.sinks = {}
        self.overlay = None
        if cams:
            h = 1080 if args_cli.video_res == "1080" else 720
            self.overlay = Overlay(h)
            fps = 1.0 / (PHYSICS_DT * self.capture_every)
            for name in cams:
                self.sinks[name] = VideoSink(os.path.join(out_dir, f"demo_{name}.mp4"), fps)
        self.caption2 = ""
        self.state_name = "INIT"
        self.ep_metrics = None
        # bodies for the left/right minimum-distance log
        self.lr_bodies = {s: [i for i, n in enumerate(robot.body_names) if n.startswith(f"openarm_{s}_") and
                              not re.search(r"link[0-2]$", n)] for s in ("right", "left")}

    # ---------------- low-level stepping ----------------
    def _apply_attach(self):
        if self.attached is None:
            return
        side, T_hc = self.attached
        io = self.arms[side]
        d = self.robot.data
        hand_w = d.body_pose_w.torch[:, io.hand_id]
        off_p = torch.tensor([T_hc[:3, 3].tolist()], dtype=torch.float32, device=self.dev)
        off_q = torch.tensor([quat_xyzw_from_R(T_hc[:3, :3]).tolist()], dtype=torch.float32, device=self.dev)
        pos, quat = combine_frame_transforms(hand_w[:, 0:3], hand_w[:, 3:7], off_p, off_q)
        self.cable.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1))
        self.cable.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self.dev))

    def step(self, refs):
        """refs: {side: (T_ref, q_ref)}; arms without a ref hold their last reference."""
        for side, io in self.arms.items():
            q_ref = refs[side][1] if side in refs else self.hold[side]
            T_ref = refs[side][0] if side in refs else self.models[side].fk(q_ref)
            io.command(T_ref, q_ref)
            io.grip(self.grip_q[side])
        self._apply_attach()
        self.robot.write_data_to_sim()
        self.cable.write_data_to_sim()
        on_frame = self.step_count % self.capture_every == 0
        capture = bool(self.sinks) and on_frame
        # render at the capture cadence when recording, and also in a GUI run so the window stays live
        self.sim.step(render=on_frame and (capture or not args_cli.headless))
        self.robot.update(PHYSICS_DT)
        self.cable.update(PHYSICS_DT)
        self.step_count += 1
        if capture:
            self._capture()
        if self.ep_metrics is not None:
            self._track_metrics()

    def _capture(self):
        lines = [CAPTION, self.caption2 + f" | state {self.state_name}"]
        for name, cam in self.cams.items():
            cam.update(PHYSICS_DT * self.capture_every)
            out = cam.data.output
            img = (out["rgb"] if "rgb" in out else out["rgba"]).torch[0]
            frame = np.ascontiguousarray(img.cpu().numpy()[:, :, :3]).astype(np.uint8)
            self.sinks[name].add(self.overlay(frame, lines) if self.overlay else frame)

    def _track_metrics(self):
        bp = self.robot.data.body_pose_w.torch[0, :, 0:3]
        r, l = self.lr_bodies["right"], self.lr_bodies["left"]
        if r and l:
            dmin = float(torch.cdist(bp[r], bp[l]).min())
            self.ep_metrics["min_lr_body_origin_dist_m"] = min(self.ep_metrics["min_lr_body_origin_dist_m"], dmin)
        dh = float(torch.norm(bp[self.arms["right"].hand_id] - bp[self.arms["left"].hand_id]))
        self.ep_metrics["min_hand_dist_m"] = min(self.ep_metrics["min_hand_dist_m"], dh)

    def idle(self, seconds):
        for _ in range(int(round(seconds / PHYSICS_DT))):
            if not simulation_app.is_running():
                raise KeyboardInterrupt
            self.step({})

    # ---------------- states ----------------
    def run_state(self, name, motions, grip=None, after_s=0.0, tol=(0.004, 3.0), extra_s=1.5):
        """Execute joint-path motions (dict side -> Motion) concurrently, then hold. Returns a state log row."""
        self.state_name = name
        if grip:
            self.grip_q.update(grip)
        t0 = self.step_count * PHYSICS_DT
        dur = max([m.duration for m in motions.values()] + [0.0])
        margin_min = 1e9
        t = 0.0
        while True:
            if not simulation_app.is_running():
                raise KeyboardInterrupt
            refs = {}
            for side, m in motions.items():
                q_ref = m.q_at(t)
                refs[side] = (self.models[side].fk(q_ref), q_ref)
                self.hold[side] = q_ref
            self.step(refs)
            t += PHYSICS_DT
            for side in motions:
                margin_min = min(margin_min, self.models[side].margin_deg(self.arms[side].q()[0].cpu().numpy()))
            if t >= dur + after_s:
                errs = {s: pose_error(self.arms[s].tcp_T(), m.T_goal) for s, m in motions.items()}
                if all(e[0] < tol[0] and e[1] < tol[1] for e in errs.values()) or t >= dur + after_s + extra_s:
                    break
        row = {"state": name, "t_start_s": round(t0, 2), "duration_s": round(t, 2)}
        for s, m in motions.items():
            pe, re = pose_error(self.arms[s].tcp_T(), m.T_goal)
            q = self.arms[s].q()[0].cpu().numpy()
            row[s] = {"pos_err_mm": round(pe * 1e3, 1), "rot_err_deg": round(re, 1),
                      "end_margin_deg": round(self.models[s].margin_deg(q), 1),
                      "planned_min_margin_deg": round(min(self.models[s].margin_deg(qq) for qq in m.path), 1)}
            if pe > tol[0] * 3 or re > tol[1] * 3:
                row[s]["warning"] = "target not reached"
        if motions:
            row["min_margin_deg_actual"] = round(margin_min, 1)
        print(f"  [{name:<16}] " + json.dumps({k: v for k, v in row.items() if k not in ("state",)}))
        return row

    def gripper_state(self, name, side, width, seconds, attach=None):
        self.state_name = name
        self.grip_q[side] = width
        t0 = self.step_count * PHYSICS_DT
        self.idle(seconds)
        refused = None
        if attach == "attach" and args_cli.attach == "on":
            T_hand = self.arms[side].hand_T()
            T_cable = self.cable_T()
            # glue only if the grasp point is actually between the fingers (else a miss would 'pick' remotely)
            pt = self.P.cable.conn_center if side == "right" else self.P.cable.left_pt
            off = (inv_T(self.arms[side].tcp_T()) @ np.append(apply_T(T_cable, pt), 1.0))[:3]  # in TCP frame
            # tool y = closing axis, tool x = pad width, tool z = approach (rolled grasps sit nearer the fingertip)
            if abs(off[1]) <= 0.010 and abs(off[0]) <= 0.020 and abs(off[2]) <= 0.025:
                self.attached = (side, inv_T(T_hand) @ T_cable)
            else:
                refused = [round(float(v) * 1e3, 1) for v in off]
                print(f"  [{name:<16}] attach refused: grasp miss (grasp point in TCP frame {refused} mm)")
        elif attach == "detach" and self.attached is not None and self.attached[0] == side:
            self.attached = None
        row = {"state": name, "t_start_s": round(t0, 2), "duration_s": seconds, "attached_to": None if self.attached is None
               else self.attached[0]}
        if refused is not None:
            row["attach_refused_grasp_offset_mm"] = refused
        print(f"  [{name:<16}] " + json.dumps({k: v for k, v in row.items() if k != "state"}))
        return row

    def cable_T(self):
        d = self.cable.data
        root = self.robot.data.root_pose_w.torch
        pos, quat = subtract_frame_transforms(root[:, 0:3], root[:, 3:7], d.root_pos_w.torch, d.root_quat_w.torch)
        return make_T(R_from_quat_xyzw(quat[0].cpu().numpy()), pos[0].cpu().numpy())

    # ---------------- setup ----------------
    def write_limits_and_print(self):
        ids, lims = [], []
        for side, io in self.arms.items():
            ids += list(io.joint_ids)
            lims += io.lim.tolist()
        lim_t = torch.tensor([lims], dtype=torch.float32, device=self.dev)  # (1, 14, 2)
        self.robot.write_joint_position_limit_to_sim_index(limits=lim_t, joint_ids=ids, warn_limit_violation=False)
        pl = self.robot.data.joint_pos_limits.torch[0].cpu().numpy()
        print("[INFO] robot.data.joint_pos_limits after writing LeRobot limits (arm joints in deg, fingers in m):")
        for j, n in enumerate(self.robot.joint_names):
            if "finger" in n:
                print(f"    {n:<28} [{pl[j, 0]:+.4f}, {pl[j, 1]:+.4f}] m")
            else:
                print(f"    {n:<28} [{pl[j, 0] / DEG:+7.1f}, {pl[j, 1] / DEG:+7.1f}] deg")
        return {n: [float(pl[j, 0]), float(pl[j, 1])] for j, n in enumerate(self.robot.joint_names)}

    def teleport(self, q_by_side, fingers=FINGER_OPEN):
        q_all = self.robot.data.joint_pos.torch.clone()
        for side, q in q_by_side.items():
            q_all[0, self.arms[side].joint_ids] = torch.tensor(q, dtype=torch.float32, device=self.dev)
            if len(self.arms[side].finger_ids):
                q_all[0, self.arms[side].finger_ids] = fingers
        self.robot.write_joint_position_to_sim_index(position=q_all)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q_all))
        self.robot.set_joint_position_target_index(target=q_all)
        self.robot.reset()
        for side, q in q_by_side.items():
            self.hold[side] = np.asarray(q, dtype=float)

    def check_fingers(self):
        """Measure the finger-link origin gap in the USD and print the implied pad gap at q = 0 (STL pad offset)."""
        out = {}
        names = self.robot.body_names
        pos = self.robot.data.body_pos_w.torch[0].cpu().numpy()
        jp = self.robot.data.joint_pos.torch[0].cpu().numpy()
        halves = []
        for side, io in self.arms.items():
            fb = [i for i, n in enumerate(names) if n.startswith(f"openarm_{side}_") and "finger" in n]
            if len(fb) != 2 or not len(io.finger_ids):
                print(f"[WARN] {side}: finger bodies {[names[i] for i in fb]} - cannot measure the finger gap")
                continue
            R_h = self.arms[side].hand_T()[:3, :3]  # hand frame in root frame; root = world (fixed at origin)
            dy = abs(float((R_h.T @ (pos[fb[0]] - pos[fb[1]]))[1]))
            q = float(np.mean(jp[io.finger_ids]))
            half0 = dy / 2.0 - q
            halves.append(half0)
            out[side] = {"origin_gap_m": round(dy, 5), "finger_q_m": round(q, 5), "origin_half_gap_q0_m": round(half0, 5),
                         "pad_gap_q0_m": round(finger_pad_gap(0.0, half0), 5)}
            print(f"[INFO] {side} fingers: origin gap {dy * 1e3:.2f} mm at q={q * 1e3:.1f} mm -> origin half-gap at q=0 "
                  f"{half0 * 1e3:.2f} mm (URDF 5.00); pad gap at TCP height at q=0 ~ {finger_pad_gap(0.0, half0) * 1e3:+.1f} mm "
                  f"(finger.stl; negative = pads overlap = closes on anything)")
            for lbl, w in (("connector width", self.P.cable.conn_size[1]), ("cable diameter", self.P.cable.d)):
                if finger_pad_gap(0.0, half0) >= w:
                    print(f"[WARN] {side}: pad gap at q=0 >= {lbl} {w * 1e3:.1f} mm -> --attach off cannot hold the cable "
                          f"with this hand")
        if halves and abs(np.mean(halves) - FINGER_ORIGIN_HALF_GAP) < 0.004:
            self.finger_half_gap = float(np.mean(halves))
        elif halves:
            print("[WARN] measured finger origin half-gap far from the URDF 5 mm (different gripper/USD?); keeping 5 mm")
        return out

    def kinematic_settle(self, n=3):
        """Step without dynamics of interest (targets = state) so PhysX refreshes link poses/Jacobians."""
        for _ in range(n):
            self.robot.write_data_to_sim()
            self.sim.step(render=not args_cli.headless)
            self.robot.update(PHYSICS_DT)
            self.cable.update(PHYSICS_DT)

    def setup(self, rng):
        print(BANNER)
        print(f"[INFO] fixed base: {self.robot.is_fixed_base}  (num_base_dofs={self.robot.num_base_dofs})")
        if not self.robot.is_fixed_base:
            print("[WARN] articulation is NOT fixed-base: the robot base is only held by gravity-off; results may drift.")
        print(f"[INFO] joints: {self.robot.joint_names}")
        print(f"[INFO] bodies: {self.robot.body_names}")
        limits = self.write_limits_and_print()
        self.kinematic_settle(5)
        meta = {"limits_in_sim": limits, "models": {}, "fingers": self.check_fingers()}
        self.models = {}
        for side, io in self.arms.items():
            norms = io.build_model()
            self.models[side] = io.model
            m = io.model
            print(f"[INFO] {side}: PoE model from PhysX Jacobian; axis norms {np.round(norms, 4).tolist()} (expect 1.0)")
            for i in range(7):
                print(f"    j{i + 1}: axis {np.round(m.w[i], 3).tolist()} through {np.round(m.p[i], 4).tolist()}")
            # shoulder check vs URDF / workbench.yaml (j2 axis expected through (-0.0301, +-0.1535, 0.698), along x)
            exp_p = np.array([-0.0301, (1.0 if side == "left" else -1.0) * 0.1535, 0.698])
            w2, p2 = m.w[1], m.p[1]
            dist = float(np.linalg.norm(np.cross(exp_p - p2, w2)))
            print(f"    shoulder check: distance of URDF j2 point {exp_p.tolist()} to model j2 axis = {dist * 1e3:.1f} mm, "
                  f"|axis.x| = {abs(w2[0]):.3f} (URDF: 1.0)")
            meta["models"][side] = {"axes": m.w.round(5).tolist(), "points": m.p.round(5).tolist(),
                                    "shoulder_j2_dist_mm": round(dist * 1e3, 2)}
        self.q_ready = compute_ready(self.models, self.P, rng)
        self.teleport({"right": self.q_ready["right"], "left": self.q_ready["left"]})
        self.kinematic_settle(5)
        for side in ("right", "left"):
            pe, re = pose_error(self.arms[side].tcp_T(), self.models[side].fk(self.q_ready[side]))
            print(f"[INFO] READY {side}: q = {np.round(self.q_ready[side] / DEG, 1).tolist()} deg, "
                  f"model-vs-sim TCP error {pe * 1e3:.2f} mm / {re:.2f} deg")
            meta["models"][side]["ready_q_deg"] = (self.q_ready[side] / DEG).round(2).tolist()
            meta["models"][side]["ready_model_vs_sim"] = [round(pe * 1e3, 3), round(re, 3)]
        return meta

    # ---------------- episodes ----------------
    def reset_episode(self, drop):
        P, cg = self.P, self.P.cable
        self.attached = None
        self.grip_q = {"right": FINGER_OPEN, "left": FINGER_OPEN}
        self.teleport({"right": self.q_ready["right"], "left": self.q_ready["left"]})
        T = cg.rest_pose(drop["center_xy"], drop["body_yaw"], P.top_z)
        T[2, 3] = P.top_z + P.drop_h
        pose = torch.tensor([T[:3, 3].tolist() + quat_xyzw_from_R(T[:3, :3]).tolist()], dtype=torch.float32, device=self.dev)
        self.cable.write_root_pose_to_sim_index(root_pose=pose)
        self.cable.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self.dev))
        self.cable.reset()

    def episode(self, ep, seed):
        P = self.P
        rng = np.random.default_rng(seed)
        if args_cli.no_resample:
            drop, rejected = sample_drop(rng, P), []
        else:  # re-draw drops that the right arm cannot pick and bring to the handover under the LeRobot limits
            drop, rejected = sample_feasible_drop(rng, P, self.models, self.q_ready, P.max_drop_resample)
        self.caption2 = f"episode {ep} | seed {seed} | attach {args_cli.attach} | partition {args_cli.partition}"
        print(f"[EP {ep}] seed {seed}: drop connector at {np.round(drop['conn_xy'], 3).tolist()}, "
              f"body yaw {drop['body_yaw'] / DEG:+.1f} deg (re-drawn {len(rejected)}x for IK feasibility)")
        self.reset_episode(drop)
        self.ep_metrics = {"min_hand_dist_m": 1e9, "min_lr_body_origin_dist_m": 1e9}
        log = {"episode": ep, "seed": seed, "attach": args_cli.attach, "partition": args_cli.partition, "drop": drop,
               "drop_rejected_infeasible": len(rejected), "states": [], "success": False, "fail_reason": None}
        states = log["states"]
        R, L = self.models["right"], self.models["left"]
        ioR, ioL = self.arms["right"], self.arms["left"]
        spd = args_cli.speed

        def q_of(io):
            return io.q()[0].cpu().numpy().astype(float)

        try:
            self.state_name = "READY (cable settling)"
            self.idle(P.settle_s)
            T_c = self.cable_T()
            log["settled_cable_xyz"] = T_c[:3, 3].round(4).tolist()
            states.append({"state": "READY", "settle_s": P.settle_s})
            pick = plan_right_pick(R, q_of(ioR), T_c, P, rng, allow_roll=P.allow_roll)
            if not pick["ok"]:
                log["fail_reason"] = "R_APPROACH: no feasible grasp chain (IK under LeRobot limits)"
                return log
            log["grasp"] = {k: pick[k] for k in ("pitch_deg", "roll_deg", "approach_h", "lift_m")}
            log["grasp"]["planned_min_margin_deg"] = round(pick["margin"], 1)
            pre = FINGER_PRESHAPE if pick["roll_deg"] != 0.0 else FINGER_OPEN
            states.append(self.run_state("R_APPROACH", {"right": Motion(R, pick["approach"], spd)}, grip={"right": pre}))
            states.append(self.run_state("R_DESCEND", {"right": Motion(R, pick["descend"], spd)}, tol=(0.002, 2.0)))
            w = P.cable.conn_size[1]
            # attach off: q = 0 (pad target overlaps by ~5.5 mm -> full squeeze); attach on: pads 0.5 mm into each side
            close_r = 0.0 if args_cli.attach == "off" else finger_q_for_gap(w - 0.001, self.finger_half_gap)
            states.append(self.gripper_state("R_CLOSE", "right", close_r, 0.6, attach="attach"))
            lift = plan_right_lift(R, q_of(ioR), pick["T_grasp"], P)
            if not lift.ok:
                lift = pick["lift"]
            states.append(self.run_state("R_LIFT", {"right": Motion(R, lift, spd)}))
            if self.cable_T()[2, 3] < P.top_z + 0.03:
                log["fail_reason"] = "R_LIFT: cable not lifted (grasp slipped)"
                return log
            T_c_in_r = inv_T(ioR.tcp_T()) @ self.cable_T()  # measured (ground-truth) cable pose in the right TCP frame
            ho = plan_right_handover(R, q_of(ioR), T_c_in_r, P, rng)
            if not ho.ok:
                log["fail_reason"] = "R_TO_HANDOVER: IK"
                return log
            row = self.run_state("R_TO_HANDOVER", {"right": Motion(R, ho, spd)})
            row["planned"] = {k: ho.info.get(k) for k in ("min_z_above_table", "torso_clearance_m", "handover_x_shift_m")}
            states.append(row)
            lp = plan_left_pick(L, q_of(ioL), self.cable_T(), P, rng)
            if not lp["ok"]:
                log["fail_reason"] = "L_APPROACH: IK"
                return log
            states.append(self.run_state("L_APPROACH", {"left": Motion(L, [lp["approach"], lp["insert"]], spd)}, tol=(0.002, 2.0)))
            close_l = 0.0 if args_cli.attach == "off" else finger_q_for_gap(P.cable.d - 0.001, self.finger_half_gap)
            states.append(self.gripper_state("L_CLOSE", "left", close_l, 0.6, attach="attach"))
            self.state_name = "HOLD"
            self.idle(0.5)
            states.append({"state": "HOLD", "duration_s": 0.5})
            states.append(self.gripper_state("R_OPEN", "right", FINGER_OPEN, 0.5, attach="detach"))
            a, b = plan_retreat(R, q_of(ioR), self.q_ready["right"], P.retreats, 0.0, rng)
            row = self.run_state("R_RETREAT", {"right": Motion(R, [a, b], spd)})
            row["back_m"] = a.info.get("back_m")
            states.append(row)
            p_lg = apply_T(self.cable_T(), P.cable.left_pt)
            if np.linalg.norm(p_lg - ioL.tcp_T()[:3, 3]) > 0.03:
                log["fail_reason"] = "HANDOVER: cable not in the left hand"
                return log
            T_c_in_l = inv_T(ioL.tcp_T()) @ self.cable_T()
            pl = plan_left_place(L, q_of(ioL), T_c_in_l, P, rng)
            if not pl["ok"]:
                log["fail_reason"] = "L_TO_BOX: IK"
                return log
            log["place"] = {"label": pl["label"], "planned_min_margin_deg": round(pl["margin"], 1),
                            "planned_min_z_above_table": round(pl["min_z_above_table"], 3)}
            states.append(self.run_state("L_TO_BOX", {"left": Motion(L, [pl["to_above"], pl["descend"]], spd)}, tol=(0.003, 2.0)))
            states.append(self.gripper_state("L_OPEN", "left", FINGER_OPEN, 0.5, attach="detach"))
            a, b = plan_retreat(L, q_of(ioL), self.q_ready["left"], P.left_retreats, 0.06, rng)
            states.append(self.run_state("L_RETREAT", {"left": Motion(L, [a, b], spd)}))
            self.state_name = "DONE"
            self.idle(2.0)
            states.append({"state": "DONE"})
        finally:
            self.attached = None
            T_c = self.cable_T()
            (ix0, ix1), (iy0, iy1) = P.box_inner()
            c = T_c[:3, 3]
            inset = 0.005
            inside = bool(ix0 + inset <= c[0] <= ix1 - inset and iy0 + inset <= c[1] <= iy1 - inset)
            below = bool(P.top_z < c[2] < P.top_z + P.wall_h)
            log["final_cable_center_xyz"] = c.round(4).tolist()
            log["success_detail"] = {"inside_xy_inset_5mm": inside, "below_rim": below}
            log["success"] = bool(inside and below and log["fail_reason"] is None)
            log["min_hand_dist_m"] = round(self.ep_metrics["min_hand_dist_m"], 4)
            log["min_lr_body_origin_dist_m"] = round(self.ep_metrics["min_lr_body_origin_dist_m"], 4)
            margins = [s.get("min_margin_deg_actual") for s in states if isinstance(s, dict) and "min_margin_deg_actual" in s]
            log["min_joint_margin_deg"] = min(margins) if margins else None
            self.ep_metrics = None
        return log

    def dry_run(self, n_eps, base_seed, teleport_check):
        rows = []
        for ep in range(n_eps):
            seed = base_seed + ep
            rng = np.random.default_rng(seed)
            if args_cli.no_resample:  # same sampler as episode() so the dry run describes the episodes that will play
                drop, rejected = sample_drop(rng, self.P), []
            else:
                drop, rejected = sample_feasible_drop(rng, self.P, self.models, self.q_ready, self.P.max_drop_resample)
            T_c = self.P.cable.rest_pose(drop["center_xy"], drop["body_yaw"], self.P.top_z)
            out, ok = plan_episode_nominal(self.models, self.P, T_c, self.q_ready, rng)
            print(f"\n[DRY {ep}] seed {seed} conn {np.round(drop['conn_xy'], 3).tolist()} yaw {drop['body_yaw'] / DEG:+.1f} deg "
                  f"(re-drawn {len(rejected)}x) -> {'ALL WAYPOINTS FEASIBLE' if ok else 'INFEASIBLE'}")
            print(f"    {'state':<15}{'side':<7}{'ok':<6}{'pos mm':>8}{'rot deg':>9}{'end mrg':>9}{'min mrg':>9}  target xyz / notes")
            for r in out:
                note = {k: v for k, v in r.items() if k in ("grasp_pitch_deg", "grasp_roll_deg", "approach_h", "lift_m", "place",
                                                             "min_z_above_table", "torso_clearance_m", "handover_x_shift_m",
                                                             "hand_origin_dist_m", "tcp_dist_m", "release_cable_xyz")}
                print(f"    {r['state']:<15}{r['side']:<7}{str(r['ok']):<6}{r.get('pos_err_mm', float('nan')):>8}"
                      f"{r.get('rot_err_deg', float('nan')):>9}{r.get('end_margin_deg', float('nan')):>9}"
                      f"{r.get('min_margin_deg', float('nan')):>9}  {r.get('target_xyz')} {note if note else ''}")
                if teleport_check and r.get("q_end_deg") is not None:
                    side = r["side"]
                    q = np.array(r["q_end_deg"]) * DEG
                    self.teleport({side: q})
                    self.kinematic_settle(2)
                    pe, re = pose_error(self.arms[side].tcp_T(), self.models[side].fk(q))
                    r["teleport_model_vs_sim"] = [round(pe * 1e3, 3), round(re, 3)]
                    if pe > 2e-3 or re > 1.0:
                        print(f"      [WARN] teleport check {side}: model vs sim TCP {pe * 1e3:.2f} mm / {re:.2f} deg")
            self.teleport({"right": self.q_ready["right"], "left": self.q_ready["left"]})
            self.kinematic_settle(2)
            rows.append({"episode": ep, "seed": seed, "drop": drop, "drop_rejected_infeasible": len(rejected),
                         "all_feasible": ok, "waypoints": out})
        return rows


def main():
    usd_path = USD_PATH  # checked before AppLauncher (_resolve_inputs)
    with open(WORKBENCH_PATH, encoding="utf-8") as f:
        wb = yaml.safe_load(f)
    P = Params(wb)
    P.yaw_range = (args_cli.yaw_min_deg * DEG, args_cli.yaw_max_deg * DEG)
    pref = (args_cli.grasp_pitch_deg * DEG, 0.0)
    P.grasp_cands = [pref] + [c for c in P.grasp_cands if abs(c[0] - pref[0]) > 1e-6 or c[1] != 0.0]
    P.allow_roll = args_cli.attach == "on"  # rolled grasps put the connector at the fingertip: attach mode only
    P.place_mode = args_cli.place_mode
    P.place_elevs = [args_cli.place_elev_deg * DEG] + [e for e in (60 * DEG, 75 * DEG, 45 * DEG) if abs(e - args_cli.place_elev_deg * DEG) > 1e-6]
    P.place_azim = None if args_cli.place_azim_deg is None else args_cli.place_azim_deg * DEG
    partition_on = args_cli.partition == "on"
    if partition_on != bool((wb.get("partition") or {}).get("sim_demo", False)):
        print(f"[INFO] --partition {args_cli.partition} (workbench partition.sim_demo = {(wb.get('partition') or {}).get('sim_demo')})")

    out_dir = os.path.expanduser(args_cli.out_dir or os.path.join("~/openarm-out/demo", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] output dir: {out_dir}")

    sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.25, -0.85, 1.05], [0.28, 0.0, 0.42])
    robot, cable, cams = design_scene(P, wb, usd_path, partition_on, video=(args_cli.record_video == "on"))
    sim.reset()
    print("[INFO]: Setup complete...")
    if cams:
        dev = sim.device
        cams["oblique"].set_world_poses_from_view(torch.tensor([[1.25, -0.85, 1.05]], device=dev),
                                                  torch.tensor([[0.28, 0.0, 0.42]], device=dev))
        if "top" in cams:
            cams["top"].set_world_poses_from_view(torch.tensor([[0.27, 0.0, 1.55]], device=dev),
                                                  torch.tensor([[0.33, 0.0, P.top_z]], device=dev))

    demo = Demo(sim, robot, cable, cams, P, out_dir)
    meta = {"args": {k: v for k, v in vars(args_cli).items() if isinstance(v, (str, int, float, bool, type(None)))},
            "usd": usd_path, "banner": BANNER, "caption": CAPTION, "physics_dt": PHYSICS_DT,
            "cable_mass_kg": P.cable_mass, "cable_geom": {"length": P.cable.L, "d": P.cable.d, "connector": P.cable.conn_size,
                                                          "left_grasp_from_connector_m": P.cable.left_dist}}
    # READY uses a FIXED seed (independent of --seed) so episode i of '--seed 0' == episode 0 of '--seed i'.
    rng = np.random.default_rng(READY_SEED)
    try:
        meta.update(demo.setup(rng))
        with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, default=str)
        if args_cli.dry_run_kinematics:
            rows = demo.dry_run(args_cli.episodes, args_cli.seed, not args_cli.no_teleport_check)
            with open(os.path.join(out_dir, "dry_run.jsonl"), "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, default=str) + "\n")
            n_ok = sum(r["all_feasible"] for r in rows)
            print(f"\n[DRY RUN] {n_ok}/{len(rows)} seeds have every waypoint IK-feasible under LeRobot limits -> {out_dir}/dry_run.jsonl")
            return
        if cams:
            for _ in range(20):  # RTX warm-up so the first video frames are not black
                sim.render()
        results = []
        with open(os.path.join(out_dir, "episodes.jsonl"), "w", encoding="utf-8") as f:
            for ep in range(args_cli.episodes):
                if not simulation_app.is_running():
                    break
                t_wall = time.time()
                try:
                    log = demo.episode(ep, args_cli.seed + ep)
                except KeyboardInterrupt:
                    raise
                except Exception as e:  # keep the remaining episodes (and the video) going
                    traceback.print_exc()
                    demo.attached, demo.ep_metrics = None, None
                    log = {"episode": ep, "seed": args_cli.seed + ep, "success": False, "fail_reason": f"exception: {e!r}",
                           "min_hand_dist_m": None, "min_joint_margin_deg": None}
                log["wall_s"] = round(time.time() - t_wall, 1)
                results.append(log)
                f.write(json.dumps(log, default=str) + "\n")
                f.flush()
                print(f"[EP {ep}] success={log['success']} reason={log['fail_reason']} final={log.get('final_cable_center_xyz')} "
                      f"min_hand_dist={log['min_hand_dist_m']} min_margin={log['min_joint_margin_deg']}")
        n_ok = sum(r["success"] for r in results)
        print(f"\n[SUMMARY] success {n_ok}/{len(results)} (attach={args_cli.attach}) -> {out_dir}/episodes.jsonl")
        print(CAPTION)
    except KeyboardInterrupt:
        print("[INFO] interrupted")
    finally:
        for s in demo.sinks.values():
            s.close()


if __name__ == "__main__":
    try:
        main()
    finally:  # always close Kit, also on sys.exit / exceptions (an unclosed Kit app can hang at shutdown)
        simulation_app.close()
