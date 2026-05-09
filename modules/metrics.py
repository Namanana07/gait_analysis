"""
Biomechanical metrics in treadmill (ChArUco board) coordinates:
joint angles, foot clearance, step length, and cadence.
Angles are reported in degrees; distances in meters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .pose_estimation import KEYPOINT_INDEX
from .reconstruction import Pose3D
from .calibration import point_to_plane_distance


def _unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two vectors (0–180)."""
    ua, ub = _unit(a), _unit(b)
    c = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def build_treadmill_frame(
    forward_hint: np.ndarray,
    world_up: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orthonormal basis (forward, left, up) in world(board) frame.
    forward_hint: rough walking direction (3,) in world coords
    world_up: unit normal of belt plane pointing away from belt surface
    """
    u = _unit(world_up)
    f = _unit(np.asarray(forward_hint, dtype=np.float64) - np.dot(forward_hint, u) * u)
    if np.linalg.norm(f) < 1e-6:
        f = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        f = _unit(f - np.dot(f, u) * u)
        if np.linalg.norm(f) < 1e-6:
            f = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            f = _unit(f - np.dot(f, u) * u)
    left = _unit(np.cross(u, f))
    forward = _unit(np.cross(left, u))
    return forward, left, u


def trunk_pitch_deg_in_sagittal(trunk_vec: np.ndarray, forward: np.ndarray, up: np.ndarray) -> float:
    """Upper-body pitch: deviation of trunk from vertical within sagittal plane."""
    left = _unit(np.cross(up, forward))
    trunk_proj = trunk_vec - np.dot(trunk_vec, left) * left
    pitch = _angle_deg(trunk_proj, up)
    sign = np.sign(np.dot(trunk_proj, forward))
    return pitch * sign


def hip_abduction_deg(thigh_vec: np.ndarray, forward: np.ndarray, up: np.ndarray) -> float:
    """Approximate hip abduction+: lateral deviation of thigh out of nominal sagittal plane."""
    left = _unit(np.cross(up, forward))
    thigh_horiz = thigh_vec - np.dot(thigh_vec, up) * up
    thigh_fwd_comp = np.dot(thigh_horiz, forward)
    thigh_lat_comp = np.dot(thigh_horiz, left)
    lat_plane_norm = np.sqrt(max(thigh_fwd_comp ** 2 + thigh_lat_comp ** 2, 1e-12))
    return float(np.degrees(np.arcsin(np.clip(thigh_lat_comp / lat_plane_norm, -1.0, 1.0))))


def hip_flexion_deg(thigh_vec: np.ndarray, torso_vec: np.ndarray, forward: np.ndarray, up: np.ndarray) -> float:
    """Hip flex/extension in sagittal plane (180 - angle between thigh & torso projections)."""
    left = _unit(np.cross(up, forward))
    def proj_sag(v):
        return v - np.dot(v, left) * left
    t_th = proj_sag(thigh_vec)
    t_to = proj_sag(torso_vec)
    ang = _angle_deg(t_th, t_to)
    flex = 180.0 - ang
    return flex


def knee_flexion_deg(thigh_vec: np.ndarray, shank_vec: np.ndarray) -> float:
    return 180.0 - _angle_deg(thigh_vec, shank_vec)


def ankle_angle_deg(shank_vec: np.ndarray, foot_vec: np.ndarray, forward: np.ndarray, up: np.ndarray) -> float:
    """Plantarflexion+: angle between shank and foot projections in sagittal plane."""
    left = _unit(np.cross(up, forward))

    def proj_sag(v):
        return v - np.dot(v, left) * left

    return _angle_deg(proj_sag(shank_vec), proj_sag(foot_vec))


@dataclass
class FrameMetrics:
    upper_body_pitch_deg: float = np.nan
    hip_adduction_deg: float = np.nan
    hip_flexion_deg: float = np.nan
    knee_flexion_deg: float = np.nan
    ankle_angle_deg: float = np.nan
    foot_clearance_m: float = np.nan
    step_length_m: float = np.nan
    step_frequency_spm: float = np.nan


@dataclass
class GaitTracker:
    """Heel-strike based step length and cadence; peak foot clearance during swing."""

    belt_normal_world: np.ndarray
    belt_point_world: np.ndarray
    forward: np.ndarray
    airborne_thresh_m: float = 0.04
    contact_thresh_m: float = 0.018

    last_h_l: Optional[float] = None
    last_h_r: Optional[float] = None

    airborne_l: bool = False
    airborne_r: bool = False

    swing_peak_clearance_m: float = 0.0

    heel_strike_times: List[float] = field(default_factory=list)
    last_fwd_at_hs_l: Optional[float] = None
    last_fwd_at_hs_r: Optional[float] = None

    last_step_length_m: float = np.nan
    last_step_frequency_spm: float = np.nan
    last_completed_swing_peak_m: float = np.nan

    def _best_clearance_m(self, pose: Pose3D, ids: List[int]) -> float:
        vals = []
        for i in ids:
            if pose.valid_mask[i]:
                vals.append(
                    point_to_plane_distance(
                        pose.keypoints_3d[i],
                        self.belt_normal_world,
                        self.belt_point_world,
                    )
                )
        return float(max(vals)) if vals else np.nan

    def _forward_coord(self, p: np.ndarray) -> float:
        return float(np.dot(p, self.forward))

    def update(self, pose: Pose3D, time_s: float) -> Tuple[float, float]:
        """
        Update internal gait state; returns (left_clearance_m, right_clearance_m).
        """
        left_ids = [
            KEYPOINT_INDEX["left_heel"],
            KEYPOINT_INDEX["left_big_toe"],
            KEYPOINT_INDEX["left_ankle"],
        ]
        right_ids = [
            KEYPOINT_INDEX["right_heel"],
            KEYPOINT_INDEX["right_big_toe"],
            KEYPOINT_INDEX["right_ankle"],
        ]
        h_l = self._best_clearance_m(pose, left_ids)
        h_r = self._best_clearance_m(pose, right_ids)

        def step_side(side: str, h: float, prev_h: Optional[float], airborne: bool) -> Tuple[Optional[float], bool]:
            if np.isnan(h):
                return prev_h, airborne
            if h > self.airborne_thresh_m:
                airborne = True
            if airborne:
                self.swing_peak_clearance_m = max(self.swing_peak_clearance_m, h)
            if prev_h is not None and not np.isnan(prev_h):
                contacts = h < self.contact_thresh_m and prev_h >= self.contact_thresh_m
                if airborne and contacts:
                    self._heel_strike(side, time_s, pose)
                    airborne = False
                    self.swing_peak_clearance_m = 0.0
            return h, airborne

        self.last_h_l, self.airborne_l = step_side("L", h_l, self.last_h_l, self.airborne_l)
        self.last_h_r, self.airborne_r = step_side("R", h_r, self.last_h_r, self.airborne_r)

        return h_l, h_r

    def _hip_forward(self, pose: Pose3D, side: str) -> Optional[float]:
        hi = KEYPOINT_INDEX["right_hip"] if side == "R" else KEYPOINT_INDEX["left_hip"]
        if not pose.valid_mask[hi]:
            return None
        return self._forward_coord(pose.keypoints_3d[hi])

    def _heel_strike(self, side: str, t: float, pose: Pose3D) -> None:
        self.last_completed_swing_peak_m = self.swing_peak_clearance_m
        self.heel_strike_times.append(t)
        fwd = self._hip_forward(pose, side)
        if fwd is None:
            self._update_cadence()
            return
        if side == "L" and self.last_fwd_at_hs_r is not None:
            self.last_step_length_m = abs(fwd - self.last_fwd_at_hs_r)
        elif side == "R" and self.last_fwd_at_hs_l is not None:
            self.last_step_length_m = abs(fwd - self.last_fwd_at_hs_l)
        if side == "L":
            self.last_fwd_at_hs_l = fwd
        else:
            self.last_fwd_at_hs_r = fwd
        self._update_cadence()

    def _update_cadence(self) -> None:
        hs = self.heel_strike_times
        if len(hs) < 2:
            return
        dt = hs[-1] - hs[-2]
        if dt > 1e-6:
            self.last_step_frequency_spm = 60.0 / dt


def _limb_metrics(
    pose: Pose3D,
    hi: int,
    ki: int,
    ai: int,
    ti: int,
    hi_opp: int,
    belt_normal_world: np.ndarray,
    belt_point_world: np.ndarray,
    fwd: np.ndarray,
    up: np.ndarray,
    mid_sh: Optional[np.ndarray],
) -> Optional[FrameMetrics]:
    m = FrameMetrics()
    if not (pose.valid_mask[hi] and pose.valid_mask[ki] and pose.valid_mask[ai]):
        return None
    hip = pose.keypoints_3d[hi]
    knee = pose.keypoints_3d[ki]
    ankle = pose.keypoints_3d[ai]
    thigh = knee - hip
    shank = ankle - knee
    toe = pose.keypoints_3d[ti] if pose.valid_mask[ti] else ankle
    foot = toe - ankle
    if mid_sh is not None:
        torso_ref = mid_sh - hip
    elif pose.valid_mask[hi_opp]:
        torso_ref = hip - pose.keypoints_3d[hi_opp]
    else:
        torso_ref = up

    m.hip_adduction_deg = hip_abduction_deg(thigh, fwd, up)
    m.hip_flexion_deg = hip_flexion_deg(thigh, torso_ref, fwd, up)
    m.knee_flexion_deg = knee_flexion_deg(thigh, shank)
    m.ankle_angle_deg = ankle_angle_deg(shank, foot, fwd, up)

    heel = KEYPOINT_INDEX["right_heel"] if hi == KEYPOINT_INDEX["right_hip"] else KEYPOINT_INDEX["left_heel"]
    clears = []
    for i in (ai, ti, heel):
        if pose.valid_mask[i]:
            clears.append(
                point_to_plane_distance(
                    pose.keypoints_3d[i],
                    np.asarray(belt_normal_world, dtype=np.float64),
                    np.asarray(belt_point_world, dtype=np.float64),
                )
            )
    if clears:
        m.foot_clearance_m = float(max(clears))
    return m


def compute_frame_metrics(
    pose: Pose3D,
    belt_normal_world: np.ndarray,
    belt_point_world: np.ndarray,
    forward_hint: np.ndarray,
    bilateral_average: bool = True,
) -> FrameMetrics:
    """
    Joint metrics in world/board frame.
    By default averages left/right limb angles; foot clearance is max of both feet.
    """
    m = FrameMetrics()
    up = _unit(np.asarray(belt_normal_world, dtype=np.float64))
    fwd, left, _u = build_treadmill_frame(forward_hint, up)
    del left, _u

    LS, RS = KEYPOINT_INDEX["left_shoulder"], KEYPOINT_INDEX["right_shoulder"]
    LH, RH = KEYPOINT_INDEX["left_hip"], KEYPOINT_INDEX["right_hip"]
    LK, RK = KEYPOINT_INDEX["left_knee"], KEYPOINT_INDEX["right_knee"]
    LA, RA = KEYPOINT_INDEX["left_ankle"], KEYPOINT_INDEX["right_ankle"]
    LT, RT = KEYPOINT_INDEX["left_big_toe"], KEYPOINT_INDEX["right_big_toe"]

    def mid(i, j):
        if pose.valid_mask[i] and pose.valid_mask[j]:
            return (pose.keypoints_3d[i] + pose.keypoints_3d[j]) * 0.5
        return None

    mid_sh = mid(LS, RS)
    mid_hip = mid(LH, RH)
    if mid_sh is not None and mid_hip is not None:
        trunk = mid_sh - mid_hip
        m.upper_body_pitch_deg = trunk_pitch_deg_in_sagittal(trunk, fwd, up)

    def run_side(hi, ki, ai, ti, opp):
        return _limb_metrics(
            pose,
            hi,
            ki,
            ai,
            ti,
            opp,
            belt_normal_world,
            belt_point_world,
            fwd,
            up,
            mid_sh,
        )

    vals = []
    rl = run_side(RH, RK, RA, RT, LH)
    ll = run_side(LH, LK, LA, LT, RH)
    for v in (ll, rl):
        if v is not None:
            vals.append(v)
    if not vals:
        return m
    src = vals[0] if len(vals) == 1 or not bilateral_average else None
    if src is not None:
        m.hip_adduction_deg = src.hip_adduction_deg
        m.hip_flexion_deg = src.hip_flexion_deg
        m.knee_flexion_deg = src.knee_flexion_deg
        m.ankle_angle_deg = src.ankle_angle_deg
        m.foot_clearance_m = src.foot_clearance_m
        return m

    m.hip_adduction_deg = float(np.nanmean([v.hip_adduction_deg for v in vals]))
    m.hip_flexion_deg = float(np.nanmean([v.hip_flexion_deg for v in vals]))
    m.knee_flexion_deg = float(np.nanmean([v.knee_flexion_deg for v in vals]))
    m.ankle_angle_deg = float(np.nanmean([v.ankle_angle_deg for v in vals]))

    clears = [v.foot_clearance_m for v in vals if not np.isnan(v.foot_clearance_m)]
    m.foot_clearance_m = float(max(clears)) if clears else np.nan

    return m
