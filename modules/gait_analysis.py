"""
Gait Analysis Module.
Computes biomechanical parameters from 3D joint positions:
- Upper body pitch angle (上半身俯仰角)
- Hip adduction/abduction angle (髋关节内收外展角度)
- Hip flexion/extension angle (髋关节屈伸角度)
- Knee flexion/extension angle (膝关节屈伸角度)
- Ankle angle (踝关节角度)
- Maximum foot clearance (足最大离地高度)
- Step length (步长)
- Cadence (步频)
"""

import logging
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from scipy import signal
from scipy.signal import butter, filtfilt, savgol_filter

from .reconstruction import Pose3D
from .pose_estimation import KEYPOINT_INDEX, NUM_KEYPOINTS
from .calibration import point_to_plane_distance

logger = logging.getLogger(__name__)


@dataclass
class GaitParameters:
    """Gait parameters for a single frame."""
    timestamp_us: int = 0
    # Angles in degrees
    upper_body_pitch: float = np.nan        # 上半身俯仰角
    hip_adduction_abduction: float = np.nan # 髋关节内收外展角度
    hip_flexion_extension: float = np.nan   # 髋关节屈伸角度
    knee_flexion_extension: float = np.nan  # 膝关节屈伸角度
    ankle_angle: float = np.nan             # 踝关节角度
    foot_clearance: float = np.nan          # 足离地高度 (meters)
    step_length: float = np.nan             # 步长 (meters)
    cadence: float = np.nan                 # 步频 (steps/min)


@dataclass
class GaitCycleData:
    """Aggregated data for one gait cycle."""
    cycle_index: int = 0
    start_frame: int = 0
    end_frame: int = 0
    start_time_us: int = 0
    end_time_us: int = 0
    side: str = "right"  # Which leg initiated the cycle
    # Peak/mean values
    upper_body_pitch_mean: float = np.nan
    hip_adduction_abduction_max: float = np.nan
    hip_flexion_extension_max: float = np.nan
    knee_flexion_extension_max: float = np.nan
    ankle_angle_max: float = np.nan
    foot_clearance_max: float = np.nan
    step_length: float = np.nan
    cadence: float = np.nan


@dataclass
class GaitEvent:
    """A gait event (heel strike or toe off)."""
    frame_index: int
    timestamp_us: int
    event_type: str  # "heel_strike" or "toe_off"
    side: str  # "left" or "right"
    foot_position: np.ndarray = field(default_factory=lambda: np.zeros(3))


class GaitAnalyzer:
    """
    Analyzes gait biomechanics from 3D pose sequences.
    """

    def __init__(self, config: dict, fps: float = 30.0):
        """
        Args:
            config: Gait analysis configuration
            fps: Video frame rate
        """
        self.config = config
        self.fps = fps
        self.dt = 1.0 / fps

        # Filter settings
        filter_config = config.get("filter", {})
        self.filter_enabled = filter_config.get("enabled", True)
        self.filter_type = filter_config.get("type", "butterworth")
        self.cutoff_freq = filter_config.get("cutoff_freq", 6.0)
        self.filter_order = filter_config.get("order", 4)
        self.savgol_window = filter_config.get("savgol_window", 11)
        self.savgol_polyorder = filter_config.get("savgol_polyorder", 3)

        # Event detection
        event_config = config.get("event_detection", {})
        self.event_method = event_config.get("method", "velocity")
        self.velocity_threshold = event_config.get("velocity_threshold", 0.05)
        self.height_threshold = event_config.get("height_threshold", 0.02)

        # Which side to analyze
        self.analyze_side = config.get("analyze_side", "both")

        # Belt plane parameters (set after calibration)
        self.belt_plane_normal = np.array([0, 0, 1])
        self.belt_plane_point = np.array([0, 0, 0])

        # Data storage
        self._pose_history: List[Pose3D] = []
        self._timestamps: List[int] = []
        self._gait_events: List[GaitEvent] = []

    def set_belt_plane(self, normal: np.ndarray, point: np.ndarray):
        """Set the treadmill belt plane for foot clearance calculation."""
        self.belt_plane_normal = normal / np.linalg.norm(normal)
        self.belt_plane_point = point

    def add_frame(self, pose: Pose3D, timestamp_us: int):
        """Add a frame's pose data for analysis."""
        self._pose_history.append(pose)
        self._timestamps.append(timestamp_us)

    def compute_frame_parameters(self, pose: Pose3D, 
                                  prev_pose: Optional[Pose3D] = None
                                  ) -> GaitParameters:
        """
        Compute all gait parameters for a single frame.
        
        Args:
            pose: Current frame's 3D pose
            prev_pose: Previous frame's 3D pose (for velocity-based metrics)
        Returns:
            GaitParameters for this frame
        """
        params = GaitParameters()
        kp = pose.keypoints_3d
        valid = pose.valid_mask

        # === Upper Body Pitch Angle (上半身俯仰角) ===
        params.upper_body_pitch = self._compute_upper_body_pitch(kp, valid)

        # === Hip Angles (use dominant/analyzed side) ===
        if self.analyze_side in ["right", "both"]:
            params.hip_adduction_abduction = self._compute_hip_adduction(
                kp, valid, side="right"
            )
            params.hip_flexion_extension = self._compute_hip_flexion(
                kp, valid, side="right"
            )
            params.knee_flexion_extension = self._compute_knee_flexion(
                kp, valid, side="right"
            )
            params.ankle_angle = self._compute_ankle_angle(
                kp, valid, side="right"
            )
            params.foot_clearance = self._compute_foot_clearance(
                kp, valid, side="right"
            )
        elif self.analyze_side == "left":
            params.hip_adduction_abduction = self._compute_hip_adduction(
                kp, valid, side="left"
            )
            params.hip_flexion_extension = self._compute_hip_flexion(
                kp, valid, side="left"
            )
            params.knee_flexion_extension = self._compute_knee_flexion(
                kp, valid, side="left"
            )
            params.ankle_angle = self._compute_ankle_angle(
                kp, valid, side="left"
            )
            params.foot_clearance = self._compute_foot_clearance(
                kp, valid, side="left"
            )

        return params

    def _compute_upper_body_pitch(self, kp: np.ndarray, 
                                   valid: np.ndarray) -> float:
        """
        Compute upper body pitch angle.
        Defined as the angle between the trunk vector (hip_center -> neck)
        and the vertical axis (belt plane normal).
        
        Positive = leaning forward, Negative = leaning backward.
        """
        neck_idx = KEYPOINT_INDEX["neck"]
        hip_idx = KEYPOINT_INDEX["hip_center"]

        if not (valid[neck_idx] and valid[hip_idx]):
            # Try shoulder midpoint and hip midpoint
            ls = KEYPOINT_INDEX["left_shoulder"]
            rs = KEYPOINT_INDEX["right_shoulder"]
            lh = KEYPOINT_INDEX["left_hip"]
            rh = KEYPOINT_INDEX["right_hip"]
            
            if valid[ls] and valid[rs] and valid[lh] and valid[rh]:
                neck_pos = (kp[ls] + kp[rs]) / 2
                hip_pos = (kp[lh] + kp[rh]) / 2
            else:
                return np.nan
        else:
            neck_pos = kp[neck_idx]
            hip_pos = kp[hip_idx]

        # Trunk vector (pointing upward from hip to neck)
        trunk_vec = neck_pos - hip_pos
        trunk_vec_norm = np.linalg.norm(trunk_vec)
        if trunk_vec_norm < 1e-6:
            return np.nan

        trunk_vec = trunk_vec / trunk_vec_norm

        # Vertical axis (belt plane normal, pointing up)
        vertical = self.belt_plane_normal

        # Pitch angle = angle from vertical
        cos_angle = np.clip(np.dot(trunk_vec, vertical), -1, 1)
        angle = np.degrees(np.arccos(cos_angle))

        # Convention: 0 = perfectly upright, positive = forward lean
        # The trunk should normally be close to vertical (small angle)
        pitch = 90.0 - angle  # Convert so 0 = upright

        return pitch

    def _compute_hip_adduction(self, kp: np.ndarray, valid: np.ndarray,
                                side: str = "right") -> float:
        """
        Compute hip adduction/abduction angle.
        Measured in the frontal plane.
        Positive = adduction, Negative = abduction.
        """
        if side == "right":
            hip_idx = KEYPOINT_INDEX["right_hip"]
            knee_idx = KEYPOINT_INDEX["right_knee"]
            opp_hip_idx = KEYPOINT_INDEX["left_hip"]
        else:
            hip_idx = KEYPOINT_INDEX["left_hip"]
            knee_idx = KEYPOINT_INDEX["left_knee"]
            opp_hip_idx = KEYPOINT_INDEX["right_hip"]

        if not (valid[hip_idx] and valid[knee_idx] and valid[opp_hip_idx]):
            return np.nan

        hip = kp[hip_idx]
        knee = kp[knee_idx]
        opp_hip = kp[opp_hip_idx]

        # Define frontal plane using hip-to-hip vector and vertical
        hip_axis = opp_hip - hip  # Medial-lateral direction
        hip_axis_norm = np.linalg.norm(hip_axis)
        if hip_axis_norm < 1e-6:
            return np.nan
        
        # Vertical axis
        vertical = self.belt_plane_normal

        # Sagittal normal (perpendicular to frontal plane)
        sagittal_normal = np.cross(hip_axis, vertical)
        sagittal_normal = sagittal_normal / (np.linalg.norm(sagittal_normal) + 1e-8)

        # Thigh vector
        thigh = knee - hip
        thigh_norm = np.linalg.norm(thigh)
        if thigh_norm < 1e-6:
            return np.nan

        # Project thigh onto frontal plane
        thigh_frontal = thigh - np.dot(thigh, sagittal_normal) * sagittal_normal
        thigh_frontal_norm = np.linalg.norm(thigh_frontal)
        if thigh_frontal_norm < 1e-6:
            return np.nan
        thigh_frontal = thigh_frontal / thigh_frontal_norm

        # Angle between projected thigh and downward vertical
        down = -vertical
        cos_angle = np.clip(np.dot(thigh_frontal, down), -1, 1)
        angle = np.degrees(np.arccos(cos_angle))

        # Determine sign: adduction (toward midline) is positive
        cross = np.cross(down, thigh_frontal)
        sign = np.sign(np.dot(cross, sagittal_normal))
        
        if side == "right":
            return -sign * angle  # Right leg: medial = positive
        else:
            return sign * angle   # Left leg: medial = positive

    def _compute_hip_flexion(self, kp: np.ndarray, valid: np.ndarray,
                              side: str = "right") -> float:
        """
        Compute hip flexion/extension angle.
        Measured in the sagittal plane.
        Positive = flexion (thigh forward), Negative = extension (thigh backward).
        """
        if side == "right":
            hip_idx = KEYPOINT_INDEX["right_hip"]
            knee_idx = KEYPOINT_INDEX["right_knee"]
            opp_hip_idx = KEYPOINT_INDEX["left_hip"]
        else:
            hip_idx = KEYPOINT_INDEX["left_hip"]
            knee_idx = KEYPOINT_INDEX["left_knee"]
            opp_hip_idx = KEYPOINT_INDEX["right_hip"]

        neck_idx = KEYPOINT_INDEX["neck"]
        hip_center_idx = KEYPOINT_INDEX["hip_center"]

        if not (valid[hip_idx] and valid[knee_idx]):
            return np.nan

        hip = kp[hip_idx]
        knee = kp[knee_idx]

        # Thigh vector
        thigh = knee - hip
        thigh_norm = np.linalg.norm(thigh)
        if thigh_norm < 1e-6:
            return np.nan
        thigh = thigh / thigh_norm

        # Trunk reference vector (vertical, pointing down)
        # Use the trunk direction or simply the negative belt normal
        trunk_dir = -self.belt_plane_normal  # Downward

        # For sagittal plane, we need the forward direction
        # Forward is perpendicular to both vertical and medial-lateral axis
        if valid[opp_hip_idx]:
            hip_axis = kp[opp_hip_idx] - hip  # Medial-lateral
            forward = np.cross(self.belt_plane_normal, hip_axis)
            forward_norm = np.linalg.norm(forward)
            if forward_norm > 1e-6:
                forward = forward / forward_norm
            else:
                forward = np.array([1, 0, 0])  # Default
        else:
            forward = np.array([1, 0, 0])

        # Angle of thigh relative to vertical in sagittal plane
        cos_angle = np.clip(np.dot(thigh, trunk_dir), -1, 1)
        angle = np.degrees(np.arccos(cos_angle))

        # Sign: flexion (forward) = positive
        sign = np.sign(np.dot(thigh, forward))

        return sign * angle

    def _compute_knee_flexion(self, kp: np.ndarray, valid: np.ndarray,
                               side: str = "right") -> float:
        """
        Compute knee flexion/extension angle.
        Angle between thigh and shank.
        0 = fully extended, positive = flexion.
        """
        if side == "right":
            hip_idx = KEYPOINT_INDEX["right_hip"]
            knee_idx = KEYPOINT_INDEX["right_knee"]
            ankle_idx = KEYPOINT_INDEX["right_ankle"]
        else:
            hip_idx = KEYPOINT_INDEX["left_hip"]
            knee_idx = KEYPOINT_INDEX["left_knee"]
            ankle_idx = KEYPOINT_INDEX["left_ankle"]

        if not (valid[hip_idx] and valid[knee_idx] and valid[ankle_idx]):
            return np.nan

        hip = kp[hip_idx]
        knee = kp[knee_idx]
        ankle = kp[ankle_idx]

        # Thigh vector (knee to hip)
        thigh = hip - knee
        thigh_norm = np.linalg.norm(thigh)
        if thigh_norm < 1e-6:
            return np.nan

        # Shank vector (knee to ankle)
        shank = ankle - knee
        shank_norm = np.linalg.norm(shank)
        if shank_norm < 1e-6:
            return np.nan

        # Angle between thigh and shank
        cos_angle = np.clip(
            np.dot(thigh, shank) / (thigh_norm * shank_norm), -1, 1
        )
        angle = np.degrees(np.arccos(cos_angle))

        # Knee flexion = 180 - angle (0 when fully extended)
        flexion = 180.0 - angle

        return flexion

    def _compute_ankle_angle(self, kp: np.ndarray, valid: np.ndarray,
                              side: str = "right") -> float:
        """
        Compute ankle dorsiflexion/plantarflexion angle.
        Angle between shank and foot.
        Positive = dorsiflexion, Negative = plantarflexion.
        90 degrees = neutral position.
        """
        if side == "right":
            knee_idx = KEYPOINT_INDEX["right_knee"]
            ankle_idx = KEYPOINT_INDEX["right_ankle"]
            toe_idx = KEYPOINT_INDEX["right_big_toe"]
            heel_idx = KEYPOINT_INDEX["right_heel"]
        else:
            knee_idx = KEYPOINT_INDEX["left_knee"]
            ankle_idx = KEYPOINT_INDEX["left_ankle"]
            toe_idx = KEYPOINT_INDEX["left_big_toe"]
            heel_idx = KEYPOINT_INDEX["left_heel"]

        if not (valid[knee_idx] and valid[ankle_idx]):
            return np.nan

        knee = kp[knee_idx]
        ankle = kp[ankle_idx]

        # Shank vector (ankle to knee, pointing up)
        shank = knee - ankle
        shank_norm = np.linalg.norm(shank)
        if shank_norm < 1e-6:
            return np.nan
        shank = shank / shank_norm

        # Foot vector
        if valid[toe_idx] and valid[heel_idx]:
            foot = kp[toe_idx] - kp[heel_idx]
        elif valid[toe_idx]:
            foot = kp[toe_idx] - ankle
        else:
            return np.nan

        foot_norm = np.linalg.norm(foot)
        if foot_norm < 1e-6:
            return np.nan
        foot = foot / foot_norm

        # Angle between shank and foot
        cos_angle = np.clip(np.dot(shank, foot), -1, 1)
        angle = np.degrees(np.arccos(cos_angle))

        # Convention: 90 = neutral, <90 = dorsiflexion, >90 = plantarflexion
        # Return relative to neutral (positive = dorsiflexion)
        ankle_angle = 90.0 - angle

        return ankle_angle

    def _compute_foot_clearance(self, kp: np.ndarray, valid: np.ndarray,
                                 side: str = "right") -> float:
        """
        Compute foot height above treadmill belt (clearance).
        Uses the lowest foot point (toe or ankle).
        """
        if side == "right":
            ankle_idx = KEYPOINT_INDEX["right_ankle"]
            toe_idx = KEYPOINT_INDEX["right_big_toe"]
            heel_idx = KEYPOINT_INDEX["right_heel"]
        else:
            ankle_idx = KEYPOINT_INDEX["left_ankle"]
            toe_idx = KEYPOINT_INDEX["left_big_toe"]
            heel_idx = KEYPOINT_INDEX["left_heel"]

        # Find the lowest valid foot point
        foot_points = []
        for idx in [toe_idx, heel_idx, ankle_idx]:
            if valid[idx]:
                foot_points.append(kp[idx])

        if not foot_points:
            return np.nan

        # Compute height above belt plane for each point
        heights = []
        for point in foot_points:
            h = point_to_plane_distance(point, self.belt_plane_normal, self.belt_plane_point)
            heights.append(h)

        # Return minimum height (lowest point of foot)
        min_height = min(heights)
        
        # Ensure non-negative (foot should be at or above belt)
        return max(0.0, min_height)

    def detect_gait_events(self) -> List[GaitEvent]:
        """
        Detect gait events (heel strikes and toe offs) from pose history.
        Uses either velocity-based or position-based method.
        
        Returns:
            List of detected gait events
        """
        if len(self._pose_history) < 10:
            logger.warning("Not enough frames for gait event detection")
            return []

        events = []

        if self.event_method == "velocity":
            events = self._detect_events_velocity()
        else:
            events = self._detect_events_position()

        self._gait_events = events
        return events

    def _detect_events_velocity(self) -> List[GaitEvent]:
        """Detect heel strikes using foot velocity (zero-crossing method)."""
        events = []
        
        sides = []
        if self.analyze_side in ["right", "both"]:
            sides.append("right")
        if self.analyze_side in ["left", "both"]:
            sides.append("left")

        for side in sides:
            if side == "right":
                ankle_idx = KEYPOINT_INDEX["right_ankle"]
                heel_idx = KEYPOINT_INDEX["right_heel"]
            else:
                ankle_idx = KEYPOINT_INDEX["left_ankle"]
                heel_idx = KEYPOINT_INDEX["left_heel"]

            # Extract foot vertical position over time
            foot_heights = []
            for pose in self._pose_history:
                if pose.valid_mask[heel_idx]:
                    h = point_to_plane_distance(
                        pose.keypoints_3d[heel_idx],
                        self.belt_plane_normal, self.belt_plane_point
                    )
                elif pose.valid_mask[ankle_idx]:
                    h = point_to_plane_distance(
                        pose.keypoints_3d[ankle_idx],
                        self.belt_plane_normal, self.belt_plane_point
                    )
                else:
                    h = np.nan
                foot_heights.append(h)

            foot_heights = np.array(foot_heights)

            # Interpolate NaN values
            foot_heights = self._interpolate_nans(foot_heights)
            if foot_heights is None:
                continue

            # Apply low-pass filter
            if self.filter_enabled:
                foot_heights = self._apply_filter(foot_heights)

            # Compute vertical velocity
            velocity = np.gradient(foot_heights, self.dt)

            # Heel strikes: foot descending and reaches minimum height
            # Look for transitions from negative velocity to near-zero
            # when foot height is near ground level
            for i in range(2, len(velocity) - 2):
                # Heel strike: velocity crosses zero from negative to positive
                # AND foot height is close to ground
                if (velocity[i-1] < -self.velocity_threshold and 
                    velocity[i] > -self.velocity_threshold and
                    foot_heights[i] < self.height_threshold * 3):
                    
                    event = GaitEvent(
                        frame_index=i,
                        timestamp_us=self._timestamps[i] if i < len(self._timestamps) else 0,
                        event_type="heel_strike",
                        side=side,
                        foot_position=self._pose_history[i].keypoints_3d[heel_idx].copy()
                        if self._pose_history[i].valid_mask[heel_idx]
                        else np.zeros(3)
                    )
                    events.append(event)

                # Toe off: velocity crosses zero from near-zero to positive
                # when foot starts to lift
                elif (velocity[i-1] < self.velocity_threshold and
                      velocity[i] > self.velocity_threshold and
                      foot_heights[i] < self.height_threshold * 5):
                    
                    event = GaitEvent(
                        frame_index=i,
                        timestamp_us=self._timestamps[i] if i < len(self._timestamps) else 0,
                        event_type="toe_off",
                        side=side,
                        foot_position=self._pose_history[i].keypoints_3d[ankle_idx].copy()
                        if self._pose_history[i].valid_mask[ankle_idx]
                        else np.zeros(3)
                    )
                    events.append(event)

        # Sort by frame index
        events.sort(key=lambda e: e.frame_index)
        return events

    def _detect_events_position(self) -> List[GaitEvent]:
        """Detect heel strikes using foot height threshold."""
        events = []
        
        sides = []
        if self.analyze_side in ["right", "both"]:
            sides.append("right")
        if self.analyze_side in ["left", "both"]:
            sides.append("left")

        for side in sides:
            if side == "right":
                heel_idx = KEYPOINT_INDEX["right_heel"]
                ankle_idx = KEYPOINT_INDEX["right_ankle"]
            else:
                heel_idx = KEYPOINT_INDEX["left_heel"]
                ankle_idx = KEYPOINT_INDEX["left_ankle"]

            # Extract foot height
            foot_heights = []
            for pose in self._pose_history:
                if pose.valid_mask[heel_idx]:
                    h = point_to_plane_distance(
                        pose.keypoints_3d[heel_idx],
                        self.belt_plane_normal, self.belt_plane_point
                    )
                elif pose.valid_mask[ankle_idx]:
                    h = point_to_plane_distance(
                        pose.keypoints_3d[ankle_idx],
                        self.belt_plane_normal, self.belt_plane_point
                    )
                else:
                    h = np.nan
                foot_heights.append(h)

            foot_heights = np.array(foot_heights)
            foot_heights = self._interpolate_nans(foot_heights)
            if foot_heights is None:
                continue

            if self.filter_enabled:
                foot_heights = self._apply_filter(foot_heights)

            # Find local minima below threshold (heel strikes)
            is_ground = foot_heights < self.height_threshold
            
            # Find transitions: not_ground -> ground (heel strike)
            for i in range(1, len(is_ground)):
                if not is_ground[i-1] and is_ground[i]:
                    event = GaitEvent(
                        frame_index=i,
                        timestamp_us=self._timestamps[i] if i < len(self._timestamps) else 0,
                        event_type="heel_strike",
                        side=side
                    )
                    events.append(event)
                elif is_ground[i-1] and not is_ground[i]:
                    event = GaitEvent(
                        frame_index=i,
                        timestamp_us=self._timestamps[i] if i < len(self._timestamps) else 0,
                        event_type="toe_off",
                        side=side
                    )
                    events.append(event)

        events.sort(key=lambda e: e.frame_index)
        return events

    def compute_step_length(self, event1: GaitEvent, event2: GaitEvent) -> float:
        """
        Compute step length between two consecutive heel strikes.
        Step length = horizontal distance between foot positions at heel strikes.
        
        Note: On a treadmill, step length is the distance between feet at contact,
        not the distance traveled over ground.
        """
        if event1.event_type != "heel_strike" or event2.event_type != "heel_strike":
            return np.nan

        # Get foot positions at both events
        idx1 = event1.frame_index
        idx2 = event2.frame_index

        if idx1 >= len(self._pose_history) or idx2 >= len(self._pose_history):
            return np.nan

        pose1 = self._pose_history[idx1]
        pose2 = self._pose_history[idx2]

        # Use heel or ankle position
        side1_heel = KEYPOINT_INDEX[f"{event1.side}_heel"]
        side2_heel = KEYPOINT_INDEX[f"{event2.side}_heel"]
        side1_ankle = KEYPOINT_INDEX[f"{event1.side}_ankle"]
        side2_ankle = KEYPOINT_INDEX[f"{event2.side}_ankle"]

        if pose1.valid_mask[side1_heel]:
            pos1 = pose1.keypoints_3d[side1_heel]
        elif pose1.valid_mask[side1_ankle]:
            pos1 = pose1.keypoints_3d[side1_ankle]
        else:
            return np.nan

        if pose2.valid_mask[side2_heel]:
            pos2 = pose2.keypoints_3d[side2_heel]
        elif pose2.valid_mask[side2_ankle]:
            pos2 = pose2.keypoints_3d[side2_ankle]
        else:
            return np.nan

        # Project to belt plane (horizontal distance)
        # Remove vertical component
        diff = pos2 - pos1
        vertical_component = np.dot(diff, self.belt_plane_normal) * self.belt_plane_normal
        horizontal_diff = diff - vertical_component

        return float(np.linalg.norm(horizontal_diff))

    def compute_cadence(self, events: List[GaitEvent], 
                        window_frames: int = 0) -> float:
        """
        Compute cadence (steps per minute) from heel strike events.
        
        Args:
            events: List of heel strike events
            window_frames: If > 0, compute cadence over this window
        Returns:
            Cadence in steps/min
        """
        heel_strikes = [e for e in events if e.event_type == "heel_strike"]
        
        if len(heel_strikes) < 2:
            return np.nan

        # Compute average time between heel strikes
        intervals = []
        for i in range(1, len(heel_strikes)):
            dt_us = heel_strikes[i].timestamp_us - heel_strikes[i-1].timestamp_us
            if dt_us > 0:
                intervals.append(dt_us / 1e6)  # Convert to seconds

        if not intervals:
            return np.nan

        avg_interval = np.mean(intervals)
        if avg_interval > 0:
            cadence = 60.0 / avg_interval  # Steps per minute
            return cadence

        return np.nan

    def compute_all_parameters(self) -> Tuple[List[GaitParameters], List[GaitCycleData]]:
        """
        Compute all gait parameters for the entire recording.
        
        Returns:
            (frame_parameters, cycle_data)
            - frame_parameters: Per-frame gait parameters
            - cycle_data: Per-gait-cycle aggregated data
        """
        if len(self._pose_history) == 0:
            return [], []

        # === Compute per-frame parameters ===
        frame_params = []
        for i, pose in enumerate(self._pose_history):
            prev_pose = self._pose_history[i-1] if i > 0 else None
            params = self.compute_frame_parameters(pose, prev_pose)
            params.timestamp_us = self._timestamps[i] if i < len(self._timestamps) else 0
            frame_params.append(params)

        # === Detect gait events ===
        events = self.detect_gait_events()
        logger.info(f"Detected {len(events)} gait events")

        # === Compute step length and cadence ===
        heel_strikes = [e for e in events if e.event_type == "heel_strike"]

        # Assign step length to frames
        for i in range(1, len(heel_strikes)):
            step_len = self.compute_step_length(heel_strikes[i-1], heel_strikes[i])
            # Assign to frames in this step
            start_frame = heel_strikes[i-1].frame_index
            end_frame = heel_strikes[i].frame_index
            for f in range(start_frame, min(end_frame + 1, len(frame_params))):
                frame_params[f].step_length = step_len

        # Compute running cadence
        cadence = self.compute_cadence(events)
        for params in frame_params:
            params.cadence = cadence

        # === Compute per-cycle data ===
        cycle_data = self._compute_cycle_data(frame_params, heel_strikes)

        # === Apply filtering to frame-level data ===
        if self.filter_enabled:
            frame_params = self._filter_frame_parameters(frame_params)

        return frame_params, cycle_data

    def _compute_cycle_data(self, frame_params: List[GaitParameters],
                            heel_strikes: List[GaitEvent]) -> List[GaitCycleData]:
        """Compute per-gait-cycle aggregated data."""
        cycle_data = []

        for i in range(1, len(heel_strikes)):
            start = heel_strikes[i-1].frame_index
            end = heel_strikes[i].frame_index

            if end <= start or end >= len(frame_params):
                continue

            cycle_params = frame_params[start:end+1]

            # Extract arrays for aggregation
            pitches = [p.upper_body_pitch for p in cycle_params if not np.isnan(p.upper_body_pitch)]
            hip_add = [p.hip_adduction_abduction for p in cycle_params if not np.isnan(p.hip_adduction_abduction)]
            hip_flex = [p.hip_flexion_extension for p in cycle_params if not np.isnan(p.hip_flexion_extension)]
            knee_flex = [p.knee_flexion_extension for p in cycle_params if not np.isnan(p.knee_flexion_extension)]
            ankle = [p.ankle_angle for p in cycle_params if not np.isnan(p.ankle_angle)]
            clearance = [p.foot_clearance for p in cycle_params if not np.isnan(p.foot_clearance)]

            cycle = GaitCycleData(
                cycle_index=i-1,
                start_frame=start,
                end_frame=end,
                start_time_us=heel_strikes[i-1].timestamp_us,
                end_time_us=heel_strikes[i].timestamp_us,
                side=heel_strikes[i-1].side,
                upper_body_pitch_mean=np.mean(pitches) if pitches else np.nan,
                hip_adduction_abduction_max=np.max(np.abs(hip_add)) if hip_add else np.nan,
                hip_flexion_extension_max=np.max(hip_flex) if hip_flex else np.nan,
                knee_flexion_extension_max=np.max(knee_flex) if knee_flex else np.nan,
                ankle_angle_max=np.max(ankle) if ankle else np.nan,
                foot_clearance_max=np.max(clearance) if clearance else np.nan,
                step_length=frame_params[start].step_length if start < len(frame_params) else np.nan,
                cadence=frame_params[start].cadence if start < len(frame_params) else np.nan,
            )
            cycle_data.append(cycle)

        return cycle_data

    def _filter_frame_parameters(self, params: List[GaitParameters]) -> List[GaitParameters]:
        """Apply smoothing filter to frame-level parameters."""
        if len(params) < self.savgol_window:
            return params

        # Extract each parameter as array
        fields = ['upper_body_pitch', 'hip_adduction_abduction', 
                  'hip_flexion_extension', 'knee_flexion_extension',
                  'ankle_angle', 'foot_clearance']

        for field in fields:
            values = np.array([getattr(p, field) for p in params])
            filtered = self._interpolate_and_filter(values)
            if filtered is not None:
                for i, p in enumerate(params):
                    setattr(p, field, filtered[i])

        return params

    def _interpolate_and_filter(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Interpolate NaN values and apply filter."""
        data = self._interpolate_nans(data)
        if data is None:
            return None
        return self._apply_filter(data)

    def _interpolate_nans(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Interpolate NaN values in a 1D array."""
        nans = np.isnan(data)
        if np.all(nans):
            return None
        if not np.any(nans):
            return data.copy()

        # Linear interpolation
        result = data.copy()
        valid_idx = np.where(~nans)[0]
        nan_idx = np.where(nans)[0]

        if len(valid_idx) < 2:
            return None

        result[nan_idx] = np.interp(nan_idx, valid_idx, data[valid_idx])
        return result

    def _apply_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply low-pass filter to data."""
        if len(data) < 12:
            return data

        try:
            if self.filter_type == "butterworth":
                nyquist = self.fps / 2
                normalized_cutoff = min(self.cutoff_freq / nyquist, 0.99)
                b, a = butter(self.filter_order, normalized_cutoff, btype='low')
                return filtfilt(b, a, data)
            elif self.filter_type == "savgol":
                window = min(self.savgol_window, len(data))
                if window % 2 == 0:
                    window -= 1
                return savgol_filter(data, window, self.savgol_polyorder)
        except Exception as e:
            logger.warning(f"Filtering failed: {e}")

        return data
