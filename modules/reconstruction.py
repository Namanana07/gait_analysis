"""
3D Reconstruction Module.
Supports stereo triangulation, depth-based backprojection, and fusion of both methods.
Transforms 2D pose detections into 3D world coordinates.
"""

import logging
import numpy as np
import cv2
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field

from .pose_estimation import PoseResult, NUM_KEYPOINTS, KEYPOINT_INDEX
from .calibration import CalibrationResult, point_to_plane_distance

logger = logging.getLogger(__name__)


@dataclass
class Pose3D:
    """3D pose of a person in world coordinates."""
    keypoints_3d: np.ndarray          # (N, 3) world coordinates in meters
    confidence: np.ndarray            # (N,) confidence per keypoint
    valid_mask: np.ndarray            # (N,) bool - whether keypoint is valid
    method: str = "unknown"           # "triangulation", "depth", or "fusion"


class StereoTriangulator:
    """
    Triangulates 3D points from stereo 2D observations.
    Uses DLT (Direct Linear Transform) method.
    """

    def __init__(self, calibration: CalibrationResult):
        self.calib = calibration
        
        # Build projection matrices
        # P = K @ [R | t] where [R|t] transforms world to camera
        self.P1 = self.calib.K1 @ np.hstack([self.calib.R1, self.calib.t1.reshape(3, 1)])
        self.P2 = self.calib.K2 @ np.hstack([self.calib.R2, self.calib.t2.reshape(3, 1)])

    def triangulate_point(self, pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:
        """
        Triangulate a single 3D point from two 2D observations.
        
        Args:
            pt1: 2D point in camera 1 (x, y)
            pt2: 2D point in camera 2 (x, y)
        Returns:
            3D point in world coordinates (x, y, z)
        """
        # Build A matrix for DLT
        A = np.zeros((4, 4))
        A[0] = pt1[0] * self.P1[2] - self.P1[0]
        A[1] = pt1[1] * self.P1[2] - self.P1[1]
        A[2] = pt2[0] * self.P2[2] - self.P2[0]
        A[3] = pt2[1] * self.P2[2] - self.P2[1]

        # SVD solution
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X[:3] / X[3]  # Homogeneous -> Euclidean

        return X

    def triangulate_pose(self, pose1: PoseResult, pose2: PoseResult,
                         min_confidence: float = 0.3,
                         reproj_threshold: float = 10.0) -> Pose3D:
        """
        Triangulate all keypoints of a person from two camera views.
        
        Args:
            pose1: Pose detection from camera 1
            pose2: Pose detection from camera 2
            min_confidence: Minimum confidence to use a keypoint
            reproj_threshold: Maximum reprojection error (pixels)
        Returns:
            Pose3D with triangulated keypoints
        """
        keypoints_3d = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)
        confidence = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        valid_mask = np.zeros(NUM_KEYPOINTS, dtype=bool)

        for i in range(NUM_KEYPOINTS):
            if pose1.confidence[i] < min_confidence or \
               pose2.confidence[i] < min_confidence:
                continue

            pt1 = pose1.keypoints[i]
            pt2 = pose2.keypoints[i]

            # Triangulate
            point_3d = self.triangulate_point(pt1, pt2)

            # Compute reprojection error
            reproj_err = self._reprojection_error(point_3d, pt1, pt2)

            if reproj_err < reproj_threshold:
                keypoints_3d[i] = point_3d
                confidence[i] = min(pose1.confidence[i], pose2.confidence[i])
                valid_mask[i] = True
            else:
                logger.debug(f"Keypoint {i} rejected: reproj error = {reproj_err:.1f}")

        return Pose3D(
            keypoints_3d=keypoints_3d,
            confidence=confidence,
            valid_mask=valid_mask,
            method="triangulation"
        )

    def _reprojection_error(self, point_3d: np.ndarray,
                            pt1: np.ndarray, pt2: np.ndarray) -> float:
        """Compute mean reprojection error across both cameras."""
        # Project to camera 1
        X_h = np.append(point_3d, 1.0)
        proj1 = self.P1 @ X_h
        proj1 = proj1[:2] / proj1[2]

        # Project to camera 2
        proj2 = self.P2 @ X_h
        proj2 = proj2[:2] / proj2[2]

        err1 = np.linalg.norm(proj1 - pt1)
        err2 = np.linalg.norm(proj2 - pt2)

        return (err1 + err2) / 2


class DepthBackprojector:
    """
    Backprojects 2D keypoints to 3D using depth maps.
    Works with a single camera but requires accurate depth data.
    """

    def __init__(self, calibration: CalibrationResult, depth_scale: float = 0.001):
        """
        Args:
            calibration: Camera calibration result
            depth_scale: Conversion factor from depth values to meters
        """
        self.calib = calibration
        self.depth_scale = depth_scale

    def backproject_point(self, x: float, y: float, depth: float,
                          K: np.ndarray, R: np.ndarray, 
                          t: np.ndarray) -> np.ndarray:
        """
        Backproject a 2D point with depth to 3D world coordinates.
        
        Args:
            x, y: Pixel coordinates
            depth: Depth value in meters
            K: Camera intrinsic matrix
            R: Rotation (world to camera)
            t: Translation (world to camera)
        Returns:
            3D point in world coordinates
        """
        # Pixel to camera coordinates
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        X_cam = (x - cx) * depth / fx
        Y_cam = (y - cy) * depth / fy
        Z_cam = depth

        point_cam = np.array([X_cam, Y_cam, Z_cam])

        # Camera to world coordinates: X_world = R^T @ (X_cam - t)
        point_world = R.T @ (point_cam - t.flatten())

        return point_world

    def get_depth_at_point(self, depth_map: np.ndarray, x: float, y: float,
                           search_radius: int = 5,
                           min_depth: float = 0.3,
                           max_depth: float = 5.0) -> Optional[float]:
        """
        Get depth value at a pixel location with neighborhood search.
        
        Args:
            depth_map: Depth image (H, W) in raw units
            x, y: Pixel coordinates
            search_radius: Radius for neighborhood search
            min_depth: Minimum valid depth in meters
            max_depth: Maximum valid depth in meters
        Returns:
            Depth in meters, or None if invalid
        """
        h, w = depth_map.shape
        ix, iy = int(round(x)), int(round(y))

        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return None

        # Search in neighborhood for valid depth
        y_min = max(0, iy - search_radius)
        y_max = min(h, iy + search_radius + 1)
        x_min = max(0, ix - search_radius)
        x_max = min(w, ix + search_radius + 1)

        patch = depth_map[y_min:y_max, x_min:x_max].astype(np.float64)
        patch_meters = patch * self.depth_scale

        # Filter valid depths
        valid = (patch_meters > min_depth) & (patch_meters < max_depth)
        if not np.any(valid):
            return None

        # Use median of valid depths (robust to outliers)
        return float(np.median(patch_meters[valid]))

    def backproject_pose(self, pose: PoseResult, depth_map: np.ndarray,
                         camera_idx: int = 0,
                         min_confidence: float = 0.3,
                         search_radius: int = 5,
                         min_depth: float = 0.3,
                         max_depth: float = 5.0) -> Pose3D:
        """
        Backproject all keypoints of a pose using depth map.
        
        Args:
            pose: 2D pose detection
            depth_map: Depth image
            camera_idx: Which camera (0 or 1) for selecting correct parameters
            min_confidence: Minimum keypoint confidence
            search_radius: Depth lookup radius
            min_depth: Minimum valid depth (meters)
            max_depth: Maximum valid depth (meters)
        Returns:
            Pose3D with backprojected keypoints
        """
        if camera_idx == 0:
            K = self.calib.K1
            R = self.calib.R1
            t = self.calib.t1
        else:
            K = self.calib.K2
            R = self.calib.R2
            t = self.calib.t2

        keypoints_3d = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)
        confidence = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        valid_mask = np.zeros(NUM_KEYPOINTS, dtype=bool)

        for i in range(NUM_KEYPOINTS):
            if pose.confidence[i] < min_confidence:
                continue

            x, y = pose.keypoints[i]
            depth = self.get_depth_at_point(
                depth_map, x, y, search_radius, min_depth, max_depth
            )

            if depth is None:
                continue

            point_3d = self.backproject_point(x, y, depth, K, R, t)
            keypoints_3d[i] = point_3d
            confidence[i] = pose.confidence[i]
            valid_mask[i] = True

        return Pose3D(
            keypoints_3d=keypoints_3d,
            confidence=confidence,
            valid_mask=valid_mask,
            method="depth"
        )


class PoseReconstructor:
    """
    High-level 3D pose reconstruction combining triangulation and depth methods.
    """

    def __init__(self, calibration: CalibrationResult, config: dict):
        """
        Args:
            calibration: Stereo calibration result
            config: Reconstruction configuration
        """
        self.calib = calibration
        self.config = config
        self.method = config.get("method", "fusion")

        self.triangulator = StereoTriangulator(calibration)
        
        depth_scale = config.get("depth_scale", 0.001)
        self.depth_projector = DepthBackprojector(calibration, depth_scale)

        # Fusion weights
        fusion_config = config.get("fusion", {})
        self.tri_weight = fusion_config.get("triangulation_weight", 0.6)
        self.depth_weight = fusion_config.get("depth_weight", 0.4)

        # Thresholds
        tri_config = config.get("triangulation", {})
        self.min_confidence = tri_config.get("min_confidence", 0.3)
        self.reproj_threshold = tri_config.get("reproj_error_threshold", 10.0)

        depth_config = config.get("depth", {})
        self.search_radius = depth_config.get("search_radius", 5)
        self.max_depth = depth_config.get("max_depth", 5.0)
        self.min_depth = depth_config.get("min_depth", 0.3)

    def reconstruct(self, pose1: PoseResult, pose2: PoseResult,
                    depth1: Optional[np.ndarray] = None,
                    depth2: Optional[np.ndarray] = None) -> Pose3D:
        """
        Reconstruct 3D pose from dual camera observations.
        
        Args:
            pose1: 2D pose from camera 1
            pose2: 2D pose from camera 2
            depth1: Depth map from camera 1 (optional)
            depth2: Depth map from camera 2 (optional)
        Returns:
            Pose3D in world coordinates
        """
        if self.method == "triangulation":
            return self.triangulator.triangulate_pose(
                pose1, pose2, self.min_confidence, self.reproj_threshold
            )
        elif self.method == "depth":
            return self._reconstruct_depth(pose1, pose2, depth1, depth2)
        elif self.method == "fusion":
            return self._reconstruct_fusion(pose1, pose2, depth1, depth2)
        else:
            logger.warning(f"Unknown method '{self.method}', using triangulation")
            return self.triangulator.triangulate_pose(
                pose1, pose2, self.min_confidence, self.reproj_threshold
            )

    def _reconstruct_depth(self, pose1, pose2, depth1, depth2) -> Pose3D:
        """Reconstruct using depth backprojection (best depth used)."""
        results = []

        if depth1 is not None:
            p3d_1 = self.depth_projector.backproject_pose(
                pose1, depth1, camera_idx=0,
                min_confidence=self.min_confidence,
                search_radius=self.search_radius,
                min_depth=self.min_depth,
                max_depth=self.max_depth
            )
            results.append(p3d_1)

        if depth2 is not None:
            p3d_2 = self.depth_projector.backproject_pose(
                pose2, depth2, camera_idx=1,
                min_confidence=self.min_confidence,
                search_radius=self.search_radius,
                min_depth=self.min_depth,
                max_depth=self.max_depth
            )
            results.append(p3d_2)

        if len(results) == 0:
            return Pose3D(
                keypoints_3d=np.zeros((NUM_KEYPOINTS, 3)),
                confidence=np.zeros(NUM_KEYPOINTS),
                valid_mask=np.zeros(NUM_KEYPOINTS, dtype=bool),
                method="depth"
            )
        elif len(results) == 1:
            return results[0]
        else:
            # Merge: use higher confidence
            return self._merge_depth_poses(results[0], results[1])

    def _reconstruct_fusion(self, pose1, pose2, depth1, depth2) -> Pose3D:
        """Fuse triangulation and depth methods."""
        # Triangulation
        tri_pose = self.triangulator.triangulate_pose(
            pose1, pose2, self.min_confidence, self.reproj_threshold
        )

        # Depth-based
        depth_pose = self._reconstruct_depth(pose1, pose2, depth1, depth2)

        # Weighted fusion
        keypoints_3d = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)
        confidence = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        valid_mask = np.zeros(NUM_KEYPOINTS, dtype=bool)

        for i in range(NUM_KEYPOINTS):
            tri_valid = tri_pose.valid_mask[i]
            depth_valid = depth_pose.valid_mask[i]

            if tri_valid and depth_valid:
                # Check consistency
                dist = np.linalg.norm(tri_pose.keypoints_3d[i] - depth_pose.keypoints_3d[i])
                if dist < 0.1:  # Within 10cm, fuse
                    w_tri = self.tri_weight * tri_pose.confidence[i]
                    w_dep = self.depth_weight * depth_pose.confidence[i]
                    total_w = w_tri + w_dep
                    
                    keypoints_3d[i] = (
                        w_tri * tri_pose.keypoints_3d[i] +
                        w_dep * depth_pose.keypoints_3d[i]
                    ) / total_w
                    confidence[i] = total_w / (self.tri_weight + self.depth_weight)
                else:
                    # Large discrepancy, prefer triangulation (more reliable for pose)
                    keypoints_3d[i] = tri_pose.keypoints_3d[i]
                    confidence[i] = tri_pose.confidence[i] * 0.8
                valid_mask[i] = True

            elif tri_valid:
                keypoints_3d[i] = tri_pose.keypoints_3d[i]
                confidence[i] = tri_pose.confidence[i]
                valid_mask[i] = True

            elif depth_valid:
                keypoints_3d[i] = depth_pose.keypoints_3d[i]
                confidence[i] = depth_pose.confidence[i]
                valid_mask[i] = True

        return Pose3D(
            keypoints_3d=keypoints_3d,
            confidence=confidence,
            valid_mask=valid_mask,
            method="fusion"
        )

    def _merge_depth_poses(self, pose_a: Pose3D, pose_b: Pose3D) -> Pose3D:
        """Merge two depth-based poses, preferring higher confidence."""
        keypoints_3d = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)
        confidence = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        valid_mask = np.zeros(NUM_KEYPOINTS, dtype=bool)

        for i in range(NUM_KEYPOINTS):
            if pose_a.valid_mask[i] and pose_b.valid_mask[i]:
                if pose_a.confidence[i] >= pose_b.confidence[i]:
                    keypoints_3d[i] = pose_a.keypoints_3d[i]
                    confidence[i] = pose_a.confidence[i]
                else:
                    keypoints_3d[i] = pose_b.keypoints_3d[i]
                    confidence[i] = pose_b.confidence[i]
                valid_mask[i] = True
            elif pose_a.valid_mask[i]:
                keypoints_3d[i] = pose_a.keypoints_3d[i]
                confidence[i] = pose_a.confidence[i]
                valid_mask[i] = True
            elif pose_b.valid_mask[i]:
                keypoints_3d[i] = pose_b.keypoints_3d[i]
                confidence[i] = pose_b.confidence[i]
                valid_mask[i] = True

        return Pose3D(
            keypoints_3d=keypoints_3d,
            confidence=confidence,
            valid_mask=valid_mask,
            method="depth"
        )
