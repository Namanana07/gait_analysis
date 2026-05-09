"""
Camera Calibration Module using ChArUco markers.
Handles:
- ChArUco marker detection in RGB frames
- Camera extrinsic calibration (relative pose between two cameras)
- Treadmill belt plane estimation
- Coordinate system alignment
"""

import logging
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of stereo calibration."""
    # Camera 1 extrinsics (world frame = treadmill marker plane)
    R1: np.ndarray  # 3x3 rotation matrix
    t1: np.ndarray  # 3x1 translation vector
    # Camera 2 extrinsics
    R2: np.ndarray
    t2: np.ndarray
    # Relative transformation (cam2 in cam1 frame)
    R_rel: np.ndarray
    t_rel: np.ndarray
    # Treadmill belt plane in world frame (normal vector and point)
    belt_plane_normal: np.ndarray  # Unit normal (pointing up)
    belt_plane_point: np.ndarray   # A point on the plane
    # Reprojection error
    reproj_error: float
    # Camera intrinsics used
    K1: np.ndarray
    K2: np.ndarray
    dist1: Optional[np.ndarray]
    dist2: Optional[np.ndarray]


class CharucoCalibrator:
    """
    Calibrator using ChArUco markers on treadmill frame.
    Detects markers, estimates camera poses, and computes treadmill plane.
    """

    # ArUco dictionary mapping
    DICT_MAP = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    }

    def __init__(self, config: dict):
        """
        Initialize calibrator with ChArUco configuration.
        
        Args:
            config: ChArUco configuration dictionary from config.yaml
        """
        self.config = config
        dict_name = config.get("dictionary", "DICT_4X4_50")
        
        if dict_name not in self.DICT_MAP:
            raise ValueError(f"Unknown ArUco dictionary: {dict_name}. "
                           f"Available: {list(self.DICT_MAP.keys())}")

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.DICT_MAP[dict_name])
        
        # Create ChArUco board
        self.squares_x = config.get("squares_x", 3)
        self.squares_y = config.get("squares_y", 2)
        self.square_length = config.get("square_length", 0.04)
        self.marker_length = config.get("marker_length", 0.03)
        
        self.charuco_board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            self.aruco_dict
        )
        
        # Belt plane offset (downward from markers)
        self.belt_offset = config.get("belt_plane_offset", 0.015)
        
        # Detector parameters
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.adaptiveThreshConstant = 7
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 23
        self.detector_params.adaptiveThreshWinSizeStep = 10
        
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.detector_params
        )

        # Calibration data storage
        self._cam1_rvecs = []
        self._cam1_tvecs = []
        self._cam2_rvecs = []
        self._cam2_tvecs = []
        self._valid_frames = 0

    def reset(self) -> None:
        self._cam1_rvecs.clear()
        self._cam1_tvecs.clear()
        self._cam2_rvecs.clear()
        self._cam2_tvecs.clear()
        self._valid_frames = 0

    def detect_charuco(self, image: np.ndarray, 
                       camera_matrix: np.ndarray,
                       dist_coeffs: Optional[np.ndarray] = None
                       ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect ChArUco corners in an image.
        
        Args:
            image: RGB or grayscale image
            camera_matrix: 3x3 intrinsic matrix
            dist_coeffs: Distortion coefficients
            
        Returns:
            (charuco_corners, charuco_ids) or (None, None) if detection fails
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Detect ArUco markers
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

        if ids is None or len(ids) < 2:
            return None, None

        # Refine and interpolate ChArUco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.charuco_board,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs
        )

        if retval < 4:  # Need at least 4 corners for pose estimation
            return None, None

        return charuco_corners, charuco_ids

    def estimate_pose(self, charuco_corners: np.ndarray,
                      charuco_ids: np.ndarray,
                      camera_matrix: np.ndarray,
                      dist_coeffs: Optional[np.ndarray] = None
                      ) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Estimate camera pose from ChArUco corners.
        
        Returns:
            (success, rvec, tvec)
        """
        if dist_coeffs is None:
            dist_coeffs = np.zeros(5)

        success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, self.charuco_board,
            camera_matrix, dist_coeffs,
            None, None
        )

        if success:
            return True, rvec, tvec
        return False, np.zeros(3), np.zeros(3)

    def add_calibration_frame(self, 
                              image1: np.ndarray, image2: np.ndarray,
                              K1: np.ndarray, K2: np.ndarray,
                              dist1: Optional[np.ndarray] = None,
                              dist2: Optional[np.ndarray] = None) -> bool:
        """
        Add a frame pair for calibration.
        Both cameras must detect the ChArUco board.
        
        Returns:
            True if both cameras detected the board successfully
        """
        # Detect in camera 1
        corners1, ids1 = self.detect_charuco(image1, K1, dist1)
        if corners1 is None:
            return False

        # Detect in camera 2
        corners2, ids2 = self.detect_charuco(image2, K2, dist2)
        if corners2 is None:
            return False

        # Estimate pose for both cameras
        ok1, rvec1, tvec1 = self.estimate_pose(corners1, ids1, K1, dist1)
        ok2, rvec2, tvec2 = self.estimate_pose(corners2, ids2, K2, dist2)

        if not (ok1 and ok2):
            return False

        self._cam1_rvecs.append(rvec1)
        self._cam1_tvecs.append(tvec1)
        self._cam2_rvecs.append(rvec2)
        self._cam2_tvecs.append(tvec2)
        self._valid_frames += 1

        logger.debug(f"Calibration frame {self._valid_frames} added successfully")
        return True

    def add_calibration_frame_single(
        self,
        image: np.ndarray,
        K: np.ndarray,
        dist: Optional[np.ndarray] = None,
    ) -> bool:
        """Collect ChArUco pose from a single camera (no stereo pair)."""
        corners, ids = self.detect_charuco(image, K, dist)
        if corners is None:
            return False
        ok, rvec, tvec = self.estimate_pose(corners, ids, K, dist)
        if not ok:
            return False
        self._cam1_rvecs.append(rvec)
        self._cam1_tvecs.append(tvec)
        self._valid_frames += 1
        logger.debug(f"Single-camera calibration frame {self._valid_frames} added")
        return True

    def compute_calibration(self, K1: np.ndarray, K2: np.ndarray,
                            dist1: Optional[np.ndarray] = None,
                            dist2: Optional[np.ndarray] = None
                            ) -> Optional[CalibrationResult]:
        """
        Compute final calibration from collected frames.
        
        Returns:
            CalibrationResult or None if calibration fails
        """
        if self._valid_frames < 3:
            logger.error(f"Not enough valid frames for calibration: "
                        f"{self._valid_frames} < 3")
            return None

        # Average the pose estimates for robustness
        # Convert rvecs to rotation matrices and average using SVD
        R1_avg, t1_avg = self._average_poses(self._cam1_rvecs, self._cam1_tvecs)
        R2_avg, t2_avg = self._average_poses(self._cam2_rvecs, self._cam2_tvecs)

        # Compute relative transformation: T_2in1 = T_1_to_world^-1 * T_2_to_world
        # But since both are expressed as board-to-camera transforms:
        # R_cam_from_board, t_cam_from_board
        # World frame = board frame (marker plane)
        
        # Camera 1 in world: R1, t1 transform world->cam1
        # Camera 2 in world: R2, t2 transform world->cam2
        # Relative: cam2 in cam1 frame
        R_rel = R2_avg @ R1_avg.T
        t_rel = t2_avg - R_rel @ t1_avg

        # Compute treadmill belt plane
        # The marker plane is at z=0 in world coordinates
        # Belt plane is offset downward by belt_offset
        # In world frame, "down" is typically -Y or +Z depending on setup
        # We'll define the marker plane normal as the Z-axis of the board
        belt_plane_normal = np.array([0, 0, 1], dtype=np.float64)  # Board Z-axis = up
        belt_plane_point = np.array([0, 0, -self.belt_offset], dtype=np.float64)

        # Compute reprojection error
        reproj_error = self._compute_reprojection_error(
            R1_avg, t1_avg, R2_avg, t2_avg, K1, K2, dist1, dist2
        )

        logger.info(f"Calibration complete. Reprojection error: {reproj_error:.3f} px")
        logger.info(f"Belt plane offset: {self.belt_offset*100:.1f} cm below markers")

        return CalibrationResult(
            R1=R1_avg, t1=t1_avg,
            R2=R2_avg, t2=t2_avg,
            R_rel=R_rel, t_rel=t_rel,
            belt_plane_normal=belt_plane_normal,
            belt_plane_point=belt_plane_point,
            reproj_error=reproj_error,
            K1=K1, K2=K2,
            dist1=dist1, dist2=dist2
        )

    def compute_calibration_single(
        self,
        K: np.ndarray,
        dist: Optional[np.ndarray] = None,
    ) -> Optional[CalibrationResult]:
        """Build calibration from single-camera ChArUco poses (stereo fields mirror cam1)."""
        if self._valid_frames < 3:
            logger.error(
                "Not enough single-camera frames for calibration: %s < 3",
                self._valid_frames,
            )
            return None
        dist = dist if dist is not None else np.zeros(5)
        R1, t1 = self._average_poses(self._cam1_rvecs, self._cam1_tvecs)
        belt_plane_normal = np.array([0, 0, 1], dtype=np.float64)
        belt_plane_point = np.array([0, 0, -self.belt_offset], dtype=np.float64)
        reproj_error = self._compute_reprojection_error(
            R1, t1, R1, t1, K, K, dist, dist
        )
        return CalibrationResult(
            R1=R1,
            t1=t1,
            R2=R1.copy(),
            t2=t1.copy(),
            R_rel=np.eye(3, dtype=np.float64),
            t_rel=np.zeros((3, 1), dtype=np.float64),
            belt_plane_normal=belt_plane_normal,
            belt_plane_point=belt_plane_point,
            reproj_error=float(reproj_error),
            K1=K.copy(),
            K2=K.copy(),
            dist1=np.array(dist),
            dist2=np.array(dist),
        )

    def _average_poses(self, rvecs: List[np.ndarray], 
                       tvecs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Average multiple pose estimates using rotation averaging."""
        # Convert rvecs to rotation matrices
        rotations = []
        for rvec in rvecs:
            R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
            rotations.append(R)

        # Average rotation using SVD (geodesic mean approximation)
        R_sum = np.zeros((3, 3))
        for R in rotations:
            R_sum += R
        U, S, Vt = np.linalg.svd(R_sum)
        R_avg = U @ Vt
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R_avg) < 0:
            U[:, -1] *= -1
            R_avg = U @ Vt

        # Average translation
        t_avg = np.mean(tvecs, axis=0).reshape(3, 1)

        return R_avg, t_avg

    def _compute_reprojection_error(self, R1, t1, R2, t2, K1, K2, dist1, dist2):
        """Estimate reprojection error using board corner projections."""
        # Generate 3D board points
        obj_points = self.charuco_board.getChessboardCorners()
        
        if dist1 is None:
            dist1 = np.zeros(5)
        if dist2 is None:
            dist2 = np.zeros(5)

        # Project to camera 1
        rvec1, _ = cv2.Rodrigues(R1)
        proj1, _ = cv2.projectPoints(obj_points, rvec1, t1, K1, dist1)

        # Project to camera 2
        rvec2, _ = cv2.Rodrigues(R2)
        proj2, _ = cv2.projectPoints(obj_points, rvec2, t2, K2, dist2)

        # Since we don't have ground truth 2D points here,
        # estimate error from consistency between frames
        errors = []
        for i in range(min(len(self._cam1_rvecs), 5)):
            R1_i, _ = cv2.Rodrigues(self._cam1_rvecs[i].reshape(3, 1))
            t1_i = self._cam1_tvecs[i].reshape(3, 1)
            proj_i, _ = cv2.projectPoints(obj_points, 
                                          self._cam1_rvecs[i].reshape(3, 1),
                                          t1_i, K1, dist1)
            proj_avg, _ = cv2.projectPoints(obj_points, rvec1, t1, K1, dist1)
            err = np.sqrt(np.mean((proj_i - proj_avg) ** 2))
            errors.append(err)

        return np.mean(errors) if errors else 0.0

    def calibrate_single_frame(self, image: np.ndarray,
                               camera_matrix: np.ndarray,
                               dist_coeffs: Optional[np.ndarray] = None
                               ) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Quick single-frame calibration for runtime use.
        Returns camera pose relative to the ChArUco board.
        """
        corners, ids = self.detect_charuco(image, camera_matrix, dist_coeffs)
        if corners is None:
            return False, np.eye(3), np.zeros((3, 1))
        
        return self.estimate_pose(corners, ids, camera_matrix, dist_coeffs)

    def save_calibration(self, result: CalibrationResult, filepath: str):
        """Save calibration result to file."""
        data = {
            'R1': result.R1.tolist(),
            't1': result.t1.tolist(),
            'R2': result.R2.tolist(),
            't2': result.t2.tolist(),
            'R_rel': result.R_rel.tolist(),
            't_rel': result.t_rel.tolist(),
            'belt_plane_normal': result.belt_plane_normal.tolist(),
            'belt_plane_point': result.belt_plane_point.tolist(),
            'reproj_error': result.reproj_error,
            'K1': result.K1.tolist(),
            'K2': result.K2.tolist(),
            'dist1': result.dist1.tolist() if result.dist1 is not None else None,
            'dist2': result.dist2.tolist() if result.dist2 is not None else None,
        }
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Calibration saved to {filepath}")

    @staticmethod
    def load_calibration(filepath: str) -> Optional[CalibrationResult]:
        """Load calibration result from file."""
        import json
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            return CalibrationResult(
                R1=np.array(data['R1']),
                t1=np.array(data['t1']),
                R2=np.array(data['R2']),
                t2=np.array(data['t2']),
                R_rel=np.array(data['R_rel']),
                t_rel=np.array(data['t_rel']),
                belt_plane_normal=np.array(data['belt_plane_normal']),
                belt_plane_point=np.array(data['belt_plane_point']),
                reproj_error=data['reproj_error'],
                K1=np.array(data['K1']),
                K2=np.array(data['K2']),
                dist1=np.array(data['dist1']) if data['dist1'] else None,
                dist2=np.array(data['dist2']) if data['dist2'] else None,
            )
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return None


def compute_treadmill_plane(R_cam: np.ndarray, t_cam: np.ndarray,
                            belt_offset: float = 0.015
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute treadmill belt plane in camera coordinates.
    
    The ChArUco markers define a plane. The belt is 'belt_offset' meters
    below this plane (in the board's -Z direction).
    
    Args:
        R_cam: Rotation from board frame to camera frame (3x3)
        t_cam: Translation from board frame to camera frame (3x1)
        belt_offset: Distance below marker plane (meters)
    
    Returns:
        (plane_normal_cam, plane_point_cam) in camera coordinates
    """
    # Board Z-axis in camera frame
    board_z_in_cam = R_cam[:, 2]  # Third column of rotation matrix
    
    # Plane normal (pointing away from belt surface, i.e., upward)
    plane_normal_cam = board_z_in_cam / np.linalg.norm(board_z_in_cam)
    
    # A point on the belt plane: origin of board shifted down by belt_offset
    board_origin_in_cam = t_cam.flatten()
    plane_point_cam = board_origin_in_cam - belt_offset * plane_normal_cam
    
    return plane_normal_cam, plane_point_cam


def point_to_plane_distance(point: np.ndarray, 
                            plane_normal: np.ndarray,
                            plane_point: np.ndarray) -> float:
    """
    Compute signed distance from a 3D point to a plane.
    Positive = above the plane (in normal direction).
    
    Args:
        point: 3D point (3,)
        plane_normal: Unit normal vector of the plane (3,)
        plane_point: A point on the plane (3,)
    
    Returns:
        Signed distance (positive above, negative below)
    """
    return np.dot(point - plane_point, plane_normal)
