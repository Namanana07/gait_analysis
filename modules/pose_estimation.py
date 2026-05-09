"""
Pose Estimation Module supporting MediaPipe and MMPose backends.
Provides unified interface for 2D human pose detection with GPU acceleration.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)


# Unified keypoint definition (HALPE 26 compatible + extra)
# This maps to a common set of body landmarks used for gait analysis
KEYPOINT_NAMES = [
    "nose",              # 0
    "left_eye",          # 1
    "right_eye",         # 2
    "left_ear",          # 3
    "right_ear",         # 4
    "left_shoulder",     # 5
    "right_shoulder",    # 6
    "left_elbow",        # 7
    "right_elbow",       # 8
    "left_wrist",        # 9
    "right_wrist",       # 10
    "left_hip",          # 11
    "right_hip",         # 12
    "left_knee",         # 13
    "right_knee",        # 14
    "left_ankle",        # 15
    "right_ankle",       # 16
    "head_top",          # 17
    "neck",              # 18
    "hip_center",        # 19 (pelvis)
    "left_big_toe",      # 20
    "right_big_toe",     # 21
    "left_small_toe",    # 22
    "right_small_toe",   # 23
    "left_heel",         # 24
    "right_heel",        # 25
]

KEYPOINT_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
NUM_KEYPOINTS = len(KEYPOINT_NAMES)


def fill_derived_keypoints(keypoints: np.ndarray, confidence: np.ndarray) -> None:
    """Neck and hip_center from shoulders/hips when missing."""
    if confidence[KEYPOINT_INDEX["neck"]] < 0.1:
        ls = KEYPOINT_INDEX["left_shoulder"]
        rs = KEYPOINT_INDEX["right_shoulder"]
        if confidence[ls] > 0.3 and confidence[rs] > 0.3:
            keypoints[KEYPOINT_INDEX["neck"]] = (keypoints[ls] + keypoints[rs]) / 2
            confidence[KEYPOINT_INDEX["neck"]] = min(confidence[ls], confidence[rs])
    if confidence[KEYPOINT_INDEX["hip_center"]] < 0.1:
        lh = KEYPOINT_INDEX["left_hip"]
        rh = KEYPOINT_INDEX["right_hip"]
        if confidence[lh] > 0.3 and confidence[rh] > 0.3:
            keypoints[KEYPOINT_INDEX["hip_center"]] = (keypoints[lh] + keypoints[rh]) / 2
            confidence[KEYPOINT_INDEX["hip_center"]] = min(confidence[lh], confidence[rh])


@dataclass
class PoseResult:
    """Result of pose estimation for a single person."""
    keypoints: np.ndarray          # (N, 2) array of (x, y) pixel coordinates
    confidence: np.ndarray         # (N,) confidence scores per keypoint
    bbox: Optional[np.ndarray] = None  # [x1, y1, x2, y2] bounding box
    person_score: float = 0.0      # Overall detection confidence


@dataclass
class FramePoseResult:
    """All pose detections in a single frame."""
    persons: List[PoseResult] = field(default_factory=list)
    frame_index: int = 0
    # The primary person (closest to center / largest bbox)
    primary_person_idx: int = 0


class BasePoseEstimator(ABC):
    """Abstract base class for pose estimators."""

    @abstractmethod
    def initialize(self, config: dict) -> bool:
        """Initialize the model. Returns True on success."""
        pass

    @abstractmethod
    def estimate(self, image: np.ndarray) -> FramePoseResult:
        """
        Estimate poses in an image.
        
        Args:
            image: RGB image (H, W, 3) uint8
        Returns:
            FramePoseResult with detected persons
        """
        pass

    @abstractmethod
    def release(self):
        """Release model resources."""
        pass

    def select_primary_person(self, result: FramePoseResult, 
                              image_shape: Tuple[int, int]) -> int:
        """
        Select the primary person (the one on the treadmill).
        Uses bbox size and center proximity as heuristics.
        """
        if len(result.persons) == 0:
            return 0
        if len(result.persons) == 1:
            return 0

        h, w = image_shape[:2]
        center = np.array([w / 2, h / 2])
        
        best_idx = 0
        best_score = -1

        for i, person in enumerate(result.persons):
            if person.bbox is not None:
                bbox = person.bbox
                bbox_center = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
                bbox_area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                
                # Score: larger bbox + closer to center
                dist = np.linalg.norm(bbox_center - center)
                score = bbox_area / (dist + 1)
            else:
                # Use mean keypoint position
                valid = person.confidence > 0.3
                if np.any(valid):
                    kp_center = person.keypoints[valid].mean(axis=0)
                    dist = np.linalg.norm(kp_center - center)
                    score = np.sum(valid) / (dist + 1)
                else:
                    score = 0

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx


class MediaPipePoseEstimator(BasePoseEstimator):
    """
    Pose estimator using Google MediaPipe.
    Fast and lightweight, provides 33 body landmarks.
    """

    # Mapping from MediaPipe landmark indices to our unified format
    MP_TO_UNIFIED = {
        0: KEYPOINT_INDEX["nose"],
        2: KEYPOINT_INDEX["left_eye"],
        5: KEYPOINT_INDEX["right_eye"],
        7: KEYPOINT_INDEX["left_ear"],
        8: KEYPOINT_INDEX["right_ear"],
        11: KEYPOINT_INDEX["left_shoulder"],
        12: KEYPOINT_INDEX["right_shoulder"],
        13: KEYPOINT_INDEX["left_elbow"],
        14: KEYPOINT_INDEX["right_elbow"],
        15: KEYPOINT_INDEX["left_wrist"],
        16: KEYPOINT_INDEX["right_wrist"],
        23: KEYPOINT_INDEX["left_hip"],
        24: KEYPOINT_INDEX["right_hip"],
        25: KEYPOINT_INDEX["left_knee"],
        26: KEYPOINT_INDEX["right_knee"],
        27: KEYPOINT_INDEX["left_ankle"],
        28: KEYPOINT_INDEX["right_ankle"],
        31: KEYPOINT_INDEX["left_big_toe"],
        32: KEYPOINT_INDEX["right_big_toe"],
        29: KEYPOINT_INDEX["left_heel"],
        30: KEYPOINT_INDEX["right_heel"],
    }

    def __init__(self):
        self._pose = None
        self._config = {}

    def initialize(self, config: dict) -> bool:
        try:
            import mediapipe as mp
            
            mp_config = config.get("mediapipe", {})
            
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=mp_config.get("model_complexity", 2),
                enable_segmentation=False,
                min_detection_confidence=mp_config.get("min_detection_confidence", 0.5),
                min_tracking_confidence=mp_config.get("min_tracking_confidence", 0.5),
            )
            
            self._config = mp_config
            logger.info("MediaPipe Pose initialized successfully")
            return True

        except ImportError:
            logger.error("MediaPipe not installed. Install with: pip install mediapipe")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            return False

    def estimate(self, image: np.ndarray) -> FramePoseResult:
        if self._pose is None:
            return FramePoseResult()

        # MediaPipe expects RGB
        results = self._pose.process(image)

        frame_result = FramePoseResult()

        if results.pose_landmarks:
            h, w = image.shape[:2]
            landmarks = results.pose_landmarks.landmark

            keypoints = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
            confidence = np.zeros(NUM_KEYPOINTS, dtype=np.float32)

            for mp_idx, unified_idx in self.MP_TO_UNIFIED.items():
                lm = landmarks[mp_idx]
                keypoints[unified_idx] = [lm.x * w, lm.y * h]
                confidence[unified_idx] = lm.visibility

            # Compute derived keypoints
            # Neck = midpoint of shoulders
            if confidence[KEYPOINT_INDEX["left_shoulder"]] > 0.3 and \
               confidence[KEYPOINT_INDEX["right_shoulder"]] > 0.3:
                keypoints[KEYPOINT_INDEX["neck"]] = (
                    keypoints[KEYPOINT_INDEX["left_shoulder"]] +
                    keypoints[KEYPOINT_INDEX["right_shoulder"]]
                ) / 2
                confidence[KEYPOINT_INDEX["neck"]] = min(
                    confidence[KEYPOINT_INDEX["left_shoulder"]],
                    confidence[KEYPOINT_INDEX["right_shoulder"]]
                )

            # Hip center = midpoint of hips
            if confidence[KEYPOINT_INDEX["left_hip"]] > 0.3 and \
               confidence[KEYPOINT_INDEX["right_hip"]] > 0.3:
                keypoints[KEYPOINT_INDEX["hip_center"]] = (
                    keypoints[KEYPOINT_INDEX["left_hip"]] +
                    keypoints[KEYPOINT_INDEX["right_hip"]]
                ) / 2
                confidence[KEYPOINT_INDEX["hip_center"]] = min(
                    confidence[KEYPOINT_INDEX["left_hip"]],
                    confidence[KEYPOINT_INDEX["right_hip"]]
                )

            # Head top approximation from nose
            if confidence[KEYPOINT_INDEX["nose"]] > 0.3:
                nose = keypoints[KEYPOINT_INDEX["nose"]]
                # Approximate head top as slightly above nose
                if confidence[KEYPOINT_INDEX["neck"]] > 0.3:
                    neck = keypoints[KEYPOINT_INDEX["neck"]]
                    head_dir = nose - neck
                    keypoints[KEYPOINT_INDEX["head_top"]] = nose + head_dir * 0.5
                    confidence[KEYPOINT_INDEX["head_top"]] = confidence[KEYPOINT_INDEX["nose"]] * 0.8

            person = PoseResult(
                keypoints=keypoints,
                confidence=confidence,
                person_score=np.mean(confidence[confidence > 0])
            )
            frame_result.persons.append(person)

        return frame_result

    def release(self):
        if self._pose:
            self._pose.close()
            self._pose = None


class MMPosePoseEstimator(BasePoseEstimator):
    """
    Pose estimator using MMPose/RTMPose.
    High accuracy with GPU acceleration via NVIDIA A5000.
    """

    # RTMPose HALPE26 keypoint mapping to unified format
    HALPE26_TO_UNIFIED = {
        0: KEYPOINT_INDEX["nose"],
        1: KEYPOINT_INDEX["left_eye"],
        2: KEYPOINT_INDEX["right_eye"],
        3: KEYPOINT_INDEX["left_ear"],
        4: KEYPOINT_INDEX["right_ear"],
        5: KEYPOINT_INDEX["left_shoulder"],
        6: KEYPOINT_INDEX["right_shoulder"],
        7: KEYPOINT_INDEX["left_elbow"],
        8: KEYPOINT_INDEX["right_elbow"],
        9: KEYPOINT_INDEX["left_wrist"],
        10: KEYPOINT_INDEX["right_wrist"],
        11: KEYPOINT_INDEX["left_hip"],
        12: KEYPOINT_INDEX["right_hip"],
        13: KEYPOINT_INDEX["left_knee"],
        14: KEYPOINT_INDEX["right_knee"],
        15: KEYPOINT_INDEX["left_ankle"],
        16: KEYPOINT_INDEX["right_ankle"],
        17: KEYPOINT_INDEX["head_top"],
        18: KEYPOINT_INDEX["neck"],
        19: KEYPOINT_INDEX["hip_center"],
        20: KEYPOINT_INDEX["left_big_toe"],
        21: KEYPOINT_INDEX["right_big_toe"],
        22: KEYPOINT_INDEX["left_small_toe"],
        23: KEYPOINT_INDEX["right_small_toe"],
        24: KEYPOINT_INDEX["left_heel"],
        25: KEYPOINT_INDEX["right_heel"],
    }

    # COCO 17 keypoint mapping (fallback)
    COCO17_TO_UNIFIED = {
        0: KEYPOINT_INDEX["nose"],
        1: KEYPOINT_INDEX["left_eye"],
        2: KEYPOINT_INDEX["right_eye"],
        3: KEYPOINT_INDEX["left_ear"],
        4: KEYPOINT_INDEX["right_ear"],
        5: KEYPOINT_INDEX["left_shoulder"],
        6: KEYPOINT_INDEX["right_shoulder"],
        7: KEYPOINT_INDEX["left_elbow"],
        8: KEYPOINT_INDEX["right_elbow"],
        9: KEYPOINT_INDEX["left_wrist"],
        10: KEYPOINT_INDEX["right_wrist"],
        11: KEYPOINT_INDEX["left_hip"],
        12: KEYPOINT_INDEX["right_hip"],
        13: KEYPOINT_INDEX["left_knee"],
        14: KEYPOINT_INDEX["right_knee"],
        15: KEYPOINT_INDEX["left_ankle"],
        16: KEYPOINT_INDEX["right_ankle"],
    }

    def __init__(self):
        self._detector = None
        self._pose_estimator = None
        self._config = {}
        self._device = "cuda:0"
        self._keypoint_mapping = self.HALPE26_TO_UNIFIED

    def initialize(self, config: dict) -> bool:
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self._device = "cpu"
            else:
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"Using GPU: {device_name}")
                
            mmpose_config = config.get("mmpose", {})
            self._device = mmpose_config.get("device", "cuda:0")
            model_dir = mmpose_config.get("model_dir", "./models")

            # Try to use mmpose inferencer (simpler API)
            return self._init_with_inferencer(mmpose_config, model_dir)

        except ImportError as e:
            logger.error(f"MMPose dependencies not installed: {e}")
            logger.error("Install with: pip install mmpose mmdet mmcv mmengine")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize MMPose: {e}")
            return False

    def _init_with_inferencer(self, config: dict, model_dir: str) -> bool:
        """Initialize using MMPose Inferencer API (recommended)."""
        try:
            from mmpose.apis import MMPoseInferencer
            
            # Use RTMPose with HALPE26 keypoints for full body including feet
            pose_config = config.get("pose_config", 
                "rtmpose-l_8xb256-420e_body8-halpe26-256x192.py")
            pose_checkpoint = config.get("pose_checkpoint",
                "rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth")
            
            det_config = config.get("det_config",
                "rtmdet_m_640-8xb32_coco-person.py")
            det_checkpoint = config.get("det_checkpoint",
                "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth")

            self._inferencer = MMPoseInferencer(
                pose2d=pose_config,
                pose2d_weights=pose_checkpoint,
                det_model=det_config,
                det_weights=det_checkpoint,
                device=self._device
            )
            
            # Determine keypoint format
            if "halpe26" in pose_config.lower():
                self._keypoint_mapping = self.HALPE26_TO_UNIFIED
                self._num_model_kpts = 26
            else:
                self._keypoint_mapping = self.COCO17_TO_UNIFIED
                self._num_model_kpts = 17

            logger.info(f"MMPose Inferencer initialized on {self._device}")
            logger.info(f"Pose model: {pose_config}")
            logger.info(f"Detection model: {det_config}")
            return True

        except Exception as e:
            logger.warning(f"MMPose Inferencer failed: {e}")
            return self._init_with_topdown(config, model_dir)

    def _init_with_topdown(self, config: dict, model_dir: str) -> bool:
        """Initialize using top-down API (fallback)."""
        try:
            from mmpose.apis import init_model as init_pose_model
            from mmpose.apis import inference_topdown
            from mmdet.apis import init_detector, inference_detector

            det_config = config.get("det_config",
                "rtmdet_m_640-8xb32_coco-person.py")
            det_checkpoint = config.get("det_checkpoint",
                "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth")
            pose_config = config.get("pose_config",
                "rtmpose-l_8xb256-420e_body8-halpe26-256x192.py")
            pose_checkpoint = config.get("pose_checkpoint",
                "rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth")

            import os
            det_config_path = os.path.join(model_dir, det_config)
            det_ckpt_path = os.path.join(model_dir, det_checkpoint)
            pose_config_path = os.path.join(model_dir, pose_config)
            pose_ckpt_path = os.path.join(model_dir, pose_checkpoint)

            self._detector = init_detector(
                det_config_path, det_ckpt_path, device=self._device
            )
            self._pose_estimator = init_pose_model(
                pose_config_path, pose_ckpt_path, device=self._device
            )

            self._det_score_thr = config.get("det_score_thr", 0.5)
            
            if "halpe26" in pose_config.lower():
                self._keypoint_mapping = self.HALPE26_TO_UNIFIED
                self._num_model_kpts = 26
            else:
                self._keypoint_mapping = self.COCO17_TO_UNIFIED
                self._num_model_kpts = 17

            logger.info(f"MMPose top-down API initialized on {self._device}")
            return True

        except Exception as e:
            logger.error(f"MMPose top-down init failed: {e}")
            return False

    def estimate(self, image: np.ndarray) -> FramePoseResult:
        """Run pose estimation on an image."""
        frame_result = FramePoseResult()

        try:
            if hasattr(self, '_inferencer') and self._inferencer is not None:
                return self._estimate_inferencer(image)
            elif self._detector is not None and self._pose_estimator is not None:
                return self._estimate_topdown(image)
        except Exception as e:
            logger.warning(f"Pose estimation error: {e}")

        return frame_result

    def _estimate_inferencer(self, image: np.ndarray) -> FramePoseResult:
        """Estimate using MMPose Inferencer."""
        import cv2
        
        # MMPose expects BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        result_generator = self._inferencer(image_bgr, return_vis=False)
        result = next(result_generator)
        
        frame_result = FramePoseResult()
        
        predictions = result.get('predictions', [[]])[0]
        
        for pred in predictions:
            keypoints_raw = np.array(pred['keypoints'])  # (N, 2)
            scores_raw = np.array(pred['keypoint_scores'])  # (N,)
            bbox = np.array(pred.get('bbox', [None]))[0]
            
            # Map to unified format
            keypoints = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
            confidence = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
            
            for model_idx, unified_idx in self._keypoint_mapping.items():
                if model_idx < len(keypoints_raw):
                    keypoints[unified_idx] = keypoints_raw[model_idx]
                    confidence[unified_idx] = scores_raw[model_idx]
            
            # Compute derived keypoints if not directly available
            fill_derived_keypoints(keypoints, confidence)
            
            person = PoseResult(
                keypoints=keypoints,
                confidence=confidence,
                bbox=bbox,
                person_score=float(np.mean(scores_raw))
            )
            frame_result.persons.append(person)
        
        if frame_result.persons:
            frame_result.primary_person_idx = self.select_primary_person(
                frame_result, image.shape
            )
        
        return frame_result

    def _estimate_topdown(self, image: np.ndarray) -> FramePoseResult:
        """Estimate using top-down API."""
        import cv2
        from mmdet.apis import inference_detector
        from mmpose.apis import inference_topdown
        from mmpose.structures import merge_data_samples
        
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detect persons
        det_result = inference_detector(self._detector, image_bgr)
        
        # Filter person detections
        pred_instances = det_result.pred_instances
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
        
        # Keep only person class (label 0) with sufficient score
        person_mask = (labels == 0) & (scores > self._det_score_thr)
        person_bboxes = bboxes[person_mask]
        
        if len(person_bboxes) == 0:
            return FramePoseResult()
        
        # Run pose estimation
        pose_results = inference_topdown(
            self._pose_estimator, image_bgr, person_bboxes
        )
        
        frame_result = FramePoseResult()
        
        for i, pose_result in enumerate(pose_results):
            pred_instances = pose_result.pred_instances
            keypoints_raw = pred_instances.keypoints[0].cpu().numpy()
            scores_raw = pred_instances.keypoint_scores[0].cpu().numpy()
            
            keypoints = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
            confidence = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
            
            for model_idx, unified_idx in self._keypoint_mapping.items():
                if model_idx < len(keypoints_raw):
                    keypoints[unified_idx] = keypoints_raw[model_idx]
                    confidence[unified_idx] = scores_raw[model_idx]
            
            fill_derived_keypoints(keypoints, confidence)
            
            person = PoseResult(
                keypoints=keypoints,
                confidence=confidence,
                bbox=person_bboxes[i] if i < len(person_bboxes) else None,
                person_score=float(scores[person_mask][i]) if i < np.sum(person_mask) else 0.0
            )
            frame_result.persons.append(person)
        
        if frame_result.persons:
            frame_result.primary_person_idx = self.select_primary_person(
                frame_result, image.shape
            )
        
        return frame_result

    def release(self):
        """Release GPU resources."""
        self._detector = None
        self._pose_estimator = None
        if hasattr(self, '_inferencer'):
            self._inferencer = None
        
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass


class UltralyticsPoseEstimator(BasePoseEstimator):
    """
    YOLOv8/YOLO11-pose via Ultralytics (CUDA-friendly on Windows + NVIDIA).
    """

    def __init__(self):
        self._model = None
        self._device = "0"
        self._imgsz = 640
        self._conf = 0.25

    def initialize(self, config: dict) -> bool:
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed. pip install ultralytics")
            return False
        ucfg = config.get("ultralytics", {})
        weights = ucfg.get("weights", "yolo11n-pose.pt")
        self._device = ucfg.get("device", "0")
        self._imgsz = int(ucfg.get("imgsz", 640))
        self._conf = float(ucfg.get("conf", 0.25))
        try:
            self._model = YOLO(weights)
            self._model.to(self._device)
            logger.info("Ultralytics pose model loaded: %s on %s", weights, self._device)
            return True
        except Exception as e:
            logger.error("Ultralytics init failed: %s", e)
            return False

    def estimate(self, image: np.ndarray) -> FramePoseResult:
        if self._model is None:
            return FramePoseResult()
        import cv2

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self._model.predict(
            bgr,
            imgsz=self._imgsz,
            conf=self._conf,
            verbose=False,
            device=self._device,
        )
        frame_result = FramePoseResult()
        if not results:
            return frame_result
        r = results[0]
        if r.keypoints is None or r.keypoints.xy is None:
            return frame_result
        xy = r.keypoints.xy.cpu().numpy()
        if xy.size == 0:
            return frame_result
        sc = r.keypoints.conf
        if sc is not None:
            sc = sc.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None
        box_sc = r.boxes.conf.cpu().numpy() if r.boxes is not None else None

        for pi in range(xy.shape[0]):
            raw = xy[pi]
            n = raw.shape[0]
            scores = sc[pi] if sc is not None else np.ones(n, dtype=np.float32)
            mapping = (
                MMPosePoseEstimator.COCO17_TO_UNIFIED
                if n <= 17
                else MMPosePoseEstimator.HALPE26_TO_UNIFIED
            )
            keypoints = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
            confidence = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
            for model_idx, unified_idx in mapping.items():
                if model_idx < n:
                    keypoints[unified_idx] = raw[model_idx]
                    confidence[unified_idx] = float(scores[model_idx])
            fill_derived_keypoints(keypoints, confidence)
            bbox = boxes[pi] if boxes is not None and pi < boxes.shape[0] else None
            pscore = (
                float(box_sc[pi])
                if box_sc is not None and pi < box_sc.shape[0]
                else float(np.mean(scores))
            )
            frame_result.persons.append(
                PoseResult(keypoints=keypoints, confidence=confidence, bbox=bbox, person_score=pscore)
            )
        if frame_result.persons:
            frame_result.primary_person_idx = self.select_primary_person(frame_result, image.shape)
        return frame_result

    def release(self):
        self._model = None


def create_pose_estimator(config: dict) -> BasePoseEstimator:
    """
    Factory function to create pose estimator based on configuration.
    
    Args:
        config: Pose estimation configuration dictionary
    Returns:
        Initialized BasePoseEstimator instance
    """
    backend = config.get("backend", "mmpose")
    
    if backend == "mediapipe":
        estimator = MediaPipePoseEstimator()
    elif backend == "mmpose":
        estimator = MMPosePoseEstimator()
    elif backend == "ultralytics":
        estimator = UltralyticsPoseEstimator()
    else:
        logger.warning(f"Unknown backend '{backend}', defaulting to mmpose")
        estimator = MMPosePoseEstimator()
    
    if not estimator.initialize(config):
        logger.error(f"Failed to initialize {backend} backend")
        # Try fallback
        if backend == "mmpose":
            logger.info("Falling back to MediaPipe")
            estimator = MediaPipePoseEstimator()
            if not estimator.initialize(config):
                raise RuntimeError("All pose estimation backends failed to initialize")
    
    return estimator
