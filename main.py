"""
Main Pipeline for Dual-Camera Treadmill Gait Analysis.
Orchestrates the full processing workflow:
1. Load configuration
2. Open video files
3. Calibrate cameras using ChArUco markers
4. Run pose estimation on each frame
5. Reconstruct 3D poses
6. Analyze gait biomechanics
7. Output results to CSV
"""

import os
import sys
import time
import logging
import argparse
import yaml
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.video_reader import (
    create_video_reader, DualCameraReader, CameraIntrinsics, FrameData
)
from modules.calibration import CharucoCalibrator, CalibrationResult
from modules.pose_estimation import (
    create_pose_estimator, FramePoseResult, KEYPOINT_INDEX
)
from modules.reconstruction import PoseReconstructor, Pose3D
from modules.gait_analysis import GaitAnalyzer, GaitParameters, GaitCycleData
from modules.output import CSVWriter


# Configure logging
def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )


logger = logging.getLogger(__name__)


class GaitAnalysisPipeline:
    """
    Main pipeline class that orchestrates the complete gait analysis workflow.
    """

    def __init__(self, config_path: str):
        """
        Initialize pipeline with configuration file.
        
        Args:
            config_path: Path to config.yaml
        """
        self.config = self._load_config(config_path)
        self.calibration: Optional[CalibrationResult] = None
        self.pose_estimator = None
        self.reconstructor = None
        self.gait_analyzer = None

    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        logger.info(f"Configuration loaded from {config_path}")
        return config

    def run(self, video1_path: str = None, video2_path: str = None,
            output_dir: str = None):
        """
        Run the complete gait analysis pipeline.
        
        Args:
            video1_path: Override camera 1 video path
            video2_path: Override camera 2 video path
            output_dir: Override output directory
        """
        start_time = time.time()

        # Resolve paths
        cam1_path = video1_path or self.config['video']['camera1_path']
        cam2_path = video2_path or self.config['video']['camera2_path']
        out_dir = output_dir or self.config['output'].get('output_dir', './output')

        if not cam1_path or not cam2_path:
            raise ValueError("Both camera1_path and camera2_path must be specified")

        logger.info("=" * 60)
        logger.info("  Dual-Camera Treadmill Gait Analysis")
        logger.info("=" * 60)
        logger.info(f"Camera 1: {cam1_path}")
        logger.info(f"Camera 2: {cam2_path}")
        logger.info(f"Output: {out_dir}")
        logger.info("")

        # === Step 1: Open videos ===
        logger.info("[Step 1/6] Opening video files...")
        backend = self.config['video'].get('reader_backend', 'orbbec')
        dual_reader = DualCameraReader(backend=backend)

        if not dual_reader.open(cam1_path, cam2_path):
            raise RuntimeError("Failed to open video files")

        fps = dual_reader.reader1.get_fps()
        logger.info(f"  FPS: {fps}")

        # Get camera intrinsics
        K1, dist1 = self._get_intrinsics(dual_reader.reader1, 0)
        K2, dist2 = self._get_intrinsics(dual_reader.reader2, 1)
        logger.info(f"  Camera 1 intrinsics: fx={K1[0,0]:.1f}, fy={K1[1,1]:.1f}")
        logger.info(f"  Camera 2 intrinsics: fx={K2[0,0]:.1f}, fy={K2[1,1]:.1f}")

        # === Step 2: Calibration ===
        logger.info("\n[Step 2/6] Camera calibration via ChArUco markers...")
        calib_file = os.path.join(out_dir, "calibration.json")
        
        if os.path.exists(calib_file) and self.config['output'].get('save_calibration', True):
            logger.info(f"  Loading existing calibration from {calib_file}")
            self.calibration = CharucoCalibrator.load_calibration(calib_file)
        
        if self.calibration is None:
            self.calibration = self._run_calibration(
                dual_reader, K1, K2, dist1, dist2
            )
            if self.calibration is None:
                raise RuntimeError("Calibration failed")
            
            # Save calibration
            os.makedirs(out_dir, exist_ok=True)
            calibrator = CharucoCalibrator(self.config['charuco'])
            calibrator.save_calibration(self.calibration, calib_file)

        # === Step 3: Initialize pose estimator ===
        logger.info("\n[Step 3/6] Initializing pose estimation...")
        self.pose_estimator = create_pose_estimator(self.config['pose'])

        # === Step 4: Initialize 3D reconstructor ===
        logger.info("\n[Step 4/6] Setting up 3D reconstruction...")
        recon_config = self.config.get('reconstruction', {})
        recon_config['depth_scale'] = self.config['camera'].get('depth_scale', 0.001)
        self.reconstructor = PoseReconstructor(self.calibration, recon_config)

        # === Step 5: Initialize gait analyzer ===
        logger.info("\n[Step 5/6] Initializing gait analyzer...")
        self.gait_analyzer = GaitAnalyzer(self.config.get('gait', {}), fps=fps)
        self.gait_analyzer.set_belt_plane(
            self.calibration.belt_plane_normal,
            self.calibration.belt_plane_point
        )

        # === Step 6: Process frames ===
        logger.info("\n[Step 6/6] Processing video frames...")
        
        # Re-open videos for processing (after calibration consumed some frames)
        dual_reader.close()
        dual_reader = DualCameraReader(backend=backend)
        if not dual_reader.open(cam1_path, cam2_path):
            raise RuntimeError("Failed to re-open video files")

        timestamps = []
        frame_count = 0
        failed_frames = 0

        try:
            for frame1, frame2 in dual_reader.synchronized_frames():
                frame_count += 1

                if frame_count % 100 == 0:
                    logger.info(f"  Processing frame {frame_count}...")

                # Skip frames without color data
                if frame1.color is None or frame2.color is None:
                    failed_frames += 1
                    continue

                try:
                    # Run pose estimation on both views
                    pose_result1 = self.pose_estimator.estimate(frame1.color)
                    pose_result2 = self.pose_estimator.estimate(frame2.color)

                    # Get primary person from each view
                    if not pose_result1.persons or not pose_result2.persons:
                        failed_frames += 1
                        # Add empty pose to maintain frame alignment
                        empty_pose = Pose3D(
                            keypoints_3d=np.zeros((26, 3)),
                            confidence=np.zeros(26),
                            valid_mask=np.zeros(26, dtype=bool),
                            method="none"
                        )
                        self.gait_analyzer.add_frame(empty_pose, frame1.timestamp_us)
                        timestamps.append(frame1.timestamp_us)
                        continue

                    person1 = pose_result1.persons[pose_result1.primary_person_idx]
                    person2 = pose_result2.persons[pose_result2.primary_person_idx]

                    # 3D reconstruction
                    pose_3d = self.reconstructor.reconstruct(
                        person1, person2,
                        depth1=frame1.depth,
                        depth2=frame2.depth
                    )

                    # Add to gait analyzer
                    self.gait_analyzer.add_frame(pose_3d, frame1.timestamp_us)
                    timestamps.append(frame1.timestamp_us)

                except Exception as e:
                    logger.debug(f"Frame {frame_count} processing error: {e}")
                    failed_frames += 1
                    # Add placeholder
                    empty_pose = Pose3D(
                        keypoints_3d=np.zeros((26, 3)),
                        confidence=np.zeros(26),
                        valid_mask=np.zeros(26, dtype=bool),
                        method="none"
                    )
                    self.gait_analyzer.add_frame(empty_pose, frame1.timestamp_us)
                    timestamps.append(frame1.timestamp_us)

        finally:
            dual_reader.close()

        logger.info(f"  Processed {frame_count} frames ({failed_frames} failed)")

        # === Compute gait parameters ===
        logger.info("\nComputing gait parameters...")
        frame_params, cycle_data = self.gait_analyzer.compute_all_parameters()

        # === Write output ===
        logger.info("\nWriting results...")
        
        # Determine recording start time (from first frame or current time)
        recording_start = datetime.now()  # Default
        # If available from video metadata, use that instead

        csv_writer = CSVWriter(out_dir, start_time=recording_start, fps=fps)
        
        frame_csv_name = self.config['output'].get('frame_csv', 'gait_frame_data.csv')
        cycle_csv_name = self.config['output'].get('cycle_csv', 'gait_cycle_data.csv')

        frame_csv_path = csv_writer.write_frame_csv(
            frame_params, timestamps, filename=frame_csv_name
        )
        cycle_csv_path = csv_writer.write_cycle_csv(
            cycle_data, filename=cycle_csv_name
        )
        summary_path = csv_writer.write_summary(frame_params, cycle_data)

        # === Summary ===
        elapsed = time.time() - start_time
        logger.info("")
        logger.info("=" * 60)
        logger.info("  Analysis Complete!")
        logger.info("=" * 60)
        logger.info(f"  Total frames processed: {frame_count}")
        logger.info(f"  Gait cycles detected: {len(cycle_data)}")
        logger.info(f"  Processing time: {elapsed:.1f} seconds")
        logger.info(f"  Frame-level CSV: {frame_csv_path}")
        logger.info(f"  Cycle-level CSV: {cycle_csv_path}")
        logger.info(f"  Summary: {summary_path}")
        logger.info("=" * 60)

        # Release resources
        if self.pose_estimator:
            self.pose_estimator.release()

        return frame_csv_path, cycle_csv_path

    def _get_intrinsics(self, reader, camera_idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get camera intrinsic matrix from reader or config."""
        # Try from reader
        intrinsics = reader.get_color_intrinsics()
        if intrinsics:
            K = intrinsics.to_matrix()
            dist = intrinsics.distortion_coeffs
            return K, dist

        # Try from config
        cam_config = self.config.get('camera', {})
        intr_path = cam_config.get(f'camera{camera_idx+1}_intrinsics')
        if intr_path and os.path.exists(intr_path):
            import json
            with open(intr_path, 'r') as f:
                data = json.load(f)
            K = np.array(data['camera_matrix'])
            dist = np.array(data.get('distortion_coefficients', [0]*5))
            return K, dist

        # Default fallback (will need calibration)
        logger.warning(f"Using default intrinsics for camera {camera_idx+1}. "
                      "Results may be inaccurate.")
        K = np.array([
            [600, 0, 320],
            [0, 600, 240],
            [0, 0, 1]
        ], dtype=np.float64)
        return K, np.zeros(5)

    def _run_calibration(self, dual_reader: DualCameraReader,
                         K1: np.ndarray, K2: np.ndarray,
                         dist1: Optional[np.ndarray],
                         dist2: Optional[np.ndarray]) -> Optional[CalibrationResult]:
        """
        Run automatic calibration using ChArUco markers in the video.
        Samples frames from the beginning of the video.
        """
        charuco_config = self.config.get('charuco', {})
        calibrator = CharucoCalibrator(charuco_config)
        
        target_frames = charuco_config.get('calibration_frames', 30)
        sample_interval = charuco_config.get('calibration_interval', 10)
        
        logger.info(f"  Attempting calibration with {target_frames} frames "
                   f"(sampling every {sample_interval} frames)")

        frame_idx = 0
        valid_count = 0

        for frame1, frame2 in dual_reader.synchronized_frames():
            frame_idx += 1

            # Only sample every N frames
            if frame_idx % sample_interval != 0:
                continue

            if frame1.color is None or frame2.color is None:
                continue

            # Try to add calibration frame
            success = calibrator.add_calibration_frame(
                frame1.color, frame2.color, K1, K2, dist1, dist2
            )
            if success:
                valid_count += 1
                logger.debug(f"  Calibration frame {valid_count}/{target_frames} added")

            if valid_count >= target_frames:
                break

            # Don't process more than 1000 frames for calibration
            if frame_idx > 1000:
                break

        logger.info(f"  Collected {valid_count} valid calibration frames")

        if valid_count < 3:
            logger.error("  Not enough valid calibration frames!")
            return None

        # Compute calibration
        result = calibrator.compute_calibration(K1, K2, dist1, dist2)
        if result:
            logger.info(f"  Calibration successful! Reproj error: {result.reproj_error:.3f} px")
        
        return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dual-Camera Treadmill Gait Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with config file
  python main.py --config config.yaml --video1 cam1.mkv --video2 cam2.mkv

  # Specify output directory
  python main.py --config config.yaml --video1 cam1.mkv --video2 cam2.mkv --output ./results

  # Use MediaPipe backend
  python main.py --config config.yaml --video1 cam1.mkv --video2 cam2.mkv --pose-backend mediapipe

  # Verbose output
  python main.py --config config.yaml --video1 cam1.mkv --video2 cam2.mkv --log-level DEBUG
        """
    )

    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--video1', '-v1', type=str,
                       help='Path to camera 1 MKV video')
    parser.add_argument('--video2', '-v2', type=str,
                       help='Path to camera 2 MKV video')
    parser.add_argument('--output', '-o', type=str,
                       help='Output directory path')
    parser.add_argument('--pose-backend', type=str, choices=['mediapipe', 'mmpose'],
                       help='Override pose estimation backend')
    parser.add_argument('--recon-method', type=str, 
                       choices=['triangulation', 'depth', 'fusion'],
                       help='Override 3D reconstruction method')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--calibration-file', type=str,
                       help='Path to existing calibration JSON file')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Create pipeline
    try:
        pipeline = GaitAnalysisPipeline(args.config)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Apply command-line overrides
    if args.pose_backend:
        pipeline.config['pose']['backend'] = args.pose_backend
    if args.recon_method:
        pipeline.config['reconstruction']['method'] = args.recon_method
    if args.calibration_file:
        calib = CharucoCalibrator.load_calibration(args.calibration_file)
        if calib:
            pipeline.calibration = calib
            logger.info(f"Loaded calibration from {args.calibration_file}")
        else:
            logger.error(f"Failed to load calibration from {args.calibration_file}")
            sys.exit(1)

    # Run pipeline
    try:
        pipeline.run(
            video1_path=args.video1,
            video2_path=args.video2,
            output_dir=args.output
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
