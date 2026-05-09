"""
Video Reader Module for Orbbec Femto Mega MKV files.
Supports multiple backends: OrbbecSDK, Open3D, FFmpeg.
Extracts RGB frames, Depth frames, and timestamps from MKV recordings.
"""

import os
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Generator
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Container for a single frame of RGB-D data."""
    index: int
    timestamp_us: int           # Timestamp in microseconds from recording start
    system_time: Optional[datetime]  # Absolute system time if available
    color: Optional[np.ndarray]  # RGB image (H, W, 3) uint8
    depth: Optional[np.ndarray]  # Depth image (H, W) uint16 in mm
    color_intrinsics: Optional[dict] = None
    depth_intrinsics: Optional[dict] = None


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_coeffs: Optional[np.ndarray] = None  # k1, k2, p1, p2, k3...

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)


class BaseVideoReader(ABC):
    """Abstract base class for video readers."""

    @abstractmethod
    def open(self, filepath: str) -> bool:
        """Open a video file. Returns True on success."""
        pass

    @abstractmethod
    def close(self):
        """Release resources."""
        pass

    @abstractmethod
    def get_frame_count(self) -> int:
        """Get total number of frames."""
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """Get frames per second."""
        pass

    @abstractmethod
    def get_color_intrinsics(self) -> Optional[CameraIntrinsics]:
        """Get color camera intrinsics."""
        pass

    @abstractmethod
    def get_depth_intrinsics(self) -> Optional[CameraIntrinsics]:
        """Get depth camera intrinsics."""
        pass

    @abstractmethod
    def read_frame(self) -> Optional[FrameData]:
        """Read the next frame. Returns None at end of file."""
        pass

    @abstractmethod
    def seek(self, frame_index: int) -> bool:
        """Seek to a specific frame index."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def frames(self) -> Generator[FrameData, None, None]:
        """Generator that yields all frames."""
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame


class OrbbecVideoReader(BaseVideoReader):
    """
    Video reader using Orbbec SDK (pyorbbecsdk).
    Best for reading MKV files recorded with Orbbec Femto Mega.
    """

    def __init__(self):
        self._pipeline = None
        self._playback = None
        self._frame_index = 0
        self._total_frames = 0
        self._fps = 30.0
        self._color_intrinsics = None
        self._depth_intrinsics = None
        self._start_time = None
        self._first_frame_wall_time = None
        self._first_timestamp_us = None

    def open(self, filepath: str) -> bool:
        try:
            from pyorbbecsdk import Pipeline, Config, OBSensorType
            from pyorbbecsdk import OBFormat, PlaybackDevice

            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return False

            self._pipeline = Pipeline(filepath)
            playback = self._pipeline.get_playback()
            self._playback = playback

            # Get device info and camera parameters
            device_info = playback.get_device_info()
            camera_param = self._pipeline.get_camera_param()

            # Extract color intrinsics
            color_intr = camera_param.rgb_intrinsic
            self._color_intrinsics = CameraIntrinsics(
                fx=color_intr.fx, fy=color_intr.fy,
                cx=color_intr.cx, cy=color_intr.cy,
                width=color_intr.width, height=color_intr.height,
                distortion_coeffs=np.array(color_intr.distortion[:5])
            )

            # Extract depth intrinsics
            depth_intr = camera_param.depth_intrinsic
            self._depth_intrinsics = CameraIntrinsics(
                fx=depth_intr.fx, fy=depth_intr.fy,
                cx=depth_intr.cx, cy=depth_intr.cy,
                width=depth_intr.width, height=depth_intr.height,
                distortion_coeffs=np.array(depth_intr.distortion[:5])
            )

            # Get duration info
            duration_ms = playback.get_duration()
            self._fps = 30.0  # Default, will be updated from actual timestamps
            self._total_frames = int(duration_ms / 1000.0 * self._fps)

            # Optional: wall-clock anchor from SDK (if exposed by your pyorbbecsdk build)
            self._recording_start_wall = None
            for attr in (
                "get_start_time_system_time_us",
                "get_start_system_time_us",
                "get_start_time_us",
            ):
                if hasattr(playback, attr):
                    try:
                        us = getattr(playback, attr)()
                        if us and us > 0:
                            self._recording_start_wall = datetime.fromtimestamp(us / 1e6)
                            logger.info(f"Playback wall-clock start: {self._recording_start_wall}")
                            break
                    except Exception:
                        pass

            # Start pipeline for playback
            config = Config()
            self._pipeline.start(config)

            logger.info(f"Opened Orbbec MKV: {filepath}")
            logger.info(f"Duration: {duration_ms}ms, Est. frames: {self._total_frames}")
            return True

        except ImportError:
            logger.error("pyorbbecsdk not installed. Install with: pip install pyorbbecsdk")
            return False
        except Exception as e:
            logger.error(f"Failed to open Orbbec MKV: {e}")
            return False

    def close(self):
        if self._pipeline:
            try:
                self._pipeline.stop()
            except:
                pass
            self._pipeline = None

    def get_frame_count(self) -> int:
        return self._total_frames

    def get_fps(self) -> float:
        return self._fps

    def get_color_intrinsics(self) -> Optional[CameraIntrinsics]:
        return self._color_intrinsics

    def get_depth_intrinsics(self) -> Optional[CameraIntrinsics]:
        return self._depth_intrinsics

    def read_frame(self) -> Optional[FrameData]:
        try:
            frameset = self._pipeline.wait_for_frames(1000)
            if frameset is None:
                return None

            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()

            color_image = None
            depth_image = None
            timestamp_us = 0

            if color_frame:
                color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
                color_image = color_data.reshape(
                    color_frame.get_height(), color_frame.get_width(), -1
                )
                if color_image.shape[2] == 4:  # BGRA -> RGB
                    color_image = color_image[:, :, :3][:, :, ::-1]
                elif color_image.shape[2] == 3:  # BGR -> RGB
                    color_image = color_image[:, :, ::-1].copy()
                timestamp_us = color_frame.get_timestamp_us()

            if depth_frame:
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_image = depth_data.reshape(
                    depth_frame.get_height(), depth_frame.get_width()
                )
                if timestamp_us == 0:
                    timestamp_us = depth_frame.get_timestamp_us()

            system_time = None
            if color_frame:
                for attr in (
                    "get_global_timestamp_us",
                    "get_system_timestamp_us",
                    "get_global_time_stamp_us",
                ):
                    if hasattr(color_frame, attr):
                        try:
                            gts = int(getattr(color_frame, attr)())
                            if gts > 1_000_000_000_000:
                                system_time = datetime.fromtimestamp(gts / 1e6)
                                break
                            if gts > 1_000_000_000:
                                system_time = datetime.fromtimestamp(gts)
                                break
                        except Exception:
                            pass
            if system_time is None and self._recording_start_wall and timestamp_us > 0:
                system_time = self._recording_start_wall + timedelta(microseconds=timestamp_us)
            if system_time is None and timestamp_us > 0:
                if self._first_frame_wall_time is None:
                    self._first_frame_wall_time = datetime.now()
                    self._first_timestamp_us = timestamp_us
                delta = timestamp_us - self._first_timestamp_us
                system_time = self._first_frame_wall_time + timedelta(microseconds=delta)

            frame_data = FrameData(
                index=self._frame_index,
                timestamp_us=timestamp_us,
                system_time=system_time,
                color=color_image,
                depth=depth_image
            )
            self._frame_index += 1
            return frame_data

        except Exception as e:
            logger.debug(f"End of stream or error: {e}")
            return None

    def seek(self, frame_index: int) -> bool:
        # Orbbec SDK playback seek by timestamp
        if self._playback:
            try:
                target_us = int(frame_index / self._fps * 1e6)
                self._playback.seek(target_us)
                self._frame_index = frame_index
                if frame_index == 0:
                    self._first_frame_wall_time = None
                    self._first_timestamp_us = None
                return True
            except:
                return False
        return False


class Open3DVideoReader(BaseVideoReader):
    """
    Video reader using Open3D's AzureKinect MKV reader.
    Compatible with Orbbec Femto Mega recordings.
    """

    def __init__(self):
        self._reader = None
        self._frame_index = 0
        self._total_frames = 0
        self._fps = 30.0
        self._color_intrinsics = None
        self._depth_intrinsics = None
        self._metadata = None

    def open(self, filepath: str) -> bool:
        try:
            import open3d as o3d

            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return False

            self._reader = o3d.io.AzureKinectMKVReader()
            if not self._reader.open(filepath):
                logger.error(f"Failed to open MKV with Open3D: {filepath}")
                return False

            self._metadata = self._reader.get_metadata()

            # Extract intrinsics from metadata
            intrinsic = self._metadata.intrinsics
            w = intrinsic.width
            h = intrinsic.height
            intr_matrix = intrinsic.intrinsic_matrix

            self._color_intrinsics = CameraIntrinsics(
                fx=intr_matrix[0, 0], fy=intr_matrix[1, 1],
                cx=intr_matrix[0, 2], cy=intr_matrix[1, 2],
                width=w, height=h
            )
            self._depth_intrinsics = self._color_intrinsics  # Aligned

            # Estimate total frames (read through once if needed)
            self._fps = 30.0
            logger.info(f"Opened MKV with Open3D: {filepath}")
            return True

        except ImportError:
            logger.error("Open3D not installed. Install with: pip install open3d")
            return False
        except Exception as e:
            logger.error(f"Failed to open MKV with Open3D: {e}")
            return False

    def close(self):
        if self._reader:
            self._reader.close()
            self._reader = None

    def get_frame_count(self) -> int:
        return self._total_frames

    def get_fps(self) -> float:
        return self._fps

    def get_color_intrinsics(self) -> Optional[CameraIntrinsics]:
        return self._color_intrinsics

    def get_depth_intrinsics(self) -> Optional[CameraIntrinsics]:
        return self._depth_intrinsics

    def read_frame(self) -> Optional[FrameData]:
        try:
            import open3d as o3d

            if not self._reader.is_eof():
                rgbd = self._reader.next_frame()
                if rgbd is None:
                    return None

                color_image = np.asarray(rgbd.color).copy()
                depth_image = np.asarray(rgbd.depth).copy().astype(np.uint16)

                timestamp_us = int(self._frame_index / self._fps * 1e6)

                frame_data = FrameData(
                    index=self._frame_index,
                    timestamp_us=timestamp_us,
                    system_time=None,
                    color=color_image,
                    depth=depth_image
                )
                self._frame_index += 1
                return frame_data
            return None

        except Exception as e:
            logger.debug(f"Error reading frame: {e}")
            return None

    def seek(self, frame_index: int) -> bool:
        # Open3D MKV reader doesn't support direct seeking
        # Reset and skip frames
        return False


class FFmpegVideoReader(BaseVideoReader):
    """
    Video reader using FFmpeg/OpenCV for the color stream
    and raw binary extraction for depth.
    Fallback when SDK readers are not available.
    """

    def __init__(self, depth_scale: float = 1.0):
        self._color_cap = None
        self._depth_cap = None
        self._frame_index = 0
        self._total_frames = 0
        self._fps = 30.0
        self._color_intrinsics = None
        self._depth_intrinsics = None
        self._depth_scale = depth_scale
        self._filepath = ""
        self._timestamps = []

    def open(self, filepath: str) -> bool:
        try:
            import cv2
            import subprocess
            import json

            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return False

            self._filepath = filepath

            # Use ffprobe to get stream info
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-show_format', filepath
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                probe_data = json.loads(result.stdout)
                streams = probe_data.get('streams', [])
                for stream in streams:
                    if stream.get('codec_type') == 'video':
                        # Get FPS from first video stream
                        fps_str = stream.get('r_frame_rate', '30/1')
                        num, den = map(int, fps_str.split('/'))
                        self._fps = num / den if den > 0 else 30.0
                        break

            # Open color stream with OpenCV
            self._color_cap = cv2.VideoCapture(filepath)
            if not self._color_cap.isOpened():
                logger.error(f"Failed to open video with OpenCV: {filepath}")
                return False

            self._total_frames = int(self._color_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self._color_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._color_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Default intrinsics (user should provide actual values)
            self._color_intrinsics = CameraIntrinsics(
                fx=width, fy=width,
                cx=width / 2, cy=height / 2,
                width=width, height=height
            )
            self._depth_intrinsics = self._color_intrinsics

            logger.info(f"Opened MKV with FFmpeg/OpenCV: {filepath}")
            logger.info(f"Frames: {self._total_frames}, FPS: {self._fps}")
            logger.warning("Depth stream may not be available via FFmpeg backend. "
                          "Consider using 'orbbec' or 'open3d' backend for full RGB-D.")
            return True

        except Exception as e:
            logger.error(f"Failed to open with FFmpeg: {e}")
            return False

    def close(self):
        if self._color_cap:
            self._color_cap.release()
            self._color_cap = None

    def get_frame_count(self) -> int:
        return self._total_frames

    def get_fps(self) -> float:
        return self._fps

    def get_color_intrinsics(self) -> Optional[CameraIntrinsics]:
        return self._color_intrinsics

    def get_depth_intrinsics(self) -> Optional[CameraIntrinsics]:
        return self._depth_intrinsics

    def read_frame(self) -> Optional[FrameData]:
        import cv2

        if self._color_cap is None:
            return None

        ret, frame = self._color_cap.read()
        if not ret:
            return None

        # Convert BGR to RGB
        color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_us = int(self._color_cap.get(cv2.CAP_PROP_POS_MSEC) * 1000)

        frame_data = FrameData(
            index=self._frame_index,
            timestamp_us=timestamp_us,
            system_time=None,
            color=color_image,
            depth=None  # Depth not available via this backend
        )
        self._frame_index += 1
        return frame_data

    def seek(self, frame_index: int) -> bool:
        if self._color_cap:
            self._color_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            self._frame_index = frame_index
            return True
        return False


def create_video_reader(backend: str = "orbbec", **kwargs) -> BaseVideoReader:
    """
    Factory function to create the appropriate video reader.
    
    Args:
        backend: "orbbec", "open3d", or "ffmpeg"
    Returns:
        BaseVideoReader instance
    """
    if backend == "orbbec":
        return OrbbecVideoReader()
    elif backend == "open3d":
        return Open3DVideoReader()
    elif backend == "ffmpeg":
        return FFmpegVideoReader(**kwargs)
    else:
        logger.warning(f"Unknown backend '{backend}', falling back to ffmpeg")
        return FFmpegVideoReader(**kwargs)


class DualCameraReader:
    """
    Synchronized reader for two camera MKV files.
    Aligns frames by timestamp for stereo processing.
    """

    def __init__(self, backend: str = "orbbec"):
        self.reader1 = create_video_reader(backend)
        self.reader2 = create_video_reader(backend)
        self._sync_threshold_us = 16000  # 16ms sync threshold (~1 frame at 60fps)

    def open(self, filepath1: str, filepath2: str) -> bool:
        """Open both camera files."""
        ok1 = self.reader1.open(filepath1)
        ok2 = self.reader2.open(filepath2)
        if not (ok1 and ok2):
            logger.error("Failed to open one or both video files")
            self.close()
            return False
        return True

    def close(self):
        """Close both readers."""
        self.reader1.close()
        self.reader2.close()

    def synchronized_frames(self) -> Generator[Tuple[FrameData, FrameData], None, None]:
        """
        Yield synchronized frame pairs from both cameras.
        Uses timestamp-based synchronization.
        """
        frame1 = self.reader1.read_frame()
        frame2 = self.reader2.read_frame()

        while frame1 is not None and frame2 is not None:
            diff = abs(frame1.timestamp_us - frame2.timestamp_us)

            if diff <= self._sync_threshold_us:
                # Frames are synchronized
                yield frame1, frame2
                frame1 = self.reader1.read_frame()
                frame2 = self.reader2.read_frame()
            elif frame1.timestamp_us < frame2.timestamp_us:
                # Camera 1 is behind, advance it
                frame1 = self.reader1.read_frame()
            else:
                # Camera 2 is behind, advance it
                frame2 = self.reader2.read_frame()

    def rewind(self) -> bool:
        """Seek both readers to the first frame when the backend supports seeking."""
        ok1 = self.reader1.seek(0)
        ok2 = self.reader2.seek(0)
        return bool(ok1 and ok2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
