"""
Dual Orbbec Femto Mega MKV gait analysis -> per-frame CSV (system time + 8 metrics).

Windows: use Python 3.10+, install CUDA PyTorch if using MMPose; Ultralytics works with
`pip install ultralytics` and GPU via the same CUDA stack.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from modules.calibration import CalibrationResult, CharucoCalibrator
from modules.metrics import GaitTracker, compute_frame_metrics
from modules.pose_estimation import PoseResult, create_pose_estimator
from modules.reconstruction import PoseReconstructor
from modules.video_reader import CameraIntrinsics, DualCameraReader, create_video_reader

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "time",
    "upper_body_pitch_deg",
    "hip_adduction_abduction_deg",
    "hip_flexion_extension_deg",
    "knee_flexion_extension_deg",
    "ankle_angle_deg",
    "foot_max_clearance_m",
    "step_length_m",
    "step_frequency_steps_per_min",
]


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _K_and_dist(ci: CameraIntrinsics) -> Tuple[np.ndarray, np.ndarray]:
    K = ci.to_matrix()
    dist = ci.distortion_coeffs
    if dist is None:
        dist = np.zeros(5, dtype=np.float64)
    else:
        dist = np.asarray(dist, dtype=np.float64).reshape(-1)
        if len(dist) < 5:
            d = np.zeros(5, dtype=np.float64)
            d[: len(dist)] = dist
            dist = d
    return K, dist


def _calibrate_stereo(
    dual: DualCameraReader,
    charuco_cfg: dict,
    K1: np.ndarray,
    K2: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    n_frames: int,
    interval: int,
) -> Optional[CalibrationResult]:
    cal = CharucoCalibrator(charuco_cfg)
    added = 0
    for i, (f1, f2) in enumerate(dual.synchronized_frames()):
        if i % interval != 0:
            continue
        if f1.color is None or f2.color is None:
            continue
        if cal.add_calibration_frame(f1.color, f2.color, K1, K2, d1, d2):
            added += 1
            logger.info("Calibration stereo frame %s / %s", added, n_frames)
        if added >= n_frames:
            break
    if added < 3:
        logger.error("Stereo ChArUco calibration failed (need >= 3 valid frames).")
        return None
    return cal.compute_calibration(K1, K2, d1, d2)


def _calibrate_single_cam(
    reader,
    charuco_cfg: dict,
    K: np.ndarray,
    d: np.ndarray,
    n_frames: int,
    interval: int,
) -> Optional[CalibrationResult]:
    cal = CharucoCalibrator(charuco_cfg)
    added = 0
    for i, frame in enumerate(reader.frames()):
        if i % interval != 0:
            continue
        if frame.color is None:
            continue
        if cal.add_calibration_frame_single(frame.color, K, d):
            added += 1
            logger.info("Calibration single-camera frame %s / %s", added, n_frames)
        if added >= n_frames:
            break
    if added < 3:
        logger.error("Single-camera ChArUco calibration failed (need >= 3 valid frames).")
        return None
    return cal.compute_calibration_single(K, d)


def _format_time(t: Optional[datetime]) -> str:
    if t is None:
        return ""
    return t.strftime("%Y-%m-%d %H:%M:%S.%f")


def _primary_pose(fr, shape) -> Optional[PoseResult]:
    if not fr.persons:
        return None
    idx = int(np.clip(fr.primary_person_idx, 0, len(fr.persons) - 1))
    return fr.persons[idx]


def _forward_hint_world(cfg: dict, calib: CalibrationResult) -> np.ndarray:
    g = cfg.get("gait", {})
    h = g.get("forward_hint_board", [1.0, 0.0, 0.0])
    return np.asarray(h, dtype=np.float64).reshape(3)


def _belt_world(calib: CalibrationResult) -> Tuple[np.ndarray, np.ndarray]:
    n = np.asarray(calib.belt_plane_normal, dtype=np.float64).reshape(3)
    p = np.asarray(calib.belt_plane_point, dtype=np.float64).reshape(3)
    return n, p


def _maybe_smooth(columns: Dict[str, np.ndarray], filt: dict) -> None:
    if not filt.get("enabled", False):
        return
    try:
        from scipy.signal import butter, filtfilt, savgol_filter
    except ImportError:
        logger.warning("scipy not installed; skipping temporal filter.")
        return
    keys = [
        "upper_body_pitch_deg",
        "hip_adduction_abduction_deg",
        "hip_flexion_extension_deg",
        "knee_flexion_extension_deg",
        "ankle_angle_deg",
        "foot_max_clearance_m",
    ]
    n = len(next(iter(columns.values())))
    if n < 5:
        return
    ftype = filt.get("type", "savgol")
    if ftype == "savgol":
        w = min(int(filt.get("savgol_window", 11)), n // 2 * 2 + 1)
        if w < 5:
            return
        po = int(filt.get("savgol_polyorder", 3))
        for k in keys:
            if k not in columns:
                continue
            x = columns[k].astype(float)
            mask = np.isfinite(x)
            if np.sum(mask) < w:
                continue
            columns[k][:] = savgol_filter(np.nan_to_num(x, nan=np.nanmedian(x)), w, po)
    elif ftype == "butterworth":
        fps = float(filt.get("assumed_fps", 30.0))
        cut = float(filt.get("cutoff_freq", 6.0))
        order = int(filt.get("order", 4))
        ny = 0.5 * fps
        b, a = butter(order, cut / ny, btype="low")
        for k in keys:
            if k not in columns:
                continue
            x = np.nan_to_num(columns[k].astype(float), nan=np.nanmedian(columns[k]))
            if len(x) > 3 * order:
                columns[k][:] = filtfilt(b, a, x)


def run_dual_mkv(cfg: dict) -> str:
    vcfg = cfg["video"]
    backend = vcfg.get("reader_backend", "orbbec")
    path1 = vcfg["camera1_path"]
    path2 = vcfg.get("camera2_path") or ""

    char_cfg = cfg["charuco"]
    n_cal = int(char_cfg.get("calibration_frames", 30))
    cal_int = int(char_cfg.get("calibration_interval", 10))

    out_dir = cfg.get("output", {}).get("output_dir", "./output")
    os.makedirs(out_dir, exist_ok=True)
    csv_name = cfg.get("output", {}).get("frame_csv", "gait_frame_data.csv")
    out_csv = os.path.join(out_dir, csv_name)

    pose_est = create_pose_estimator(cfg["pose"])
    rows_buffer: List[dict] = []
    columns: Dict[str, List[float]] = {c: [] for c in CSV_COLUMNS[1:]}

    try:
        if path2:
            dual = DualCameraReader(backend)
            if not dual.open(path1, path2):
                raise RuntimeError("Could not open one or both MKV files.")
            K1, d1 = _K_and_dist(dual.reader1.get_color_intrinsics())
            K2, d2 = _K_and_dist(dual.reader2.get_color_intrinsics())

            calib = _calibrate_stereo(dual, char_cfg, K1, K2, d1, d2, n_cal, cal_int)
            if calib is None:
                raise RuntimeError("Calibration failed.")

            if not dual.rewind():
                logger.warning("Seek to start not supported; reopening MKV files.")
                dual.close()
                if not dual.open(path1, path2):
                    raise RuntimeError("Reopen after calibration failed.")

            recon = PoseReconstructor(calib, cfg["reconstruction"])
            fwd_hint = _forward_hint_world(cfg, calib)
            belt_n, belt_p = _belt_world(calib)
            gcfg = cfg.get("gait", {})
            tracker = GaitTracker(
                belt_normal_world=belt_n,
                belt_point_world=belt_p,
                forward=fwd_hint,
                airborne_thresh_m=float(gcfg.get("airborne_thresh_m", 0.04)),
                contact_thresh_m=float(gcfg.get("contact_thresh_m", 0.018)),
            )

            for f1, f2 in dual.synchronized_frames():
                if f1.color is None:
                    continue
                t_wall = f1.system_time or f2.system_time
                t_s = (f1.timestamp_us or 0) / 1e6

                p1 = pose_est.estimate(f1.color)
                p2 = pose_est.estimate(f2.color) if f2.color is not None else p1
                pr1 = _primary_pose(p1, f1.color.shape)
                pr2 = _primary_pose(p2, f2.color.shape if f2.color is not None else f1.color.shape)
                if pr1 is None or pr2 is None:
                    row = {c: np.nan for c in CSV_COLUMNS[1:]}
                    row["time"] = _format_time(t_wall)
                    rows_buffer.append(row)
                    for c in CSV_COLUMNS[1:]:
                        columns[c].append(float("nan"))
                    continue

                pose3d = recon.reconstruct(pr1, pr2, f1.depth, f2.depth)
                tracker.update(pose3d, t_s)
                metrics = compute_frame_metrics(
                    pose3d, belt_n, belt_p, fwd_hint, bilateral_average=True
                )

                if tracker.airborne_l or tracker.airborne_r:
                    foot_max = max(
                        metrics.foot_clearance_m if not np.isnan(metrics.foot_clearance_m) else 0.0,
                        tracker.swing_peak_clearance_m,
                    )
                else:
                    foot_max = (
                        tracker.last_completed_swing_peak_m
                        if not np.isnan(tracker.last_completed_swing_peak_m)
                        else metrics.foot_clearance_m
                    )

                row = {
                    "time": _format_time(t_wall),
                    "upper_body_pitch_deg": metrics.upper_body_pitch_deg,
                    "hip_adduction_abduction_deg": metrics.hip_adduction_deg,
                    "hip_flexion_extension_deg": metrics.hip_flexion_deg,
                    "knee_flexion_extension_deg": metrics.knee_flexion_deg,
                    "ankle_angle_deg": metrics.ankle_angle_deg,
                    "foot_max_clearance_m": foot_max,
                    "step_length_m": tracker.last_step_length_m,
                    "step_frequency_steps_per_min": tracker.last_step_frequency_spm,
                }
                rows_buffer.append(row)
                for k in CSV_COLUMNS[1:]:
                    columns[k].append(float(row[k]) if row[k] is not None else float("nan"))

            dual.close()
        else:
            reader = create_video_reader(backend)
            if not reader.open(path1):
                raise RuntimeError("Could not open MKV.")
            K, d = _K_and_dist(reader.get_color_intrinsics())
            calib = _calibrate_single_cam(reader, char_cfg, K, d, n_cal, cal_int)
            if calib is None:
                raise RuntimeError("Calibration failed.")
            if not reader.seek(0):
                reader.close()
                reader = create_video_reader(backend)
                if not reader.open(path1):
                    raise RuntimeError("Reopen after calibration failed.")

            rcfg = dict(cfg["reconstruction"])
            rcfg["method"] = "depth"
            recon = PoseReconstructor(calib, rcfg)
            fwd_hint = _forward_hint_world(cfg, calib)
            belt_n, belt_p = _belt_world(calib)
            gcfg = cfg.get("gait", {})
            tracker = GaitTracker(
                belt_normal_world=belt_n,
                belt_point_world=belt_p,
                forward=fwd_hint,
                airborne_thresh_m=float(gcfg.get("airborne_thresh_m", 0.04)),
                contact_thresh_m=float(gcfg.get("contact_thresh_m", 0.018)),
            )

            for frame in reader.frames():
                if frame.color is None:
                    continue
                t_wall = frame.system_time
                t_s = (frame.timestamp_us or 0) / 1e6
                p1 = pose_est.estimate(frame.color)
                pr1 = _primary_pose(p1, frame.color.shape)
                if pr1 is None:
                    row = {c: np.nan for c in CSV_COLUMNS[1:]}
                    row["time"] = _format_time(t_wall)
                    rows_buffer.append(row)
                    for c in CSV_COLUMNS[1:]:
                        columns[c].append(float("nan"))
                    continue

                pose3d = recon.reconstruct(pr1, pr1, frame.depth, None)
                tracker.update(pose3d, t_s)
                metrics = compute_frame_metrics(
                    pose3d, belt_n, belt_p, fwd_hint, bilateral_average=True
                )
                if tracker.airborne_l or tracker.airborne_r:
                    foot_max = max(
                        metrics.foot_clearance_m if not np.isnan(metrics.foot_clearance_m) else 0.0,
                        tracker.swing_peak_clearance_m,
                    )
                else:
                    foot_max = (
                        tracker.last_completed_swing_peak_m
                        if not np.isnan(tracker.last_completed_swing_peak_m)
                        else metrics.foot_clearance_m
                    )
                row = {
                    "time": _format_time(t_wall),
                    "upper_body_pitch_deg": metrics.upper_body_pitch_deg,
                    "hip_adduction_abduction_deg": metrics.hip_adduction_deg,
                    "hip_flexion_extension_deg": metrics.hip_flexion_deg,
                    "knee_flexion_extension_deg": metrics.knee_flexion_deg,
                    "ankle_angle_deg": metrics.ankle_angle_deg,
                    "foot_max_clearance_m": foot_max,
                    "step_length_m": tracker.last_step_length_m,
                    "step_frequency_steps_per_min": tracker.last_step_frequency_spm,
                }
                rows_buffer.append(row)
                for k in CSV_COLUMNS[1:]:
                    columns[k].append(float(row[k]) if row[k] is not None else float("nan"))
            reader.close()

    finally:
        pose_est.release()

    col_arr = {k: np.array(v, dtype=np.float64) for k, v in columns.items()}
    _maybe_smooth(col_arr, cfg.get("gait", {}).get("filter", {}))

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for i, row in enumerate(rows_buffer):
            out_row = {"time": row["time"]}
            for c in CSV_COLUMNS[1:]:
                val = col_arr[c][i] if i < len(col_arr[c]) else float("nan")
                out_row[c] = val if np.isfinite(val) else ""
            w.writerow(out_row)

    logger.info("Wrote %s", out_csv)
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Femto Mega treadmill gait CSV export")
    ap.add_argument("--config", default="config.yaml", help="YAML config path")
    args = ap.parse_args()
    cfg_path = args.config
    if not os.path.isfile(cfg_path):
        logger.error("Config not found: %s", cfg_path)
        sys.exit(1)
    cfg = _load_config(cfg_path)
    run_dual_mkv(cfg)


if __name__ == "__main__":
    main()
