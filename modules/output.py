"""
Output Module.
Generates CSV files for gait analysis results:
- Frame-level CSV: one row per video frame
- Gait cycle CSV: one row per gait cycle (step)
"""

import os
import logging
import csv
import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta

from .gait_analysis import GaitParameters, GaitCycleData

logger = logging.getLogger(__name__)


class CSVWriter:
    """Writes gait analysis results to CSV files."""

    # Column headers for frame-level CSV
    FRAME_HEADERS = [
        "时间",                    # Time (system time format)
        "上半身俯仰角(°)",          # Upper body pitch angle
        "髋关节内收外展角度(°)",     # Hip adduction/abduction
        "髋关节屈伸角度(°)",        # Hip flexion/extension
        "膝关节屈伸角度(°)",        # Knee flexion/extension
        "踝关节角度(°)",            # Ankle angle
        "足最大离地高度(m)",         # Foot clearance
        "步长(m)",                  # Step length
        "步频(steps/min)",          # Cadence
    ]

    # Column headers for cycle-level CSV
    CYCLE_HEADERS = [
        "周期序号",                 # Cycle index
        "开始时间",                 # Start time
        "结束时间",                 # End time
        "侧别",                    # Side (left/right)
        "上半身俯仰角均值(°)",      # Mean upper body pitch
        "髋关节内收外展角度最大值(°)", # Max hip adduction/abduction
        "髋关节屈伸角度最大值(°)",   # Max hip flexion/extension
        "膝关节屈伸角度最大值(°)",   # Max knee flexion/extension
        "踝关节角度最大值(°)",       # Max ankle angle
        "足最大离地高度(m)",         # Max foot clearance
        "步长(m)",                  # Step length
        "步频(steps/min)",          # Cadence
    ]

    def __init__(self, output_dir: str, 
                 start_time: Optional[datetime] = None,
                 fps: float = 30.0):
        """
        Args:
            output_dir: Directory to save CSV files
            start_time: Recording start system time (for absolute timestamps)
            fps: Video frame rate
        """
        self.output_dir = output_dir
        self.start_time = start_time or datetime.now()
        self.fps = fps
        
        os.makedirs(output_dir, exist_ok=True)

    def timestamp_to_system_time(self, timestamp_us: int) -> str:
        """
        Convert frame timestamp (microseconds) to system time string.
        Format: YYYY-MM-DD HH:MM:SS.mmm
        """
        time_offset = timedelta(microseconds=timestamp_us)
        absolute_time = self.start_time + time_offset
        return absolute_time.strftime("%Y-%m-%d %H:%M:%S.") + \
               f"{absolute_time.microsecond // 1000:03d}"

    def write_frame_csv(self, parameters: List[GaitParameters], 
                        timestamps_us: List[int],
                        filename: str = "gait_frame_data.csv"):
        """
        Write frame-level gait parameters to CSV.
        
        Args:
            parameters: List of GaitParameters (one per frame)
            timestamps_us: Frame timestamps in microseconds
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(self.FRAME_HEADERS)

            for i, params in enumerate(parameters):
                timestamp_us = timestamps_us[i] if i < len(timestamps_us) else 0
                time_str = self.timestamp_to_system_time(timestamp_us)

                row = [
                    time_str,
                    f"{params.upper_body_pitch:.2f}" if not np.isnan(params.upper_body_pitch) else "",
                    f"{params.hip_adduction_abduction:.2f}" if not np.isnan(params.hip_adduction_abduction) else "",
                    f"{params.hip_flexion_extension:.2f}" if not np.isnan(params.hip_flexion_extension) else "",
                    f"{params.knee_flexion_extension:.2f}" if not np.isnan(params.knee_flexion_extension) else "",
                    f"{params.ankle_angle:.2f}" if not np.isnan(params.ankle_angle) else "",
                    f"{params.foot_clearance:.4f}" if not np.isnan(params.foot_clearance) else "",
                    f"{params.step_length:.4f}" if not np.isnan(params.step_length) else "",
                    f"{params.cadence:.1f}" if not np.isnan(params.cadence) else "",
                ]
                writer.writerow(row)

        logger.info(f"Frame-level CSV saved: {filepath} ({len(parameters)} rows)")
        return filepath

    def write_cycle_csv(self, cycles: List[GaitCycleData],
                        filename: str = "gait_cycle_data.csv"):
        """
        Write gait-cycle-level data to CSV.
        
        Args:
            cycles: List of GaitCycleData (one per gait cycle)
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(self.CYCLE_HEADERS)

            for cycle in cycles:
                start_time_str = self.timestamp_to_system_time(cycle.start_time_us)
                end_time_str = self.timestamp_to_system_time(cycle.end_time_us)
                side_str = "右" if cycle.side == "right" else "左"

                row = [
                    cycle.cycle_index + 1,  # 1-based indexing
                    start_time_str,
                    end_time_str,
                    side_str,
                    f"{cycle.upper_body_pitch_mean:.2f}" if not np.isnan(cycle.upper_body_pitch_mean) else "",
                    f"{cycle.hip_adduction_abduction_max:.2f}" if not np.isnan(cycle.hip_adduction_abduction_max) else "",
                    f"{cycle.hip_flexion_extension_max:.2f}" if not np.isnan(cycle.hip_flexion_extension_max) else "",
                    f"{cycle.knee_flexion_extension_max:.2f}" if not np.isnan(cycle.knee_flexion_extension_max) else "",
                    f"{cycle.ankle_angle_max:.2f}" if not np.isnan(cycle.ankle_angle_max) else "",
                    f"{cycle.foot_clearance_max:.4f}" if not np.isnan(cycle.foot_clearance_max) else "",
                    f"{cycle.step_length:.4f}" if not np.isnan(cycle.step_length) else "",
                    f"{cycle.cadence:.1f}" if not np.isnan(cycle.cadence) else "",
                ]
                writer.writerow(row)

        logger.info(f"Gait cycle CSV saved: {filepath} ({len(cycles)} cycles)")
        return filepath

    def write_summary(self, frame_params: List[GaitParameters],
                      cycle_data: List[GaitCycleData],
                      filename: str = "analysis_summary.txt"):
        """Write a text summary of the analysis."""
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("  步态分析结果摘要 (Gait Analysis Summary)\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"分析帧数: {len(frame_params)}\n")
            f.write(f"帧率: {self.fps:.1f} fps\n")
            duration_s = len(frame_params) / self.fps
            f.write(f"总时长: {duration_s:.1f} 秒\n")
            f.write(f"检测到的步态周期: {len(cycle_data)}\n\n")

            if cycle_data:
                f.write("-" * 40 + "\n")
                f.write("  各参数统计 (均值 ± 标准差)\n")
                f.write("-" * 40 + "\n\n")

                # Compute statistics from cycle data
                fields = [
                    ("上半身俯仰角", "upper_body_pitch_mean", "°"),
                    ("髋关节内收外展", "hip_adduction_abduction_max", "°"),
                    ("髋关节屈伸", "hip_flexion_extension_max", "°"),
                    ("膝关节屈伸", "knee_flexion_extension_max", "°"),
                    ("踝关节角度", "ankle_angle_max", "°"),
                    ("足最大离地高度", "foot_clearance_max", "m"),
                    ("步长", "step_length", "m"),
                    ("步频", "cadence", "steps/min"),
                ]

                for name, attr, unit in fields:
                    values = [getattr(c, attr) for c in cycle_data 
                             if not np.isnan(getattr(c, attr))]
                    if values:
                        mean = np.mean(values)
                        std = np.std(values)
                        f.write(f"  {name}: {mean:.2f} ± {std:.2f} {unit}\n")
                    else:
                        f.write(f"  {name}: N/A\n")

                f.write("\n")

            f.write("=" * 60 + "\n")

        logger.info(f"Summary saved: {filepath}")
        return filepath
