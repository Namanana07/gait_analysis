# 双相机跑步机步态分析系统

基于奥比中光 Femto Mega 双相机的跑步机步态分析程序，支持从 MKV 格式 RGB-D 视频中提取人体运动姿态参数。

## 系统要求

- Windows 10/11 64-bit
- Python 3.9 - 3.11
- NVIDIA GPU (A5000 或同等级别，CUDA 12.x)
- 8GB+ GPU 显存（推荐）
- 奥比中光 Femto Mega 相机 x2

## 安装步骤

### 1. 创建 Python 虚拟环境

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. 安装 PyTorch (CUDA)

```bash
# 确认 CUDA 版本后选择对应命令
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装基础依赖

```bash
pip install -r requirements.txt
```

### 4. 安装 MMPose (可选，推荐用于高精度)

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
mim install "mmpose>=1.0.0"
```

### 5. 安装奥比中光 SDK (可选)

从 [Orbbec SDK Release](https://github.com/orbbec/pyorbbecsdk/releases) 下载并安装 `pyorbbecsdk`：

```bash
pip install pyorbbecsdk
```

### 6. 下载模型文件 (MMPose 后端)

```bash
mkdir models
cd models
# RTMDet 人体检测模型
mim download mmdet --config rtmdet_m_640-8xb32_coco-person --dest .
# RTMPose 姿态估计模型 (HALPE26 全身关键点)
mim download mmpose --config rtmpose-l_8xb256-420e_body8-halpe26-256x192 --dest .
```

## 使用方法

### 基本用法

```bash
python main.py --config config.yaml --video1 camera1.mkv --video2 camera2.mkv
```

### 完整命令行参数

```bash
python main.py \
    --config config.yaml \
    --video1 path/to/camera1.mkv \
    --video2 path/to/camera2.mkv \
    --output ./output \
    --pose-backend mmpose \
    --recon-method fusion \
    --log-level INFO
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config, -c` | 配置文件路径 | `config.yaml` |
| `--video1, -v1` | 相机1视频路径 | 配置文件中指定 |
| `--video2, -v2` | 相机2视频路径 | 配置文件中指定 |
| `--output, -o` | 输出目录 | `./output` |
| `--pose-backend` | 姿态估计后端 (`mmpose`/`mediapipe`) | `mmpose` |
| `--recon-method` | 3D重建方法 (`triangulation`/`depth`/`fusion`) | `fusion` |
| `--log-level` | 日志级别 | `INFO` |
| `--calibration-file` | 已有标定文件路径 | 自动标定 |

## 配置说明

编辑 `config.yaml` 进行详细配置：

### ChArUco 标定参数

```yaml
charuco:
  dictionary: "DICT_4X4_50"   # ArUco 字典类型
  squares_x: 3                 # 棋盘格 X 方向格数
  squares_y: 2                 # 棋盘格 Y 方向格数
  square_length: 0.04          # 方格边长 (米)
  marker_length: 0.03          # 标记边长 (米)
  belt_plane_offset: 0.015     # 履带平面偏移 (1.5cm)
```

请根据您实际使用的 ChArUco 标记尺寸修改上述参数。

### 姿态估计后端选择

- **MMPose/RTMPose** (推荐): 精度高，GPU 加速，支持 HALPE26 全身 26 关键点（含足部）
- **MediaPipe**: 轻量级，安装简单，33 个关键点，但精度略低

## 输出文件

运行后在输出目录生成：

### 逐帧数据 (`gait_frame_data.csv`)

| 列名 | 说明 |
|------|------|
| 时间 | 系统时间 (YYYY-MM-DD HH:MM:SS.mmm) |
| 上半身俯仰角(°) | 躯干相对垂直方向的前倾/后仰角 |
| 髋关节内收外展角度(°) | 额状面内大腿偏移角度 |
| 髋关节屈伸角度(°) | 矢状面内大腿前后摆动角度 |
| 膝关节屈伸角度(°) | 膝关节弯曲角度 |
| 踝关节角度(°) | 踝关节背屈/跖屈角度 |
| 足最大离地高度(m) | 足部距离履带平面高度 |
| 步长(m) | 相邻两次触地间的水平距离 |
| 步频(steps/min) | 每分钟步数 |

### 步态周期汇总 (`gait_cycle_data.csv`)

每个步态周期（一步）输出一行汇总数据，包含各参数的峰值/均值。

### 分析摘要 (`analysis_summary.txt`)

包含整体统计信息（均值 ± 标准差）。

## 处理流程

```
视频输入 (MKV RGB-D)
    ↓
ChArUco 标定 → 相机外参 + 跑步机平面
    ↓
姿态估计 (MMPose/MediaPipe) → 2D 关键点
    ↓
3D 重建 (三角化 + 深度融合) → 3D 关键点
    ↓
步态分析 → 生物力学参数计算
    ↓
CSV 输出 (逐帧 + 步态周期)
```

## 项目结构

```
gait_analysis/
├── config.yaml                # 配置文件
├── main.py                    # 主程序入口
├── requirements.txt           # Python 依赖
├── README.md                  # 本文件
└── modules/
    ├── __init__.py
    ├── video_reader.py        # MKV 视频读取 (多后端)
    ├── calibration.py         # ChArUco 相机标定
    ├── pose_estimation.py     # 人体姿态估计 (MediaPipe/MMPose)
    ├── reconstruction.py      # 3D 关键点重建
    ├── gait_analysis.py       # 步态生物力学分析
    └── output.py              # CSV 结果输出
```

## 注意事项

1. **ChArUco 标定**: 确保 6 个 ChArUco 标记在跑步机边框上清晰可见，且两台相机都能看到至少部分标记
2. **相机安装**: 两台相机应从不同角度拍摄跑步机，建议一前一侧或两侧对称放置
3. **光照条件**: 保持均匀光照，避免强反射影响深度数据质量
4. **GPU 显存**: MMPose 后端约需 2-4GB 显存，A5000 (24GB) 完全足够
5. **首次运行**: 第一次运行会自动下载模型文件（如使用 mim），请确保网络连接

## 常见问题

**Q: 标定失败怎么办？**  
A: 确保 ChArUco 标记在视频开头清晰可见，调整 `calibration_frames` 和 `calibration_interval` 参数。

**Q: 使用 MediaPipe 时足部关键点不准？**  
A: MediaPipe 的足部关键点（脚趾/脚跟）精度有限，建议使用 MMPose 的 HALPE26 模型以获得更好的足部检测效果。

**Q: 深度图噪声大怎么处理？**  
A: 调整 `reconstruction.depth.search_radius` 增大搜索半径，或切换到 `triangulation` 方法依赖纯视觉三角化。
