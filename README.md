# Fly MOT

Fly MOT 是一个面向果蝇视频的多目标检测、身份关联、轨迹修复、事件诊断和评估项目。当前推荐工作流以 `src/` 下的模块化管线为准，并提供本地 Web GUI 作为可视化控制台。视频格式支持 MP4、AVI 等常见格式，输出视频统一写为 .mp4，编码为 mp4v。

## Features

- YOLO 果蝇检测，支持低置信度 rescue、ROI rescue 和增强图像补检。
- 多目标跟踪，包含运动门控、外观/形状/空间/时间特征融合、身份槽位和 slot stickiness。
- 轨迹后处理，包含全局 ReID、长间隔桥接、插值、短弱轨迹过滤和 rescue-heavy 轨迹过滤。
- 事件诊断，包含 crossing、interaction、merged state、possible ID swap 等输出。
- 指标评估，支持基础 tracking 指标、点匹配评估、Point-HOTA 分项和 hard-case 分析。
- ReID 数据准备与外观编码器训练。
- 本地 FastAPI + HTML/CSS/JS + Three.js Web GUI，支持启动任务、查看日志、下载结果、播放结果视频和 3D 轨迹可视化。
- OpenCV 标注工具 `GT_generator.py`，用于生成 `coords/*.csv` 格式的点标注。

## Project Layout

```text
.
├─ main.py                    # src/main.py 的兼容入口，便于直接运行默认pipeline
├─ src/
│  ├─ main.py                 # MOT 主流程入口：run_pipeline(cfg)
│  ├─ config.py               # dataclass 配置和 GUI 覆盖参数白名单
│  ├─ association/            # 级联匹配、代价融合、全局 ReID、插值和 slot stickiness
│  ├─ core/                   # Track、Detection、FrameResult 等核心数据结构
│  ├─ detector/               # YOLO 检测封装、低置信度 rescue 和 ROI crop
│  ├─ evaluation/             # 指标、benchmark、ID switch 和冲突诊断工具
│  ├─ events/                 # crossing、interaction、merged state 等事件诊断
│  ├─ identity/               # 外观、形状、空间、时间身份特征和 ReID 编码器
│  ├─ io_utils/               # CSV、视频、缓存和日志读写工具
│  ├─ local_gui/              # Electron 本地桌面 GUI
│  ├─ motion/                 # Kalman 滤波和运动学特征
│  ├─ preprocessing/          # 检测归一化、质量过滤和预处理
│  ├─ qt_gui/                 # 原生 PyQt6 桌面 GUI
│  ├─ render/                 # 跟踪结果视频渲染和配色
│  ├─ tracklet/               # TrackBuilder、轨迹特征、split/merge 和后处理
│  ├─ training/               # ReID 数据准备、采样器、损失函数和编码器训练
│  └─ web/                    # FastAPI 本地 Web GUI 后端、runner、parser 和静态前端
├─ tests/                     # 单元测试和 smoke tests
├─ coords/                    # GT 标注 CSV 与中间轨迹 CSV
├─ outputs/                   # 默认运行输出目录
├─ class.txt                  # YOLO 类别名称，每行一个类别
├─ data.yaml                  # YOLO 数据集路径、类别数和类别名配置
├─ package.json               # Electron 本地 GUI 的 npm 脚本和依赖声明
├─ package-lock.json          # npm 依赖锁定文件
├─ pyproject.toml             # Python 依赖声明和项目元数据
├─ uv.lock                    # uv 锁定文件
├─ GT_generator.py            # OpenCV 点标注工具
└─ train.py                   # YOLO 训练脚本
```

## Requirements

- Python `>=3.10`
- `uv`
- Windows和Linux 均可运行核心 Python 管线；CUDA 可选
- NVIDIA GPU 可显著加速模型训练与推理。

项目依赖见 `pyproject.toml`，核心包括：

- `ultralytics`
- `torch` / `torchvision`
- `opencv-python`
- `numpy` / `scipy` / `pandas`
- `fastapi` / `uvicorn` / `python-multipart`
- `PyQt6`

## Installation

系统环境配置：

1. 安装 Python 3.10 或更高版本。
链接：<https://www.python.org/downloads/latest/python3.12/>
2. 安装 NVIDIA GPU 驱动（如果使用 CUDA），在官网根据实际情况选择相应选项。
链接：<https://www.nvidia.com/Download/index.aspx>
3. 安装 Node.js 22.2.3 或更高版本（如果使用 Electron 本地 GUI），使用官网默认选项，然后点击 Windows Installer 安装。
链接：<https://nodejs.org/en/download/>

```powershell
pip install uv
uv python install
uv sync
```

如果需要使用 CUDA，请确认 PyTorch 与本机 CUDA 环境匹配。当前 `pyproject.toml` 配置了 PyTorch CUDA 12.8 索引。

## Quick Start

运行默认 MOT 管线：

```powershell
uv run python main.py
```

默认配置来自 `src/config.py`：

- 输入视频：`min_test.mp4`
- 默认检测模型：`best.pt`
- 默认 GT：`coords/gt.csv`
- 默认输出：`outputs/`

结果会写入：

```text
outputs/
├─ csv/
│  ├─ detections.csv #重要
│  ├─ tracks.csv     #重要
│  ├─ track_stats.csv
│  ├─ events.csv     #关注interaction帧
│  ├─ metrics.csv
│  ├─ recall_audit.csv
│  ├─ stage_metrics.csv
│  └─ hard_case_summary.csv
├─ logs/run.log
└─ videos/result.mp4
```

## Local Web GUI

启动本地控制台：

```powershell
uv run uvicorn src.web.server:app --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000
```

Web GUI 支持：

- 选择或上传视频
- 选择 `.pt` 权重
- 调整置信度、IoU、输入尺寸、最大检测数、果蝇数量、ID 槽位和最大帧数
- 后台运行 `run_pipeline(cfg)`
- 查看运行进度和日志
- 播放渲染后的视频
- 用 Three.js 查看 3D 轨迹
- 下载 CSV、日志和结果视频
- 切换 `显微`、`明亮`、`高对比` 主题

GUI 运行输出会写入 `outputs/gui/runs/{job_id}/`，不会覆盖默认 `outputs/`。

## Configuration

主要配置都在 `src/config.py` 中，常用项包括：

- `DetectionConfig.model_path`
- `DetectionConfig.conf_thres`
- `DetectionConfig.iou_thres`
- `DetectionConfig.imgsz`
- `TrackConfig.num_flies`
- `TrackConfig.identity_slots`
- `RuntimeConfig.video_path`
- `RuntimeConfig.output_root`
- `RuntimeConfig.max_frames`
- `RuntimeConfig.save_video`
- `EvaluationConfig.enabled`
- `EvaluationConfig.gt_csv_path`

代码中也提供 `apply_overrides(cfg, overrides)`，Web GUI 的后端会用白名单方式覆盖这些字段。

## Ground Truth Annotation

`GT_generator.py` 是 OpenCV 点标注工具。使用前请先编辑文件顶部常量：

```python
VIDEO_PATH = r"D:\fly\min_test.mp4"
OUTPUT_GT_CSV = r"D:\fly\coords\gt.csv"
NUM_FLIES = 6 #视频中的目标数量
```

启动：

```powershell
uv run python GT_generator.py
```

使用说明：

- 鼠标左键：选择一个点
- 数字键：输入目标 ID，范围为 `0` 到 `NUM_FLIES - 1`
- `Enter` / `Space`：将已输入的 ID 分配给当前选中的点
- `Backspace`：编辑已输入的 ID
- `Esc`：清空已输入的 ID
- `n`：下一帧
- `b`：上一帧
- `f`：快进
- `r`：快退
- `i`：继承上一帧标注
- `x`：删除当前帧的指定 ID
- `d`：删除当前帧最近一次分配的 ID
- `c`：清空当前帧
- `s`：保存 CSV
- `j`：跳转到指定帧
- `h`：隐藏 / 显示屏幕帮助
- `q`：保存并退出

GT CSV 格式：

```csv
frame,id,x,y
0,0,1049.25,390.37
```

## Training

YOLO 检测模型训练入口：

```powershell
uv run python train.py
```

`train.py` 当前使用硬编码路径和参数；训练前请检查：

- `data.yaml` 中的 `train` 和 `val` 路径
- 模型名称和类别数
- `epochs`(训练轮数)
- `batch`(批大小)
- `imgsz`(输入尺寸)
- `device`(训练设备)

ReID 数据准备：

```powershell
uv run python -m training.prepare_reid_dataset
```

ReID / appearance bundle 训练：

```powershell
uv run python -m training.train_encoder
```

训练输出默认写入 `outputs/models/` 和 `outputs/reid_data/`。

## Local Desktop GUI

除 Web GUI 外，项目还提供一个 Electron 本地桌面控制台。它复用 Web GUI 的 HTML/CSS/JS/Three.js 视觉系统，但不启动 FastAPI 后端；任务运行、日志读取、结果文件和视频转码都由 Electron 主进程在本机完成。

首次使用先安装桌面端依赖或直接使用安装包：

```powershell
npm.cmd install
```

启动本地 GUI：

```powershell
npm.cmd run local-gui
```

桌面 GUI 支持选择项目内视频或外部视频、选择 `.pt` 权重与可选 GT CSV、调整检测/跟踪/渲染/评估参数、启动/取消任务、播放结果视频、查看 Three.js 3D 轨迹、打开 CSV/日志/视频文件，以及切换 `显微`、`明亮`、`高对比` 主题。

桌面 GUI 输出写入 `outputs/local_gui/runs/{job_id}/`，不会覆盖默认 `outputs/` 或 Web GUI 的 `outputs/gui/`。

使用安装包时，默认输出路径为 C:\Users\<USER>\Desktop\Fly MOT Electron，请根据需要在安装时更改安装路径。安装完后，桌面会创建一个 `Fly MOT Electron` 快捷方式，双击即可启动 GUI。GUI 默认文件输出路径为 C:\Users\<USER>\AppData\Local\FlyMOT\outputs\local_gui\runs\{job_id}，请根据需要在 GUI 内更改输出路径。

数据缓存位于 C:\Users\seanl\AppData\Roaming\fly-mot-electron-cpu。

## Local PyQt6 GUI

项目还提供一个不依赖 WebEngine 的原生 PyQt6 控制台。它不启动后端服务，直接用 Qt Widgets/QSS 构建界面，通过 `QProcess` 调用现有 `src.web.runner`，并在本地读取日志、CSV、结果视频和轨迹。

启动 PyQt6 GUI：

```powershell
uv run python -m src.qt_gui.app
```

如果 `uv run` 遇到本机缓存权限问题，也可以直接使用虚拟环境解释器：

```powershell
.venv\Scripts\python.exe -m src.qt_gui.app
```

PyQt6 GUI 支持与 Web GUI 基本相同的任务流程：选择项目内或外部视频、选择 `.pt` 权重、可选 GT CSV、调整检测/跟踪/渲染/评估参数、启动/取消任务、查看日志和指标、播放结果视频、查看本地绘制的轨迹视图、打开输出文件，并切换 `显微`、`明亮`、`高对比` QSS 主题。

PyQt6 GUI 输出写入 `outputs/qt_gui/runs/{job_id}/`。

## Testing

运行全部测试：

```powershell
uv run python -m unittest discover -s tests
```

只运行 Web GUI 相关测试：

```powershell
uv run python -m unittest discover -s tests -p test_web_gui.py
```

测试覆盖了 association、preprocessing、event、metrics、postprocess、ReID 数据准备、pipeline smoke 和 Web GUI parser/API helper。

## Citation

如果你在论文或报告中使用本项目，建议记录：

- 使用的版本
- 输入视频和 GT CSV
- 各模型权重
- 关键配置覆盖项
- `outputs/csv/metrics.csv`
- `outputs/csv/stage_metrics.csv`
