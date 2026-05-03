# eyeTool Deployment Guide

eyeTool is a Python/OpenCV camera monitoring and recording application for the FriendlyElec NanoPi M6. It supports single-camera viewing, multi-camera grid display, RK3588 NPU person detection, alarm-zone setup, preprocessing controls, per-slot recording, display selection, configuration persistence, and external-library logging.

## Target Runtime

- **Board:** FriendlyElec NanoPi M6 / RK3588S, arm64
- **OS:** FriendlyElec Ubuntu 24.04 Desktop
- **Python:** 3.12+
- **Display:** local X11/XWayland display, commonly `:0` or `:1`
- **Camera:** USB UVC camera or MIPI-CSI camera exposed through `/dev/video*`
- **NPU runtime:** RKNNLite with `librknnrt.so`

## Repository Layout

```text
eyeTool/
├── main.py                         # Application entry point
├── cli.py                          # CLI argument parser
├── requirements.txt                # Python runtime dependencies
├── core/                           # Camera, config, display, zones, hotplug
├── detection/                      # RKNN YOLOv8 inference and pipeline
├── preprocessing/                  # Per-stream preprocessing model
├── streams/                        # Multi-camera stream management/compositor
├── ui/                             # Runtime menus, monitor, TUI modules
├── utils/                          # External logging and terminal helpers
└── logs/                           # Runtime external-library logs
```

## Fresh Deployment

### 1. Install OS packages

```bash
sudo apt update
sudo apt install -y \
    python3 python3-pip python3-venv \
    v4l-utils xauth \
    libgl1 libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good
```

Add the runtime user to the `video` group:

```bash
sudo usermod -aG video $USER
```

Log out and back in, or reboot, before running eyeTool.

### 2. Create the Python environment

Run these commands from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `opencv-python` cannot install an arm64 wheel, use the system package fallback:

```bash
deactivate
sudo apt install -y python3-opencv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install numpy scipy 'rknn-toolkit-lite2>=2.3.0'
```

### 3. Install the RKNN model

Detection expects `yolov8n.rknn` to be available relative to `detection/rknn_yolov8.py`. The default lookup checks:

```text
eyeTool/detection/yolov8n.rknn
```

The model is not committed to the repository. Build it offline and copy it to the target device:

```bash
scp yolov8n.rknn pi@<nanopi-ip>:~/eyeTool/eyeTool/detection/yolov8n.rknn
```

If detection fails to initialize, confirm that `librknnrt.so` is installed by the FriendlyElec RKNN runtime package.

### 4. Verify cameras

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

On NanoPi M6 images, MIPI/ISP nodes may occupy many `/dev/video*` paths. A USB UVC webcam commonly appears at a higher index such as `/dev/video20`. eyeTool defaults to `/dev/video-camera0` only when it exists and reports discrete capture formats, otherwise it tries to auto-detect a USB camera and then falls back to OpenCV index `0`.

## Running eyeTool

### Interactive runtime menu

```bash
source .venv/bin/activate
python main.py
```

The active menu options are:

1. Live camera feed (single)
2. Multi-camera feed (with recording)
3. Setup alarm zones
4. Image preprocessing
5. Monitoring TUI
6. Configuration (save/restore default)
7. Capture single image
8. Probe camera (no GUI)
9. Select display target
10. Detection settings
11. Recording settings
12. Exit

### Direct CLI modes

```bash
python main.py --mode feed
python main.py --mode capture --output shot.jpg
python main.py --mode probe
python main.py --mode record-multi
```

Select a camera explicitly:

```bash
python main.py --device /dev/video-camera0
python main.py --device /dev/video20
python main.py --device 0
```

Select a display explicitly:

```bash
python main.py --display :0
python main.py --display :1
```

Enable console mirroring for external-library logs:

```bash
python main.py --debug
```

## Configuration Files

Runtime configuration defaults to the `core/` directory:

```text
core/manufacturer_default.json      # Shipped baseline settings
core/user_settings.json             # User delta from manufacturer defaults
core/zones.json                     # Slot bindings, polygons, preprocessing
core/manufacturer_zones.json        # Optional archived manufacturer zones
```

Set `EYETOOL_CONFIG_DIR` to place these files elsewhere:

```bash
export EYETOOL_CONFIG_DIR="$HOME/.config/eyeTool"
python main.py
```

The configuration menu can save the current runtime state as the manufacturer baseline, restore the manufacturer baseline, clear user setting overrides, and show active config paths.

## Recording Deployment Notes

Recording settings live in `core/manufacturer_default.json` and user overrides:

- **Primary directory:** `/media/pi/6333-3864`
- **Fallback directory:** `~/Videos`
- **Default segment duration:** 2 minutes
- **Preferred codec:** `mpp_h264`
- **Cleanup threshold:** delete older recordings when disk usage exceeds the configured threshold

Before using multi-camera recording, mount the intended storage device and confirm the runtime user can write to it.

## Display and SSH Operation

eyeTool uses OpenCV GUI windows through X11/XWayland. Display selection follows the implementation order:

1. `--display` command-line argument, when provided
2. saved `display.target` from config
3. existing parent-shell `DISPLAY`
4. first socket found under `/tmp/.X11-unix/`

The display helper attempts to merge GNOME/mutter XWayland authentication into `~/.Xauthority`. If GUI windows still fail from SSH, run this once on the NanoPi local desktop:

```bash
xhost +local:
```

Disable it later if needed:

```bash
xhost -local:
```

## Runtime Logs

Each run creates a timestamped external-library log under:

```text
logs/YYYY-MM-DD-HHMMSS.log
```

OpenCV, V4L2, RKNN, GStreamer, and MPP messages may be redirected there instead of the console. Use `--debug` to mirror those messages to the console during troubleshooting.
