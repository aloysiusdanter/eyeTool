# eyeTool

A Python application for camera image loading and processing using OpenCV,
targeted at the **FriendlyElec NanoPi M6** single-board computer.

## Features

- Live camera feed display
- Single image capture
- Simple command-line interface

## Target Hardware

This project is developed for and tested on the FriendlyElec NanoPi M6.

- **Board:** FriendlyElec NanoPi M6 (Rockchip RK3588S, 8-core Cortex-A76/A55,
  Mali-G610 MP4 GPU, 6 TOPS NPU, LPDDR5 RAM)
- **Storage:** microSD Class 10, 8 GB or larger (SDHC), or eMMC module
- **Power:** USB-C PD adapter, 10 W or greater (6-20 V input)
- **Display:** the embedded display attached to the NanoPi M6 is used for both
  development and runtime. OpenCV debug windows (`cv2.imshow`) will pop up on
  this display; no headless mode is required.
- **Camera:** a camera is *required*. Either option works:
  - a **USB UVC webcam** plugged into a USB-A port (e.g. Logitech C920), or
  - a **MIPI-CSI camera** (e.g. FriendlyElec CAM415) on the 4-lane MIPI-CSI
    connector.
- **Input:** USB keyboard/mouse, or an SSH session with display forwarding
  (see [Remote development over SSH](#remote-development-over-ssh) below).

> Camera choice is still TBD. Both USB UVC and MIPI-CSI cameras expose
> `/dev/video*` nodes on the FriendlyElec images, so the Python code does not
> need to change based on the choice; only the device index may differ.

## Operating System

The officially supported and recommended image is the one pre-installed on the
board from FriendlyElec:

- **FriendlyElec Ubuntu 24.04 Desktop (arm64)** for RK3588S
- Default accounts: `pi` / `pi`, and `root` / `fa`
- Architecture: `aarch64` (arm64)

Other FriendlyElec images (Debian 11 Desktop, FriendlyCore, Armbian) are not
targeted by this project.

## System Prerequisites

Install the required OS packages on a fresh FriendlyElec Ubuntu 24.04 image:

```bash
sudo apt update
sudo apt install -y \
    python3 python3-pip python3-venv \
    v4l-utils \
    libgl1 libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good
```

- `v4l-utils` provides `v4l2-ctl` to list and inspect cameras.
- `libgl1` and the `libx*` packages are needed by the `opencv-python` wheel to
  open GUI windows via `cv2.imshow`.
- GStreamer packages enable the camera pipelines documented in the NanoPi M6
  wiki (useful for debugging the camera outside of Python).

Add your runtime user to the `video` group so `/dev/video*` is accessible
without root:

```bash
sudo usermod -aG video $USER
```

Log out and back in (or reboot) for the group change to take effect.

### Python runtime

FriendlyElec Ubuntu 24.04 ships with **Python 3.12** as the system interpreter,
which meets this project's minimum requirement.

If you want a newer interpreter (**Python 3.14** is preferred when available),
install it alongside the system Python. On Ubuntu 24.04 arm64 use either the
deadsnakes PPA or `pyenv`:

```bash
# Option A: deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.14 python3.14-venv python3.14-dev

# Option B: pyenv (build from source)
curl -fsSL https://pyenv.run | bash
# then follow the printed shell-init instructions, and:
pyenv install 3.14
pyenv local  3.14
```

Do not replace the system `python3`; only use the new interpreter to create a
virtual environment for this project.

## Camera Setup and Verification

After connecting the camera (USB or MIPI-CSI), verify it is detected before
running `main.py`:

```bash
# List all video capture devices
v4l2-ctl --list-devices

# Inspect the formats/resolutions of a specific device
v4l2-ctl -d /dev/video0 --list-formats-ext
```

If multiple `/dev/video*` nodes appear (common on RK3588S because the ISP
registers several nodes), adjust the index passed to `cv2.VideoCapture(...)`
in `main.py` to the one that corresponds to your actual camera.

> **USB camera tip:** On a fresh FriendlyElec image the MIPI-CSI ISP pipeline
> already occupies `/dev/video0` through `/dev/video19`. A USB UVC webcam
> typically appears as `/dev/video20` (or `/dev/video21`). Use
> `v4l2-ctl --list-devices` to find the exact node labelled with your webcam's
> vendor name, then pass it to eyeTool with `--device /dev/video20`.

> **Auto-resolution:** When eyeTool opens a USB camera, it automatically calls
> `v4l2-ctl --list-formats-ext` to enumerate every supported
> `(width × height × fps)` combination and selects the one with the highest
> score. For example, the DECXIN webcam chooses **MJPEG 1280×720 @ 60 fps**
> over YUYV 1280×720 @ 10 fps. MIPI-CSI (`/dev/video-camera0`) is left at its
> driver default. If `v4l2-ctl` is unavailable the camera opens with OpenCV
> defaults (graceful fallback).

Optional GStreamer preview (USB UVC example), useful to confirm the camera
works independently of OpenCV. Replace `/dev/video20` with your actual node:

```bash
gst-launch-1.0 v4l2src device=/dev/video20 ! \
    image/jpeg,width=1280,height=720,framerate=30/1 ! \
    jpegdec ! videoconvert ! glimagesink
```

## Installation

1. Navigate to the project directory:
   ```bash
   cd eyeTool
   ```

2. Create and activate a virtual environment (use `python3.14` instead of
   `python3` if you installed Python 3.14 as described above):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   If `pip` cannot find an arm64 wheel for `opencv-python` and tries to build
   from source, fall back to the Debian package instead:
   ```bash
   deactivate
   sudo apt install -y python3-opencv
   # and re-create the venv with --system-site-packages if needed:
   python3 -m venv --system-site-packages .venv
   source .venv/bin/activate
   pip install numpy==1.26.4
   ```

## Usage

Make sure the embedded display is active, then run:

```bash
python main.py
```

> If you are connected over SSH, see [Remote development over SSH](#remote-development-over-ssh)
> below before running GUI modes (`feed` or `capture`).

By default `main.py` uses the NanoPi M6 MIPI-CSI symlink
`/dev/video-camera0` if present, otherwise falls back to OpenCV index `0`.
Override with `--device`:

```bash
python main.py --device /dev/video-camera0      # MIPI-CSI (explicit)
python main.py --device 0                       # OpenCV index 0
python main.py --device /dev/video1             # specific node
```

Run modes (skip the interactive menu):

```bash
python main.py --mode feed                      # live feed window
python main.py --mode capture --output shot.jpg # capture one frame
python main.py --mode probe                     # grab 1 frame, no GUI
```

### Menu Options

1. **Live camera feed** - Displays real-time camera feed on the embedded
   display. Press `q` to quit.
2. **Capture single image** - Capture and save an image. Press `SPACE` to
   capture, `q` to quit.
3. **Probe camera (no GUI)** - Open the camera, grab one frame, and print
   its resolution and reported FPS. Useful over SSH without X.
4. **Exit** - Close the application.
5. **Select display target** - Choose which X display (`cv2.imshow` windows
   are sent to. Lists all detected displays; `:0` is the built-in LCD,
   `:1` is the HDMI output (when connected).

### Display target

eyeTool can direct `cv2.imshow` windows to any X display connected to the
NanoPi M6. The NanoPi M6 runs **GNOME on Wayland**; XWayland provides X11
compatibility sockets that Qt5 (bundled with the `opencv-python` wheel) uses
to open windows.

| Display | Output |
|---------|--------|
| `:0`    | Built-in LCD (default) |
| `:1`    | HDMI monitor (when connected) |

**Selecting the display at startup (CLI):**

```bash
python main.py --display :0        # built-in LCD (default)
python main.py --display :1        # HDMI monitor
```

**Selecting the display at runtime (menu option 5):**

From the interactive menu, choose option **5. Select display target**. The
menu shows all detected X displays with labels and marks the current one.
Type the number or the display string (e.g. `:1`) to switch.

Auto-detection runs at startup: if `$DISPLAY` is not set (typical over SSH),
eyeTool scans `/tmp/.X11-unix/` for XWayland sockets and picks `:0` (the
built-in LCD) automatically.

### Remote development over SSH

The FriendlyElec Ubuntu 24.04 Desktop image runs a **Wayland** compositor.
The `opencv-python` pip wheel bundles a **Qt5** backend that speaks X11 only;
it connects to Wayland through XWayland automatically.

When you SSH in as `pi`, eyeTool auto-detects the XWayland display and sets
`DISPLAY=:0` for you. You do not need to export `DISPLAY` manually.

However, the local X session may still require an X authority cookie. If
`cv2.imshow` fails with *"Authorization required"* or *"could not connect to
display"*:

1. **On the NanoPi display itself** (local keyboard or a separate VNC
   session), open a terminal and run:
   ```bash
   xhost +local:
   ```
   This grants any local user (including your SSH session) permission to open
   windows on the running display.

2. **From your SSH session**, launch eyeTool normally:
   ```bash
   python main.py --mode feed --device /dev/video20
   ```

The live feed window appears on the NanoPi's embedded display. Press `q` to
quit.

> **Security note:** `xhost +local:` is convenient for development on a
> trusted local network. Disable it again with `xhost -local:` when you are
> done, or restrict access to a specific user if multiple accounts log in
> locally.

## Requirements

- FriendlyElec NanoPi M6 (RK3588S, arm64) with its embedded display
- FriendlyElec Ubuntu 24.04 Desktop (pre-installed image)
- Python 3.12+ (Python 3.14 preferred when available)
- OpenCV 4.9.0.80
- NumPy 1.26.4
- A USB UVC webcam or a MIPI-CSI camera
- System packages listed under **System Prerequisites**

## Project Structure

```
eyeTool/
|-- main.py              # Main application script
|-- requirements.txt     # Python dependencies
|-- README.md            # Project documentation
```
