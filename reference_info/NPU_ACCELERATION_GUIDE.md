# YOLOv8 NPU Acceleration Guide for RK3588

This guide explains how to accelerate YOLOv8n on NanoPi M6 (RK3588S) using RKNN runtime to achieve 50-100+ FPS instead of 3 FPS.

## Overview

- **Current**: CPU-only inference with ultralytics (~3 FPS)
- **Target**: NPU-accelerated inference with RKNN runtime (50-100+ FPS)
- **Hardware**: RK3588S NPU (6 TOPS)
- **Model**: YOLOv8n (nano, ~6MB)

## Prerequisites

### Development PC (for model conversion)

- `rknn-toolkit2` is **Linux-only** (manylinux x86_64 / aarch64 wheels; no
  Windows wheel exists). On Windows 11, run the conversion inside
  **WSL2 Ubuntu 22.04** (what this repo uses) or Docker.
- Python: 3.8–3.12 supported; **3.10** used here.
- Internet access (rknn-toolkit2 + YOLOv8 ONNX download).
- WinSCP / `scp` (for transferring the final `.rknn` to the NanoPi).

### NanoPi M6 (for inference)

- OS: FriendlyElec Ubuntu 24.04 Desktop
- Python: 3.12+
- SSH access

## Part 1: Model Conversion (WSL2 Ubuntu on the Windows PC)

> **See `README.md` in this repo for the fully working, reproducible pipeline.**
> The steps below are kept for context.

### Step 1.1: Provision WSL Ubuntu 22.04

```powershell
wsl --install -d Ubuntu-22.04 --no-launch
wsl -d Ubuntu-22.04 --user root -- bash -lc "useradd -m -s /bin/bash -G sudo kit && echo 'kit:kit' | chpasswd && echo 'kit ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/kit && chmod 440 /etc/sudoers.d/kit && printf '[user]\ndefault=kit\n[boot]\nsystemd=true\n' > /etc/wsl.conf"
wsl --terminate Ubuntu-22.04
```

Then inside Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-dev libgl1 libglib2.0-0
python3 -m venv ~/rknn-env
source ~/rknn-env/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt   # pins onnx==1.14.1 (avoids upstream bug)
```

**Why not native Windows?** `rknn-toolkit2` is only published as manylinux
wheels – `pip install rknn-toolkit2` on Windows fails to find a wheel.

### Step 1.2: Get the RKNN-optimized YOLOv8n ONNX

Use airockchip's pre-built ONNX (its head is modified to three groups of
`(box-DFL, per-class-conf, summed-conf)` — much NPU-friendlier than the
stock ultralytics export):

```bash
python scripts/download_assets.py
# writes: model/yolov8n.onnx  +  20 COCO calibration images  +  model/dataset.txt
```

### Step 1.3: Conversion script

The ready-to-run script is `scripts/convert_yolo_to_rknn.py`. It mirrors the
official airockchip reference (`rknn_model_zoo/examples/yolov8/python/convert.py`).

### Step 1.4: Quantization dataset

`scripts/download_assets.py` already fetches airockchip's 20-image COCO subset
and writes `model/dataset.txt`. Replace it with your own images for better
domain accuracy.

### Step 1.5: Run conversion

```bash
python scripts/convert_yolo_to_rknn.py --platform rk3588 --dtype i8
```

Expected output (tail):
```
--> Config model
--> Loading ONNX model
--> Building RKNN model (this can take a few minutes)
I rknn building done.
--> Exporting to .../model/yolov8n.rknn
SUCCESS: .../model/yolov8n.rknn (4,327,755 bytes / 4.13 MB)
```

The file `model/yolov8n.rknn` (~4 MB, i8) is the converted model.

### Step 1.6: Transfer model to NanoPi

Use WinSCP or similar SFTP client:

1. Open WinSCP
2. Connect to NanoPi M6:
   - Host: `<nanopi-ip>`
   - Username: `pi`
   - Password: `<your-password>`
3. Navigate to `~/eyeTool/eyeTool/`
4. Upload `yolov8n.rknn` to that directory

Or use PowerShell (if OpenSSH is installed on Windows):

```powershell
scp yolov8n.rknn pi@<nanopi-ip>:~/eyeTool/eyeTool/
```

## Part 2: Install RKNN Runtime on NanoPi M6

### Step 2.1: Install rknn-runtime

```bash
# SSH into NanoPi M6
ssh pi@<nanopi-ip>

# Install from system package (if available)
sudo apt update
sudo apt install rknn-runtime

# Or install Python package
cd ~/eyeTool/eyeTool
source .venv/bin/activate
pip install rknn-runtime
```

### Step 2.2: Verify installation

```bash
python -c "from rknnlite.api import RKNNLite; print('RKNN runtime OK')"
```

## Part 3: Rewrite Detection Code

### Step 3.1: Update main.py imports

Replace ultralytics import with RKNN:

```python
# Remove:
# from ultralytics import YOLO

# Add:
from rknnlite.api import RKNNLite
```

### Step 3.2: Replace detection function

Replace `draw_detections()` and `_get_yolo_model()` with RKNN version:

```python
_rknn_model: RKNNLite | None = None
_rknn_initialized = False

def _init_rknn() -> RKNNLite:
    """Initialize RKNN runtime with yolov8n.rknn model."""
    global _rknn_model, _rknn_initialized
    if _rknn_initialized:
        return _rknn_model

    print("Initializing RKNN runtime...")
    _rknn_model = RKNNLite()
    ret = _rknn_model.init_runtime('yolov8n.rknn')
    if ret != 0:
        print("RKNN initialization failed!")
        raise RuntimeError("RKNN init failed")
    print("RKNN runtime initialized.")
    _rknn_initialized = True
    return _rknn_model

def draw_detections_rknn(frame: np.ndarray, confidence: float = 0.5) -> int:
    """Run YOLOv8n detection using RKNN NPU and draw bounding boxes.

    Returns number of persons detected.
    """
    rknn = _init_rknn()

    # Preprocess: resize to 640x640 (YOLOv8 input size)
    input_size = 640
    img = cv2.resize(frame, (input_size, input_size))
    img = np.expand_dims(img, 0)  # Add batch dimension
    img = img.astype(np.float32)

    # Inference
    outputs = rknn.inference(inputs=[img])

    # Postprocess (simplified - needs NMS implementation)
    # RKNN outputs raw tensors, need to parse and apply NMS
    # This is a simplified version - full implementation requires:
    # - Parse output tensors (boxes, scores, classes)
    # - Apply Non-Maximum Suppression (NMS)
    # - Filter by confidence threshold

    # TODO: Implement proper postprocessing based on YOLOv8 output format
    # Reference: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/engine/results.py

    count = 0  # Placeholder
    return count
```

### Step 3.3: Implement postprocessing

RKNN outputs raw tensors. You need to implement proper postprocessing:

Key steps:
1. Parse output tensors (YOLOv8 outputs 3 feature maps)
2. Decode boxes from DFL (Distribution Focal Loss)
3. Apply confidence threshold
4. Apply NMS (Non-Maximum Suppression)

Reference implementations:
- https://github.com/rockchip-linux/rknn-toolkit2/tree/master/rknn/api/rknn2
- https://github.com/ultralytics/ultralytics

### Step 3.4: Update load_camera_feed()

Replace the ultralytics detection call:

```python
# Remove:
# if _detection_enabled:
#     draw_detections(frame, _detection_confidence)

# Add:
if _detection_enabled:
    draw_detections_rknn(frame, _detection_confidence)
```

## Part 4: Multi-threading (Optional for further optimization)

For maximum performance (100+ FPS), implement multi-threading pipeline:

```python
import threading
import queue

class DetectionPipeline:
    def __init__(self):
        self.preprocess_queue = queue.Queue(maxsize=2)
        self.infer_queue = queue.Queue(maxsize=2)
        self.postprocess_queue = queue.Queue(maxsize=2)

    def preprocess_thread(self):
        while running:
            frame = self.preprocess_queue.get()
            # Resize, normalize
            processed = preprocess(frame)
            self.infer_queue.put(processed)

    def infer_thread(self):
        rknn = _init_rknn()
        while running:
            processed = self.infer_queue.get()
            outputs = rknn.inference(inputs=[processed])
            self.postprocess_queue.put(outputs)

    def postprocess_thread(self):
        while running:
            outputs = self.postprocess_queue.get()
            # NMS, draw boxes
            detections = postprocess(outputs, frame)
```

## Part 5: Testing

### Step 5.1: Test RKNN model

```bash
cd ~/eyeTool/eyeTool
source .venv/bin/activate
python main.py --mode feed
```

Expected:
- Model loads faster than ultralytics
- FPS should be 20-50+ (single-thread)
- 100+ FPS with multi-threading

### Step 5.2: Troubleshooting

**Error: "RKNN initialization failed"**
- Check `yolov8n.rknn` file exists
- Verify rknn-runtime installed correctly
- Check model was converted for RK3588 target

**Error: "Segmentation fault"**
- RKNN runtime version mismatch
- Reinstall rknn-runtime

**Poor accuracy**
- Use better quantization dataset
- Increase quantization dataset size
- Check input preprocessing matches training

## Part 6: Performance Tuning

### Adjust detection resolution

Lower input resolution for faster inference:

```python
input_size = 320  # Instead of 640
```

### Skip frames

Detect every N frames:

```python
frame_count += 1
if frame_count % 2 == 0:  # Detect every 2 frames
    detections = detect(frame)
else:
    # Use previous detections
    pass
```

### Use lighter model

Try YOLOv8n-tiny or YOLOv8n-quantized variants.

## References

- RKNN Toolkit2: https://github.com/rockchip-linux/rknn-toolkit2
- YOLOv8: https://github.com/ultralytics/ultralytics
- Reference implementation (173 FPS): https://github.com/ppogg/YOLOv8-rknn-multi-thread
- Multi-threading reference: https://github.com/leafqycc/rknn-cpp-Multithreading

## Summary

1. **PC**: Install rknn-toolkit2, convert yolov8n.pt → yolov8n.rknn
2. **NanoPi**: Install rknn-runtime, transfer .rknn model
3. **Code**: Replace ultralytics with RKNN API, implement postprocessing
4. **Optional**: Multi-threading for 100+ FPS
5. **Test**: Verify FPS improvement from 3 → 50-100+

Expected performance: **50-100+ FPS** (vs current 3 FPS)
