# Conversion Playbook: YOLOv8n → RKNN (RK3588) on Windows 11

This is a **redo guide** for the conversion that produced `model/yolov8n.rknn`
in this workspace. Following it from a clean Windows 11 box should reproduce
the exact same artifact in roughly 15–25 minutes (most of it download time).

## TL;DR

```powershell
# 1. WSL Ubuntu (one-time, ~2 min)
wsl --install -d Ubuntu-22.04 --no-launch
wsl -d Ubuntu-22.04 --user root -- bash -lc "useradd -m -s /bin/bash -G sudo kit && echo 'kit:kit' | chpasswd && echo 'kit ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/kit && chmod 440 /etc/sudoers.d/kit && printf '[user]\ndefault=kit\n[boot]\nsystemd=true\n' > /etc/wsl.conf"
wsl --terminate Ubuntu-22.04
```

```bash
# 2. Inside WSL (~5–10 min)
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-dev libgl1 libglib2.0-0
python3 -m venv ~/rknn-env
source ~/rknn-env/bin/activate
pip install --upgrade pip wheel
pip install -r /mnt/c/Users/12103/CascadeProjects/rknn_conversion/requirements.txt

# 3. Build (~3–5 min)
cd /mnt/c/Users/12103/CascadeProjects/rknn_conversion
python scripts/download_assets.py
python scripts/convert_yolo_to_rknn.py --platform rk3588 --dtype i8
# -> model/yolov8n.rknn  (4.13 MB)
```

That's it. The rest of this document explains *why* each step is the way it
is, so when something breaks you know where to poke.

---

## 1. Why a Linux VM is required

`rknn-toolkit2` ships **only as `manylinux` wheels** (x86_64 / aarch64). It is
not a typo, not a packaging oversight, not a "for now" — Rockchip just doesn't
build a Windows wheel. `pip install rknn-toolkit2` on native Windows fails
with "no matching distribution".

Three valid hosts for the converter:

| Host                       | Verdict                                  |
|----------------------------|------------------------------------------|
| Native Windows             | ❌ no wheel                              |
| **WSL2 Ubuntu 22.04**      | ✅ this playbook                         |
| Docker (Linux container)   | ✅ works; heavier                        |
| A real Linux box           | ✅ obviously                             |

The lite **runtime** (`rknn-toolkit-lite2`) used on the NanoPi for inference
is a separate, much smaller package and is not needed on the dev PC.

## 2. Why Ubuntu 22.04 specifically

- `rknn-toolkit2` 2.3.2 is currently published for CPython 3.6, 3.8, 3.10, 3.11, 3.12.
- Ubuntu 22.04 ships **Python 3.10.12** by default — minimum friction, no PPA.
- 24.04 (Python 3.12) also works but adds nothing here.

## 3. Why the WSL install command looks the way it does

The plain `wsl --install -d Ubuntu-22.04` opens an interactive console asking
for username/password as its **first action**. If you don't watch it, that
prompt blocks indefinitely (and inside an automation tool, it wedges).

The `--no-launch` variant downloads the rootfs without opening that console.
We then provision the user in one shot via `useradd` while running as `root`
(WSL gives you a passwordless root shell on a brand-new distro), and write
`/etc/wsl.conf` so the next `wsl -d Ubuntu-22.04` defaults to the new user.

> **Why username `kit` / password `kit`?** Empirically, when Cascade tries the
> system-default username `2103` on this machine, the WSL bootstrap UI hangs.
> Using a clean ASCII-letter username sidesteps the bug entirely.

## 4. Why `onnx==1.14.1` is pinned

`rknn-toolkit2 2.3.2`'s metadata says `onnx>=1.16.1`. Its actual code calls
`onnx.mapping.TENSOR_TYPE_TO_NP_TYPE`, a symbol that **was removed in
`onnx 1.15`**. So the metadata and the source disagree.

Symptom if you don't pin:

```
AttributeError: module 'onnx' has no attribute 'mapping'
  File "rknn/api/base_utils.py", line 34, in to_np_type
```

Pinning `onnx==1.14.1` is the smallest fix. `protobuf<4` and `numpy<2` follow
from onnx 1.14's own constraints. pip will print a "dependency conflict"
warning — ignore it; the toolkit runs correctly.

## 5. Why we don't use `ultralytics yolo export format=onnx`

The official ultralytics ONNX export emits **one big concatenated output**
that bakes in argmax / sigmoid / coordinate decoding. Two problems for RKNN:

1. Several of those ops fall back to CPU on the NPU, killing throughput.
2. The output layout is *not* what airockchip's reference postprocess
   expects, so any matching `yolov8.py` you copy from the model zoo is wrong.

Airockchip maintains a fork (`airockchip/ultralytics_yolov8`) whose head is
sliced into **three feature-map groups**, each emitting:

- `[1, 64, H, W]` — DFL distribution for box edges (4 sides × 16 bins)
- `[1, 80, H, W]` — per-class confidence (raw, pre-sigmoid)
- `[1,  1, H, W]` — pre-computed class-confidence sum (used as a fast
  early-reject score during NMS)

…for `H, W ∈ {(80,80), (40,40), (20,20)}` — the strides 8, 16, 32 used with
640×640 input. That's **9 output tensors total**.

Rather than build the fork ourselves, `scripts/download_assets.py` downloads
airockchip's pre-built ONNX directly from Rockchip's own CDN. It's exactly
what the `rknn_model_zoo/examples/yolov8` reference uses.

## 6. Why these calibration images

`do_quantization=True` does **post-training int8 quantization**, which
needs ~10–30 representative inputs to pick activation scales. We use the
20-image COCO subset that airockchip ships with the model zoo
(`datasets/COCO/subset/`) — same images Rockchip uses to publish the
reference accuracy numbers, so our quantization should match theirs.

Three quantization warnings in the build output are normal:

```
W build: found outlier value, this may affect quantization accuracy
        model.0.conv.weight       2.44   2.47   -17.494
        model.22.cv3.X.1.conv.weight  ...
```

These come from the YOLOv8n weights themselves (a few channels in the first
conv and the detection head are statistically unusual). Per-channel quant
handles them; mAP loss vs FP16 is typically <1 point on COCO.

To use your **own** calibration images later (e.g. real frames from the
NanoPi camera), drop JPEGs into `model/subset/` and rewrite `model/dataset.txt`
to list them, one path per line, relative to that file's directory.

## 7. Files in this workspace

```
rknn_conversion/
├── CONVERSION_PLAYBOOK.md          ← this file
├── HANDOFF_TO_NANOPI.md            ← what to send to the NanoPi side
├── README.md                       ← short overview
├── NPU_ACCELERATION_GUIDE.md       ← original (corrected) end-to-end guide
├── RKNN_CONVERSION_SUMMARY.md      ← project-status note
├── requirements.txt                ← pinned deps (onnx==1.14.1)
├── requirements.lock.txt           ← `pip freeze` of the working venv
├── scripts/
│   ├── check_env.ps1               ← prints WSL / admin / Python info
│   ├── verify_rknn.py              ← imports `rknn.api.RKNN` as a sanity check
│   ├── download_assets.py          ← fetches ONNX + 20 calibration images
│   └── convert_yolo_to_rknn.py     ← ONNX → RKNN
└── model/
    ├── yolov8n.onnx                ← airockchip-optimized, 9-output head
    ├── yolov8n.rknn                ← THE DELIVERABLE (i8, rk3588, 4.13 MB)
    ├── dataset.txt                 ← lists ./subset/*.jpg
    └── subset/                     ← 20 COCO calibration images
```

## 8. Sanity check after a rebuild

```bash
# Should be ~4.1 MB and recently mtime'd
ls -la model/yolov8n.rknn

# Check it loads round-trip without a real NPU (CPU simulator):
python - <<'PY'
from rknn.api import RKNN
r = RKNN(verbose=False)
assert r.load_rknn('model/yolov8n.rknn') == 0
# init_runtime(target=None) uses the simulator and is enough to validate the file
assert r.init_runtime() == 0
print("rknn file is valid")
r.release()
PY
```

If you see a different size from 4,327,755 bytes, that's *probably* fine —
it can drift slightly between rknn-toolkit2 patch releases. Anything in the
3.5 – 5 MB range for `yolov8n.rknn` (i8, rk3588) is normal.

## 9. Common breakage and fixes

| Symptom                                                        | Fix                                                                  |
|----------------------------------------------------------------|----------------------------------------------------------------------|
| `pip install rknn-toolkit2` fails: "no matching distribution"  | You're on native Windows. Run inside WSL.                            |
| `AttributeError: module 'onnx' has no attribute 'mapping'`     | `pip install 'onnx==1.14.1' 'protobuf<4' 'numpy<2'`                 |
| WSL install hangs at "Enter new UNIX username:"                | Use the `--no-launch` flow in §1.                                    |
| `rknn-toolkit2` complains about `libGL.so.1`                   | `sudo apt install -y libgl1 libglib2.0-0`                           |
| Conversion finishes but `mAP` looks awful in inference         | Calibration set mismatch. Use real-domain frames in `model/subset/`. |
| Build fails with "input shape mismatch"                        | Don't pass an ultralytics-exported ONNX; use the airockchip one.     |

## 10. Next: deploy the model

See `HANDOFF_TO_NANOPI.md` — written for the developer (or AI assistant)
working in the NanoPi-side repo (`eyeTool`).
