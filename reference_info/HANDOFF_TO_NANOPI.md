# Handoff: Wire the new RKNN model into `eyeTool` on the NanoPi M6

> **Audience:** the Cascade session (or human) working in the **`eyeTool`
> repo on the NanoPi M6**, picking up after the conversion task in the
> sibling project `rknn_conversion/`.
>
> **Status of upstream work:** ✅ done — `yolov8n.rknn` is built and ready.
> **Status of this work:** ⏳ pending — replace the CPU `ultralytics` path
> in `eyeTool/main.py` with NPU inference using this `.rknn` file.

## 1. What you're getting

A single artefact:

| File                | Size      | Properties                                              |
|---------------------|-----------|---------------------------------------------------------|
| `yolov8n.rknn`      | 4.13 MB   | int8, target `rk3588`, 9-output head, input 640×640×3  |

It was produced by `rknn-toolkit2 2.3.2` from airockchip's RKNN-optimised
YOLOv8n ONNX, calibrated with the official 20-image COCO subset. The full
build log, scripts, and `pip freeze` are in the sibling
`rknn_conversion/` repo (`CONVERSION_PLAYBOOK.md`).

**Place it at** `~/eyeTool/eyeTool/yolov8n.rknn` on the NanoPi.

From the dev PC (Windows PowerShell):

```powershell
scp rknn_conversion\model\yolov8n.rknn pi@<nanopi-ip>:~/eyeTool/eyeTool/yolov8n.rknn
```

## 2. Model spec — read this before you write a postprocess

This is **not** a stock ultralytics export. The head was sliced for NPU
efficiency, so the postprocess must match.

### Input

- Name: `images`
- Shape: `(1, 3, 640, 640)` (NCHW) — but the toolkit was configured with
  `mean=[0,0,0]`, `std=[255,255,255]`, so the runtime accepts a **uint8
  HWC tensor of shape `(640, 640, 3)`** and handles normalization to `[0,1]`
  internally. Pass it through `rknn_lite.inference(inputs=[bgr_to_rgb(img)])`.
- Channel order: **RGB** (the airockchip ONNX expects RGB just like vanilla
  YOLOv8). Convert from OpenCV's BGR before inference.
- Letterbox to 640×640 with grey (114,114,114) padding to preserve aspect
  ratio; keep the scale + pad offsets so you can map detections back to the
  original frame.

### Outputs (9 tensors)

For each of the three FPN levels (strides 8, 16, 32 → grid sizes 80, 40, 20
at 640 input), the model emits **three tensors in this order**:

| Index | Shape           | Meaning                                              |
|-------|-----------------|------------------------------------------------------|
|   0   | `(1, 64, 80, 80)` | stride 8 — DFL box distribution (4 sides × 16 bins) |
|   1   | `(1, 80, 80, 80)` | stride 8 — per-class raw confidence (pre-sigmoid)   |
|   2   | `(1,  1, 80, 80)` | stride 8 — class-sum (used as fast prefilter)       |
|   3   | `(1, 64, 40, 40)` | stride 16 — DFL                                      |
|   4   | `(1, 80, 40, 40)` | stride 16 — class confidence                         |
|   5   | `(1,  1, 40, 40)` | stride 16 — class-sum                                |
|   6   | `(1, 64, 20, 20)` | stride 32 — DFL                                      |
|   7   | `(1, 80, 20, 20)` | stride 32 — class confidence                         |
|   8   | `(1,  1, 20, 20)` | stride 32 — class-sum                                |

> **Verify order on first run** by printing `[o.shape for o in outputs]`.
> If your build of rknn-toolkit2 ever swapped the order, fix the indexing
> before debugging postprocess math.

`class-sum` is `Σ_c sigmoid(class_conf[:, c, :, :])` precomputed in fp on
the NPU; airockchip uses it to skip cells where no class can possibly win
the NMS, before doing the heavier per-class sigmoid + DFL decode. You may
ignore it if you prefer simpler code (decode all cells); you'll lose a few
ms but that's it.

### Decoded coordinate convention

After DFL decode + stride scaling, boxes are **xyxy in 640-space**. Undo
the letterbox to map back to the original frame.

## 3. Concrete tasks

The previous session left detection wired through `ultralytics.YOLO`. The
two functions to replace in `eyeTool/main.py` are `_get_yolo_model()` and
`draw_detections()`. The `load_camera_feed()` call site is already
gated on `_detection_enabled` and uses a confidence threshold from the
settings menu — keep that contract intact.

### Task list

1. **Install the runtime on the NanoPi.** Inside the existing
   `~/eyeTool/eyeTool/.venv`:

   ```bash
   pip install rknn-toolkit-lite2
   # NOT rknn-toolkit2 — that's the converter and won't install on aarch64 anyway
   python -c "from rknnlite.api import RKNNLite; print('ok')"
   ```

2. **Add `yolov8n.rknn` to the deploy.** Drop the file next to `main.py`,
   add it to `.gitignore` (it's a binary build artefact; rebuild from the
   sibling repo when needed), and add a `Makefile`/`README` line documenting
   how to refresh it.

3. **Replace the model loader.** Singleton pattern, cheap to call:

   ```python
   from rknnlite.api import RKNNLite

   _RKNN = None

   def _get_rknn() -> RKNNLite:
       global _RKNN
       if _RKNN is not None:
           return _RKNN
       r = RKNNLite()
       if r.load_rknn("yolov8n.rknn") != 0:
           raise RuntimeError("load_rknn failed")
       # NPU_CORE_AUTO lets the runtime pick a free core; on RK3588 the
       # NPU has 3 cores so this is a free ~3x with multiple consumers.
       if r.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
           raise RuntimeError("init_runtime failed")
       _RKNN = r
       return _RKNN
   ```

4. **Implement the postprocess.** Don't reinvent it — port the official
   reference. The two files you need from the airockchip model zoo:

   - <https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8/python/yolov8.py>
     (look at `post_process`, `dfl`, `box_process`, and `nms_boxes`)
   - <https://github.com/airockchip/rknn_model_zoo/blob/main/py_utils/coco_utils.py>
     (only the COCO class names — you only need index 0 = "person" for
     `eyeTool`'s human-detection use case)

   Drop them into `eyeTool/rknn_yolov8.py` and adapt the entry point to
   match the rest of `main.py`'s style. The whole module is ~150 lines.

5. **Replace `draw_detections()`** with a thin wrapper:

   ```python
   def draw_detections(frame, confidence: float = 0.5) -> int:
       """Run YOLOv8 on `frame` (BGR uint8), draw person boxes, return count."""
       rknn = _get_rknn()

       letterboxed, scale, (pad_w, pad_h) = _letterbox(frame, 640)
       rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
       outputs = rknn.inference(inputs=[rgb])

       boxes, scores, classes = post_process(outputs, conf_thres=confidence)
       persons = 0
       for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
           if cls != 0:        # COCO "person"
               continue
           # Map back to original frame coords:
           x1 = int((x1 - pad_w) / scale); x2 = int((x2 - pad_w) / scale)
           y1 = int((y1 - pad_h) / scale); y2 = int((y2 - pad_h) / scale)
           cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
           cv2.putText(frame, f"person {score:.2f}", (x1, max(0, y1-6)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
           persons += 1
       return persons
   ```

6. **Delete the `ultralytics` import and dependency.** Remove from
   `requirements.txt`, `pip uninstall ultralytics torch torchvision` to
   reclaim disk and import time.

7. **Sanity-test on a still image first**, before the camera loop:

   ```bash
   python -c "import cv2; from main import draw_detections; \
              f = cv2.imread('test_person.jpg'); n = draw_detections(f); \
              print('persons:', n); cv2.imwrite('out.jpg', f)"
   ```

8. **Then re-enable the live feed:**

   ```bash
   python main.py --mode feed
   ```

   Expected: 30–50 FPS single-threaded. If the prior 3 FPS metric came
   from a 1080p capture, you may also be limited by capture/display, not
   inference; print three timestamps (capture, inference, draw) per frame
   to see where time goes.

## 4. Performance lever sequence

Apply in this order — each multiplies on top of the previous:

| Lever                                                | Expected gain   |
|------------------------------------------------------|-----------------|
| Baseline: `core_mask=NPU_CORE_AUTO`, single thread   | 30–50 FPS       |
| Three-stage pipeline: capture / infer / draw threads | +30–50 %        |
| Use all three NPU cores via three `RKNNLite` instances and a round-robin queue | up to ~150 FPS |
| Drop input to 480×480 (rebuild the rknn from the same ONNX) | +30 %     |

Don't bother with multi-threading until you've proved the single-thread
path is correct — debugging a wrong postprocess inside a pipeline is
miserable.

## 5. Things easy to get wrong

- **BGR vs RGB.** OpenCV gives you BGR; the model wants RGB. Forgetting
  this gives ~10–20 % mAP loss and very confused detections.
- **Letterbox scale arithmetic.** Off-by-one in the unpad math draws
  boxes a few pixels off — looks like a postprocess bug, isn't.
- **Confidence threshold convention.** The airockchip postprocess applies
  `sigmoid` *after* slicing class scores; if you apply it twice you'll
  see all detections die.
- **Quantized output dtype.** With `int8` outputs, the runtime returns
  fp32 numpy arrays already de-quantized — don't re-scale them.
- **The 9th output (class-sum)** is `(1, 1, H, W)`, not `(1, 80, H, W)`.
  If your decoder treats it like the class tensor, scores will be nonsense.

## 6. Acceptance criteria

- `python main.py --mode feed` runs without `ultralytics` installed.
- Per-frame log line shows inference time ≤ 25 ms (i.e. ≥ 40 FPS).
- Detection rectangles are visually correct on a person walking past.
- Settings menu sliders (confidence, target FPS, toggle) still work.
- `git status` shows: removed ultralytics import, added `yolov8n.rknn`
  (gitignored), added `rknn_yolov8.py`, updated `requirements.txt`.

## 7. If you get stuck

- **`load_rknn` returns non-zero.** Check that `librknnrt.so` exists on
  the NanoPi (`/usr/lib/librknnrt.so`). FriendlyElec ships it; if missing,
  `sudo apt install rknpu2-runtime` or copy from the
  [rknn-toolkit2 runtime release](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/runtime).
- **`init_runtime` returns -1 with a "driver version mismatch".** Update
  the kernel module: `sudo apt upgrade rknpu-driver` and reboot.
- **The first inference is slow (~500 ms) but subsequent ones are fast.**
  That's normal — first call JITs and warms the NPU. Run one dummy frame
  during startup.

## 8. Reference

- Source ONNX (with optimized head): airockchip/rknn_model_zoo,
  `examples/yolov8/model/yolov8n.onnx`.
- Reference postprocess to port: airockchip/rknn_model_zoo,
  `examples/yolov8/python/yolov8.py`.
- Multi-thread / multi-core C++ example (if you ever rewrite the hot path
  in C++): <https://github.com/leafqycc/rknn-cpp-Multithreading>.
- Build provenance: see `rknn_conversion/CONVERSION_PLAYBOOK.md`,
  `rknn_conversion/requirements.lock.txt`.

Good luck — the conversion side is the hard part, and that's done.
This is mostly translating one well-documented postprocess.
