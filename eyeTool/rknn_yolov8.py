"""YOLOv8 NPU inference and postprocessing for the airockchip RKNN model.

This module wraps the RKNNLite runtime and ports the airockchip
`rknn_model_zoo/examples/yolov8/python/yolov8.py` postprocess to pure
NumPy (no PyTorch dependency).

Model spec (see reference_info/HANDOFF_TO_NANOPI.md):
- Input:  uint8 HWC RGB tensor of shape (640, 640, 3); the runtime
  normalizes via mean=[0,0,0], std=[255,255,255].
- Outputs: 9 tensors, three per FPN level (strides 8, 16, 32):
    - (1, 64, H, W) -- DFL box distribution
    - (1, 80, H, W) -- per-class raw confidence (post-sigmoid in this
      airockchip head; we treat as already-sigmoided scores)
    - (1,  1, H, W) -- class-sum prefilter (ignored here)
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np
from scipy.special import softmax as scipy_softmax
from rknnlite.api import RKNNLite

# Detection constants
INPUT_SIZE = 640
NMS_THRESH = 0.45
DFL_LEN = 16  # 64 channels / 4 sides

_RKNN_SINGLE: RKNNLite | None = None
_RKNN_ALL_CORES: RKNNLite | None = None

# Timing stats for profiling
_npu_time_sum = 0.0
_postprocess_time_sum = 0.0
_npu_count = 0
_last_stats_ts = 0.0


def _get_rknn(model_path: str = "yolov8n.rknn") -> RKNNLite:
    """Singleton RKNNLite loader (single core, AUTO)."""
    global _RKNN_SINGLE
    if _RKNN_SINGLE is not None:
        return _RKNN_SINGLE
    if not os.path.isabs(model_path):
        # Resolve relative to this file so it works from any CWD
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(here, model_path)
        if os.path.exists(candidate):
            model_path = candidate
    print(f"Loading RKNN model: {model_path}")
    r = RKNNLite()
    if r.load_rknn(model_path) != 0:
        raise RuntimeError(f"load_rknn failed for {model_path}")
    if r.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
        raise RuntimeError("init_runtime failed")
    print("RKNN runtime initialized (NPU_CORE_AUTO).")
    _RKNN_SINGLE = r
    return _RKNN_SINGLE


def _get_rknn_all_cores(model_path: str = "yolov8n.rknn") -> RKNNLite:
    """Singleton RKNNLite loader using all three NPU cores simultaneously.

    core_mask=7 = NPU_CORE_0|NPU_CORE_1|NPU_CORE_2 tells the RKNN runtime
    to distribute one inference across all 3 cores, cutting per-inference
    latency by ~3x compared to single-core AUTO.
    """
    global _RKNN_ALL_CORES
    if _RKNN_ALL_CORES is not None:
        return _RKNN_ALL_CORES
    if not os.path.isabs(model_path):
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(here, model_path)
        if os.path.exists(candidate):
            model_path = candidate
    print(f"Loading RKNN model (all-core): {model_path}")
    r = RKNNLite()
    if r.load_rknn(model_path) != 0:
        raise RuntimeError(f"load_rknn failed for {model_path}")
    # mask 7 = NPU_CORE_0 | NPU_CORE_1 | NPU_CORE_2
    if r.init_runtime(core_mask=7) != 0:
        raise RuntimeError("init_runtime failed for all-core mode")
    print("RKNN runtime initialized (NPU_CORE_0|1|2, all-core).")
    _RKNN_ALL_CORES = r
    return _RKNN_ALL_CORES


def letterbox(im: np.ndarray, new_size: int = INPUT_SIZE,
              pad_color: tuple[int, int, int] = (114, 114, 114)
              ) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize *im* to (new_size, new_size) preserving aspect ratio with padding.

    Returns (padded_image, scale, (pad_w, pad_h)).
    """
    h, w = im.shape[:2]
    scale = min(new_size / w, new_size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (new_size - new_w) // 2
    pad_h = (new_size - new_h) // 2
    padded = np.full((new_size, new_size, 3), pad_color, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return padded, scale, (pad_w, pad_h)


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def _dfl(position: np.ndarray) -> np.ndarray:
    """DFL decode: scipy softmax + weighted sum along axis 2."""
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = scipy_softmax(y, axis=2)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(axis=2)


def _box_process(position: np.ndarray) -> np.ndarray:
    """Decode box DFL output into xyxy in 640-input space."""
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1).astype(np.float32)
    stride = np.array([INPUT_SIZE // grid_h, INPUT_SIZE // grid_w]).reshape(1, 2, 1, 1)

    pos = _dfl(position)
    box_xy = grid + 0.5 - pos[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + pos[:, 2:4, :, :]
    return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)


def _sp_flatten(_in: np.ndarray) -> np.ndarray:
    ch = _in.shape[1]
    return _in.transpose(0, 2, 3, 1).reshape(-1, ch)


def _nms_boxes(boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """OpenCV NMS (C++ optimized). boxes: xyxy."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    # OpenCV NMSBoxes requires [x, y, w, h] format
    xywh = boxes.astype(np.float32).copy()
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # w = x2 - x1
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # h = y2 - y1
    # score_threshold=0.0: boxes already filtered upstream by conf_thres
    indices = cv2.dnn.NMSBoxes(xywh.tolist(), scores.astype(np.float32).tolist(),
                               0.0, float(NMS_THRESH))
    if len(indices) == 0:
        return np.array([], dtype=np.int64)
    return np.array(indices, dtype=np.int64).flatten()


def post_process(outputs: list[np.ndarray], conf_thres: float = 0.5
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode 9-tensor airockchip YOLOv8 outputs into (boxes, classes, scores).

    Boxes are xyxy in 640-input space. Returns empty arrays if nothing
    survives filtering.
    """
    global _postprocess_time_sum, _npu_count, _last_stats_ts
    t0 = time.monotonic()
    branches = 3
    pair = len(outputs) // branches  # = 3 (dfl, class_conf, class_sum)
    boxes_l, classes_conf_l = [], []
    for i in range(branches):
        boxes_l.append(_box_process(outputs[pair * i]))
        classes_conf_l.append(outputs[pair * i + 1])

    boxes = np.concatenate([_sp_flatten(b) for b in boxes_l])
    classes_conf = np.concatenate([_sp_flatten(c) for c in classes_conf_l])

    # The airockchip head emits already-sigmoided per-class scores.
    class_max = np.max(classes_conf, axis=-1)
    classes = np.argmax(classes_conf, axis=-1)

    pos = np.where(class_max >= conf_thres)
    if pos[0].size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return np.empty((0, 4), dtype=np.float32), empty.astype(np.int64), empty

    boxes = boxes[pos]
    classes = classes[pos]
    scores = class_max[pos]

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes.tolist()):
        m = classes == c
        b = boxes[m]; s = scores[m]
        keep = _nms_boxes(b, s)
        if keep.size:
            nboxes.append(b[keep])
            nclasses.append(np.full(keep.size, c, dtype=np.int64))
            nscores.append(s[keep])

    if not nboxes:
        empty = np.empty((0,), dtype=np.float32)
        _postprocess_time_sum += time.monotonic() - t0
        return np.empty((0, 4), dtype=np.float32), empty.astype(np.int64), empty

    result = np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)
    _postprocess_time_sum += time.monotonic() - t0
    return result


def infer(frame_bgr: np.ndarray, conf_thres: float = 0.5,
          rknn: RKNNLite | None = None
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, tuple[int, int]]:
    """Run NPU inference on a BGR frame.

    Returns (boxes_xyxy_640, classes, scores, scale, (pad_w, pad_h)).
    Use scale and pad to map boxes back to the original frame.
    If *rknn* is None, uses the singleton AUTO instance.
    """
    global _npu_time_sum, _npu_count, _last_stats_ts

    if rknn is None:
        rknn = _get_rknn()
    padded, scale, (pad_w, pad_h) = letterbox(frame_bgr, INPUT_SIZE)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    # Time NPU inference
    t_npu = time.monotonic()
    # RKNNLite expects a 4D NHWC tensor; add the batch dim.
    outputs = rknn.inference(inputs=[np.expand_dims(rgb, axis=0)])
    _npu_time_sum += time.monotonic() - t_npu

    boxes, classes, scores = post_process(outputs, conf_thres=conf_thres)

    _npu_count += 1
    # Print stats every 100 inferences
    if _npu_count % 100 == 0:
        avg_npu = _npu_time_sum / _npu_count * 1000  # ms
        avg_post = _postprocess_time_sum / _npu_count * 1000  # ms
        print(f"NPU profile: avg inference {avg_npu:.1f}ms  avg postprocess {avg_post:.1f}ms  total {avg_npu + avg_post:.1f}ms")

    return boxes, classes, scores, scale, (pad_w, pad_h)


def warmup() -> None:
    """Run one dummy inference to JIT/warm the NPU. ~500 ms first call."""
    rknn = _get_rknn()
    dummy = np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    rknn.inference(inputs=[dummy])


def warmup_all_cores() -> None:
    """Warm up the all-core RKNN instance."""
    rknn = _get_rknn_all_cores()
    dummy = np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    rknn.inference(inputs=[dummy])
    print("  Warmed up all-core RKNN instance")
