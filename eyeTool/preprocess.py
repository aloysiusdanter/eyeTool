"""Per-camera image preprocessing (brightness / contrast / saturation /
gamma).

Tuned for the RK3588 CPU budget at 25-60 fps on 720p frames:

* The combined brightness + contrast + gamma transform is collapsed
  into a single 256-entry LUT and applied with ``cv2.LUT`` (one
  vectorised C call, ~0.5 ms on a 1280x720 frame).
* Saturation requires a BGR<->HSV round-trip (~3-5 ms) so it is
  *only* applied when the user actually changed it; the common
  "saturation == 1.0" case is a no-op.

A ``Preprocess`` instance is callable. Plug it into
``FrameSource(preprocess=...)`` and any frame that comes out of the
capture worker is run through the LUT/HSV pipeline before either the
detector or the compositor see it -- so the detector sees exactly
what the user sees, and the per-slot controls are honoured everywhere
without any extra code at the consumer side.

Stored format (``zones.json[slots][N].preprocessing``)::

    {
        "brightness": 0.0,    # additive, range [-1.0, +1.0]  (0 = off)
        "contrast":   1.0,    # multiplicative around 128, [0.0, 3.0]
        "saturation": 1.0,    # HSV S scale,                  [0.0, 3.0]
        "gamma":      1.0     # gamma correction,             [0.1, 3.0]
    }
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# Sliders / config bounds (mirrored by the trackbar UI)
BRIGHTNESS_RANGE = (-1.0, 1.0)
CONTRAST_RANGE = (0.0, 3.0)
SATURATION_RANGE = (0.0, 3.0)
GAMMA_RANGE = (0.1, 3.0)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


@dataclass
class Preprocess:
    brightness: float = 0.0   # [-1, 1]
    contrast: float = 1.0     # [0, 3]
    saturation: float = 1.0   # [0, 3]
    gamma: float = 1.0        # [0.1, 3]

    def __post_init__(self) -> None:
        self.brightness = _clamp(self.brightness, *BRIGHTNESS_RANGE)
        self.contrast = _clamp(self.contrast, *CONTRAST_RANGE)
        self.saturation = _clamp(self.saturation, *SATURATION_RANGE)
        self.gamma = _clamp(self.gamma, *GAMMA_RANGE)
        self._build_lut()

    # --- LUT --------------------------------------------------------
    def _build_lut(self) -> None:
        x = np.arange(256, dtype=np.float32) / 255.0
        # gamma first (operates on linearish range)
        x = np.power(np.clip(x, 1e-6, 1.0), 1.0 / max(0.1, self.gamma))
        x *= 255.0
        # contrast is a scale around mid-grey
        x = (x - 128.0) * self.contrast + 128.0
        # brightness is an additive offset (full ±255 at ±1.0)
        x = x + self.brightness * 255.0
        self._lut = np.clip(x, 0, 255).astype(np.uint8)

    # --- runtime helpers --------------------------------------------
    def is_identity(self) -> bool:
        """True when this Preprocess is a no-op (skip the work)."""
        return (abs(self.brightness) < 1e-3 and
                abs(self.contrast - 1.0) < 1e-3 and
                abs(self.saturation - 1.0) < 1e-3 and
                abs(self.gamma - 1.0) < 1e-3)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing in-place-friendly fashion. Always returns
        a (possibly new) frame; caller can use it directly."""
        if self.is_identity() or frame is None or frame.size == 0:
            return frame
        out = cv2.LUT(frame, self._lut)
        if abs(self.saturation - 1.0) >= 1e-3:
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
            s = hsv[:, :, 1].astype(np.float32) * self.saturation
            hsv[:, :, 1] = np.clip(s, 0, 255).astype(np.uint8)
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return out

    # --- (de)serialisation ------------------------------------------
    @classmethod
    def from_dict(cls, d: dict | None) -> "Preprocess":
        if not d:
            return cls()
        return cls(
            brightness=float(d.get("brightness", 0.0)),
            contrast=float(d.get("contrast", 1.0)),
            saturation=float(d.get("saturation", 1.0)),
            gamma=float(d.get("gamma", 1.0)),
        )

    def to_dict(self) -> dict:
        return {
            "brightness": round(self.brightness, 3),
            "contrast": round(self.contrast, 3),
            "saturation": round(self.saturation, 3),
            "gamma": round(self.gamma, 3),
        }
