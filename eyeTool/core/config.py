"""3-tier JSON configuration for eyeTool.

Layered overlay (lowest precedence first):
  1. ``manufacturer_default.json`` -- shipped baseline, never edited at runtime.
  2. ``user_settings.json``        -- user tweaks (detection / preprocessing).
  3. ``zones.json``                -- per-stream slot bindings + polygons.

The merged result is exposed via ``Config.get(...)`` with dotted keys, e.g.
``cfg.get("detection.confidence")``. ``Config.save_user()`` writes the
*delta* (everything that differs from the manufacturer default) back to
``user_settings.json``; ``Config.save_zones()`` writes the zone slot map
to ``zones.json``.

Files live in the same directory as the running code by default; override
with ``EYETOOL_CONFIG_DIR`` if you want to keep them in ``~/.config``.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from typing import Any

# ---------------------------------------------------------------------------
# Built-in manufacturer defaults
# ---------------------------------------------------------------------------
# Written verbatim to manufacturer_default.json on first run if the file is
# missing. Never mutated at runtime. Edit this dict to ship new defaults.
MANUFACTURER_DEFAULTS: dict[str, Any] = {
    "display": {
        "width": 1280,
        "height": 800,
        "target_fps": 30,
        "show_perf_panel": True,
        # Preferred X DISPLAY string (e.g. ":0" for the built-in LCD,
        # ":1" for the HDMI/secondary). Empty string = auto-detect on
        # startup. Overridden by the --display CLI flag if given.
        "target": ":1",
    },
    "detection": {
        "enabled": True,
        "confidence": 0.5,
        "detect_every_n": 1,
        "use_multi_core": False,
        "person_class_id": 0,
    },
    "streams": {
        "max_streams": 4,
        "grid": "2x2",
        "watchdog_stall_s": 2.0,
        "min_capture_w": 1280,
        "min_capture_h": 720,
    },
    "preprocessing": {
        "rotation_deg": 0.0,
        "exposure_delta": 0,
        "contrast": 1.0,
    },
    "alarm": {
        "color_inside": [0, 0, 255],   # BGR red
        "color_outside": [0, 255, 0],  # BGR green
        "color_stale": [0, 255, 255],  # BGR yellow
        "stale_threshold_s": 0.2,
    },
    "recording": {
        "enabled": True,
        "save_dir": "/media/pi/6333-3864",
        "fallback_dir": "~/Videos",
        "segment_duration_min": 2,
        "codec": "mpp_h264",  # Options: mpp_h264 (HW), mp4v, mjpg, avc1
        "storage_threshold_percent": 20,  # Delete when disk is 20% full
    },
}


def _config_dir() -> str:
    override = os.environ.get("EYETOOL_CONFIG_DIR")
    if override:
        os.makedirs(override, exist_ok=True)
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    return here


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Return a new dict with *overlay* recursively merged on top of *base*."""
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _diff(default: dict, current: dict) -> dict:
    """Return the subset of *current* that differs from *default* (recursive)."""
    out: dict[str, Any] = {}
    for k, v in current.items():
        if k not in default:
            out[k] = copy.deepcopy(v)
            continue
        dv = default[k]
        if isinstance(v, dict) and isinstance(dv, dict):
            sub = _diff(dv, v)
            if sub:
                out[k] = sub
        elif v != dv:
            out[k] = copy.deepcopy(v)
    return out


class Config:
    """Thread-safe layered config holder.

    The ``settings`` view = manufacturer ⊕ user (detection/preprocessing/...).
    The ``zones`` view    = per-slot dict {slot_id: SlotConfig}.
    """

    MANUFACTURER_FILE = "manufacturer_default.json"
    MANUFACTURER_ZONES_FILE = "manufacturer_zones.json"
    USER_FILE = "user_settings.json"
    ZONES_FILE = "zones.json"

    def __init__(self, config_dir: str | None = None) -> None:
        self._dir = config_dir or _config_dir()
        self._lock = threading.Lock()
        self._manufacturer: dict[str, Any] = {}
        self._user: dict[str, Any] = {}
        self._zones: dict[str, Any] = {}
        self._merged: dict[str, Any] = {}
        self.reload()

    # --- paths -------------------------------------------------------
    @property
    def manufacturer_path(self) -> str:
        return os.path.join(self._dir, self.MANUFACTURER_FILE)

    @property
    def user_path(self) -> str:
        return os.path.join(self._dir, self.USER_FILE)

    @property
    def zones_path(self) -> str:
        return os.path.join(self._dir, self.ZONES_FILE)

    @property
    def manufacturer_zones_path(self) -> str:
        return os.path.join(self._dir, self.MANUFACTURER_ZONES_FILE)

    # --- io ----------------------------------------------------------
    def reload(self) -> None:
        with self._lock:
            self._manufacturer = self._load_or_seed(
                self.manufacturer_path, MANUFACTURER_DEFAULTS)
            self._user = self._load(self.user_path, default={})
            self._zones = self._load(self.zones_path,
                                     default={"slots": {}})
            self._merged = _deep_merge(self._manufacturer, self._user)

    @staticmethod
    def _load(path: str, default: dict) -> dict:
        if not os.path.exists(path):
            return copy.deepcopy(default)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"{path}: top-level must be an object")
            return data
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"[config] failed to read {path}: {e}; using defaults")
            return copy.deepcopy(default)

    def _load_or_seed(self, path: str, seed: dict) -> dict:
        if not os.path.exists(path):
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(seed, f, indent=2)
                print(f"[config] wrote seed defaults to {path}")
            except OSError as e:
                print(f"[config] could not seed {path}: {e}")
            return copy.deepcopy(seed)
        return self._load(path, default=seed)

    # --- accessors ---------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        """Look up a dotted key in the merged settings, e.g. 'detection.confidence'."""
        with self._lock:
            cur: Any = self._merged
            for part in dotted.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return default
                cur = cur[part]
            return copy.deepcopy(cur)

    def set(self, dotted: str, value: Any) -> None:
        """Update a dotted key in the live merged view (and the user delta)."""
        parts = dotted.split(".")
        with self._lock:
            for tgt in (self._merged, self._user):
                cur = tgt
                for p in parts[:-1]:
                    if p not in cur or not isinstance(cur[p], dict):
                        cur[p] = {}
                    cur = cur[p]
                cur[parts[-1]] = copy.deepcopy(value)

    def settings_view(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._merged)

    # --- zones / slots ----------------------------------------------
    def slot(self, slot_id: int) -> dict[str, Any] | None:
        with self._lock:
            slots = self._zones.get("slots", {})
            data = slots.get(str(slot_id))
            return copy.deepcopy(data) if data else None

    def all_slots(self) -> dict[int, dict[str, Any]]:
        with self._lock:
            slots = self._zones.get("slots", {})
            return {int(k): copy.deepcopy(v) for k, v in slots.items() if v}

    def update_slot(self, slot_id: int, data: dict[str, Any] | None) -> None:
        """Replace the full configuration of *slot_id* (None removes it)."""
        with self._lock:
            self._zones.setdefault("slots", {})
            if data is None:
                self._zones["slots"].pop(str(slot_id), None)
            else:
                self._zones["slots"][str(slot_id)] = copy.deepcopy(data)

    # --- persistence -------------------------------------------------
    def save_user(self) -> None:
        """Persist the *delta* between merged settings and manufacturer to user_settings.json."""
        with self._lock:
            delta = _diff(self._manufacturer, self._merged)
            self._user = delta
            self._atomic_write(self.user_path, delta)

    def save_zones(self) -> None:
        with self._lock:
            self._atomic_write(self.zones_path, self._zones)

    def save_all(self) -> None:
        self.save_user()
        self.save_zones()

    # --- factory / manufacturer-default operations -------------------
    def save_as_manufacturer_default(self, include_zones: bool = True) -> None:
        """Promote the *current* state to be the new manufacturer baseline.

        Behaviour:
          * Current merged settings are written to ``manufacturer_default.json``.
          * ``user_settings.json`` is reduced to an empty delta (deleted),
            because the manufacturer view now matches what was active.
          * If ``include_zones`` (default), the current zones map is also
            archived to ``manufacturer_zones.json`` so a future
            ``restore_manufacturer_default`` can bring the same slot
            bindings + polygons back.

        After this call the live merged view and zones map are unchanged
        from the user's perspective; only the *baseline* moved.
        """
        with self._lock:
            self._atomic_write(self.manufacturer_path, self._merged)
            self._manufacturer = copy.deepcopy(self._merged)
            self._user = {}
            try:
                os.remove(self.user_path)
            except FileNotFoundError:
                pass
            except OSError as e:
                print(f"[config] could not remove {self.user_path}: {e}")
            if include_zones:
                self._atomic_write(self.manufacturer_zones_path, self._zones)

    def restore_manufacturer_default(self, include_zones: bool = True) -> None:
        """Drop user overrides and revert to the manufacturer baseline.

          * ``user_settings.json`` is deleted; merged view becomes
            == manufacturer.
          * If ``include_zones`` and ``manufacturer_zones.json`` exists,
            the zones map is restored from it. Otherwise zones.json is
            cleared (no slot bindings, no polygons).
        """
        with self._lock:
            try:
                os.remove(self.user_path)
            except FileNotFoundError:
                pass
            except OSError as e:
                print(f"[config] could not remove {self.user_path}: {e}")
            self._user = {}
            self._merged = copy.deepcopy(self._manufacturer)
            if include_zones:
                if os.path.exists(self.manufacturer_zones_path):
                    self._zones = self._load(self.manufacturer_zones_path,
                                             default={"slots": {}})
                else:
                    self._zones = {"slots": {}}
                self._atomic_write(self.zones_path, self._zones)

    def clear_user_overrides(self) -> None:
        """Wipe ``user_settings.json`` only (keep zones intact)."""
        with self._lock:
            try:
                os.remove(self.user_path)
            except FileNotFoundError:
                pass
            except OSError as e:
                print(f"[config] could not remove {self.user_path}: {e}")
            self._user = {}
            self._merged = copy.deepcopy(self._manufacturer)

    def has_manufacturer_zones(self) -> bool:
        return os.path.exists(self.manufacturer_zones_path)

    @staticmethod
    def _atomic_write(path: str, data: dict) -> None:
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
            os.replace(tmp, path)
        except OSError as e:
            print(f"[config] failed to write {path}: {e}")
            try:
                os.remove(tmp)
            except OSError:
                pass


# Module-level singleton (lazy)
_config_singleton: Config | None = None


def get_config() -> Config:
    global _config_singleton
    if _config_singleton is None:
        _config_singleton = Config()
    return _config_singleton
