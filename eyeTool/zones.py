"""Per-stream alarm-zone polygons + point-in-polygon test.

A ``Zone`` is a closed polygon in *original-frame* coordinates (the same
space the camera produces frames in -- not tile space, not inference
space). Storing it that way means the zone survives:
  * tile resize / letterboxing in the compositor,
  * detection-input letterboxing,
  * any future change of display resolution.

Public helpers:
  ``load_zones(cfg)``           snapshot of {slot_id: Zone} from config.
  ``Zone.contains(x, y)``       True if the test point is inside or on
                                the polygon edge.
  ``Zone.draw_on_tile(...)``    render translucent fill + outline mapped
                                from frame-space onto the compositor tile.
  ``foot_point(box)``           bottom-centre of an ``(x1,y1,x2,y2)``
                                box, used as the surveillance test point.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# BGR. Light orange ~ #FFB347 = (71, 179, 255). Slightly more saturated
# outline so the polygon edge stays visible on bright video backgrounds.
COLOR_FILL = (71, 179, 255)
COLOR_OUTLINE = (0, 140, 255)
FILL_ALPHA = 0.22


@dataclass
class Zone:
    polygon: list[tuple[int, int]]   # at least 3 vertices in frame coords
    name: str = ""

    def __post_init__(self) -> None:
        # Pre-compute a numpy contour for cv2 calls (kept in frame space)
        if len(self.polygon) >= 3:
            self._contour = np.asarray(self.polygon, dtype=np.float32)
        else:
            self._contour = None

    @property
    def is_valid(self) -> bool:
        return self._contour is not None

    # ---- geometry --------------------------------------------------
    def contains(self, x: float, y: float) -> bool:
        """True if (x, y) is inside or on the polygon edge.

        Uses ``cv2.pointPolygonTest`` (C++ ray casting, signed distance).
        Returns True on the boundary (>= 0) so 'touching' counts as inside.
        """
        if self._contour is None:
            return False
        d = cv2.pointPolygonTest(self._contour, (float(x), float(y)), False)
        return d >= 0

    def intersects_bbox(self, x1: float, y1: float,
                        x2: float, y2: float) -> bool:
        """True if the axis-aligned bounding box (x1,y1)-(x2,y2) overlaps
        or touches the polygon (in original-frame coordinates).

        Three checks cover every case for arbitrary simple polygons:
          1. Any **bbox corner** lies inside / on the polygon.
          2. Any **polygon vertex** lies inside the bbox.
          3. Any **bbox edge** crosses any **polygon edge**
             (covers the X-shaped no-vertex-contained case).

        Returns True on touch (>= 0) so just-grazing counts as inside,
        matching the wishlist for surveillance alarm semantics.
        """
        if self._contour is None:
            return False
        # Normalise corners
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # 1) bbox corners inside polygon (boundary counts)
        bbox_corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        for cx, cy in bbox_corners:
            if cv2.pointPolygonTest(self._contour, (float(cx), float(cy)),
                                    False) >= 0:
                return True

        # 2) polygon vertices inside bbox
        for vx, vy in self.polygon:
            if x1 <= vx <= x2 and y1 <= vy <= y2:
                return True

        # 3) bbox edges crossing polygon edges
        bbox_edges = (
            (bbox_corners[0], bbox_corners[1]),
            (bbox_corners[1], bbox_corners[2]),
            (bbox_corners[2], bbox_corners[3]),
            (bbox_corners[3], bbox_corners[0]),
        )
        n = len(self.polygon)
        for i in range(n):
            p1 = self.polygon[i]
            p2 = self.polygon[(i + 1) % n]
            for a, b in bbox_edges:
                if _segments_intersect(a, b, p1, p2):
                    return True
        return False

    # ---- rendering -------------------------------------------------
    def draw_on_tile(self, tile: np.ndarray,
                     tile_scale: float, off_x: int, off_y: int) -> None:
        """Map polygon from frame-space into the tile and draw it.

        ``tile_scale``, ``off_x``, ``off_y`` come from the compositor's
        letterbox of the original frame into this tile (same trio used by
        ``overlay_tile_detections``).
        """
        if self._contour is None:
            return
        pts = self._contour.copy()
        pts[:, 0] = pts[:, 0] * tile_scale + off_x
        pts[:, 1] = pts[:, 1] * tile_scale + off_y
        pts_i = pts.astype(np.int32)
        # Translucent fill via per-channel addWeighted on a copy of the tile.
        overlay = tile.copy()
        cv2.fillPoly(overlay, [pts_i], COLOR_FILL)
        cv2.addWeighted(overlay, FILL_ALPHA, tile, 1.0 - FILL_ALPHA, 0, dst=tile)
        # Solid outline on top
        cv2.polylines(tile, [pts_i], isClosed=True,
                      color=COLOR_OUTLINE, thickness=1, lineType=cv2.LINE_AA)


def foot_point(box) -> tuple[float, float]:
    """Bottom-centre of an (x1, y1, x2, y2) bbox -- the standard ground-
    plane surveillance test point for upright people."""
    x1, y1, x2, y2 = box
    return (float(x1 + x2) * 0.5, float(y2))


def _ccw(p, q, r) -> float:
    """Signed twice-area of triangle (p, q, r); >0 if CCW, <0 if CW, 0 colinear."""
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _on_segment(p, q, r) -> bool:
    """True if r lies on the (closed) segment pq, given the three are colinear."""
    return (min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and
            min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))


def _segments_intersect(a1, a2, b1, b2) -> bool:
    """True if segment a1-a2 intersects (or touches) segment b1-b2.

    Standard CCW-cross-product test plus colinear endpoint-on-segment
    fallback so that just-touching counts as intersection (consistent with
    the wishlist's "or touching" requirement).
    """
    d1 = _ccw(b1, b2, a1)
    d2 = _ccw(b1, b2, a2)
    d3 = _ccw(a1, a2, b1)
    d4 = _ccw(a1, a2, b2)
    if (((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and
            ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))):
        return True
    # Colinear & overlapping cases
    if d1 == 0 and _on_segment(b1, b2, a1):
        return True
    if d2 == 0 and _on_segment(b1, b2, a2):
        return True
    if d3 == 0 and _on_segment(a1, a2, b1):
        return True
    if d4 == 0 and _on_segment(a1, a2, b2):
        return True
    return False


def load_zones(cfg) -> dict[int, Zone]:
    """Snapshot of {slot_id: Zone} from a ``Config`` object.

    Slots without a polygon (or with < 3 vertices) are simply absent from
    the returned dict, so callers can treat ``zones.get(slot_id)`` as
    "is there a zone defined here?".
    """
    out: dict[int, Zone] = {}
    for sid, slot_cfg in cfg.all_slots().items():
        poly = slot_cfg.get("polygon") if slot_cfg else None
        if not poly or len(poly) < 3:
            continue
        try:
            pts = [(int(p[0]), int(p[1])) for p in poly]
        except (TypeError, ValueError, IndexError):
            print(f"[zones] slot {sid}: invalid polygon, skipping")
            continue
        zone = Zone(polygon=pts, name=slot_cfg.get("name", ""))
        if zone.is_valid:
            out[int(sid)] = zone
    return out
