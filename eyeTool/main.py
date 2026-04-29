#!/usr/bin/env python3
"""eyeTool - CLI entry point.

Target platform: FriendlyElec NanoPi M6 (Rockchip RK3588S, aarch64)
running FriendlyElec Ubuntu 24.04 Desktop. See README.md for the full
list of prerequisites.
"""

import sys

from cli import parse_args
from core.camera import resolve_camera_source
from core.display import auto_set_display, set_display


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.display is not None:
        set_display(args.display)
    else:
        auto_set_display()

    source = resolve_camera_source(args.device)

    if args.mode == "menu":
        try:
            from ui.tui.app import run_tui
            run_tui(source, args.output)
        except ImportError:
            from ui.menus import interactive_menu
            interactive_menu(source, args.output)
    elif args.mode == "feed":
        from ui.menus import load_camera_feed
        load_camera_feed(source)
    elif args.mode == "capture":
        from ui.menus import capture_single_image
        capture_single_image(source, args.output)
    elif args.mode == "probe":
        from ui.menus import probe_camera
        probe_camera(source)
    elif args.mode == "record-multi":
        from ui.menus import record_multi_camera_feed
        record_multi_camera_feed()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        sys.exit(1)
