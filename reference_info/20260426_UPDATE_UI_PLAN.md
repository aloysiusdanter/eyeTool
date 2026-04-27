# htop-Style Unified TUI Redesign + Module Reorganization

This plan redesigns the entire eyeTool application interface to use a unified htop-style curses TUI with multiple modes/tabs, replacing the current text-based menu system while preserving all existing functionality. Additionally, it reorganizes the project modules for better maintainability and clarity.

## Design Decisions Made

### Video Display Strategy
- **Choice**: Keep OpenCV-based features (polygon editor, preprocess editor, multi-camera feed) as separate GUI windows
- **Rationale**: These require pixel-perfect video rendering that curses cannot provide
- **TUI Integration**: The TUI will show status messages and instructions when these features are active, maintaining visual consistency
- **User Flow**: User selects feature from TUI → OpenCV window opens → TUI shows "Feature active, press Q to return" message → User closes feature → Returns to TUI

### Main Layout Structure
```
┌─────────────────────────────────────────────────────┐
│ Top Panel: Camera Status | Slot Bindings | Metrics  │ ← 3-4 lines
├─────────────────────────────────────────────────────┤
│                                                     │
│ Main Panel: Interactive Content Area                │ ← Variable height
│ (Current mode-specific content)                     │
│                                                     │
├─────────────────────────────────────────────────────┤
│ Bottom Panel: Event Log | Help/Shortcuts            │ ← 4-5 lines
└─────────────────────────────────────────────────────┘
```

**Top Panel Components**:
- Camera status: Connected cameras count, device nodes, models
- Slot bindings: Which port_path is bound to which slot (0-3)
- Detection metrics: Detection ON/OFF, confidence, FPS, persons detected

**Main Panel**: Dynamic content based on current mode/tab
- Main Menu: List of 11 options with arrow navigation
- Sub-menus: Setup zones, preprocessing, config, etc.
- Status messages: When OpenCV features are active
- Interactive forms: Input prompts rendered in curses

**Bottom Panel**:
- Left: Rolling event log (last 12 events like current monitor.py)
- Right: Context-sensitive help/shortcuts for current mode

### Navigation Model
- **Style**: Multiple modes/tabs (like htop's F-key screens)
- **Input**: Arrow keys (Up/Down to navigate, Enter to select, Esc to go back)
- **Reason for arrow keys**: More extensible than F-keys or number keys as features grow
- **Mode transitions**: Main menu → Sub-menu → Feature execution → Return

### Real-time Updates
- **Refresh rate**: 400ms - same as current monitor.py
- **What updates**:
  - Top panel: Camera hotplug detection, slot states, detection metrics
  - Bottom panel: Event log
  - Main panel: Only when content changes (menu selection, status updates)

## Module Reorganization Plan

### Current Structure Issues
The current project has several organizational problems:
- **main.py is monolithic**: 1081 lines containing CLI, menus, display detection, camera feeds, zone setup, preprocessing, configuration
- **Flat module structure**: 18 top-level modules with no logical grouping
- **Mixed concerns**: UI, core logic, and CLI entry point intermingled
- **Hard to navigate**: Related functionality scattered across files

### Proposed New Structure
```
eyeTool/
├── __init__.py
├── main.py                      # CLI entry point only (~50 lines)
├── cli.py                       # Argument parsing and CLI mode dispatch
│
├── core/                        # Core functionality
│   ├── __init__.py
│   ├── camera.py                # Camera utilities (from camera.py)
│   ├── config.py                # Configuration management (from config.py)
│   ├── hotplug.py               # USB hotplug detection (from hotplug.py)
│   ├── display.py               # Display detection (extracted from main.py)
│   └── zones.py                 # Zone management (from zones.py)
│
├── detection/                   # Detection and inference
│   ├── __init__.py
│   ├── pipeline.py              # Frame pipeline and detection (from pipeline.py)
│   └── rknn_yolov8.py           # NPU inference (from rknn_yolov8.py)
│
├── streams/                      # Stream management
│   ├── __init__.py
│   ├── stream.py                # Stream management (from stream.py)
│   └── compositor.py            # Video compositing (from compositor.py)
│
├── preprocessing/               # Image preprocessing
│   ├── __init__.py
│   └── preprocess.py            # Preprocessing logic (from preprocess.py)
│
├── ui/                          # All UI-related code
│   ├── __init__.py
│   ├── tui/                     # New curses TUI
│   │   ├── __init__.py
│   │   ├── app.py               # Main TUI application class
│   │   ├── panels.py            # Panel rendering (top, main, bottom)
│   │   ├── modes.py             # Mode definitions (main menu, sub-menus)
│   │   └── layout.py            # Layout management
│   ├── editors/                 # OpenCV-based editors
│   │   ├── __init__.py
│   │   ├── polygon_editor.py    # Polygon editor (from polygon_editor.py)
│   │   └── preprocess_editor.py # Preprocessing editor (from preprocess_editor.py)
│   ├── menus.py                 # Menu logic (extracted from main.py)
│   └── monitor.py               # Curses monitoring dashboard (from monitor.py)
│
└── utils/                       # Shared utilities
    ├── __init__.py
    └── terminal_input.py        # Terminal input handling (from terminal_input.py)
```

### Reorganization Benefits
1. **Logical grouping**: Related functionality grouped into packages
2. **Smaller modules**: main.py reduced from 1081 to ~50 lines
3. **Clear separation**: UI, core logic, and CLI entry point separated
4. **Easier navigation**: Package structure makes code organization obvious
5. **Better testability**: Smaller, focused modules easier to test
6. **Scalability**: Easy to add new features in appropriate packages

### Migration Strategy

#### Phase 0: Preparation
- Create new package directories with __init__.py files
- Update imports in all files to use new structure
- Ensure all existing tests pass

#### Phase 1: Extract Core (core/)
- Move camera.py → core/camera.py
- Move config.py → core/config.py
- Move hotplug.py → core/hotplug.py
- Move zones.py → core/zones.py
- Extract display detection functions from main.py → core/display.py
  - `detect_x_displays()`
  - `_merge_mutter_xauth()`
  - `set_display()`
  - `auto_set_display()`
  - `select_display_menu()`

#### Phase 2: Extract Detection (detection/)
- Move pipeline.py → detection/pipeline.py
- Move rknn_yolov8.py → detection/rknn_yolov8.py

#### Phase 3: Extract Streams (streams/)
- Move stream.py → streams/stream.py
- Move compositor.py → streams/compositor.py

#### Phase 4: Extract Preprocessing (preprocessing/)
- Move preprocess.py → preprocessing/preprocess.py

#### Phase 5: Extract UI (ui/)
- Move monitor.py → ui/monitor.py
- Move polygon_editor.py → ui/editors/polygon_editor.py
- Move preprocess_editor.py → ui/editors/preprocess_editor.py
- Extract menu functions from main.py → ui/menus.py
  - `setup_zones_menu()`
  - `setup_zone_for_slot()`
  - `preprocess_settings_menu()`
  - `setup_preprocess_for_slot()`
  - `configuration_menu()`
  - `detection_settings_menu()`

#### Phase 6: Extract Utilities (utils/)
- Move terminal_input.py → utils/terminal_input.py

#### Phase 7: Create CLI Entry Point (cli.py + main.py)
- Extract argument parsing from main.py → cli.py
  - `parse_args()`
- Simplify main.py to:
  - Import from cli
  - Call appropriate function based on mode
  - Keep only essential entry point logic (~50 lines)

#### Phase 8: Update All Imports
- Update imports in all files to use new package structure
- Example: `from camera import open_camera` → `from core.camera import open_camera`
- Example: `from pipeline import Detector` → `from detection.pipeline import Detector`
- Test all functionality to ensure imports work correctly

### Import Migration Examples

**Before**:
```python
from camera import open_camera
from pipeline import Detector
from config import get_config
```

**After**:
```python
from core.camera import open_camera
from detection.pipeline import Detector
from core.config import get_config
```

**Before** (in main.py):
```python
from polygon_editor import run as run_polygon_editor
from preprocess_editor import run as run_pp_editor
```

**After**:
```python
from ui.editors.polygon_editor import run as run_polygon_editor
from ui.editors.preprocess_editor import run as run_pp_editor
```

## Architecture

### New File Structure (with TUI)
```
eyeTool/
├── core/                        # Core functionality
│   ├── camera.py
│   ├── config.py
│   ├── hotplug.py
│   ├── display.py
│   └── zones.py
├── detection/                   # Detection and inference
│   ├── pipeline.py
│   └── rknn_yolov8.py
├── streams/                      # Stream management
│   ├── stream.py
│   └── compositor.py
├── preprocessing/               # Image preprocessing
│   └── preprocess.py
├── ui/                          # All UI-related code
│   ├── tui/                     # New curses TUI
│   │   ├── app.py
│   │   ├── panels.py
│   │   ├── modes.py
│   │   └── layout.py
│   ├── editors/
│   │   ├── polygon_editor.py
│   │   └── preprocess_editor.py
│   ├── menus.py                 # Menu logic (extracted from main.py)
│   └── monitor.py
├── utils/                       # Shared utilities
│   └── terminal_input.py
├── cli.py                       # CLI argument parsing
└── main.py                      # CLI entry point (~50 lines)
```

### Component Responsibilities

**tui/app.py**:
- Main curses loop
- Input handling (arrow keys, Enter, Esc, q/Q)
- Mode switching logic
- Refresh scheduling
- Integration with existing eyeTool modules

**tui/panels.py**:
- `draw_top_panel()`: Camera status, slots, metrics
- `draw_main_panel()`: Mode-specific content
- `draw_bottom_panel()`: Event log, help
- Helper functions for consistent styling (colors, borders)

**tui/modes.py**:
- Mode classes for each screen:
  - `MainMenuMode`: 11-option main menu
  - `ZonesMenuMode`: Slot picker for zone setup
  - `PreprocessMenuMode`: Slot picker for preprocessing
  - `ConfigMenuMode`: Configuration options
  - `DisplayMenuMode`: Display selection
  - `DetectionMenuMode`: Detection settings
  - `StatusMode`: Shows messages when OpenCV features are active
- Each mode implements:
  - `render()`: Draw content to main panel
  - `handle_input()`: Process key presses
  - `get_help()`: Return help text for bottom panel

**tui/layout.py**:
- Calculate panel heights based on terminal size
- Handle terminal resize
- Ensure minimum terminal size requirements (24x80)

**ui/menus.py**:
- Extracted menu functions from main.py:
  - `setup_zones_menu()`
  - `setup_zone_for_slot()`
  - `preprocess_settings_menu()`
  - `setup_preprocess_for_slot()`
  - `configuration_menu()`
  - `detection_settings_menu()`
- These functions will be called from TUI modes

**core/display.py**:
- Extracted display functions from main.py:
  - `detect_x_displays()`
  - `_merge_mutter_xauth()`
  - `set_display()`
  - `auto_set_display()`
  - `select_display_menu()`

**cli.py**:
- Extracted argument parsing from main.py:
  - `parse_args()`
- CLI mode dispatch logic

**main.py** (simplified):
- Import from cli
- Call appropriate function based on mode
- Keep only essential entry point logic (~50 lines)

### Integration with Existing Code

**main.py changes**:
- Remove `interactive_menu()` function (text-based)
- Add `run_tui()` function that launches the new curses TUI
- Extract menu functions to ui/menus.py
- Extract display functions to core/display.py
- Extract argument parsing to cli.py
- Keep all feature functions in ui/menus.py:
  - `load_camera_feed()`, `load_multi_camera_feed()`
  - `setup_zone_for_slot()`, `setup_preprocess_for_slot()`
  - `capture_single_image()`, `probe_camera()`
  - `configuration_menu()`, `detection_settings_menu()`
- These functions will be called from TUI modes

**monitor.py** → ui/monitor.py:
- Keep as standalone tool for debugging (user might want passive monitoring in separate SSH window)
- Move to ui/ package for consistency

**terminal_input.py** → utils/terminal_input.py:
- Keep unchanged (used by polygon_editor.py and preprocess_editor.py)
- TUI will use curses input directly
- Move to utils/ package

## Implementation Phases

### Phase 0: Module Reorganization (Foundation)
- Create new package directories (core/, detection/, streams/, preprocessing/, ui/, utils/)
- Add __init__.py files to each package
- Move modules to appropriate packages:
  - camera.py → core/camera.py
  - config.py → core/config.py
  - hotplug.py → core/hotplug.py
  - zones.py → core/zones.py
  - pipeline.py → detection/pipeline.py
  - rknn_yolov8.py → detection/rknn_yolov8.py
  - stream.py → streams/stream.py
  - compositor.py → streams/compositor.py
  - preprocess.py → preprocessing/preprocess.py
  - monitor.py → ui/monitor.py
  - polygon_editor.py → ui/editors/polygon_editor.py
  - preprocess_editor.py → ui/editors/preprocess_editor.py
  - terminal_input.py → utils/terminal_input.py
- Update all imports throughout the codebase
- Test that all existing functionality still works

### Phase 1: Extract Display Logic
- Extract display functions from main.py → core/display.py:
  - `detect_x_displays()`
  - `_merge_mutter_xauth()`
  - `set_display()`
  - `auto_set_display()`
  - `select_display_menu()`
- Update imports in main.py
- Test display selection functionality

### Phase 2: Extract Menu Logic
- Extract menu functions from main.py → ui/menus.py:
  - `setup_zones_menu()`
  - `setup_zone_for_slot()`
  - `preprocess_settings_menu()`
  - `setup_preprocess_for_slot()`
  - `configuration_menu()`
  - `detection_settings_menu()`
- Keep camera feed functions in ui/menus.py:
  - `load_camera_feed()`
  - `load_multi_camera_feed()`
  - `capture_single_image()`
  - `probe_camera()`
- Update imports in main.py
- Test all menu functionality

### Phase 3: Extract CLI Logic
- Extract argument parsing from main.py → cli.py:
  - `parse_args()`
- Simplify main.py to CLI entry point only (~50 lines)
- Test all CLI modes (menu, feed, capture, probe)

### Phase 4: TUI Foundation (tui/app.py + layout.py)
- Create TUI application class with curses wrapper
- Implement panel layout calculation
- Add basic input loop (arrow keys, Enter, Esc, q)
- Add color scheme (green/red/yellow for status)
- Test with placeholder panels

### Phase 5: Top and Bottom Panels (tui/panels.py)
- Implement `draw_top_panel()`:
  - Call `hotplug.list_cameras()` for camera status
  - Read config for slot bindings
  - Show detection state from global variables
- Implement `draw_bottom_panel()`:
  - Rolling event log (deque, max 12 events)
  - Context-sensitive help text
- Auto-refresh every 400ms

### Phase 6: Main Menu Mode (tui/modes.py - MainMenuMode)
- Render 11 menu options with arrow navigation
- Highlight selected item
- Handle Enter to execute feature
- Handle Esc to exit
- Map each option to functions in ui/menus.py

### Phase 7: Sub-menu Modes
- Implement `ZonesMenuMode`: Slot picker (0-3 + Back)
- Implement `PreprocessMenuMode`: Slot picker with preprocessing values
- Implement `ConfigMenuMode`: Configuration options
- Implement `DisplayMenuMode`: Display selection
- Implement `DetectionMenuMode`: Detection settings
- All use arrow navigation + Enter

### Phase 8: Status Mode for OpenCV Features
- Implement `StatusMode` to show messages when OpenCV features are active
- Example: "Multi-camera feed running. Press Q to return."
- TUI pauses updates while OpenCV window is open
- User closes OpenCV window → TUI resumes

### Phase 9: Integration (main.py)
- Replace `interactive_menu()` with `run_tui()`
- Wire up mode transitions to functions in ui/menus.py
- Test all 11 menu options
- Ensure config persistence still works

### Phase 10: Polish
- Add keyboard shortcuts (F-keys for quick mode access?)
- Improve visual styling (borders, colors)
- Add error handling for terminal too small
- Test on NanoPi hardware (800x480 display)
- Ensure SSH compatibility

## Alternative Design Choices Considered

### Video Display Options
- **Option A (Chosen)**: Keep OpenCV, TUI shows status messages
  - Pros: Preserves pixel-perfect video, minimal code changes
  - Cons: Two different UI paradigms
- **Option B**: ASCII art in curses
  - Pros: Single UI paradigm
  - Cons: Poor video quality, complex implementation
- **Option C**: Hybrid (video in curses block characters)
  - Pros: Single UI
  - Cons: Very complex, limited resolution

### Navigation Options
- **Option A (Chosen)**: Arrow keys + Enter
  - Pros: Extensible, familiar
  - Cons: Slower than F-keys for power users
- **Option B**: F-keys (F1-F11)
  - Pros: Fast access
  - Cons: Limited to 12 options, SSH issues
- **Option C**: Number keys (1-11)
  - Pros: Fast, familiar
  - Cons: Limited to 9-10 options, less intuitive for navigation

### Layout Options
- **Option A (Chosen)**: Three panels (top/main/bottom)
  - Pros: Clear separation, like htop
  - Cons: Less space for main content
- **Option B**: Two panels (main/bottom)
  - Pros: More space for main content
  - Cons: Status/metrics clutter main panel
- **Option C**: Single panel with sections
  - Pros: Maximum space
  - Cons: Hard to read, no scrolling

### Module Organization Options
- **Option A (Chosen)**: Logical packages (core, detection, streams, ui, utils)
  - Pros: Clear separation, follows Python best practices, scalable
  - Cons: More directory levels, more __init__.py files
- **Option B**: Keep flat structure
  - Pros: Simpler structure, fewer directories
  - Cons: Hard to navigate, main.py too large, mixed concerns
- **Option C**: Minimal grouping (ui/, rest flat)
  - Pros: Some organization
  - Cons: Still flat for most modules, doesn't solve main.py size

## Testing Strategy

1. **Module Migration Tests**: After each migration phase, ensure all imports work and tests pass
2. **Unit Tests**: Test each mode class independently
3. **Integration Tests**: Test mode transitions
4. **Hardware Tests**: Test on NanoPi with 800x480 display
5. **SSH Tests**: Test over SSH without X11 forwarding
6. **Regression Tests**: Ensure all existing features still work after reorganization

## Risks and Mitigations

**Risk**: Import breakage during module reorganization
- **Mitigation**: Migrate incrementally, test after each phase, use absolute imports

**Risk**: main.py becomes too complex during extraction
- **Mitigation**: Extract to ui/menus.py first, then to cli.py, keep main.py minimal

**Risk**: Terminal size too small on NanoPi (800x480 = ~100x30 chars)
- **Mitigation**: Enforce minimum size, show error if too small, optimize layout

**Risk**: Curses compatibility issues
- **Mitigation**: Use standard curses API, test on Ubuntu 24.04, handle missing ncurses

**Risk**: Performance impact from 400ms refresh
- **Mitigation**: Only redraw changed panels, use efficient string operations

**Risk**: User confusion between TUI and OpenCV windows
- **Mitigation**: Clear status messages, consistent "press Q to return" pattern

**Risk**: Package structure adds complexity
- **Mitigation**: Clear documentation, consistent naming, __init__.py exports for convenience
