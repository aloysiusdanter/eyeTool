# Refinement Task: Multi-Stream CV System Optimization

## 1. Objective
Refactor the existing computer vision codebase to transition from a single-camera setup to a **multi-stream (up to 4) architecture**. The focus is on **resource concurrency** and **low-latency UI** for the NanoPi (800x480).

## 2. Scalable Video Compositor
- **Dynamic 2x2 Grid:** Implement/Update the tiling logic to handle $N$ cameras ($1 \le N \le 4$).
- **Fail-Safe Processing:** - Implement a watchdog to detect dropped USB streams.
    - **Optimization:** If a stream is "Unavailable," the thread must sleep or bypass the inference loop to free up CPU cycles for active streams.
- **Visuals:** Ensure labels and status messages ("Unavailable") are scaled correctly for the 400x240 quadrant size. Keep stream aspect ratio to display.

## 3. Stream Mapping & Probing
- **Flexible Mapping:** Implement a configuration layer to map physical device IDs (e.g., `/dev/videoX` or UUIDs) to specific grid positions (Top-Left, Bottom-Right, etc.).
- **Probing Function:** - Develop a background "Probe" thread that periodically checks the status of assigned but offline cameras.
    - This allows for "Hot-plugging" support—when a camera is plugged in, the probe should detect it and automatically initiate the stream into its pre-configured grid slot.

## 4. Detection & Alarm Logic
- **Inference Optimization:** Ensure the detection engine (last version) is threaded or batched to handle multiple inputs without serial blocking.
- **Priority Rendering:** - **Yellow Boxes (Outdated):** Move these to the top of the render stack (Highest Z-index).
    - **Polygon Logic:** Use a point-in-polygon algorithm (e.g., Ray Casting) to determine Red (Inside or Touching) vs Green (Outside) status.
- **Interface:** Ensure the Alarm Zone Polygon can be defined independently per camera ID.

## 5. "htop-Style" UI & Interaction
- **Static Console Frame:** Refactor the UI to use fixed coordinate updates (ncurses/blessings). Eliminate terminal scrolling.
- **Standard Monitoring Mode (htop-style):**
    - Use a fixed-position terminal UI (e.g., `ncurses`) to prevent scrolling.
    - Display real-time recognition performance metrics with a toggle (Show/Hide).
- **Interactive Configuration Mode (Triggered only for Polygon Setup):**
    - This mode is **only** active when the user explicitly triggers "Set Alarm Zone."
    - **Selection:** User selects which camera feed to configure.
    - **Interaction:** - A central **crosshair (pin-point)** appears over the selected feed.
        - `Arrow Keys`: Move pin-point by 1px.
        - `Shift + Arrow Keys`: Move pin-point by 10px.
        - `Enter`: Drop a vertex (point changes color).
        - `Esc`: Close the polygon (joining exterior points) and **exit** back to Monitoring Mode.
- **Toggle View:** Implement a hotkey to show/hide the "Recognition Performance" section.

## 6. Image Preprocessing
- Add a preprocessing pipeline stage for each feed *before* it hits the inference engine:
    - **Fine-Rotation:** (e.g., 0.5° increments).
    - **Exposure/Contrast:** Adjust using hardware-accelerated methods (if available via V4L2) or optimized NumPy operations.

## 7. Persistence & Config Portability
- **Decoupled Configs:** Ensure settings are not hardcoded. 
- **Hierarchy:** - Load `manufacturer_default.json` first.
    - Overlay `user_settings.json` (Detection/Preprocessing).
    - Overlay `zones.json` (Polygon coordinates per stream ID).
- **Save Trigger:** Implement a "Save-as-Default" function that writes the current runtime state back to the user JSON files.

## 8. Performance Constraints for NanoPi
- **Zero-Copy Goal:** Minimize frame copying between the capture thread, the inference engine, and the display compositor.
- **Thread Management:** Use a separate thread for each camera capture to prevent one slow USB bus from lagging the entire UI.