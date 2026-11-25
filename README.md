# Retrolensa

**Retrolensa** is an offline, real-time Python hand-tracking filter app powered by **Google MediaPipe**, **OpenCV**, and **NumPy**. Holding both hands up creates a dynamic 4-corner live filter portal anchored between your hands—stretching, rotating, and moving seamlessly across your webcam feed while rendering 8 live computer-vision effects strictly inside the portal.

## Features

### Spatial Hand-Tracking Controls
- **Two-Hand Dynamic Portal**: Raising both hands creates a 4-corner live frame anchored between your left and right palm centers.
- **Live Resize, Rotate & Move**: Pull your hands apart to stretch the portal, tilt one hand relative to the other to rotate it in real time, and move them over your face or background.
- **Gesture-Based Filter Switching**: Raising a single open palm cycles smoothly to the next filter with stability hold frames and a built-in cooldown (`1.5s` default) to prevent accidental skips.
- **EMA Jitter Smoothing**: Exponential moving average smoothing keeps hand anchors and portal edges steady without jitter.

### The 8 Real-Time Visual Filters
1. **Thermal** — Infrared heat-vision color mapping (`COLORMAP_JET`)
2. **Sketch** — High-contrast pencil drawing using grayscale inversion & Gaussian division
3. **Glitch** — RGB chromatic aberration channel shifting & scanline distortion
4. **Mono** — Clean high-contrast black & white
5. **Dual-Tone** — Pop-art pink and yellow gradient mapping
6. **Invert** — Full negative color inversion
7. **Sepia** — Warm vintage retro film tone
8. **Blur** — Heavy Gaussian soft-focus privacy blur

### Technical & Performance
- **100% Offline & Free**: Runs locally via MediaPipe `HandLandmarker` (`auto` GPU/CPU delegate fallback), requiring zero API keys or cloud calls.
- **Auto-Model Setup**: Automatically downloads `hand_landmarker.task` on first launch into `models/hand_landmarker.task` (or custom `--model-path`), then works fully offline.
- **Minimal HUD**: Displays active filter, 8-filter selector pips, live FPS, delegate mode, cooldown/hold progress bar, and gesture/key hints.

## Quickstart

From the `retrolensa` directory:

```powershell
# Using the prepared virtual environment
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m retrolensa
```

Or with your own Python environment:

```powershell
pip install -r requirements.txt
pip install -e .
retrolensa
```

## Controls & CLI Options

### Keyboard Shortcuts
- `1`–`8`: Jump directly to a filter
- `Space` / `N`: Next filter
- `P`: Previous filter
- `H`: Toggle minimal HUD
- `L`: Toggle hand landmark skeleton overlay
- `M`: Toggle selfie mirror flip
- `S`: Save PNG snapshot to `snapshots/`
- `Q` / `Esc`: Quit

### Useful Flags
```powershell
python -m retrolensa --camera 0 --width 1280 --height 720 --delegate auto
python -m retrolensa --filter glitch --ema-alpha 0.35 --cooldown 1.5
python -m retrolensa --source sample.mp4 --no-mirror
```
