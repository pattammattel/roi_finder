# XRF ROI Scan Planner

A small desktop tool for loading XRF image stacks, selecting regions of interest (ROIs), and generating scan plans from pixel-space boundaries.

## Quick start

This project is configured for Pixi:

```bash
pixi install
pixi run roi-planner
```

You can also run the module entry point directly:

```bash
pixi run gui
```

## Features

- Load XRF data for a scan number
- Display an element channel from a 3D XRF stack
- Auto-detect candidate ROIs or draw ROIs manually
- Adjust fine-scan parameters such as step size, dwell time, and padding
- Generate scan-plan summaries from selected ROIs
- Submit plans through a dedicated hook for beamline integration

## Phantom data

The app supports a built-in synthetic test dataset triggered by scan number `0000`. This is useful for testing the interface before connecting to real beamline data.

## Project structure

```text
src/
  roi_finder/
    __init__.py
    core.py
    main.py
    xrf_utils.py

tests/
  test_core.py
```

## Integration hooks

Real beamline data can be connected by replacing the `load_xrf_data_for_scan` and `send_scan_plans` hooks in the GUI code. The expected return type is:

```python
XRFScan(
    stack=<numpy array shape (n_elements, y, x)>,
    element_names=["Fe_K", "Cr_K", ...],
    pixel_size_um=(x_pitch, y_pitch),
    origin_um=(x_origin, y_origin),
)
```

The UI intentionally stays decoupled from databroker, pyxrf, and QueueServer so those dependencies can be introduced without changing the ROI interaction logic or plan-generation flow.

## Notes

- The project uses a `src/` package layout.
- PyQt6, pyqtgraph, NumPy, and hxntools are managed via Pixi.
- This repository is intended as a prototype and integration layer for future beamline-specific data access.
