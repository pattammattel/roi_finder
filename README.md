# XRF ROI Scan Planner

Run the prototype with Pixi:

```bash
pixi install
pixi run roi-planner
```

`pixi.toml` pins a self-contained GUI environment (Python, PyQt6, pyqtgraph,
and NumPy). The supplied `requirements.txt` is retained only as a pip fallback.

Enter any scan number and click **Load XRF data**.  Until the data-access hook is
connected, this loads a deterministic synthetic three-element XRF data set.

* **Auto**: selects connected high-intensity regions from the displayed element.
  Replace `find_rois` / `_component_boxes` with the production segmentation method.
* **Manual**: right-drag on the image to create an ROI. ROIs can be moved or resized;
  right-click an ROI to remove it.
* **Generate scan plans**: converts ROI pixel bounds to sample coordinates using
  `XRFScan.pixel_size_um` and `XRFScan.origin_um`, then previews dimensions and time.
* **Send scans**: calls the isolated `send_scan_plans(plans)` hook.

To connect beamline data, replace `load_xrf_data_for_scan`. It must return:

```python
XRFScan(stack=<numpy array shape (n_elements, y, x)>,
        element_names=["Fe_K", "Cr_K", ...],
        pixel_size_um=(x_pitch, y_pitch),
        origin_um=(x_origin, y_origin))
```

This intentionally keeps the UI layer separate from databroker, pyxrf, or QueueServer
so those APIs can be introduced without changing ROI interaction or plan generation.
