from hxntools.CompositeBroker import db
from hxntools.scan_info import get_scan_positions
import numpy as np
import datetime


def get_flyscan_dimensions(hdr):
    start_doc = hdr.start
    # 2D_FLY_PANDA: prefer 'dimensions', fallback to 'shape'
    if 'scan' in start_doc and start_doc['scan'].get('type') == '2D_FLY_PANDA':
        if 'dimensions' in start_doc:
            dim = start_doc['dimensions']
        elif 'shape' in start_doc:
            dim = start_doc['shape']
        else:
            raise ValueError("No dimensions or shape found for 2D_FLY_PANDA scan")

        return dim[::-1]
    # rel_scan: use 'shape' or 'num_points'
    elif start_doc.get('plan_name') == 'rel_scan':
        if 'shape' in start_doc:
            dim = start_doc['shape']
        elif 'num_points' in start_doc:
            dim = [start_doc['num_points']]
        else:
            raise ValueError("No shape or num_points found for rel_scan")

        return dim[::-1]
    else:
        raise ValueError("Unknown scan type for get_flyscan_dimensions")

def get_all_scalar_data(hdr):

    keys = list(hdr.table().keys())
    scalar_keys = [k for k in keys if k.startswith('sclr1') ]
    #print(f"{scalar_keys = }")
    print(f"[DATA] fetching scalar data")
    scan_dim = get_flyscan_dimensions(hdr)
    scalar_stack_list = []

    for sclr in sorted(scalar_keys):
        
        scalar = np.array(list(hdr.data(sclr))).squeeze()
        sclr_img = scalar.reshape(scan_dim)
        scalar_stack_list.append(sclr_img)

    # Stack all the 2D images along a new axis (axis=0).
    scalar_stack = np.stack(scalar_stack_list, axis=0)

    #print("3D Stack shape:", xrf_stack.shape)

    return  scalar_stack, sorted(scalar_keys)

def get_all_xrf_roi_data(hdr):


    channels = [1, 2, 3]
    keys = list(hdr.table().keys())
    roi_keys = [k for k in keys if k.startswith('Det')]
    det1_keys = [k for k in keys if k.startswith('Det1')]
    elem_list = [k.replace("Det1_", "") for k in det1_keys]

    #print(f"{elem_list = }")
    print(f"[DATA] fetching XRF ROIs")
    try:
        scan_dim = get_flyscan_dimensions(hdr)
    except Exception as e:
        print(f"[DATA ERROR] cannot get scan dimensions, defaulting to (1, n_events). Error: {e}")
        scan_dim = (1, len(hdr.table()))
    xrf_stack_list = []

    for elem in sorted(elem_list):
        roi_keys = [f'Det{chan}_{elem}' for chan in channels]
        spectrum = np.sum([np.array(list(hdr.data(roi)), dtype=np.float32).squeeze() for roi in roi_keys], axis=0)
        xrf_img = spectrum.reshape(scan_dim)
        xrf_stack_list.append(xrf_img)

    # Stack all the 2D images along a new axis (axis=0).
    xrf_stack = np.stack(xrf_stack_list, axis=0)

    #print("3D Stack shape:", xrf_stack.shape)
    return xrf_stack, sorted(elem_list)

def get_scan_details(hdr):
    start_doc = hdr.start
    param_dict = {"scan_id": start_doc.get("scan_id")}

    if 'scan' in start_doc and start_doc['scan'].get('type') == '2D_FLY_PANDA':
        scan_cfg = start_doc.get('scan', {})
        scan_input = scan_cfg.get('scan_input')
        shape = scan_cfg.get('shape')

        datetime_object = datetime.datetime.fromtimestamp(start_doc["time"])
        formatted_time = datetime_object.strftime('%Y-%m-%d %H:%M:%S')
        param_dict["time"] = formatted_time
        param_dict["motors"] = start_doc.get("motors", [])
        param_dict["scan"] = scan_cfg

        if isinstance(scan_input, (list, tuple)) and len(scan_input) >= 6:
            param_dict["scan_start1"] = float(scan_input[0])
            param_dict["scan_end1"] = float(scan_input[1])
            param_dict["num1"] = int(scan_input[2])
            param_dict["scan_start2"] = float(scan_input[3])
            param_dict["scan_end2"] = float(scan_input[4])
            param_dict["num2"] = int(scan_input[5])

        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            param_dict["shape"] = [int(shape[0]), int(shape[1])]

        for key in ("zp_theta", "mll_theta", "energy"):
            value = start_doc.get(key)
            if value is not None:
                param_dict[key] = float(value)

        for key in ("detectors", "detector_distance", "dwell", "fast_axis", "slow_axis"):
            if key in scan_cfg:
                param_dict[key] = scan_cfg[key]

        return param_dict

    elif start_doc.get('plan_name') == 'rel_scan':
        datetime_object = datetime.datetime.fromtimestamp(start_doc["time"])
        formatted_time = datetime_object.strftime('%Y-%m-%d %H:%M:%S')
        param_dict["time"] = formatted_time
        param_dict["motors"] = start_doc.get("motors", [])
        param_dict["detectors"] = start_doc.get("detectors", [])
        param_dict["num_points"] = start_doc.get("num_points", None)
        param_dict["num_intervals"] = start_doc.get("num_intervals", None)
        param_dict["plan_args"] = start_doc.get("plan_args", {})
        param_dict["plan_type"] = start_doc.get("plan_type", None)
        param_dict["plan_name"] = start_doc.get("plan_name", None)
        param_dict["scan_name"] = start_doc.get("scan_name", None)
        param_dict["sample"] = start_doc.get("sample", None)
        param_dict["PI"] = start_doc.get("PI", None)
        param_dict["experimenters"] = start_doc.get("experimenters", None)
        param_dict["shape"] = start_doc.get("shape", None)
        return param_dict

    else:
        datetime_object = datetime.datetime.fromtimestamp(start_doc["time"])
        formatted_time = datetime_object.strftime('%Y-%m-%d %H:%M:%S')
        param_dict["time"] = formatted_time
        param_dict["motors"] = start_doc.get("motors", [])
        param_dict["detectors"] = start_doc.get("detectors", [])
        return param_dict