from __future__ import annotations

from typing import Any, Sequence

from hxntools.CompositeBroker import db
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.zmq import REManagerAPI

RM = REManagerAPI()


def get_roi_positions_from_scan(scan_num, zp_flag: bool = True):
    baseline = db.get_table(db[scan_num], stream_name="baseline")
    position = baseline.iloc[0]

    if zp_flag:
        return {
            "zpssx": float(position["zpssx"]),
            "zpssy": float(position["zpssy"]),
            "zpssz": float(position["zpssz"]),
            "smarx": float(position["smarx"]),
            "smary": float(position["smary"]),
            "smarz": float(position["smarz"]),
            "zp.zpz1": float(position["zpz1"]),
            "zpsth": float(position["zpsth"]),
            "zps.zpsx": float(position["zpsx"]),
            "zps.zpsz": float(position["zpsz"]),
        }

    return {
        "dssx": float(position["dssx"]),
        "dssy": float(position["dssy"]),
        "dssz": float(position["dssz"]),
        "dsx": float(position["dsx"]),
        "dsy": float(position["dsy"]),
        "dsz": float(position["dsz"]),
        "sbz": float(position["sbz"]),
        "dsth": float(position["dsth"]),
    }


def send_fly2d_recover_and_scan(
    label,
    roi_positions,
    dets,
    mot1,
    mot1_s,
    mot1_e,
    mot1_n,
    mot2,
    mot2_s,
    mot2_e,
    mot2_n,
    exp_t,
    ic1_count=550,
    scan_time_min=5.0,
    zp_flag=True,
):
    det_names = (
        [detector.name for detector in eval(dets)]
        if isinstance(dets, str)
        else [detector.name for detector in dets]
    )
    mot1_name = mot1 if isinstance(mot1, str) else mot1.name
    mot2_name = mot2 if isinstance(mot2, str) else mot2.name

    plan = BPlan(
        "recover_pos_and_scan",
        label,
        roi_positions,
        det_names,
        mot1_name,
        mot1_s,
        mot1_e,
        mot1_n,
        mot2_name,
        mot2_s,
        mot2_e,
        mot2_n,
        exp_t,
        ic1_count,
        scan_time_min,
        zp_flag,
    )
    RM.item_add(plan)
    print(f"Added recovery + fly2d scan '{label}' to QServer queue.")

