"""Garment color fidelity: compare the dominant garment color per frame to the listing color."""
from __future__ import annotations

import numpy as np

from .. import config
from ..imaging import Frame, bgr_to_lab, delta_e, lab_to_hex, resize_max
from .report import Check

WINDOW_DE = 22.0  # pixels within this distance of the target count as garment
MIN_COVERAGE = 0.02


def _frame_color(img: np.ndarray, target: np.ndarray):
    small = resize_max(img, 360)
    lab = bgr_to_lab(small)
    lab = lab[int(lab.shape[0] * 0.25) :]  # skip the top quarter (hair, background)
    de = delta_e(lab, target)
    mask = de < WINDOW_DE
    if mask.mean() < MIN_COVERAGE:
        return None
    return np.median(lab[mask], axis=0)


def _level(value: float, warn: float, fail: float) -> str:
    return "fail" if value > fail else "warn" if value > warn else "pass"


def check_color(frames: list[Frame], target_lab: np.ndarray, cuts: list[float] | None = None,
                target_source: str = "") -> list[Check]:
    meds = [(t, _frame_color(f, target_lab)) for t, f in frames]
    found = [(t, m) for t, m in meds if m is not None]
    not_found = 1 - len(found) / len(meds)
    target_hex = lab_to_hex(target_lab)
    out: list[Check] = []

    nf_status = _level(not_found, config.NOT_FOUND_WARN, config.NOT_FOUND_FAIL)
    out.append(Check("garment_color_found", nf_status,
                     f"listing color {target_hex} {target_source}found in {len(found)}/{len(meds)} frames"
                     + ("" if nf_status == "pass" else "; possible color drift, a wrong target color, or a close-up"),
                     {"target_hex": target_hex, "not_found_fraction": round(not_found, 3)}))
    if not found or (len(found) < 3 and len(meds) >= 3):
        out.append(Check("garment_color", "skip", "too few frames with the garment color to measure"))
        return out

    arr = np.array([m for _, m in found])
    per_frame = delta_e(arr, target_lab)
    median_de = float(np.median(per_frame))
    out.append(Check("garment_color", _level(median_de, config.COLOR_WARN_DE, config.COLOR_FAIL_DE),
                     f"median distance to listing color dE={median_de:.1f} (video median {lab_to_hex(np.median(arr, axis=0))})",
                     {"median_de": round(median_de, 2), "per_frame_de": [round(float(x), 1) for x in per_frame]}))
    if len(meds) < 3:  # single image: no stability or join checks
        return out

    spread = float(delta_e(arr[:, None, :], arr[None, :, :]).max())
    out.append(Check("color_stability", _level(spread, config.DRIFT_WARN_DE, config.DRIFT_FAIL_DE),
                     f"garment color varies by dE={spread:.1f} across frames", {"spread_de": round(spread, 2)}))

    worst_jump, worst_t = 0.0, None
    for c in cuts or []:
        before = [m for t, m in found if t < c]
        after = [m for t, m in found if t >= c]
        if before and after:
            j = float(delta_e(before[-1], after[0]))
            if j > worst_jump:
                worst_jump, worst_t = j, c
    if cuts:
        out.append(Check("join_color_jump", _level(worst_jump, config.JUMP_WARN_DE, config.JUMP_FAIL_DE),
                         f"largest garment color change across a cut: dE={worst_jump:.1f}"
                         + (f" at {worst_t:.1f}s" if worst_t is not None else ""),
                         {"max_jump_de": round(worst_jump, 2)}))
    return out
