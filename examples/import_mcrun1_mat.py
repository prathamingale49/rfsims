"""Export the legacy MCrun1 MATLAB result into the rfsims trajectory format.

The raw .mat is intentionally not versioned.  Pass its path explicitly, then commit
the generated compact CSV if the test fixture needs to change.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mat_file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("examples/mcrun1_legacy_trajectory.csv"))
    parser.add_argument("--sample-period-s", type=float, default=0.2)
    args = parser.parse_args()

    sim_data = loadmat(args.mat_file, struct_as_record=False)["simData"][0, 0]
    time_s = np.asarray(sim_data.time, dtype=float).reshape(-1)
    step = max(1, round(args.sample_period_s / float(np.median(np.diff(time_s)))))
    indices = set(range(0, len(time_s), step))

    # MATLAB event indices are one-based. Keep the named events even if the
    # regular decimation would skip their exact sample.
    for field in sim_data.events[0, 0]._fieldnames:
        value = float(np.asarray(getattr(sim_data.events[0, 0], field)).reshape(-1)[0])
        if np.isfinite(value) and value >= 1:
            indices.add(min(len(time_s) - 1, int(value) - 1))
    indices = np.array(sorted(indices), dtype=int)

    pos = np.asarray(sim_data.pos, dtype=float)
    vel = np.asarray(sim_data.vel, dtype=float)
    quat = np.asarray(sim_data.quat, dtype=float)
    omega = np.asarray(sim_data.omega, dtype=float)
    temp_c = np.asarray(sim_data.temp, dtype=float).reshape(-1) - 273.15
    frame = pd.DataFrame({
        "time_s": time_s[indices],
        "north_m": pos[0, indices], "east_m": pos[1, indices], "down_m": pos[2, indices],
        "v_north_mps": vel[0, indices], "v_east_mps": vel[1, indices], "v_down_mps": vel[2, indices],
        "q1": quat[0, indices], "q2": quat[1, indices], "q3": quat[2, indices], "q4": quat[3, indices],
        "p_rad_s": omega[0, indices], "q_rad_s": omega[1, indices], "r_rad_s": omega[2, indices],
        "temp_c": temp_c[indices],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, float_format="%.7g")
    print(f"Wrote {len(frame)} samples to {args.output}")


if __name__ == "__main__":
    main()
