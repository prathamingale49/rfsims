import math

import pandas as pd

from rf_models import doppler_shift_hz, friis_cascade_noise_figure_db, fspl_db, lora_raw_bitrate_bps, lora_time_on_air_s, mismatch_loss_db, ned_geometry


def test_vswr_2_mismatch_loss():
    assert math.isclose(mismatch_loss_db(2.0), 0.5115, abs_tol=0.002)


def test_fspl_915_100km():
    assert math.isclose(fspl_db(915.0, 100.0), 131.68, abs_tol=0.05)


def test_lr2021_fast_raw_rate():
    assert math.isclose(lora_raw_bitrate_bps(5, 1000.0, 5), 125_000.0, abs_tol=1.0)


def test_170b_airtime_fast_profile():
    assert math.isclose(lora_time_on_air_s(170, 5, 1000.0, 5), 0.011976, abs_tol=1e-6)


def test_ned_geometry_vertical_rocket():
    g = ned_geometry([0], [0], [-1000], 0, 0, 0)
    assert math.isclose(float(g["range_m"][0]), 1000.0, abs_tol=1e-9)
    assert math.isclose(float(g["elevation_deg"][0]), 90.0, abs_tol=1e-9)


def test_receding_doppler_is_negative():
    assert doppler_shift_hz(915.0, 100.0) < 0


def test_friis_passive_loss_before_lna_hurts_nf():
    stages = pd.DataFrame([{"gain_db": -5.0, "nf_db": 5.0}, {"gain_db": 22.8, "nf_db": 0.66}])
    nf = friis_cascade_noise_figure_db(stages)
    assert 5.5 < nf < 6.5
