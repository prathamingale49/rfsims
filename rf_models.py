"""Pure RF/LoRa/trajectory math for the Streamlit application."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

C = 299_792_458.0
K = 1.380649e-23


def mismatch_loss_db(vswr: float) -> float:
    if vswr < 1.0:
        raise ValueError("VSWR must be >= 1")
    gamma = (vswr - 1.0) / (vswr + 1.0)
    return -10.0 * math.log10(max(1.0 - gamma**2, 1e-15))


def wavelength_m(freq_mhz: float) -> float:
    if freq_mhz <= 0:
        raise ValueError("Frequency must be > 0")
    return C / (freq_mhz * 1e6)


def fspl_db(freq_mhz: float, distance_km):
    if freq_mhz <= 0:
        raise ValueError("Frequency must be > 0")
    d = np.asarray(distance_km, dtype=float)
    d_m = np.maximum(d * 1000.0, 1e-9)
    out = 20.0 * np.log10(4.0 * math.pi * d_m / wavelength_m(freq_mhz))
    return float(out) if np.ndim(out) == 0 else out


def thermal_noise_dbm(bw_hz: float, temp_k: float = 290.0) -> float:
    if bw_hz <= 0 or temp_k <= 0:
        raise ValueError("Bandwidth and temperature must be > 0")
    return 10.0 * math.log10(K * temp_k * bw_hz * 1000.0)


def receiver_noise_floor_dbm(bw_hz: float, nf_db: float, temp_k: float = 290.0) -> float:
    return thermal_noise_dbm(bw_hz, temp_k) + nf_db


def tx_eirp_dbm(band: dict, ptx_dbm_override: float | None = None, gtx_dbi_override: float | None = None) -> float:
    ptx = float(band["ptx_dbm"] if ptx_dbm_override is None else ptx_dbm_override)
    gtx = float(band["gtx_dbi"] if gtx_dbi_override is None else gtx_dbi_override)
    mismatch = 0.0 if bool(band.get("realized_gain", False)) else mismatch_loss_db(float(band.get("tx_vswr", 1.0)))
    return ptx - float(band.get("tx_cable_db", 0.0)) - float(band.get("tx_conn_db", 0.0)) - mismatch + gtx


def received_power_dbm(band: dict, distance_km, *, ptx_dbm_override: float | None = None, gtx_dbi_override: float | None = None, extra_loss_db=0.0):
    distance = np.asarray(distance_km, dtype=float)
    atmos = float(band.get("atmos_db_per_km", 0.0)) * distance
    p = (
        tx_eirp_dbm(band, ptx_dbm_override, gtx_dbi_override)
        - fspl_db(float(band["freq_mhz"]), distance)
        - float(band.get("other_path_db", 0.0))
        - atmos
        - np.asarray(extra_loss_db, dtype=float)
        + float(band.get("grx_dbi", 0.0))
        + float(band.get("rx_frontend_gain_db", 0.0))
        - float(band.get("rx_cable_db", 0.0))
        - float(band.get("rx_conn_db", 0.0))
        - float(band.get("pol_db", 0.0))
        - float(band.get("pointing_db", 0.0))
    )
    return float(p) if np.ndim(p) == 0 else p


def lora_symbol_rate_hz(sf: int, bw_khz: float) -> float:
    return bw_khz * 1000.0 / (2**int(sf))


def lora_raw_bitrate_bps(sf: int, bw_khz: float, cr_den: int) -> float:
    if cr_den not in (5, 6, 7, 8):
        raise ValueError("Normal LoRa coding-rate denominator must be 5, 6, 7, or 8")
    return int(sf) * lora_symbol_rate_hz(sf, bw_khz) * (4.0 / int(cr_den))


def recommended_preamble_symbols(sf: int) -> int:
    return 12 if int(sf) in (5, 6) else 8


def recommended_ldro(sf: int, bw_khz: float) -> bool:
    sf = int(sf)
    bw = float(bw_khz)
    if sf <= 10 or bw >= 500.0:
        return False
    if sf == 11 and abs(bw - 250.0) <= 1.0:
        return False
    return True


def lora_time_on_air_s(payload_bytes: int, sf: int, bw_khz: float, cr_den: int, *, preamble_symbols: int | None = None, explicit_header: bool = True, crc_on: bool = True, ldro: bool | None = None) -> float:
    if payload_bytes < 0:
        raise ValueError("payload_bytes must be >= 0")
    sf = int(sf)
    cr_den = int(cr_den)
    if not 5 <= sf <= 12:
        raise ValueError("SF must be in [5, 12]")
    if cr_den not in (5, 6, 7, 8):
        raise ValueError("CR denominator must be 5, 6, 7, or 8")
    bw_hz = float(bw_khz) * 1000.0
    if bw_hz <= 0:
        raise ValueError("Bandwidth must be > 0")
    preamble = recommended_preamble_symbols(sf) if preamble_symbols is None else int(preamble_symbols)
    de = int(recommended_ldro(sf, bw_khz) if ldro is None else bool(ldro))
    ih = 0 if explicit_header else 1
    crc = 1 if crc_on else 0
    tsym = (2**sf) / bw_hz
    numerator = 8 * int(payload_bytes) - 4 * sf + 28 + 16 * crc - 20 * ih
    denominator = 4 * (sf - 2 * de)
    payload_symbols = 8 + max(math.ceil(numerator / denominator) * cr_den, 0)
    return (preamble + 4.25 + payload_symbols) * tsym


def useful_payload_rate_bps(payload_bytes: int, toa_s: float) -> float:
    return 0.0 if toa_s <= 0 else payload_bytes * 8.0 / toa_s


def max_packet_rate_hz(toa_s: float, duty_cycle_fraction: float = 1.0) -> float:
    if toa_s <= 0:
        return 0.0
    return min(max(float(duty_cycle_fraction), 0.0), 1.0) / toa_s


def mode_metrics(mode: dict, payload_bytes: int) -> dict:
    ldro_setting = mode.get("ldro", "auto")
    ldro = None if ldro_setting in (None, "auto", "Auto") else bool(ldro_setting)
    toa = lora_time_on_air_s(payload_bytes, int(mode["sf"]), float(mode["bw_khz"]), int(mode["cr_den"]), preamble_symbols=int(mode.get("preamble_symbols") or recommended_preamble_symbols(int(mode["sf"]))), explicit_header=bool(mode.get("explicit_header", True)), crc_on=bool(mode.get("crc_on", True)), ldro=ldro)
    return {
        "raw_bitrate_bps": lora_raw_bitrate_bps(int(mode["sf"]), float(mode["bw_khz"]), int(mode["cr_den"])),
        "toa_s": toa,
        "max_packet_rate_hz": max_packet_rate_hz(toa),
        "useful_rate_bps": useful_payload_rate_bps(payload_bytes, toa),
        "ldro": recommended_ldro(int(mode["sf"]), float(mode["bw_khz"])) if ldro is None else ldro,
    }


def select_best_mode(band: dict, prx_dbm: float, payload_bytes: int, required_packet_rate_hz: float, sensitivity_shift_db: float = 0.0) -> dict:
    candidates = []
    for mode in band.get("modes", []):
        if not mode.get("enabled", True):
            continue
        sens = mode.get("sensitivity_dbm")
        if sens is None or (isinstance(sens, float) and math.isnan(sens)):
            continue
        mm = mode_metrics(mode, payload_bytes)
        effective_sens = float(sens) + float(sensitivity_shift_db)
        margin = float(prx_dbm) - effective_sens
        rf_ok = margin >= float(band.get("required_margin_db", 0.0))
        rate_ok = mm["max_packet_rate_hz"] >= float(required_packet_rate_hz)
        row = {**mode, **mm, "margin_db": margin, "effective_sensitivity_dbm": effective_sens, "rf_ok": rf_ok, "rate_ok": rate_ok, "mission_ok": rf_ok and rate_ok}
        if row["mission_ok"]:
            candidates.append(row)
    if not candidates:
        return {"name": "No viable mode", "mission_ok": False, "margin_db": math.nan, "max_packet_rate_hz": 0.0, "useful_rate_bps": 0.0, "sensitivity_dbm": math.nan, "sf": math.nan, "bw_khz": 0.0, "cr_den": math.nan}
    return max(candidates, key=lambda r: (r["useful_rate_bps"], r["max_packet_rate_hz"], r["bw_khz"]))


def mode_envelope(band: dict, distances_km: Iterable[float], payload_bytes: int, required_packet_rate_hz: float) -> pd.DataFrame:
    rows = []
    for d in distances_km:
        p = float(received_power_dbm(band, float(d)))
        best = select_best_mode(band, p, payload_bytes, required_packet_rate_hz)
        rows.append({"distance_km": float(d), "prx_dbm": p, "mode": best["name"], "sf": best.get("sf", math.nan), "bw_khz": best.get("bw_khz", 0.0), "cr_den": best.get("cr_den", math.nan), "margin_db": best.get("margin_db", math.nan), "sensitivity_dbm": best.get("sensitivity_dbm", math.nan), "max_packet_rate_hz": best.get("max_packet_rate_hz", 0.0), "useful_rate_kbps": best.get("useful_rate_bps", 0.0) / 1000.0, "mission_ok": bool(best.get("mission_ok", False))})
    return pd.DataFrame(rows)


def friis_cascade_noise_figure_db(stages: pd.DataFrame) -> float:
    if stages.empty:
        return 0.0
    total_factor = None
    prior_gain = 1.0
    for _, stage in stages.iterrows():
        f = 10.0 ** (float(stage["nf_db"]) / 10.0)
        g = 10.0 ** (float(stage["gain_db"]) / 10.0)
        if total_factor is None:
            total_factor = f
        else:
            total_factor += (f - 1.0) / prior_gain
        prior_gain *= g
    return 10.0 * math.log10(max(total_factor or 1.0, 1e-15))


def ned_geometry(rocket_n_m, rocket_e_m, rocket_d_m, ground_n_m: float, ground_e_m: float, ground_d_m: float) -> dict[str, np.ndarray]:
    dn = np.asarray(rocket_n_m, dtype=float) - float(ground_n_m)
    de = np.asarray(rocket_e_m, dtype=float) - float(ground_e_m)
    dd = np.asarray(rocket_d_m, dtype=float) - float(ground_d_m)
    r = np.sqrt(dn**2 + de**2 + dd**2)
    r_safe = np.maximum(r, 1e-9)
    az = (np.degrees(np.arctan2(de, dn)) + 360.0) % 360.0
    el = np.degrees(np.arcsin(np.clip(-dd / r_safe, -1.0, 1.0)))
    return {"range_m": r, "azimuth_deg": az, "elevation_deg": el, "dn_m": dn, "de_m": de, "dd_m": dd}


def radial_velocity_mps(vn_mps, ve_mps, vd_mps, geometry: dict[str, np.ndarray]) -> np.ndarray:
    r = np.maximum(np.asarray(geometry["range_m"], dtype=float), 1e-9)
    return (np.asarray(vn_mps, dtype=float) * geometry["dn_m"] + np.asarray(ve_mps, dtype=float) * geometry["de_m"] + np.asarray(vd_mps, dtype=float) * geometry["dd_m"]) / r


def doppler_shift_hz(freq_mhz: float, radial_velocity_mps_value) -> np.ndarray:
    vr = np.asarray(radial_velocity_mps_value, dtype=float)
    return -(vr / C) * float(freq_mhz) * 1e6


def quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.asarray([qw, qx, qy, qz], dtype=float)
    n = np.linalg.norm(q)
    if n <= 0:
        raise ValueError("Quaternion norm must be nonzero")
    w, x, y, z = q / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def body_los_angles_from_quaternions(rocket_n_m, rocket_e_m, rocket_d_m, ground_n_m: float, ground_e_m: float, ground_d_m: float, q1, q2, q3, q4, *, order: str = "wxyz", direction: str = "body_to_ned") -> dict[str, np.ndarray]:
    rn = np.asarray(rocket_n_m, dtype=float)
    re = np.asarray(rocket_e_m, dtype=float)
    rd = np.asarray(rocket_d_m, dtype=float)
    qcols = [np.asarray(x, dtype=float) for x in (q1, q2, q3, q4)]
    bx = np.empty(len(rn)); by = np.empty(len(rn)); bz = np.empty(len(rn))
    for i in range(len(rn)):
        values = [col[i] for col in qcols]
        if order == "wxyz":
            qw, qx, qy, qz = values
        elif order == "xyzw":
            qx, qy, qz, qw = values
        else:
            raise ValueError("order must be 'wxyz' or 'xyzw'")
        rmat = quaternion_to_matrix(qw, qx, qy, qz)
        los_ned = np.array([float(ground_n_m) - rn[i], float(ground_e_m) - re[i], float(ground_d_m) - rd[i]])
        if direction == "body_to_ned":
            los_body = rmat.T @ los_ned
        elif direction == "ned_to_body":
            los_body = rmat @ los_ned
        else:
            raise ValueError("direction must be 'body_to_ned' or 'ned_to_body'")
        norm = max(float(np.linalg.norm(los_body)), 1e-9)
        bx[i], by[i], bz[i] = los_body / norm
    body_az = (np.degrees(np.arctan2(by, bx)) + 360.0) % 360.0
    body_el = np.degrees(np.arctan2(-bz, np.sqrt(bx**2 + by**2)))
    off_axis = np.degrees(np.arccos(np.clip(bx, -1.0, 1.0)))
    return {"body_azimuth_deg": body_az, "body_elevation_deg": body_el, "body_off_axis_deg": off_axis}


def normal_margin_percentiles(mean_margin_db, sigma_db: float, percentiles=(5, 50, 95)) -> dict[int, np.ndarray]:
    z_lookup = {1: -2.326347874, 5: -1.644853627, 10: -1.281551566, 50: 0.0, 90: 1.281551566, 95: 1.644853627, 99: 2.326347874}
    mean = np.asarray(mean_margin_db, dtype=float)
    return {p: mean + z_lookup[int(p)] * float(sigma_db) for p in percentiles}
