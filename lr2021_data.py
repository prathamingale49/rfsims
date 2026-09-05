"""LR2021 LoRa reference data used by the RF simulator.

Values in the sensitivity tables are transcribed from Semtech's
LR2021/LR2022/LR2012 Final Datasheet Rev. 2.1 (2026), Tables 3-17 and 3-18.
The published sensitivity characterization uses LoRa packets at 1% PER and
CR 4/5; the datasheet table is based on a 64-byte payload. Treat these as
preliminary design values until the final radio chain is characterized with
our own packet length, temperature range, and hardware.
"""

from __future__ import annotations

from copy import deepcopy


LR2021_SENSITIVITY_SUBGHZ_DBM = {
    1000.0: {5: -112.5, 6: -115.5, 7: -118.0, 8: -120.0, 9: -123.0, 10: -126.0, 11: -129.0, 12: -131.0},
    800.0: {5: -113.0, 6: -115.5, 7: -118.0, 8: -121.0, 9: -124.0, 10: -126.5, 11: -130.0, 12: -132.0},
    500.0: {5: -116.0, 6: -119.0, 7: -122.0, 8: -124.5, 9: -127.5, 10: -130.0, 11: -133.0, 12: -135.5},
    400.0: {5: -117.0, 6: -120.0, 7: -122.5, 8: -125.5, 9: -128.5, 10: -130.5, 11: -134.0, 12: -136.5},
    250.0: {5: -119.0, 6: -122.0, 7: -124.5, 8: -127.5, 9: -130.0, 10: -133.0, 11: -135.5, 12: -138.5},
    200.0: {5: -119.0, 6: -122.5, 7: -125.0, 8: -128.0, 9: -131.0, 10: -133.5, 11: -136.0, 12: -139.0},
    125.0: {5: -122.0, 6: -125.0, 7: -127.5, 8: -130.5, 9: -133.0, 10: -136.0, 11: -138.5, 12: -141.5},
    62.0: {5: -124.5, 6: -126.5, 7: -129.0, 8: -132.0, 9: -135.0, 10: -137.5, 11: -140.5, 12: -143.0},
    31.0: {5: -128.0, 6: -130.5, 7: -133.0, 8: -135.5, 9: -138.0, 10: -142.0, 11: -144.5, 12: -147.0},
}

LR2021_SENSITIVITY_2G4_DBM = {
    1000.0: {5: -110.0, 6: -113.0, 7: -115.5, 8: -118.5, 9: -121.5, 10: -123.5, 11: -126.5, 12: -129.5},
    800.0: {5: -111.0, 6: -114.0, 7: -116.5, 8: -119.5, 9: -122.5, 10: -125.0, 11: -128.0, 12: -130.5},
    500.0: {5: -112.5, 6: -116.0, 7: -118.5, 8: -121.5, 9: -124.0, 10: -127.0, 11: -129.5, 12: -132.5},
    400.0: {5: -114.0, 6: -117.0, 7: -119.5, 8: -123.0, 9: -125.5, 10: -128.0, 11: -131.0, 12: -134.0},
    200.0: {5: -117.0, 6: -120.0, 7: -123.0, 8: -125.5, 9: -128.5, 10: -131.0, 11: -134.0, 12: -137.0},
}

PROGRAMMABLE_LORA_BW_KHZ = [31.25, 41.67, 62.50, 83.34, 101.5625, 125.0, 203.125, 250.0, 406.25, 500.0, 812.5, 1000.0]

CANDIDATE_PROFILES = [
    {"name": "P0 FAST", "bw_khz": 1000.0, "sf": 5, "cr_den": 5},
    {"name": "P1 FAST-R", "bw_khz": 1000.0, "sf": 7, "cr_den": 5},
    {"name": "P2 MEDIUM", "bw_khz": 500.0, "sf": 7, "cr_den": 5},
    {"name": "P3 ROBUST", "bw_khz": 400.0, "sf": 9, "cr_den": 5},
    {"name": "P4 LONG", "bw_khz": 200.0, "sf": 10, "cr_den": 5},
    {"name": "P5 EXTREME 915", "bw_khz": 125.0, "sf": 12, "cr_den": 5, "subghz_only": True},
]

RECOVERY_PROFILE = {"name": "RECOVERY 915", "bw_khz": 250.0, "sf": 9, "cr_den": 5}


def _nearest_table_bandwidth(table: dict[float, dict[int, float]], bw_khz: float, tol_khz: float = 20.0) -> float | None:
    best = min(table, key=lambda x: abs(float(x) - float(bw_khz)))
    return float(best) if abs(float(best) - float(bw_khz)) <= tol_khz else None


def datasheet_sensitivity_dbm(band_kind: str, bw_khz: float, sf: int) -> float | None:
    table = LR2021_SENSITIVITY_SUBGHZ_DBM if band_kind == "subghz" else LR2021_SENSITIVITY_2G4_DBM
    key = _nearest_table_bandwidth(table, bw_khz)
    if key is None:
        return None
    return table.get(key, {}).get(int(sf))


def default_modes_for_band(band_kind: str) -> list[dict]:
    modes = []
    for profile in CANDIDATE_PROFILES:
        if profile.get("subghz_only") and band_kind != "subghz":
            continue
        p = deepcopy(profile)
        p.update(
            enabled=True,
            sensitivity_dbm=datasheet_sensitivity_dbm(band_kind, p["bw_khz"], p["sf"]),
            preamble_symbols=12 if p["sf"] in (5, 6) else 8,
            explicit_header=True,
            crc_on=True,
            ldro="auto",
        )
        modes.append(p)
    return modes
