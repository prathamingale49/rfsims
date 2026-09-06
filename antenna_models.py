"""Reusable antenna-pattern and rocket/ground geometry models.

Body coordinates use +Z toward the nose. World coordinates use North/East/Down.
The built-in patch patterns are deliberately simple axisymmetric approximations;
measured or datasheet cuts can be supplied as theta_deg/gain_dbic CSV files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rf_models import C


@dataclass(frozen=True)
class AxisymmetricPattern:
    name: str
    frequency_mhz: float
    peak_gain_dbic: float
    hpbw_deg: float
    back_gain_dbic: float
    edge_gain_dbic: float
    theta_deg: np.ndarray | None = None
    measured_gain_dbic: np.ndarray | None = None

    @classmethod
    def from_csv(cls, path: str | Path, *, name: str, frequency_mhz: float) -> "AxisymmetricPattern":
        values = np.genfromtxt(path, delimiter=",", names=True)
        theta = np.asarray(values["theta_deg"], dtype=float)
        gain = np.asarray(values["gain_dbic"], dtype=float)
        order = np.argsort(theta)
        return cls(name, frequency_mhz, float(np.max(gain)), 90.0, float(gain[order][-1]), float(np.min(gain)), theta[order], gain[order])

    def gain_dbic(self, off_boresight_deg) -> np.ndarray:
        theta = np.clip(np.asarray(off_boresight_deg, dtype=float), 0.0, 180.0)
        if self.theta_deg is not None and self.measured_gain_dbic is not None:
            return np.interp(theta, self.theta_deg, self.measured_gain_dbic)

        half_angle = np.radians(max(self.hpbw_deg, 1e-3) / 2.0)
        exponent = np.log(0.5) / np.log(max(np.cos(half_angle), 1e-6))
        front_cos = np.maximum(np.cos(np.radians(theta)), 1e-8)
        front = self.peak_gain_dbic + 10.0 * exponent * np.log10(front_cos)
        front = np.maximum(front, self.edge_gain_dbic)
        # Smooth, low back lobe. This is only a placeholder until chamber data exists.
        back_fraction = np.clip((theta - 90.0) / 90.0, 0.0, 1.0)
        back = self.edge_gain_dbic + (self.back_gain_dbic - self.edge_gain_dbic) * np.sin(0.5 * np.pi * back_fraction) ** 2
        return np.where(theta <= 90.0, front, back)


PATCH_915 = AxisymmetricPattern(
    "Taoglas ISMP.915.35.6.A.02 (parametric)", 915.0, 2.51, 100.0, -12.0, -22.0
)
PATCH_2400 = AxisymmetricPattern(
    "Pulse W3229 (parametric)", 2450.0, 6.5, 85.0, -14.0, -24.0
)
GROUND_YAGI_915 = AxisymmetricPattern(
    "17.5 dBi ground Yagi (parametric)", 915.0, 17.5, 28.0, -8.0, -20.0
)
GROUND_PATCH_2400 = AxisymmetricPattern(
    "2.4 GHz ground antenna placeholder", 2450.0, 12.0, 40.0, -8.0, -20.0
)


def unit(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norm, 1e-12)


def antenna_boresights(cant_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Opposite radial boards, both canted toward body -Z (aft)."""
    cant = np.radians(float(cant_deg))
    return (
        np.array([np.cos(cant), 0.0, -np.sin(cant)]),
        np.array([-np.cos(cant), 0.0, -np.sin(cant)]),
    )


def off_boresight_deg(direction, boresight) -> np.ndarray:
    direction_u = unit(direction)
    boresight_u = unit(boresight)
    return np.degrees(np.arccos(np.clip(np.sum(direction_u * boresight_u, axis=-1), -1.0, 1.0)))


def diversity_gains_dbic(direction_body, cant_deg: float, pattern: AxisymmetricPattern) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bore_a, bore_b = antenna_boresights(cant_deg)
    gain_a = pattern.gain_dbic(off_boresight_deg(direction_body, bore_a))
    gain_b = pattern.gain_dbic(off_boresight_deg(direction_body, bore_b))
    return gain_a, gain_b, np.maximum(gain_a, gain_b)


def attitude_body_to_ned(tilt_deg: float, tilt_azimuth_deg: float, roll_deg: float) -> np.ndarray:
    """Return a body-to-NED DCM for +Z nose, tilt from vertical, and body roll."""
    tilt = np.radians(float(tilt_deg))
    azimuth = np.radians(float(tilt_azimuth_deg))
    z_body = np.array([np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth), -np.cos(tilt)])
    horizontal = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    x_zero = horizontal - np.dot(horizontal, z_body) * z_body
    if np.linalg.norm(x_zero) < 1e-9:
        x_zero = np.array([1.0, 0.0, 0.0])
    x_zero = unit(x_zero)
    y_zero = unit(np.cross(z_body, x_zero))
    roll = np.radians(float(roll_deg))
    x_body = np.cos(roll) * x_zero + np.sin(roll) * y_zero
    y_body = -np.sin(roll) * x_zero + np.cos(roll) * y_zero
    return np.column_stack((x_body, y_body, z_body))


def quaternion_body_to_ned(q, *, order: str = "wxyz", direction: str = "body_to_ned") -> np.ndarray:
    values = np.asarray(q, dtype=float)
    if order == "xyzw":
        x, y, z, w = values
    elif order == "wxyz":
        w, x, y, z = values
    else:
        raise ValueError("order must be wxyz or xyzw")
    w, x, y, z = unit([w, x, y, z])
    matrix = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    if direction == "ned_to_body":
        matrix = matrix.T
    elif direction != "body_to_ned":
        raise ValueError("direction must be body_to_ned or ned_to_body")
    return matrix


def direction_to_ground_body(rocket_ned, ground_ned, body_to_ned: np.ndarray) -> tuple[np.ndarray, float]:
    line_ned = np.asarray(ground_ned, dtype=float) - np.asarray(rocket_ned, dtype=float)
    distance_m = float(np.linalg.norm(line_ned))
    return unit(body_to_ned.T @ line_ned), distance_m


def fspl_db(freq_mhz: float, distance_m) -> np.ndarray:
    distance = np.maximum(np.asarray(distance_m, dtype=float), 1e-6)
    wavelength = C / (float(freq_mhz) * 1e6)
    return 20.0 * np.log10(4.0 * np.pi * distance / wavelength)


def link_margin_db(*, tx_power_dbm: float, flight_gain_dbic, ground_gain_dbic, frequency_mhz: float, distance_m, fixed_losses_db: float, sensitivity_dbm: float) -> np.ndarray:
    received = float(tx_power_dbm) + np.asarray(flight_gain_dbic) + np.asarray(ground_gain_dbic) - fspl_db(frequency_mhz, distance_m) - float(fixed_losses_db)
    return received - float(sensitivity_dbm)
