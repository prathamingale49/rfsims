# RF Sims

Interactive RF link-budget, LR2021 LoRa, telemetry-throughput, and flight-trajectory trade-study tool built with Streamlit.

The project is intentionally split into a Streamlit UI (`rf.py`) and pure reusable math modules (`rf_models.py`, `lr2021_data.py`) so the same equations can later be used by post-flight scripts, Monte Carlo analysis, or a non-Streamlit frontend.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run rf.py
```

`python -m streamlit` is preferred on Windows because it avoids PATH issues when Streamlit is installed into a Python user/Conda environment.

## What is modeled now

### Link budget

- Explicit TX reference plane and RX radio-input reference plane
- VSWR → mismatch loss
- EIRP, FSPL, received power
- TX/RX cable/connector losses
- Antenna gain, polarization mismatch, pointing loss, other fixed loss
- Optional generic atmospheric `dB/km` hook (defaults to zero; not yet a physical atmosphere model)
- Optional net active RX-front-end gain

### LR2021 LoRa

- LR2021 Rev. 2.1 datasheet sensitivity tables for characterized CR 4/5 modes at sub-GHz and 2.4 GHz
- BW, SF, CR, LDRO, preamble, explicit/implicit header, and CRC
- Raw LoRa bitrate
- Standard Semtech LoRa packet time-on-air estimate
- Useful payload throughput and maximum packet rate for an arbitrary payload size
- Default application payload of **170 bytes**, based on the historical YJSP compressed telemetry packet
- Discrete candidate PHY profiles instead of arbitrary continuous retuning
- A working 915 MHz recovery-profile concept
- A profile is considered mission-viable only when it meets **both required RF link margin and required telemetry packet rate**

### Distance / dual-band adaptation

- RX power vs distance
- Maximum viable packet rate vs distance
- Maximum viable useful payload throughput vs distance
- Rough 2.4 GHz ↔ 915 MHz crossover based on highest viable packet rate

### Flight trajectory support

The app is ready for the sims team's NED state-vector export. Upload a CSV with any recognizable mapping of:

```text
time
North / East / Down position
North / East / Down velocity
q1 q2 q3 q4 quaternion
p q r angular rates
temperature (optional)
```

The app can derive slant range, ground-station azimuth/elevation, received power, recommended viable PHY profile, radial velocity, Doppler, and direction to the ground station in rocket body axes from quaternion attitude. Quaternion order and rotation direction are explicit settings because the sims team must define those conventions.

A synthetic trajectory generator is included only to exercise the UI before the real sim export exists. It is **not** a flight-dynamics model.

There is also a selectable **Legacy MC run (test model)** trajectory. It is a 5 Hz, event-preserving export of the previous vehicle's Monte Carlo result, with N/E/D position, velocity, quaternion, angular-rate, and temperature fields. The raw MATLAB file is deliberately not in Git. Rebuild the fixture with `examples/import_mcrun1_mat.py` and use the legacy run only to exercise the RF/trajectory pipeline: its lower apogee and quaternion convention are not final-flight assumptions.

### Receiver chain / uncertainty

- Friis cascaded noise-figure calculator
- First-order Gaussian-in-dB uncertainty envelope
- Clear distinction between LNA NF and system NF

## Two-board antenna cant study

The antenna study is separate from the Streamlit app:

```powershell
python antenna_pattern_3d.py --band 915
python antenna_link_geometry.py --band 915
```

`antenna_pattern_3d.py` writes an interactive six-panel 3D view with the rocket body, the two complementary PCB boresights, individual A/B lobes, the switched `max(A, B)` envelope, 0/30/45/60 degree aft-cant comparisons, and the ground-antenna pattern.

`antenna_link_geometry.py` runs the legacy MC trajectory (or another CSV) through the complete rocket/ground geometry. It writes an interactive study, a moving-lobe animation, and a per-timestep results CSV with A/B gain, selected antenna, received power, and link margin for every cant angle. It also compares arbitrary-roll worst cases and the roll/tilt envelope. Run `python antenna_link_geometry.py --help` for trajectory, attitude, RF, tracked/fixed ground pointing, and mounting options.

The built-in patterns are documented parametric stand-ins:

- Taoglas `ISMP.915.35.6.A.02`: 2.51 dBic peak on its 70 x 70 mm reference ground plane.
- Pulse `W3229`: 6.5 dBic peak, approximately 90/80 degree beamwidth, on its 70 x 70 mm reference ground plane.
- 915 MHz ground station: 17.5 dBi tracked-Yagi placeholder carried over from the existing link budget.
- 2.4 GHz ground antenna: configurable placeholder until the actual ground antenna is picked.

The patch rear/edge response and 915 MHz beamwidth are assumptions, not extracted chamber data. Both scripts accept a measured or manufacturer pattern cut with `--pattern-csv`, `--flight-pattern-csv`, or `--ground-pattern-csv`. The CSV format is:

```text
theta_deg,gain_dbic
0,2.51
...
180,-12.0
```

The first implementation treats that cut as axisymmetric. `AxisymmetricPattern` is isolated in `antenna_models.py` so it can later be replaced with a full measured `G(theta, phi)` interpolator without changing the flight/link simulation.

Manufacturer references: [Taoglas ISMP.915 datasheet](https://www.taoglas.com/datasheets/ISMP.915.35.6.A.02.pdf), [Taoglas product page](https://www.taoglas.com/product/915-ism-low-profile-pin-mount-ceramic-patch-antenna/), and [Pulse W3229 datasheet](https://productfinder.pulseelectronics.com/api/open/part-attachments/datasheet/w3229).

## LR2021 data provenance

The sensitivity tables and LoRa configuration notes are transcribed from **Semtech LR2021/LR2022/LR2012 Final Datasheet Rev. 2.1 (April 2026)**. The published sensitivities are characterization values at **1% PER with 64-byte packets** and CR 4/5. Replace them with measured end-to-end values once the LR2021 EVKs and final RF hardware are characterized.

## What still needs real data

1. LR2021 PER vs input power with the actual ~170-byte packet and each candidate profile
2. Actual TX output power vs commanded power, frequency, voltage, and temperature
3. Installed flight-antenna match and eventually directional gain/pattern/polarization
4. Actual 2.4 GHz ground antenna and feed-chain values
5. End-to-end receiver system NF/sensitivity with the final LNA/cable/filter architecture
6. Flight temperature profile and measured RF temperature coefficients
7. Final sims-team trajectory/attitude dispersions
8. Ground reflection/multipath model validated against field measurements
9. Physical atmospheric/terrain model if those terms become significant

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```
