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

### Receiver chain / uncertainty

- Friis cascaded noise-figure calculator
- First-order Gaussian-in-dB uncertainty envelope
- Clear distinction between LNA NF and system NF

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
