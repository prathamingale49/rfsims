import copy
import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

C = 299_792_458.0
K = 1.380649e-23

DEFAULT_MODES = [
    {"enabled": True, "name": "SF7 125k 4/5", "sf": 7, "bw_khz": 125.0, "cr_den": 5, "snr_req_db": -7.5, "sensitivity_dbm": None},
    {"enabled": True, "name": "SF8 125k 4/5", "sf": 8, "bw_khz": 125.0, "cr_den": 5, "snr_req_db": -10.0, "sensitivity_dbm": None},
    {"enabled": True, "name": "SF9 125k 4/5", "sf": 9, "bw_khz": 125.0, "cr_den": 5, "snr_req_db": -12.5, "sensitivity_dbm": None},
    {"enabled": True, "name": "SF10 125k 4/5", "sf": 10, "bw_khz": 125.0, "cr_den": 5, "snr_req_db": -15.0, "sensitivity_dbm": None},
    {"enabled": True, "name": "SF11 125k 4/5", "sf": 11, "bw_khz": 125.0, "cr_den": 5, "snr_req_db": -17.5, "sensitivity_dbm": None},
    {"enabled": True, "name": "SF12 125k 4/5", "sf": 12, "bw_khz": 125.0, "cr_den": 5, "snr_req_db": -20.0, "sensitivity_dbm": None},
    {"enabled": True, "name": "SF7 250k 4/5", "sf": 7, "bw_khz": 250.0, "cr_den": 5, "snr_req_db": -7.5, "sensitivity_dbm": None},
    {"enabled": True, "name": "SF7 500k 4/5", "sf": 7, "bw_khz": 500.0, "cr_den": 5, "snr_req_db": -7.5, "sensitivity_dbm": None},
]

DEFAULT_BANDS = {
    "915 MHz": dict(freq_mhz=915.0, ptx_dbm=30.5, tx_cable_db=0.2, tx_conn_db=0.145,
                    tx_vswr=2.0, gtx_dbi=2.51, realized_gain=False, grx_dbi=17.5,
                    rx_cable_db=5.0, rx_conn_db=0.145, pol_db=3.0, pointing_db=0.0,
                    other_path_db=0.0, nf_db=2.5, sensitivity_dbm=-114.0,
                    required_margin_db=15.0, temp_k=290.0, modes=copy.deepcopy(DEFAULT_MODES)),
    "2.4 GHz": dict(freq_mhz=2400.0, ptx_dbm=30.0, tx_cable_db=0.3, tx_conn_db=0.145,
                    tx_vswr=2.0, gtx_dbi=2.0, realized_gain=False, grx_dbi=17.5,
                    rx_cable_db=5.0, rx_conn_db=0.145, pol_db=3.0, pointing_db=0.0,
                    other_path_db=0.0, nf_db=2.5, sensitivity_dbm=-114.0,
                    required_margin_db=15.0, temp_k=290.0, modes=copy.deepcopy(DEFAULT_MODES)),
}


def mismatch_loss(vswr):
    if vswr < 1:
        raise ValueError("VSWR must be >= 1")
    gamma = (vswr - 1) / (vswr + 1)
    return -10 * math.log10(max(1 - gamma**2, 1e-15))


def wavelength(freq_mhz):
    return C / (freq_mhz * 1e6)


def fspl(freq_mhz, distance_km):
    d_m = np.maximum(np.asarray(distance_km, dtype=float) * 1000, 1e-9)
    return 20 * np.log10(4 * math.pi * d_m / wavelength(freq_mhz))


def eirp(b):
    lm = 0.0 if b["realized_gain"] else mismatch_loss(b["tx_vswr"])
    return b["ptx_dbm"] - b["tx_cable_db"] - b["tx_conn_db"] - lm + b["gtx_dbi"]


def prx(b, distance_km):
    return (eirp(b) - fspl(b["freq_mhz"], distance_km) - b["other_path_db"] + b["grx_dbi"]
            - b["rx_cable_db"] - b["rx_conn_db"] - b["pol_db"] - b["pointing_db"])


def noise_floor(bw_hz, nf_db, temp_k):
    return 10 * math.log10(K * temp_k * bw_hz * 1000) + nf_db


def lora_bitrate_kbps(sf, bw_khz, cr_den):
    return sf * ((bw_khz * 1e3) / (2**sf)) * (4 / cr_den) / 1e3


def mode_sensitivity(mode, band):
    if mode.get("sensitivity_dbm") is not None and not pd.isna(mode["sensitivity_dbm"]):
        return float(mode["sensitivity_dbm"])
    return noise_floor(mode["bw_khz"] * 1e3, band["nf_db"], band["temp_k"]) + mode["snr_req_db"]


def envelope(band, distances):
    out = []
    for d in distances:
        p = float(prx(band, d))
        viable = []
        for m in band["modes"]:
            if not m.get("enabled", True):
                continue
            sens = mode_sensitivity(m, band)
            margin = p - sens
            rate = lora_bitrate_kbps(int(m["sf"]), float(m["bw_khz"]), int(m["cr_den"]))
            if margin >= band["required_margin_db"]:
                viable.append((rate, m["bw_khz"], m["name"], margin, sens))
        if viable:
            rate, bw, name, margin, sens = max(viable)
        else:
            rate, bw, name, margin, sens = 0.0, 0.0, "No viable mode", np.nan, np.nan
        out.append(dict(distance_km=d, prx_dbm=p, rate_kbps=rate, bw_khz=bw,
                        mode=name, margin_db=margin, sensitivity_dbm=sens))
    return pd.DataFrame(out)


def add_profile_inputs(name, b):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**TX**")
        b["freq_mhz"] = st.number_input("Frequency (MHz)", 0.001, value=float(b["freq_mhz"]), key=f"{name}-f")
        b["ptx_dbm"] = st.number_input("TX power at reference plane (dBm)", value=float(b["ptx_dbm"]), key=f"{name}-ptx")
        b["tx_cable_db"] = st.number_input("TX cable loss (dB)", 0.0, value=float(b["tx_cable_db"]), key=f"{name}-txcbl")
        b["tx_conn_db"] = st.number_input("TX connector loss (dB)", 0.0, value=float(b["tx_conn_db"]), key=f"{name}-txconn")
        b["tx_vswr"] = st.number_input("TX antenna VSWR", 1.0, value=float(b["tx_vswr"]), key=f"{name}-vswr")
        b["gtx_dbi"] = st.number_input("TX antenna gain (dBi)", value=float(b["gtx_dbi"]), key=f"{name}-gtx")
        b["realized_gain"] = st.checkbox("TX gain is realized gain", value=bool(b["realized_gain"]), key=f"{name}-realized")
    with c2:
        st.markdown("**RX**")
        b["grx_dbi"] = st.number_input("RX antenna gain (dBi)", value=float(b["grx_dbi"]), key=f"{name}-grx")
        b["rx_cable_db"] = st.number_input("RX cable loss (dB)", 0.0, value=float(b["rx_cable_db"]), key=f"{name}-rxcbl")
        b["rx_conn_db"] = st.number_input("RX connector loss (dB)", 0.0, value=float(b["rx_conn_db"]), key=f"{name}-rxconn")
        b["nf_db"] = st.number_input("Receiver system NF (dB)", 0.0, value=float(b["nf_db"]), key=f"{name}-nf")
        b["sensitivity_dbm"] = st.number_input("Simple-mode sensitivity (dBm)", value=float(b["sensitivity_dbm"]), key=f"{name}-sens")
        b["temp_k"] = st.number_input("Noise temperature (K)", 1.0, value=float(b["temp_k"]), key=f"{name}-temp")
    with c3:
        st.markdown("**Channel / margin**")
        b["pol_db"] = st.number_input("Polarization loss (dB)", 0.0, value=float(b["pol_db"]), key=f"{name}-pol")
        b["pointing_db"] = st.number_input("Pointing loss (dB)", 0.0, value=float(b["pointing_db"]), key=f"{name}-point")
        b["other_path_db"] = st.number_input("Other path loss (dB)", 0.0, value=float(b["other_path_db"]), key=f"{name}-other")
        b["required_margin_db"] = st.number_input("Required link margin (dB)", 0.0, value=float(b["required_margin_db"]), key=f"{name}-margin")
    return b


def mode_editor(name, b):
    with st.expander("Adaptive LoRa modes"):
        st.caption("Sensitivity is calculated from kTB + NF + required SNR unless you enter a sensitivity override. Replace the placeholder SNR/sensitivity values with LR2021 datasheet or measured values before using the result as a design result.")
        df = pd.DataFrame(b["modes"])
        edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"{name}-modes")
        modes = []
        for _, r in edited.iterrows():
            if pd.isna(r.get("name")):
                continue
            sens = r.get("sensitivity_dbm")
            modes.append(dict(enabled=bool(r.get("enabled", True)), name=str(r["name"]), sf=int(r["sf"]),
                              bw_khz=float(r["bw_khz"]), cr_den=int(r["cr_den"]), snr_req_db=float(r["snr_req_db"]),
                              sensitivity_dbm=None if pd.isna(sens) else float(sens)))
        b["modes"] = modes
        if modes:
            calc = []
            for m in modes:
                calc.append({"Mode": m["name"], "BW (kHz)": m["bw_khz"], "Raw bitrate (kbps)": lora_bitrate_kbps(m["sf"], m["bw_khz"], m["cr_den"]), "Sensitivity (dBm)": mode_sensitivity(m, b)})
            st.dataframe(pd.DataFrame(calc), hide_index=True, use_container_width=True)
    return b


def line_plot(x, series, title, x_title, y_title):
    fig = go.Figure()
    for name, y in series.items():
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, hovermode="x unified")
    return fig


def init_state():
    if "bands" not in st.session_state:
        st.session_state.bands = copy.deepcopy(DEFAULT_BANDS)
    if "selected" not in st.session_state:
        st.session_state.selected = next(iter(st.session_state.bands))


def sidebar():
    with st.sidebar:
        st.header("Saved bands")
        names = list(st.session_state.bands)
        st.session_state.selected = st.selectbox("Edit band", names, index=names.index(st.session_state.selected))
        st.divider()
        new_name = st.text_input("New band name", "New band")
        source = st.selectbox("Duplicate from", names)
        if st.button("Add band", use_container_width=True):
            n = new_name.strip()
            if n and n not in st.session_state.bands:
                st.session_state.bands[n] = copy.deepcopy(st.session_state.bands[source])
                st.session_state.selected = n
                st.rerun()
            st.error("Use a unique, non-empty name.")
        if len(names) > 1 and st.button("Delete selected", use_container_width=True):
            del st.session_state.bands[st.session_state.selected]
            st.session_state.selected = next(iter(st.session_state.bands))
            st.rerun()
        st.divider()
        st.download_button("Export bands JSON", json.dumps(st.session_state.bands, indent=2), "rfsims_bands.json", "application/json", use_container_width=True)
        up = st.file_uploader("Import bands JSON", type="json")
        if up:
            try:
                loaded = json.load(up)
                if not isinstance(loaded, dict) or not loaded:
                    raise ValueError("Expected a non-empty object")
                st.session_state.bands = loaded
                st.session_state.selected = next(iter(loaded))
                st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")


def main():
    st.set_page_config(page_title="RF Sims", layout="wide")
    init_state()
    sidebar()
    st.title("RF Link Budget Simulator")
    st.caption("Real-time FSPL/link-margin trade study with saved band profiles and adaptive LoRa mode comparison.")

    name = st.session_state.selected
    band = add_profile_inputs(name, st.session_state.bands[name])
    band = mode_editor(name, band)
    st.session_state.bands[name] = band

    st.divider()
    d = st.number_input("Current TX-RX distance (km)", 0.001, value=100.0, step=1.0)
    p = float(prx(band, d))
    lm = p - band["sensitivity_dbm"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Wavelength", f"{wavelength(band['freq_mhz']):.3f} m")
    m2.metric("EIRP", f"{eirp(band):.2f} dBm")
    m3.metric("FSPL", f"{float(fspl(band['freq_mhz'], d)):.2f} dB")
    m4.metric("RX input", f"{p:.2f} dBm")
    m5.metric("Simple link margin", f"{lm:.2f} dB", f"{lm-band['required_margin_db']:.2f} dB vs req.")

    st.divider()
    c1, c2, c3 = st.columns(3)
    d0 = c1.number_input("Sweep start (km)", 0.001, value=0.5)
    d1 = c2.number_input("Sweep end (km)", 0.002, value=150.0)
    n = int(c3.number_input("Sweep points", 50, 5000, 500, 50))
    if d1 <= d0:
        st.error("Sweep end must be greater than sweep start.")
        return
    ds = np.linspace(d0, d1, n)

    tabs = st.tabs(["Link margin", "RX power", "Adaptive LoRa", "Frequency sweep"])
    with tabs[0]:
        series = {n: prx(b, ds) - b["sensitivity_dbm"] for n, b in st.session_state.bands.items()}
        st.plotly_chart(line_plot(ds, series, "Link margin vs distance", "Distance (km)", "Margin (dB)"), use_container_width=True)
    with tabs[1]:
        series = {n: prx(b, ds) for n, b in st.session_state.bands.items()}
        st.plotly_chart(line_plot(ds, series, "Receiver input power vs distance", "Distance (km)", "Power (dBm)"), use_container_width=True)
    with tabs[2]:
        st.caption("Bandwidth does not continuously shrink with distance. The tool selects the highest-throughput configured mode that still meets the required link margin.")
        envs = {n: envelope(b, ds) for n, b in st.session_state.bands.items()}
        st.plotly_chart(line_plot(ds, {n: e["rate_kbps"] for n, e in envs.items()}, "Maximum viable raw LoRa bitrate", "Distance (km)", "Raw bitrate (kbps)"), use_container_width=True)
        st.plotly_chart(line_plot(ds, {n: e["bw_khz"] for n, e in envs.items()}, "Bandwidth of highest-throughput viable mode", "Distance (km)", "Mode BW (kHz)"), use_container_width=True)
        if len(envs) >= 2:
            names = list(envs)
            a = st.selectbox("Preferred near-range band", names, index=min(1, len(names)-1))
            b_choices = [x for x in names if x != a]
            bname = st.selectbox("Fallback / long-range band", b_choices)
            mask = envs[a]["rate_kbps"].to_numpy() < envs[bname]["rate_kbps"].to_numpy()
            idx = np.flatnonzero(mask)
            if len(idx):
                i = int(idx[0])
                st.success(f"Rough swap point: {ds[i]:.1f} km. {a}: {envs[a].iloc[i]['rate_kbps']:.1f} kbps / {envs[a].iloc[i]['bw_khz']:.0f} kHz; {bname}: {envs[bname].iloc[i]['rate_kbps']:.1f} kbps / {envs[bname].iloc[i]['bw_khz']:.0f} kHz.")
            else:
                st.info("No throughput crossover in this distance sweep.")
    with tabs[3]:
        freqs = np.logspace(math.log10(100), math.log10(6000), 500)
        st.warning("This frequency sweep varies FSPL only; antenna gain, matching, filters, NF, and atmospheric loss can also vary with frequency.")
        st.plotly_chart(line_plot(freqs, {"FSPL": fspl(freqs, d)}, f"FSPL vs frequency at {d:g} km", "Frequency (MHz)", "FSPL (dB)"), use_container_width=True)

    with st.expander("Model notes"):
        st.markdown("""
- TX power is referenced to whatever plane you define; do not subtract losses already included in that measurement.
- If antenna gain is realized gain, enable the checkbox so mismatch is not counted twice.
- Receiver sensitivity is mode-dependent. The simple sensitivity box is for a single operating point; the adaptive table is for mode trades.
- Receiver NF should be a cascaded/system NF. Passive loss before the first LNA can dominate it.
- Default LoRa SNR thresholds are placeholders. Replace them with LR2021 datasheet values or measured sensitivities.
- Propagation is FSPL plus explicit losses. Add terrain, ground reflection, airframe shadowing, fading, atmosphere, and measured antenna patterns as separate models as the tool matures.
        """)


if __name__ == "__main__":
    main()
