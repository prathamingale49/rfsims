from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lr2021_data import RECOVERY_PROFILE, datasheet_sensitivity_dbm, default_modes_for_band
from rf_models import body_los_angles_from_quaternions, doppler_shift_hz, friis_cascade_noise_figure_db, fspl_db, mode_envelope, mode_metrics, ned_geometry, normal_margin_percentiles, radial_velocity_mps, received_power_dbm, recommended_preamble_symbols, receiver_noise_floor_dbm, select_best_mode, tx_eirp_dbm, wavelength_m

APP_VERSION = "0.3.0"


def make_default_band(kind: str) -> dict:
    if kind == "subghz":
        return dict(band_kind="subghz", freq_mhz=915.0, ptx_dbm=30.5, tx_cable_db=0.2, tx_conn_db=0.145, tx_vswr=2.0, gtx_dbi=2.51, realized_gain=False, grx_dbi=17.5, rx_frontend_gain_db=0.0, rx_cable_db=5.0, rx_conn_db=0.145, pol_db=3.0, pointing_db=0.0, other_path_db=0.0, atmos_db_per_km=0.0, nf_db=2.5, required_margin_db=15.0, temp_k=290.0, reference_temp_c=25.0, tx_tempco_db_per_c=0.0, sensitivity_tempco_db_per_c=0.0, modes=default_modes_for_band("subghz"))
    return dict(band_kind="2g4", freq_mhz=2400.0, ptx_dbm=30.0, tx_cable_db=0.3, tx_conn_db=0.145, tx_vswr=2.0, gtx_dbi=2.0, realized_gain=False, grx_dbi=0.0, rx_frontend_gain_db=0.0, rx_cable_db=0.5, rx_conn_db=0.145, pol_db=3.0, pointing_db=0.0, other_path_db=0.0, atmos_db_per_km=0.0, nf_db=2.5, required_margin_db=15.0, temp_k=290.0, reference_temp_c=25.0, tx_tempco_db_per_c=0.0, sensitivity_tempco_db_per_c=0.0, modes=default_modes_for_band("2g4"))


DEFAULT_BANDS = {"915 MHz": make_default_band("subghz"), "2.4 GHz": make_default_band("2g4")}


def init_state() -> None:
    if "bands" not in st.session_state:
        st.session_state.bands = copy.deepcopy(DEFAULT_BANDS)
    if "selected_band" not in st.session_state:
        st.session_state.selected_band = "915 MHz"
    if "payload_bytes" not in st.session_state:
        st.session_state.payload_bytes = 170
    if "required_packet_rate_hz" not in st.session_state:
        st.session_state.required_packet_rate_hz = 10.0


def band_sidebar() -> None:
    with st.sidebar:
        st.header("RF Sims")
        st.caption(f"v{APP_VERSION} • LR2021-focused")
        names = list(st.session_state.bands)
        current = st.session_state.selected_band if st.session_state.selected_band in names else names[0]
        st.session_state.selected_band = st.selectbox("Band profile", names, index=names.index(current))
        st.subheader("Telemetry payload")
        st.session_state.payload_bytes = int(st.number_input("Application payload (bytes)", min_value=1, max_value=255, value=int(st.session_state.payload_bytes), step=1))
        st.session_state.required_packet_rate_hz = float(st.number_input("Required update rate (Hz)", min_value=0.01, value=float(st.session_state.required_packet_rate_hz), step=0.5))
        useful = st.session_state.payload_bytes * 8 * st.session_state.required_packet_rate_hz / 1000.0
        st.caption(f"Required useful payload throughput: **{useful:.2f} kbps**")
        st.divider()
        st.subheader("Saved profiles")
        new_name = st.text_input("New band name", "New band")
        source = st.selectbox("Duplicate from", names)
        if st.button("Add band", use_container_width=True):
            candidate = new_name.strip()
            if candidate and candidate not in st.session_state.bands:
                st.session_state.bands[candidate] = copy.deepcopy(st.session_state.bands[source]); st.session_state.selected_band = candidate; st.rerun()
            else:
                st.error("Use a unique non-empty name.")
        if len(names) > 1 and st.button("Delete selected band", use_container_width=True):
            del st.session_state.bands[st.session_state.selected_band]; st.session_state.selected_band = next(iter(st.session_state.bands)); st.rerun()
        st.download_button("Export band profiles JSON", data=json.dumps(st.session_state.bands, indent=2), file_name="rfsims_bands.json", mime="application/json", use_container_width=True)
        uploaded = st.file_uploader("Import band profiles JSON", type="json", key="band-json")
        if uploaded is not None:
            try:
                loaded = json.load(uploaded)
                if not isinstance(loaded, dict) or not loaded: raise ValueError("Expected a non-empty JSON object")
                st.session_state.bands = loaded; st.session_state.selected_band = next(iter(loaded)); st.rerun()
            except Exception as exc:
                st.error(f"Import failed: {exc}")


def edit_band_inputs(name: str, band: dict) -> dict:
    st.subheader(f"{name} hardware / channel")
    if band.get("band_kind") == "2g4": st.warning("The 2.4 GHz RF hardware/antenna values are still placeholders. Do not reuse the 17.5 dBi 915 MHz Yagi gain at 2.4 GHz.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**TX reference plane**")
        band["freq_mhz"] = st.number_input("Frequency (MHz)", min_value=1.0, value=float(band["freq_mhz"]), key=f"{name}-freq")
        band["ptx_dbm"] = st.number_input("TX power (dBm)", value=float(band["ptx_dbm"]), key=f"{name}-ptx")
        band["tx_cable_db"] = st.number_input("TX cable loss (dB)", min_value=0.0, value=float(band.get("tx_cable_db", 0.0)), key=f"{name}-txcbl")
        band["tx_conn_db"] = st.number_input("TX connector loss (dB)", min_value=0.0, value=float(band.get("tx_conn_db", 0.0)), key=f"{name}-txconn")
        band["tx_vswr"] = st.number_input("TX antenna VSWR", min_value=1.0, value=float(band.get("tx_vswr", 1.0)), key=f"{name}-vswr")
        band["gtx_dbi"] = st.number_input("TX antenna gain (dBi)", value=float(band.get("gtx_dbi", 0.0)), key=f"{name}-gtx")
        band["realized_gain"] = st.checkbox("Gain is realized gain", value=bool(band.get("realized_gain", False)), key=f"{name}-realized")
    with c2:
        st.markdown("**RX path to radio input**")
        band["grx_dbi"] = st.number_input("RX antenna gain (dBi)", value=float(band.get("grx_dbi", 0.0)), key=f"{name}-grx")
        band["rx_frontend_gain_db"] = st.number_input("Net active RX frontend gain (dB)", value=float(band.get("rx_frontend_gain_db", 0.0)), key=f"{name}-rxgain")
        band["rx_cable_db"] = st.number_input("RX passive/cable loss (dB)", min_value=0.0, value=float(band.get("rx_cable_db", 0.0)), key=f"{name}-rxcbl")
        band["rx_conn_db"] = st.number_input("RX connector loss (dB)", min_value=0.0, value=float(band.get("rx_conn_db", 0.0)), key=f"{name}-rxconn")
        band["nf_db"] = st.number_input("Receiver system NF (dB)", min_value=0.0, value=float(band.get("nf_db", 0.0)), key=f"{name}-nf")
        band["temp_k"] = st.number_input("Noise temperature (K)", min_value=1.0, value=float(band.get("temp_k", 290.0)), key=f"{name}-tempk")
    with c3:
        st.markdown("**Channel / geometry losses**")
        band["pol_db"] = st.number_input("Polarization mismatch (dB)", min_value=0.0, value=float(band.get("pol_db", 0.0)), key=f"{name}-pol")
        band["pointing_db"] = st.number_input("Pointing loss (dB)", min_value=0.0, value=float(band.get("pointing_db", 0.0)), key=f"{name}-point")
        band["other_path_db"] = st.number_input("Other fixed path loss (dB)", min_value=0.0, value=float(band.get("other_path_db", 0.0)), key=f"{name}-other")
        band["atmos_db_per_km"] = st.number_input("User atmospheric attenuation (dB/km)", min_value=0.0, value=float(band.get("atmos_db_per_km", 0.0)), format="%.6f", key=f"{name}-atmos")
        band["required_margin_db"] = st.number_input("Required link margin (dB)", min_value=0.0, value=float(band.get("required_margin_db", 15.0)), key=f"{name}-margin")
    with c4:
        st.markdown("**Temperature sensitivity hooks**")
        band["reference_temp_c"] = st.number_input("Reference temperature (°C)", value=float(band.get("reference_temp_c", 25.0)), key=f"{name}-tref")
        band["tx_tempco_db_per_c"] = st.number_input("TX power drift (dB/°C)", value=float(band.get("tx_tempco_db_per_c", 0.0)), format="%.4f", key=f"{name}-tx-tempco")
        band["sensitivity_tempco_db_per_c"] = st.number_input("Sensitivity drift (dB/°C)", value=float(band.get("sensitivity_tempco_db_per_c", 0.0)), format="%.4f", key=f"{name}-sens-tempco")
        st.caption("Temperature coefficients default to zero so the simulator never invents drift.")
    return band


def edit_modes(name: str, band: dict, payload_bytes: int) -> dict:
    st.subheader("LR2021 LoRa profiles")
    source = pd.DataFrame(band.get("modes", []))
    if source.empty: source = pd.DataFrame(default_modes_for_band(band.get("band_kind", "subghz")))
    cols = ["enabled", "name", "bw_khz", "sf", "cr_den", "sensitivity_dbm", "preamble_symbols", "explicit_header", "crc_on", "ldro"]
    for col in cols:
        if col not in source: source[col] = None
    edited = st.data_editor(source[cols], num_rows="dynamic", use_container_width=True, key=f"{name}-mode-editor")
    modes = []
    for _, row in edited.iterrows():
        if pd.isna(row.get("name")): continue
        sens = row.get("sensitivity_dbm")
        modes.append({"enabled": bool(row.get("enabled", True)), "name": str(row["name"]), "bw_khz": float(row["bw_khz"]), "sf": int(row["sf"]), "cr_den": int(row["cr_den"]), "sensitivity_dbm": None if pd.isna(sens) else float(sens), "preamble_symbols": int(row.get("preamble_symbols") or recommended_preamble_symbols(int(row["sf"]))), "explicit_header": bool(row.get("explicit_header", True)), "crc_on": bool(row.get("crc_on", True)), "ldro": row.get("ldro", "auto") if not pd.isna(row.get("ldro", "auto")) else "auto"})
    band["modes"] = modes
    rows = []
    for mode in modes:
        mm = mode_metrics(mode, payload_bytes)
        rows.append({"Profile": mode["name"], "BW (kHz)": mode["bw_khz"], "SF": mode["sf"], "CR": f"4/{mode['cr_den']}", "Sensitivity (dBm)": mode.get("sensitivity_dbm"), "Raw bitrate (kbps)": mm["raw_bitrate_bps"] / 1000.0, f"{payload_bytes} B airtime (ms)": mm["toa_s"] * 1000.0, "Max packet rate (Hz)": mm["max_packet_rate_hz"], "Useful payload rate (kbps)": mm["useful_rate_bps"] / 1000.0, "LDRO": mm["ldro"]})
    if rows: st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    return band


def line_fig(x, ys, title, x_title, y_title):
    fig = go.Figure()
    for label, values in ys.items(): fig.add_trace(go.Scatter(x=x, y=values, mode="lines", name=label))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, hovermode="x unified")
    return fig


def link_operating_point(band, payload_bytes, required_packet_rate_hz):
    st.subheader("Current operating point")
    d = st.number_input("TX ↔ RX slant distance (km)", min_value=0.001, value=100.0, step=1.0, key="op-distance")
    p = float(received_power_dbm(band, d)); best = select_best_mode(band, p, payload_bytes, required_packet_rate_hz)
    cols = st.columns(6)
    cols[0].metric("Wavelength", f"{wavelength_m(band['freq_mhz']):.3f} m"); cols[1].metric("EIRP", f"{tx_eirp_dbm(band):.2f} dBm"); cols[2].metric("FSPL", f"{fspl_db(band['freq_mhz'], d):.2f} dB"); cols[3].metric("RX at radio", f"{p:.2f} dBm"); cols[4].metric("Best profile", best["name"]); cols[5].metric("Best margin" if best.get("mission_ok") else "Mission viable", f"{best['margin_db']:.1f} dB" if best.get("mission_ok") else "NO")
    table = []
    for mode in band.get("modes", []):
        if not mode.get("enabled", True): continue
        sens = mode.get("sensitivity_dbm"); mm = mode_metrics(mode, payload_bytes); margin = np.nan if sens is None else p - float(sens)
        table.append({"Profile": mode["name"], "Sensitivity (dBm)": sens, "Margin (dB)": margin, "RF margin OK": False if sens is None else margin >= band["required_margin_db"], "Max packet Hz": mm["max_packet_rate_hz"], "Rate OK": mm["max_packet_rate_hz"] >= required_packet_rate_hz, "Mission viable": False if sens is None else margin >= band["required_margin_db"] and mm["max_packet_rate_hz"] >= required_packet_rate_hz})
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)


def distance_analysis(payload_bytes, required_packet_rate_hz):
    st.subheader("Distance sweep & band crossover")
    c1, c2, c3 = st.columns(3); d0 = c1.number_input("Start distance (km)", min_value=0.001, value=0.5, key="d0"); d1 = c2.number_input("End distance (km)", min_value=0.002, value=150.0, key="d1"); n = int(c3.number_input("Sweep points", min_value=50, max_value=5000, value=600, step=50, key="dn"))
    if d1 <= d0: st.error("End distance must exceed start distance."); return
    ds = np.linspace(d0, d1, n); envs = {name: mode_envelope(b, ds, payload_bytes, required_packet_rate_hz) for name, b in st.session_state.bands.items()}
    st.plotly_chart(line_fig(ds, {name: env["prx_dbm"].to_numpy() for name, env in envs.items()}, "Received power vs distance", "Distance (km)", "RX power at radio (dBm)"), use_container_width=True)
    st.plotly_chart(line_fig(ds, {name: env["max_packet_rate_hz"].to_numpy() for name, env in envs.items()}, f"Maximum viable {payload_bytes} B packet rate", "Distance (km)", "Packets/s"), use_container_width=True)
    st.plotly_chart(line_fig(ds, {name: env["useful_rate_kbps"].to_numpy() for name, env in envs.items()}, "Maximum viable useful payload throughput", "Distance (km)", "Useful kbps"), use_container_width=True)
    if len(envs) >= 2:
        names = list(envs); c1, c2 = st.columns(2); preferred = c1.selectbox("Preferred near-range band", names, index=min(1, len(names)-1), key="cross-near"); fallback = c2.selectbox("Fallback / long-range band", names, index=0, key="cross-far")
        if preferred != fallback:
            a = envs[preferred]["max_packet_rate_hz"].to_numpy(); b = envs[fallback]["max_packet_rate_hz"].to_numpy(); idxs = np.where(a < b)[0]
            if len(idxs): st.info(f"First sweep point where **{preferred}** supports fewer {payload_bytes} B packets/s than **{fallback}**: about **{ds[int(idxs[0])]:.1f} km**. Use measured SNR/PER plus hysteresis for real switching.")
            else: st.info("No packet-rate crossover occurs inside the selected sweep.")


def receiver_nf_tool(band):
    st.subheader("Receiver cascade noise figure")
    default = pd.DataFrame([{"Stage": "Pre-LNA cable", "gain_db": -0.5, "nf_db": 0.5}, {"Stage": "LNA", "gain_db": 22.8, "nf_db": 0.66}, {"Stage": "Post-LNA cable/filter", "gain_db": -5.0, "nf_db": 5.0}, {"Stage": "LR2021 equivalent", "gain_db": 0.0, "nf_db": float(band.get("nf_db", 2.5))}])
    stages = st.data_editor(default, num_rows="dynamic", use_container_width=True, key="nf-cascade")
    try:
        nf = friis_cascade_noise_figure_db(stages); total_gain = stages["gain_db"].astype(float).sum(); c1, c2 = st.columns(2); c1.metric("Cascaded NF", f"{nf:.2f} dB"); c2.metric("Net gain", f"{total_gain:.2f} dB")
    except Exception as exc: st.error(f"Could not calculate cascade: {exc}")


def uncertainty_tool(band):
    st.subheader("Uncertainty / margin robustness")
    c1, c2, c3, c4, c5 = st.columns(5); sigmas = [c1.number_input("TX power σ (dB)", min_value=0.0, value=0.0, step=0.1), c2.number_input("TX gain σ (dB)", min_value=0.0, value=0.0, step=0.1), c3.number_input("RX gain σ (dB)", min_value=0.0, value=0.0, step=0.1), c4.number_input("Other loss σ (dB)", min_value=0.0, value=0.0, step=0.1), c5.number_input("Sensitivity σ (dB)", min_value=0.0, value=0.0, step=0.1)]
    sigma = math.sqrt(sum(float(x)**2 for x in sigmas)); d = np.linspace(1.0, 150.0, 500); p = received_power_dbm(band, d); sens_vals = [float(m["sensitivity_dbm"]) for m in band.get("modes", []) if m.get("enabled", True) and m.get("sensitivity_dbm") is not None]
    if not sens_vals: st.warning("No sensitivity values available."); return
    pct = normal_margin_percentiles(p - min(sens_vals), sigma, (5,50,95)); fig = line_fig(d, {"5th percentile": pct[5], "Median": pct[50], "95th percentile": pct[95]}, "RF margin with user-supplied dB uncertainty", "Distance (km)", "Margin (dB)"); fig.add_hline(y=float(band["required_margin_db"]), line_dash="dash", annotation_text="required margin"); st.plotly_chart(fig, use_container_width=True); st.metric("Combined σ", f"{sigma:.2f} dB")


def _normalize_col(s): return "".join(ch for ch in str(s).lower() if ch.isalnum())

def guess_column(columns, aliases):
    norm = {_normalize_col(c): c for c in columns}
    for alias in aliases:
        if _normalize_col(alias) in norm: return norm[_normalize_col(alias)]
    return None


def synthetic_trajectory():
    c1, c2, c3, c4 = st.columns(4); duration = c1.number_input("Demo flight duration (s)", min_value=10.0, value=180.0, step=10.0); apogee = c2.number_input("Demo apogee (km)", min_value=0.1, value=100.0, step=5.0); north_end = c3.number_input("Demo north drift (km)", value=10.0, step=1.0); east_end = c4.number_input("Demo east drift (km)", value=5.0, step=1.0)
    t = np.linspace(0.0, duration, 600); tau = t/duration; alt = 4*apogee*1000*tau*(1-tau); north = north_end*1000*tau; east = east_end*1000*tau; down = -alt
    return pd.DataFrame({"time_s": t, "north_m": north, "east_m": east, "down_m": down, "v_north_mps": np.gradient(north,t), "v_east_mps": np.gradient(east,t), "v_down_mps": np.gradient(down,t)})


def trajectory_analysis(band, payload_bytes, required_packet_rate_hz):
    st.subheader("Flight trajectory / geometry")
    st.caption("Ready for the sims team's NED state-vector export. The synthetic trajectory is only a UI/RF pipeline demo.")
    source = st.radio("Trajectory source", ["Synthetic demo", "Legacy MC run (test model)", "Upload CSV"], horizontal=True)
    if source == "Legacy MC run (test model)":
        df = pd.read_csv(Path("examples/mcrun1_legacy_trajectory.csv"))
        st.info("Legacy, lower-apogee Monte Carlo trajectory from the prior vehicle. Use this to exercise the RF pipeline, not as the current vehicle prediction. Position is treated as N/E/D; quaternion order/direction still need confirmation from sims.")
    elif source == "Upload CSV":
        uploaded = st.file_uploader("Trajectory CSV", type="csv", key="traj-upload")
        try: st.download_button("Download expected CSV template", Path("examples/trajectory_template.csv").read_bytes(), "trajectory_template.csv", "text/csv")
        except Exception: pass
        if uploaded is None: st.info("Expected core fields: time, N/E/D position, N/E/D velocity, quaternion, and p/q/r angular rates."); return
        df = pd.read_csv(uploaded)
    else: df = synthetic_trajectory()
    st.markdown("**Ground station relative to launch-frame origin**"); c1,c2,c3,c4 = st.columns(4); gs_n = c1.number_input("Ground station North (m)", value=0.0); gs_e = c2.number_input("Ground station East (m)", value=0.0); gs_alt = c3.number_input("Ground elevation rel. launch origin (m)", value=0.0); mast = c4.number_input("Antenna mast height (m)", min_value=0.0, value=2.0); gs_d = -(gs_alt+mast)
    aliases = {"time":["time_s","time","t"],"north":["north_m","north","n","position_north"],"east":["east_m","east","e","position_east"],"down":["down_m","down","d","position_down"],"vn":["v_north_mps","vn","velocity_north"],"ve":["v_east_mps","ve","velocity_east"],"vd":["v_down_mps","vd","velocity_down"],"q1":["q1"],"q2":["q2"],"q3":["q3"],"q4":["q4"],"temp":["temp_c","temperature_c","temperature"]}
    mapped = {k: guess_column(df.columns,v) for k,v in aliases.items()}
    with st.expander("Column mapping / state-vector conventions"):
        options = ["(none)"] + list(df.columns)
        for key,label in [("time","Time"),("north","North"),("east","East"),("down","Down"),("vn","V North"),("ve","V East"),("vd","V Down"),("q1","q1"),("q2","q2"),("q3","q3"),("q4","q4"),("temp","Temperature")]:
            idx = options.index(mapped[key]) if mapped[key] in options else 0; choice = st.selectbox(label, options, index=idx, key=f"map-{key}"); mapped[key] = None if choice == "(none)" else choice
        q_order = st.selectbox("Quaternion order", ["wxyz","xyzw"]); q_dir = st.selectbox("Quaternion rotation direction", ["body_to_ned","ned_to_body"])
    if any(mapped[k] is None for k in ("north","east","down")): st.error("North/East/Down position columns are required."); return
    t = df[mapped["time"]].to_numpy(float) if mapped["time"] else np.arange(len(df),dtype=float); geom = ned_geometry(df[mapped["north"]],df[mapped["east"]],df[mapped["down"]],gs_n,gs_e,gs_d); range_km = geom["range_m"]/1000.0
    temp_c = df[mapped["temp"]].to_numpy(float) if mapped["temp"] else np.full(len(df),float(band.get("reference_temp_c",25.0))); dtemp = temp_c-float(band.get("reference_temp_c",25.0)); ptx_shift = dtemp*float(band.get("tx_tempco_db_per_c",0.0)); sens_shift = dtemp*float(band.get("sensitivity_tempco_db_per_c",0.0))
    p_rx=np.empty(len(df)); best_names=[]; best_margin=np.empty(len(df)); best_rate=np.empty(len(df))
    for i in range(len(df)):
        p_rx[i]=float(received_power_dbm(band,range_km[i],ptx_dbm_override=float(band["ptx_dbm"])+ptx_shift[i])); best=select_best_mode(band,p_rx[i],payload_bytes,required_packet_rate_hz,sensitivity_shift_db=sens_shift[i]); best_names.append(best["name"]); best_margin[i]=best.get("margin_db",np.nan); best_rate[i]=best.get("max_packet_rate_hz",0.0)
    result=pd.DataFrame({"time_s":t,"range_km":range_km,"azimuth_deg":geom["azimuth_deg"],"elevation_deg":geom["elevation_deg"],"prx_dbm":p_rx,"best_profile":best_names,"best_margin_db":best_margin,"max_packet_rate_hz":best_rate,"temp_c":temp_c})
    if all(mapped[k] is not None for k in ("vn","ve","vd")):
        vr=radial_velocity_mps(df[mapped["vn"]],df[mapped["ve"]],df[mapped["vd"]],geom); result["radial_velocity_mps"]=vr; result["doppler_hz"]=doppler_shift_hz(band["freq_mhz"],vr)
    if all(mapped[k] is not None for k in ("q1","q2","q3","q4")):
        try:
            body=body_los_angles_from_quaternions(df[mapped["north"]],df[mapped["east"]],df[mapped["down"]],gs_n,gs_e,gs_d,df[mapped["q1"]],df[mapped["q2"]],df[mapped["q3"]],df[mapped["q4"]],order=q_order,direction=q_dir)
            for key,values in body.items(): result[key]=values
        except Exception as exc: st.warning(f"Could not evaluate body-frame LOS: {exc}")
    cols=st.columns(5); cols[0].metric("Max slant range",f"{result['range_km'].max():.1f} km"); cols[1].metric("Min elevation",f"{result['elevation_deg'].min():.1f}°"); cols[2].metric("Worst RX",f"{result['prx_dbm'].min():.1f} dBm"); cols[3].metric("Worst selected margin","—" if result["best_margin_db"].dropna().empty else f"{result['best_margin_db'].min():.1f} dB"); cols[4].metric("Peak |Doppler|",f"{result['doppler_hz'].abs().max():.0f} Hz" if "doppler_hz" in result else "need velocity")
    st.plotly_chart(line_fig(result["time_s"],{"Slant range":result["range_km"]},"Slant range vs flight time","Time (s)","Range (km)"),use_container_width=True); st.plotly_chart(line_fig(result["time_s"],{"Elevation":result["elevation_deg"],"Azimuth":result["azimuth_deg"]},"Ground-station look angles","Time (s)","Angle (deg)"),use_container_width=True); st.plotly_chart(line_fig(result["time_s"],{"RX power":result["prx_dbm"]},"Predicted received power along trajectory","Time (s)","dBm"),use_container_width=True); st.plotly_chart(line_fig(result["time_s"],{"Max viable packet rate":result["max_packet_rate_hz"]},f"Maximum viable {payload_bytes} B packet rate","Time (s)","Packets/s"),use_container_width=True)
    if "doppler_hz" in result: st.plotly_chart(line_fig(result["time_s"],{"Doppler":result["doppler_hz"]},"First-order Doppler shift","Time (s)","Hz"),use_container_width=True)
    if "body_off_axis_deg" in result: st.plotly_chart(line_fig(result["time_s"],{"Ground LOS off body +X":result["body_off_axis_deg"]},"Ground direction in rocket body frame","Time (s)","Off-axis angle (deg)"),use_container_width=True); st.caption("Body-frame angle does not yet alter antenna gain; it is ready to index a future installed pattern.")
    st.dataframe(result.head(100),hide_index=True,use_container_width=True); st.download_button("Download computed trajectory RF results",result.to_csv(index=False),"trajectory_rf_results.csv","text/csv")


def lr2021_telemetry_notes():
    st.subheader("LR2021 receiver telemetry / adaptation inputs")
    st.markdown("""Log packet SNR, average packet RSSI, LoRa-signal RSSI after despreading, packet length/coding rate, detector, CRC/header status, instantaneous RSSI/noise, RX/CRC/header-error/false-sync counters, LR2021 temperature, supply voltage, reset/error flags, and a monotonic application packet sequence number. Adapt using **SNR + PER + predicted margin**, not RSSI alone. Both ends must coordinate profile changes and return to a known 915 MHz recovery profile if a transition fails.""")
    recovery=copy.deepcopy(RECOVERY_PROFILE); recovery.update(sensitivity_dbm=datasheet_sensitivity_dbm("subghz",recovery["bw_khz"],recovery["sf"]),preamble_symbols=recommended_preamble_symbols(recovery["sf"]),explicit_header=True,crc_on=True,ldro="auto"); mm=mode_metrics(recovery,st.session_state.payload_bytes); st.info(f"Working recovery candidate: **915 MHz / {recovery['bw_khz']:.0f} kHz / SF{recovery['sf']} / CR 4/{recovery['cr_den']}**, sensitivity about **{recovery['sensitivity_dbm']:.1f} dBm**, ~**{mm['toa_s']*1000:.0f} ms** airtime for {st.session_state.payload_bytes} B. Validate before flight.")


def source_notes():
    st.subheader("Model scope / known limitations")
    st.markdown("""**Implemented now:** reference-plane link budget, mismatch/EIRP/FSPL, LR2021 sensitivity table, LoRa airtime/throughput, margin+throughput profile viability, dual-band crossover, Friis NF cascade, uncertainty bands, NED trajectory geometry, Doppler, quaternion body-frame LOS, and zero-by-default temperature hooks.

**Still needs real data:** 170 B PER/sensitivity curves, installed antenna pattern/polarization, real 2.4 GHz ground-chain values, end-to-end receiver sensitivity/NF, measured TX power vs temperature/voltage/frequency, terrain/earth-curvature where needed, physical atmospheric loss, validated ground reflection/multipath, and trajectory/attitude Monte Carlo.""")


def main():
    st.set_page_config(page_title="RF Sims", page_icon="📡", layout="wide"); init_state(); band_sidebar(); st.title("RF Link & LR2021 Telemetry Simulator"); st.caption("Live trade study for link budget, LR2021 LoRa profiles, telemetry throughput, and flight-trajectory geometry.")
    name=st.session_state.selected_band; band=st.session_state.bands[name]; tabs=st.tabs(["Link budget","LoRa profiles","Distance / crossover","Flight trajectory","Receiver NF","Uncertainty","LR2021 telemetry","Scope"])
    with tabs[0]:
        band=edit_band_inputs(name,band); st.session_state.bands[name]=band; link_operating_point(band,st.session_state.payload_bytes,st.session_state.required_packet_rate_hz); bw=st.selectbox("Noise-floor bandwidth",[1000.0,500.0,250.0,200.0,125.0,62.5,31.25],index=4); st.metric("kTB + system NF",f"{receiver_noise_floor_dbm(bw*1000,band['nf_db'],band['temp_k']):.1f} dBm")
    with tabs[1]: band=edit_modes(name,band,st.session_state.payload_bytes); st.session_state.bands[name]=band
    with tabs[2]: distance_analysis(st.session_state.payload_bytes,st.session_state.required_packet_rate_hz)
    with tabs[3]: trajectory_analysis(band,st.session_state.payload_bytes,st.session_state.required_packet_rate_hz)
    with tabs[4]: receiver_nf_tool(band)
    with tabs[5]: uncertainty_tool(band)
    with tabs[6]: lr2021_telemetry_notes()
    with tabs[7]: source_notes()


if __name__ == "__main__": main()
