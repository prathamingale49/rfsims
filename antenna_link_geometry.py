"""Rocket/ground geometry study for two switched, aft-canted RHCP patches."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from antenna_models import (
    GROUND_PATCH_2400,
    GROUND_YAGI_915,
    PATCH_2400,
    PATCH_915,
    AxisymmetricPattern,
    antenna_boresights,
    attitude_body_to_ned,
    direction_to_ground_body,
    diversity_gains_dbic,
    fspl_db,
    quaternion_body_to_ned,
    off_boresight_deg,
)


def load_pattern(path: str | None, built_in: AxisymmetricPattern) -> AxisymmetricPattern:
    if path:
        return AxisymmetricPattern.from_csv(path, name=Path(path).stem, frequency_mhz=built_in.frequency_mhz)
    return built_in


def ground_gain_for_position(position_ned, ground_ned, pattern: AxisymmetricPattern, args) -> float:
    if args.ground_pointing == "tracked":
        return float(pattern.gain_dbic(args.ground_pointing_error_deg))
    az=np.radians(args.ground_fixed_azimuth_deg); el=np.radians(args.ground_fixed_elevation_deg)
    boresight=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),-np.sin(el)])
    direction=np.asarray(position_ned,dtype=float)-np.asarray(ground_ned,dtype=float)
    return float(pattern.gain_dbic(off_boresight_deg(direction,boresight)))


def attitudes(frame: pd.DataFrame, args) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    time_s = frame["time_s"].to_numpy(float)
    if args.attitude_source == "trajectory":
        required = ["q1","q2","q3","q4"]
        if not set(required) <= set(frame.columns):
            raise ValueError("trajectory attitude needs q1/q2/q3/q4 columns")
        matrices = [quaternion_body_to_ned(frame.loc[i,required], order=args.quaternion_order,
                                           direction=args.quaternion_direction) for i in range(len(frame))]
        return matrices, np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    tilt = args.tilt_deg + args.tilt_wobble_deg * np.sin(2*np.pi*time_s/max(args.tilt_wobble_period_s,1e-6))
    roll = args.initial_roll_deg + 360.0*args.roll_rate_hz*time_s
    matrices = [attitude_body_to_ned(t, args.tilt_azimuth_deg, r) for t,r in zip(tilt,roll)]
    return matrices, tilt, np.mod(roll,360.0)


def evaluate(frame: pd.DataFrame, matrices: list[np.ndarray], cants: list[float], pattern: AxisymmetricPattern,
             ground_pattern: AxisymmetricPattern, args) -> tuple[pd.DataFrame, dict[float,dict[str,np.ndarray]]]:
    positions = frame[["north_m","east_m","down_m"]].to_numpy(float)
    ground = np.array([args.ground_north_m,args.ground_east_m,-args.ground_altitude_m])
    directions = np.empty_like(positions)
    distances = np.empty(len(frame))
    for i,(position,matrix) in enumerate(zip(positions,matrices)):
        directions[i],distances[i] = direction_to_ground_body(position,ground,matrix)
    ground_gains = np.array([ground_gain_for_position(position,ground,ground_pattern,args) for position in positions])
    base = pd.DataFrame({"time_s":frame["time_s"],"altitude_m":-frame["down_m"],"distance_m":distances})
    results = {}
    for cant in cants:
        gain_a,gain_b,selected = diversity_gains_dbic(directions,cant,pattern)
        received = args.tx_power_dbm + selected + ground_gains - fspl_db(args.frequency_mhz,distances) - args.fixed_losses_db
        results[cant] = {"gain_a":gain_a,"gain_b":gain_b,"selected_gain":selected,
                         "selected_antenna":np.where(gain_a>=gain_b,"A","B"),
                         "received_dbm":received,"margin_db":received-args.sensitivity_dbm,
                         "directions":directions,"distances":distances}
        base[f"margin_{cant:g}deg_db"] = received-args.sensitivity_dbm
    return base,results


def worst_over_roll(frame: pd.DataFrame, cants: list[float], pattern: AxisymmetricPattern,
                    ground_pattern: AxisymmetricPattern,args):
    positions=frame[["north_m","east_m","down_m"]].to_numpy(float)
    picks=np.unique(np.linspace(0,len(frame)-1,min(220,len(frame)),dtype=int))
    rolls=np.arange(0.0,360.0,5.0)
    ground=np.array([args.ground_north_m,args.ground_east_m,-args.ground_altitude_m])
    output={cant:[] for cant in cants}
    for index in picks:
        margins={cant:[] for cant in cants}
        for roll in rolls:
            matrix=attitude_body_to_ned(args.tilt_deg,args.tilt_azimuth_deg,roll)
            direction,distance=direction_to_ground_body(positions[index],ground,matrix)
            ground_gain=ground_gain_for_position(positions[index],ground,ground_pattern,args)
            for cant in cants:
                selected=diversity_gains_dbic(direction,cant,pattern)[2]
                margin=args.tx_power_dbm+selected+ground_gain-fspl_db(args.frequency_mhz,distance)-args.fixed_losses_db-args.sensitivity_dbm
                margins[cant].append(float(margin))
        for cant in cants: output[cant].append(min(margins[cant]))
    return picks,{cant:np.asarray(values) for cant,values in output.items()}


def roll_tilt_heatmap(frame: pd.DataFrame, cant: float, pattern: AxisymmetricPattern,
                      ground_pattern: AxisymmetricPattern,args):
    index=int(np.argmax(-frame["down_m"].to_numpy(float)))
    position=frame.loc[index,["north_m","east_m","down_m"]].to_numpy(float)
    ground=np.array([args.ground_north_m,args.ground_east_m,-args.ground_altitude_m])
    rolls=np.arange(0.0,361.0,5.0); tilts=np.arange(0.0,61.0,2.0)
    grid=np.empty((len(tilts),len(rolls)))
    ground_gain=ground_gain_for_position(position,ground,ground_pattern,args)
    for i,tilt in enumerate(tilts):
        for j,roll in enumerate(rolls):
            direction,distance=direction_to_ground_body(position,ground,attitude_body_to_ned(tilt,args.tilt_azimuth_deg,roll))
            selected=diversity_gains_dbic(direction,cant,pattern)[2]
            grid[i,j]=args.tx_power_dbm+selected+ground_gain-fspl_db(args.frequency_mhz,distance)-args.fixed_losses_db-args.sensitivity_dbm
    return rolls,tilts,grid,index


def study_figure(frame,summary,results,picks,roll_min,rolls,tilts,heat,cants,args,output):
    baseline=45.0 if 45.0 in cants else cants[0]
    fig=make_subplots(rows=3,cols=2,specs=[[{"type":"scene"},{"type":"xy"}],
                                           [{"type":"xy"},{"type":"xy"}],
                                           [{"type":"xy"},{"type":"heatmap"}]],
                      subplot_titles=["Rocket trajectory + ground station","Selected-diversity link margin",
                                      f"A/B/selected gain at {baseline:g}°","Worst margin over every roll angle",
                                      "Cant comparison vs altitude","Roll/tilt margin at apogee"])
    altitude=-frame["down_m"].to_numpy(float)
    fig.add_trace(go.Scatter3d(x=frame["east_m"]/1000,y=frame["north_m"]/1000,z=altitude/1000,
                               mode="lines",line=dict(color=summary[f"margin_{baseline:g}deg_db"],colorscale="Viridis",width=6,
                               colorbar=dict(title="margin dB",x=0.43)),name="trajectory"),1,1)
    fig.add_trace(go.Scatter3d(x=[args.ground_east_m/1000],y=[args.ground_north_m/1000],z=[args.ground_altitude_m/1000],
                               mode="markers",marker=dict(size=7,color="orange",symbol="diamond"),name="ground station"),1,1)
    for cant in cants:
        fig.add_trace(go.Scatter(x=frame["time_s"],y=results[cant]["margin_db"],name=f"{cant:g}°"),1,2)
    b=results[baseline]
    fig.add_trace(go.Scatter(x=frame["time_s"],y=b["gain_a"],name="A",line=dict(dash="dot")),2,1)
    fig.add_trace(go.Scatter(x=frame["time_s"],y=b["gain_b"],name="B",line=dict(dash="dash")),2,1)
    fig.add_trace(go.Scatter(x=frame["time_s"],y=b["selected_gain"],name="max(A,B)",line=dict(width=3)),2,1)
    for cant in cants:
        fig.add_trace(go.Scatter(x=altitude[picks]/1000,y=roll_min[cant],name=f"{cant:g}° worst-roll"),2,2)
        fig.add_trace(go.Scatter(x=altitude/1000,y=results[cant]["margin_db"],name=f"{cant:g}° actual roll",showlegend=False),3,1)
    fig.add_trace(go.Heatmap(x=rolls,y=tilts,z=heat,colorbar=dict(title="margin dB"),hovertemplate="roll=%{x:.0f}°<br>tilt=%{y:.0f}°<br>margin=%{z:.1f} dB<extra></extra>"),3,2)
    fig.add_trace(go.Scatter(x=[float(frame["time_s"].min()),float(frame["time_s"].max())],y=[0.0,0.0],
                             mode="lines",line=dict(dash="dash",color="red"),name="0 dB margin"),1,2)
    fig.update_xaxes(title_text="time (s)",row=1,col=2); fig.update_yaxes(title_text="margin (dB)",row=1,col=2)
    fig.update_xaxes(title_text="time (s)",row=2,col=1); fig.update_yaxes(title_text="flight gain (dBic)",row=2,col=1)
    fig.update_xaxes(title_text="altitude (km)",row=2,col=2); fig.update_yaxes(title_text="minimum margin (dB)",row=2,col=2)
    fig.update_xaxes(title_text="altitude (km)",row=3,col=1); fig.update_yaxes(title_text="margin (dB)",row=3,col=1)
    fig.update_xaxes(title_text="roll (deg)",row=3,col=2); fig.update_yaxes(title_text="tilt (deg)",row=3,col=2)
    fig.update_layout(title=f"Two-board flight antenna link study — {args.frequency_mhz:g} MHz",height=1450,width=1550,hovermode="x unified")
    output.parent.mkdir(parents=True,exist_ok=True); fig.write_html(output,include_plotlyjs="cdn")


def animation_figure(frame,matrices,pattern,cant,args,output):
    picks=np.unique(np.linspace(0,len(frame)-1,min(80,len(frame)),dtype=int))
    e=frame["east_m"].to_numpy(float)/1000; n=frame["north_m"].to_numpy(float)/1000; z=-frame["down_m"].to_numpy(float)/1000
    bore_a,bore_b=antenna_boresights(cant)
    lobe_theta=np.linspace(0.0,np.pi,12); lobe_phi=np.linspace(0.0,2.0*np.pi,20)
    lobe_tt,lobe_pp=np.meshgrid(lobe_theta,lobe_phi)
    def lobe_surface(origin,matrix,bore,color):
        z_axis=bore/np.linalg.norm(bore); reference=np.array([0.0,0.0,1.0])
        if abs(float(reference@z_axis))>0.95: reference=np.array([1.0,0.0,0.0])
        x_axis=np.cross(reference,z_axis); x_axis/=np.linalg.norm(x_axis); y_axis=np.cross(z_axis,x_axis)
        local=np.stack((np.sin(lobe_tt)*np.cos(lobe_pp),np.sin(lobe_tt)*np.sin(lobe_pp),np.cos(lobe_tt)),axis=-1)
        direction_body=local@np.column_stack((x_axis,y_axis,z_axis)).T
        theta_deg=np.degrees(lobe_tt); gain=pattern.gain_dbic(theta_deg)
        radius_km=2.2*(0.1+0.9*10**((gain-pattern.peak_gain_dbic)/20.0))
        ned=direction_body@matrix.T
        x=origin[0]+radius_km*ned[...,1]; y=origin[1]+radius_km*ned[...,0]; z=origin[2]-radius_km*ned[...,2]
        return go.Surface(x=x,y=y,z=z,surfacecolor=np.full_like(x,color),colorscale=[[0,"red" if color==0 else "blue"],[1,"red" if color==0 else "blue"]],opacity=0.28,showscale=False,hoverinfo="skip",name="A lobe" if color==0 else "B lobe")
    def traces(index):
        origin=np.array([e[index],n[index],z[index]])
        # Convert NED vectors to plot coordinates E/N/Up.
        def plot_vec(body_vec,scale=2.5):
            ned=matrices[index]@body_vec
            return origin+scale*np.array([ned[1],ned[0],-ned[2]])
        end_a=plot_vec(bore_a); end_b=plot_vec(bore_b)
        gs=np.array([args.ground_east_m/1000,args.ground_north_m/1000,args.ground_altitude_m/1000])
        if args.ground_pointing=="tracked": ground_end=origin
        else:
            az=np.radians(args.ground_fixed_azimuth_deg); el=np.radians(args.ground_fixed_elevation_deg)
            ground_end=gs+5.0*np.array([np.cos(el)*np.sin(az),np.cos(el)*np.cos(az),np.sin(el)])
        return [
            go.Scatter3d(x=[origin[0]],y=[origin[1]],z=[origin[2]],mode="markers",marker=dict(size=6,color="black"),name="rocket"),
            go.Scatter3d(x=[origin[0],end_a[0]],y=[origin[1],end_a[1]],z=[origin[2],end_a[2]],mode="lines",line=dict(color="red",width=8),name="A boresight"),
            go.Scatter3d(x=[origin[0],end_b[0]],y=[origin[1],end_b[1]],z=[origin[2],end_b[2]],mode="lines",line=dict(color="blue",width=8),name="B boresight"),
            go.Scatter3d(x=[gs[0],origin[0]],y=[gs[1],origin[1]],z=[gs[2],origin[2]],mode="lines",line=dict(color="orange",width=3,dash="dot"),name="ground LOS"),
            go.Scatter3d(x=[gs[0],ground_end[0]],y=[gs[1],ground_end[1]],z=[gs[2],ground_end[2]],mode="lines",line=dict(color="green",width=6),name="ground boresight"),
            lobe_surface(origin,matrices[index],bore_a,0),
            lobe_surface(origin,matrices[index],bore_b,1),
        ]
    fig=go.Figure(data=[go.Scatter3d(x=e,y=n,z=z,mode="lines",line=dict(color="gray",width=3),name="trajectory"),
                        go.Scatter3d(x=[args.ground_east_m/1000],y=[args.ground_north_m/1000],z=[args.ground_altitude_m/1000],mode="markers",marker=dict(size=7,color="orange"),name="ground station"),
                        *traces(picks[0])],
                  frames=[go.Frame(data=traces(i),traces=[2,3,4,5,6,7,8],name=str(k)) for k,i in enumerate(picks)])
    fig.update_layout(title=f"Moving antenna boresights — {cant:g}° aft cant",
                      scene=dict(xaxis_title="East (km)",yaxis_title="North (km)",zaxis_title="Altitude (km)",aspectmode="data"),
                      updatemenus=[dict(type="buttons",buttons=[dict(label="Play",method="animate",args=[None,{"frame":{"duration":80,"redraw":True},"fromcurrent":True}]),dict(label="Pause",method="animate",args=[[None],{"mode":"immediate"}])])],
                      sliders=[dict(steps=[dict(method="animate",args=[[str(k)],{"mode":"immediate","frame":{"duration":0,"redraw":True}}],label=f"{frame['time_s'].iloc[i]:.0f}s") for k,i in enumerate(picks)])],
                      height=850,width=1200)
    fig.write_html(output,include_plotlyjs="cdn")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory",type=Path,default=Path("examples/mcrun1_legacy_trajectory.csv"))
    parser.add_argument("--band",choices=["915","2400"],default="915")
    parser.add_argument("--cants",default="0,30,45,60")
    parser.add_argument("--flight-pattern-csv"); parser.add_argument("--ground-pattern-csv")
    parser.add_argument("--attitude-source",choices=["parametric","trajectory"],default="parametric")
    parser.add_argument("--quaternion-order",choices=["wxyz","xyzw"],default="wxyz")
    parser.add_argument("--quaternion-direction",choices=["body_to_ned","ned_to_body"],default="body_to_ned")
    parser.add_argument("--tilt-deg",type=float,default=5.0); parser.add_argument("--tilt-azimuth-deg",type=float,default=0.0)
    parser.add_argument("--tilt-wobble-deg",type=float,default=3.0); parser.add_argument("--tilt-wobble-period-s",type=float,default=20.0)
    parser.add_argument("--roll-rate-hz",type=float,default=1.0); parser.add_argument("--initial-roll-deg",type=float,default=0.0)
    parser.add_argument("--ground-north-m",type=float,default=0.0); parser.add_argument("--ground-east-m",type=float,default=0.0); parser.add_argument("--ground-altitude-m",type=float,default=0.0)
    parser.add_argument("--ground-pointing-error-deg",type=float,default=0.0)
    parser.add_argument("--ground-pointing",choices=["tracked","fixed"],default="tracked")
    parser.add_argument("--ground-fixed-azimuth-deg",type=float,default=0.0); parser.add_argument("--ground-fixed-elevation-deg",type=float,default=45.0)
    parser.add_argument("--tx-power-dbm",type=float); parser.add_argument("--sensitivity-dbm",type=float,default=-130.0)
    parser.add_argument("--fixed-losses-db",type=float,default=8.5)
    parser.add_argument("--output",type=Path,default=Path("study_outputs/antenna_link_geometry.html"))
    args=parser.parse_args()
    if args.band=="915":
        built_flight,built_ground=PATCH_915,GROUND_YAGI_915; args.frequency_mhz=915.0
        if args.tx_power_dbm is None: args.tx_power_dbm=30.5
    else:
        built_flight,built_ground=PATCH_2400,GROUND_PATCH_2400; args.frequency_mhz=2450.0
        if args.tx_power_dbm is None: args.tx_power_dbm=30.5
    flight=load_pattern(args.flight_pattern_csv,built_flight); ground=load_pattern(args.ground_pattern_csv,built_ground)
    cants=[float(value) for value in args.cants.split(",")]
    frame=pd.read_csv(args.trajectory)
    matrices,tilt,roll=attitudes(frame,args)
    summary,results=evaluate(frame,matrices,cants,flight,ground,args)
    picks,roll_min=worst_over_roll(frame,cants,flight,ground,args)
    baseline=45.0 if 45.0 in cants else cants[0]
    rolls,tilts,heat,apogee_index=roll_tilt_heatmap(frame,baseline,flight,ground,args)
    envelope_worst={cant:float(np.min(roll_tilt_heatmap(frame,cant,flight,ground,args)[2])) for cant in cants}
    study_figure(frame,summary,results,picks,roll_min,rolls,tilts,heat,cants,args,args.output)
    animation_output=args.output.with_name(args.output.stem+"_animation.html")
    results_output=args.output.with_name(args.output.stem+"_results.csv")
    recorded=frame[[column for column in ["time_s","north_m","east_m","down_m"] if column in frame]].copy()
    for cant in cants:
        key=f"{cant:g}deg"
        recorded[f"gain_a_{key}_dbic"]=results[cant]["gain_a"]
        recorded[f"gain_b_{key}_dbic"]=results[cant]["gain_b"]
        recorded[f"selected_antenna_{key}"]=results[cant]["selected_antenna"]
        recorded[f"selected_gain_{key}_dbic"]=results[cant]["selected_gain"]
        recorded[f"received_{key}_dbm"]=results[cant]["received_dbm"]
        recorded[f"margin_{key}_db"]=results[cant]["margin_db"]
    recorded.to_csv(results_output,index=False)
    animation_figure(frame,matrices,flight,baseline,args,animation_output)
    apogee=int(np.argmax(-frame["down_m"].to_numpy(float)))
    apogee_roll_values={}
    print(f"Model: {flight.name}; trajectory apogee={-frame['down_m'].min()/1000:.2f} km")
    for cant in cants:
        actual=float(np.min(results[cant]["margin_db"])); arbitrary=float(np.min(roll_min[cant]))
        at_apogee=[]
        for roll_value in np.arange(0.0,360.0,1.0):
            position=frame.loc[apogee,["north_m","east_m","down_m"]].to_numpy(float)
            direction,distance=direction_to_ground_body(position,[args.ground_north_m,args.ground_east_m,-args.ground_altitude_m],attitude_body_to_ned(args.tilt_deg,args.tilt_azimuth_deg,roll_value))
            gain=diversity_gains_dbic(direction,cant,flight)[2]
            ground_gain=ground_gain_for_position(position,[args.ground_north_m,args.ground_east_m,-args.ground_altitude_m],ground,args)
            at_apogee.append(float(args.tx_power_dbm+gain+ground_gain-fspl_db(args.frequency_mhz,distance)-args.fixed_losses_db-args.sensitivity_dbm))
        apogee_roll_values[cant]=min(at_apogee)
        print(f"{cant:>4g} deg: worst actual-roll={actual:7.2f} dB; worst roll-sweep={arbitrary:7.2f} dB; apogee worst-roll={apogee_roll_values[cant]:7.2f} dB")
    print(f"Apogee worst-roll improvement, 45 deg vs 0 deg: {apogee_roll_values.get(45.0,float('nan'))-apogee_roll_values[cants[0]]:+.2f} dB")
    print("Apogee 0-60 deg tilt / arbitrary-roll worst cases: "+", ".join(f"{cant:g} deg={envelope_worst[cant]:.2f} dB" for cant in cants))
    print(f"Apogee attitude-envelope improvement, 45 deg vs 0 deg: {envelope_worst.get(45.0,float('nan'))-envelope_worst[cants[0]]:+.2f} dB")
    worst=np.unravel_index(np.argmin(heat),heat.shape)
    print(f"{baseline:g}° apogee heatmap worst: {heat[worst]:.2f} dB at tilt={tilts[worst[0]]:.0f}°, roll={rolls[worst[1]]:.0f}°")
    print(f"Wrote {args.output}, {animation_output}, and {results_output}")


if __name__=="__main__":
    main()
