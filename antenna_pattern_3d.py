"""Interactive 3D study of the two-board switched-diversity antenna layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from antenna_models import (
    GROUND_PATCH_2400,
    GROUND_YAGI_915,
    PATCH_2400,
    PATCH_915,
    AxisymmetricPattern,
    antenna_boresights,
    diversity_gains_dbic,
    unit,
)


def local_basis(boresight: np.ndarray) -> np.ndarray:
    z_axis = unit(boresight)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, z_axis))) > 0.95:
        reference = np.array([1.0, 0.0, 0.0])
    x_axis = unit(np.cross(reference, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def pattern_surface(pattern: AxisymmetricPattern, boresight: np.ndarray, *, scale: float = 1.0, samples: int = 55):
    theta = np.linspace(0.0, np.pi, samples)
    phi = np.linspace(0.0, 2.0 * np.pi, samples * 2)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    local = np.stack((
        np.sin(theta_grid) * np.cos(phi_grid),
        np.sin(theta_grid) * np.sin(phi_grid),
        np.cos(theta_grid),
    ), axis=-1)
    directions = local @ local_basis(boresight).T
    gains = pattern.gain_dbic(np.degrees(theta_grid))
    radius = scale * (0.10 + 0.90 * 10.0 ** ((gains - pattern.peak_gain_dbic) / 20.0))
    return directions[..., 0] * radius, directions[..., 1] * radius, directions[..., 2] * radius, gains


def diversity_surface(pattern: AxisymmetricPattern, cant_deg: float, *, samples: int = 55):
    theta = np.linspace(0.0, np.pi, samples)
    phi = np.linspace(0.0, 2.0 * np.pi, samples * 2)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    directions = np.stack((
        np.sin(theta_grid) * np.cos(phi_grid),
        np.sin(theta_grid) * np.sin(phi_grid),
        np.cos(theta_grid),
    ), axis=-1)
    _, _, gain = diversity_gains_dbic(directions, cant_deg, pattern)
    radius = 0.10 + 0.90 * 10.0 ** ((gain - pattern.peak_gain_dbic) / 20.0)
    return directions[..., 0] * radius, directions[..., 1] * radius, directions[..., 2] * radius, gain


def add_surface(fig, data, row, col, name, colorscale, opacity=0.82, showscale=False):
    x, y, z, gain = data
    fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=gain, colorscale=colorscale,
                             cmin=float(np.min(gain)), cmax=float(np.max(gain)), opacity=opacity,
                             showscale=showscale, name=name, hovertemplate=f"{name}<br>gain=%{{surfacecolor:.1f}} dBic<extra></extra>"), row=row, col=col)


def add_rocket(fig, row, col, cant_deg: float):
    z = np.linspace(-1.15, 1.15, 28)
    phi = np.linspace(0.0, 2.0 * np.pi, 36)
    zz, pp = np.meshgrid(z, phi)
    radius = 0.11
    fig.add_trace(go.Surface(x=radius*np.cos(pp), y=radius*np.sin(pp), z=zz,
                             surfacecolor=np.zeros_like(zz), colorscale=[[0,"#777"],[1,"#777"]],
                             opacity=0.35, showscale=False, hoverinfo="skip", name="rocket"), row=row, col=col)
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[-1.2,1.45], mode="lines+text",
                               line=dict(color="#222", width=5), text=[None,"+Z nose"],
                               textposition="top center", showlegend=False, hoverinfo="skip"), row=row, col=col)
    for label, origin, bore, color in [
        ("A", np.array([0.13,0,0]), antenna_boresights(cant_deg)[0], "#d62728"),
        ("B", np.array([-0.13,0,0]), antenna_boresights(cant_deg)[1], "#1f77b4"),
    ]:
        end = origin + 0.68*bore
        fig.add_trace(go.Scatter3d(x=[origin[0],end[0]], y=[origin[1],end[1]], z=[origin[2],end[2]],
                                   mode="lines+text", line=dict(color=color,width=7), text=[None,f"{label} bore"],
                                   textposition="top center", showlegend=False, hoverinfo="skip"), row=row, col=col)


def load_pattern(args) -> AxisymmetricPattern:
    built_in = PATCH_915 if args.band == "915" else PATCH_2400
    if args.pattern_csv:
        return AxisymmetricPattern.from_csv(args.pattern_csv, name=Path(args.pattern_csv).stem, frequency_mhz=built_in.frequency_mhz)
    return built_in


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band", choices=["915","2400"], default="915")
    parser.add_argument("--baseline-cant", type=float, default=45.0)
    parser.add_argument("--pattern-csv", help="Optional theta_deg,gain_dbic axisymmetric pattern cut")
    parser.add_argument("--output", type=Path, default=Path("study_outputs/antenna_patterns_3d.html"))
    args = parser.parse_args()
    pattern = load_pattern(args)
    cants = [0.0, 30.0, 45.0, 60.0]
    fig = make_subplots(rows=2, cols=3, specs=[[{"type":"scene"}]*3,[{"type":"scene"}]*3],
                        subplot_titles=[f"Diversity: {c:.0f}° aft cant" for c in cants] +
                                       [f"A/B + diversity at {args.baseline_cant:.0f}°", "Ground antenna pattern"])
    positions = [(1,1),(1,2),(1,3),(2,1)]
    for cant, (row,col) in zip(cants, positions):
        add_surface(fig, diversity_surface(pattern, cant), row, col, "selected max(A,B)", "Viridis")
        add_rocket(fig, row, col, cant)

    bore_a, bore_b = antenna_boresights(args.baseline_cant)
    add_surface(fig, pattern_surface(pattern,bore_a), 2,2,"antenna A","Reds",0.38)
    add_surface(fig, pattern_surface(pattern,bore_b), 2,2,"antenna B","Blues",0.38)
    add_surface(fig, diversity_surface(pattern,args.baseline_cant),2,2,"selected max(A,B)","Viridis",0.58)
    add_rocket(fig,2,2,args.baseline_cant)

    ground = GROUND_YAGI_915 if args.band == "915" else GROUND_PATCH_2400
    add_surface(fig, pattern_surface(ground,np.array([0,0,1.0])),2,3,ground.name,"Plasma",0.85,True)
    fig.add_trace(go.Scatter3d(x=[0,0],y=[0,0],z=[0,1.25],mode="lines+text",line=dict(width=7,color="#ff7f0e"),
                               text=[None,"tracked boresight"],showlegend=False),row=2,col=3)

    scene_settings = dict(aspectmode="cube", xaxis_title="body X", yaxis_title="body Y", zaxis_title="body Z", camera_eye=dict(x=1.45,y=1.45,z=1.1))
    for i in range(1,7): fig.update_layout(**{f"scene{i if i>1 else ''}":scene_settings})
    fig.update_layout(title=f"Two-board switched-diversity pattern study — {pattern.name}", width=1700, height=1050,
                      margin=dict(l=10,r=10,t=80,b=10), showlegend=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.write_html(args.output, include_plotlyjs="cdn")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
