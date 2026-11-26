"""Comparison plots for the report (Nathaly). Matplotlib, saved as PNG.

    from eval.plots import plot_racing_lines, plot_speed_vs_s, plot_lap_times
    plot_racing_lines(track, results, "racing_lines.png")

`results` = list of dicts from eval.metrics.run_episode / load_result.
Each driver keeps the same colour in every figure (colour follows the driver).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

# Fixed colour per driver (colorblind-checked categorical order).
DRIVER_COLORS = {"pure pursuit": "#2a78d6", "mpcc": "#eb6834", "ppo": "#1baf7a", "human": "#eda100"}
_EXTRA = ["#e87ba4", "#008300"]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dd"
# one-hue sequential ramp (light -> dark blue) for speed
SPEED_CMAP = LinearSegmentedColormap.from_list("speed", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def color_for(name, i=0):
    key = name.lower()
    for k, c in DRIVER_COLORS.items():
        if key.startswith(k):
            return c
    return _EXTRA[i % len(_EXTRA)]


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def draw_track(ax, track):
    tab = track.centerline_table(s_min=0.0, s_max=track.length)
    nx, ny = -np.sin(tab["theta"]), np.cos(tab["theta"])
    ax.fill(np.r_[tab["x"] + nx * tab["w_left"], (tab["x"] - nx * tab["w_right"])[::-1]],
            np.r_[tab["y"] + ny * tab["w_left"], (tab["y"] - ny * tab["w_right"])[::-1]],
            color="#f3f2ee", zorder=0)
    ax.plot(tab["x"] + nx * tab["w_left"], tab["y"] + ny * tab["w_left"], color=MUTED, lw=1)
    ax.plot(tab["x"] - nx * tab["w_right"], tab["y"] - ny * tab["w_right"], color=MUTED, lw=1)
    ax.plot(tab["x"], tab["y"], color=MUTED, lw=0.6, ls="--")
    # start line
    i = 0
    ax.plot([tab["x"][i] + nx[i] * tab["w_left"][i], tab["x"][i] - nx[i] * tab["w_right"][i]],
            [tab["y"][i] + ny[i] * tab["w_left"][i], tab["y"][i] - ny[i] * tab["w_right"][i]], color=INK, lw=2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def plot_speed_map(track, result, path):
    """One driver: driven line coloured by speed."""
    log = result["log"]
    fig, ax = plt.subplots(figsize=(8, 7))
    _style(ax)
    draw_track(ax, track)
    sc = ax.scatter(log["x"], log["y"], c=log["vx"], s=4, cmap=SPEED_CMAP, zorder=3)
    fig.colorbar(sc, ax=ax, label="speed (m/s)", shrink=0.8)
    best = result["metrics"].get("best_lap")
    ax.set_title(f"{result['name']} on {track.name}" + (f" - best lap {best:.2f} s" if best else ""), color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_racing_lines(track, results, path):
    """All drivers' lines on one track map."""
    fig, ax = plt.subplots(figsize=(8, 7))
    _style(ax)
    draw_track(ax, track)
    for i, r in enumerate(results):
        ax.plot(r["log"]["x"], r["log"]["y"], color=color_for(r["name"], i), lw=1.5, label=r["name"], zorder=3)
    ax.legend(frameon=False, loc="best")
    ax.set_title(f"Driven lines - {track.name} track", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_speed_vs_s(results, path, track=None, profiles=None):
    """Speed and lateral offset along the lap (one lap per driver), stacked, shared s axis.
    profiles: optional {label: (s, v)} drawn dashed (e.g. Pure Pursuit's target speed)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[2, 1])
    for ax in (ax1, ax2):
        _style(ax)
    for i, r in enumerate(results):
        log = r["log"]
        m = np.asarray(log["lap"]) == 1 if (np.asarray(log["lap"]) == 1).any() else np.ones(len(log["s"]), bool)
        s = np.asarray(log["s"])[m]
        order = np.argsort(s)
        c = color_for(r["name"], i)
        ax1.plot(s[order], np.asarray(log["vx"])[m][order], color=c, lw=2, label=r["name"])
        ax2.plot(s[order], np.asarray(log["e_c"])[m][order], color=c, lw=2)
    for label, (ps, pv) in (profiles or {}).items():
        ax1.plot(ps, pv, color=MUTED, lw=1.2, ls="--", label=label)
    if track is not None:
        half = 0.75 - 0.155   # off-track line: half road - half car (shown for reference)
        for y in (half, -half):
            ax2.axhline(y, color=MUTED, lw=1, ls=":")
    ax1.set_ylabel("speed (m/s)")
    ax2.set_ylabel("offset e_c (m)\n(+ = left)")
    ax2.set_xlabel("progress along the lap s (m)")
    ax1.legend(frameon=False, ncol=4, loc="lower right")
    ax1.set_title("Speed and lateral offset along the lap (2nd lap, flying start)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lap_times(results, path):
    """Best lap per driver and track (horizontal bars, value labels)."""
    rows = [(f"{r['name']} ({r['track']})", r["metrics"].get("best_lap"), color_for(r["name"], i))
            for i, r in enumerate(results) if r["metrics"].get("best_lap")]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.5))
    _style(ax)
    ax.grid(axis="y", visible=False)
    ys = np.arange(len(rows))
    ax.barh(ys, [r[1] for r in rows], color=[r[2] for r in rows], height=0.6, edgecolor="white", linewidth=2)
    for y, (_, v, _) in zip(ys, rows):
        ax.text(v, y, f"  {v:.2f} s", va="center", color=INK)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("best lap time (s)")
    ax.set_title("Best lap time (lower is better)", color=INK)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
