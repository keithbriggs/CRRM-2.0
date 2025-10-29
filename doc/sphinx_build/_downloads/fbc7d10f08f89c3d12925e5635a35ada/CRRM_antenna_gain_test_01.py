# Ibrahim Nur 2025-09-15
# python3 CRRM_antenna_gain_test_01.py
# Demonstrates the effect of 3-sector antenna gain patterns on UE throughput

import numpy as np
from sys import path
import matplotlib.pyplot as plt
from CRRM import Sim, Params, fig_timestamp


def go():
    # Define simulation parameters for a UE moving in a circle
    r_m = 1000.0  # radius of circular path in metres
    n_steps = 360  # number of points on the circle (one per degree)
    h_bs, h_ut = 35.0, 1.8
    # Generate the circular path coordinates and angles for plotting
    thetas = np.linspace(-np.pi, np.pi, n_steps, endpoint=False)
    xs = r_m * np.cos(thetas)  # polar to cartesian
    ys = r_m * np.sin(thetas)
    degrees = np.rad2deg(thetas)
    all_tps = []
    labels = []
    for n_sec in [1, 3]:  # plotting both 1 (omnidirectional) and 3 sector cases
        labels.append(f"{n_sec}-sector")
        crrm_parameters = Params(
            cell_locations=np.array([[0.0, 0.0, h_bs]]),
            n_ues=1,
            ue_initial_locations=np.array([[r_m, 0.0, h_ut]]),
            n_sectors=n_sec,
            h_BS_default=h_bs,
            h_UT_default=h_ut,
            display_flat=True,
        )
        crrm = Sim(crrm_parameters)
        tp = np.empty(n_steps)
        for i in range(n_steps):
            pos = np.array([[xs[i], ys[i], h_ut]])
            crrm.set_ue_locations([0], pos)
            tp[i] = crrm.get_UE_throughputs()[0, 0]
        all_tps.append(tp)
    plot(
        x_data=degrees,
        y_data_list=all_tps,
        labels=labels,
        figurefnbase="img/CRRM_antenna_gain_test",
        xlabel="UE angle (degrees)",
        ylabel="throughput (Mb/s)",
        title="",  # f'Antenna pattern throughputs (UE at r={r_m:.0f}m)'
    )


def plot(x_data, y_data_list, labels, figurefnbase, xlabel="", ylabel="", title=""):
    plt.rcParams.update({"font.size": 8, "figure.autolayout": True})
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 1, 1)
    ax0.grid(linewidth=0.5, alpha=0.5)
    ax0.set_title(title)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    for y_data, label in zip(y_data_list, labels):
        ax0.plot(x_data, y_data, lw=1.5, label=label)
    ax0.legend(shadow=True, framealpha=0.5)
    ax0.set_xlim(-180, 180)
    fig.tight_layout()
    fig_timestamp(fig, author="Ibrahim Nur")
    pngfn = figurefnbase + ".png"
    fig.savefig(pngfn, dpi=200)
    print(f"eog {pngfn} &")


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=500, suppress=True)
    go()
