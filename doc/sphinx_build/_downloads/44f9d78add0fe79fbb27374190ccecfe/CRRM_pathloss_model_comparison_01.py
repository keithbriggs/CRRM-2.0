# Ibrahim Nur 2025-09-13
# python3 CRRM_pathloss_model_comparison_01.py
# Compares throughput-vs-distance for all available pathloss models

import numpy as np
from sys import path
import matplotlib.pyplot as plt
from CRRM import Sim, Params, fig_timestamp


def go():
    model_names = ("RMa", "UMa", "UMi", "power-law")
    dmin, dmax, resolution = 20.0, 10000.0, 50.0  # metres
    distances = np.arange(dmin, dmax, resolution)
    n_points = len(distances)
    h_bs, h_ut = 35.0, 1.8
    all_tps = []
    for model_name in model_names:
        print(f"Simulating {model_name}...")
        crrm_parameters = Params(
            cell_locations=np.array([[0.0, 0.0, h_bs]]),
            n_ues=1,
            ue_initial_locations=np.array([[dmin, 0.0, h_ut]]),
            h_BS_default=h_bs,
            h_UT_default=h_ut,
            pathloss_model_name=model_name,
            display_flat=True,
        )
        crrm = Sim(crrm_parameters)
        tp = np.empty(n_points)
        for i in range(n_points):
            # Get throughput at the current position, then move to the next position
            tp[i] = crrm.get_UE_throughputs()[0, 0]
            # Move radially along x-axis...
            crrm.move_ue_locations([0], np.array([resolution, 0.0]))
        all_tps.append(tp)

    plot(
        x_data=distances,
        y_data_list=all_tps,
        labels=model_names,
        figurefnbase="img/CRRM_pathloss_model_comparison",
        xlabel="distance (m)",
        ylabel="throughput (Mb/s)",
        title="",  # f'Throughput $vs$ distance by pathloss model'
    )


def plot(x_data, y_data_list, labels, figurefnbase, xlabel="", ylabel="", title=""):
    plt.rcParams.update({"font.size": 8, "figure.autolayout": True})
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 1, 1)
    ax0.grid(linewidth=0.5, alpha=0.5)
    ax0.set_title(title)
    ax0.set_xlim(min(x_data), max(x_data))
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    for y_data, label in zip(y_data_list, labels):
        ax0.plot(x_data, y_data, lw=1.5, label=label)
    ax0.legend(loc="lower left", shadow=True, framealpha=0.5)
    fig.tight_layout()
    fig_timestamp(fig, author="Ibrahim Nur")
    pngfn = figurefnbase + ".png"
    fig.savefig(pngfn, dpi=200)
    print(f"eog {pngfn} &")


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=500, suppress=True)
    go()
