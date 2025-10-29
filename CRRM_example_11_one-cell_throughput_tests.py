# Keith Briggs 2025-09-22

from time import strftime, localtime
import numpy as np
from sys import path
from CRRM import Parameters, Simulator, Logger, fig_timestamp, to_dB, from_dB


def ue_throughput_vs_distance_test():
    rng = np.random.default_rng(12345)
    crrm_parameters = Parameters(
        cell_locations=np.array([[0.0, 0.0, 20.0]]),
        n_ues=1,
        p_W=1.0e2,
        display_flat=True,
    )
    dmin, dmax, dd = 20.0, 3000.0, 10.0
    ds = np.arange(dmin, dmax, dd)
    nd = len(ds)
    tp = np.empty(nd)
    crrm_parameters.ue_locations = np.array([[dmin, 0.0, 2.0]])
    crrm = Simulator(crrm_parameters)
    crrm.set_power_matrix = np.array([[crrm_parameters.p_W]])
    print(crrm_parameters)
    d = 0.0
    for i in range(nd):
        d += dd
        crrm.move_ue_locations([0], np.array([dd, dd]))
        ue_throughput = crrm.get_UE_throughputs(0, 0)
        tp[i] = ue_throughput
    plot(
        ds,
        tp,
        figurefnbase="img/CRRM_example_11_one-cell_throughput_tests",
        xlabel="distance (m)",
        ylabel="throughput (Mb/s)",
        title="single-UE throughput predicted by CRRM_core_05 (UMa, NLOS)",
    )


# END def ue_throughput_vs_distance_test


def ue_throughput_vs_number_test(equidistant=True):
    # FIXME use add_ue method
    rng = np.random.default_rng(12345)
    ue_locations = np.empty((0, 3))
    nmax = 100
    x = []
    y = []
    for i in range(1, nmax + 1):
        if equidistant:
            new_ue_location = np.array([500.0, 0.0, 1.8])
        else:
            new_ue_location = np.array([500.0 + 100.0 * i, 0.0, 1.8])
        ue_locations = np.vstack([ue_locations, new_ue_location])
        crrm_parameters = Parameters(
            n_cell_locations=1,
            n_ues=1,
            p_W=1.0e2,
            display_flat=True,
        )
        crrm_parameters.cell_locations = np.array([[0.0, 0.0, 20.0]])
        crrm_parameters.ue_initial_locations = ue_locations
        crrm = Simulator(crrm_parameters)
        crrm.ue_locations.data = ue_locations
        crrm.tp.update()
        ue_throughputs = crrm.get_UE_throughputs()[:, 0]
        print(f"i={i} total throughput={np.sum(ue_throughputs):.1f}")
        x.append(
            [
                i,
            ]
            * i
        )
        y.append(ue_throughputs)
    z = y[0] / np.array(list(range(1, nmax + 1)))  # 1/n
    if equidistant:
        title = "UE throughput predicted by CRRM: equidistant UEs"
        figurefnbase = (
            "img/CRRM_example_11_one-cell_throughput_tests_vs_number_equal_distance"
        )
    else:
        title = "UE throughput predicted by CRRM: non-equidistant UEs"
        figurefnbase = "img/CRRM_example_11_one-cell_throughput_tests_unequal_distance"
    plot(
        x,
        y,
        z,
        figurefnbase=figurefnbase,
        xlabel="$n$ = number of UEs",
        ylabel="throughput per UE (Mb/s)",
        title=title,
        legend_ylabel="CRRM",
        legend_zlabel="$1/n$",
        log_xscale=True,
        log_yscale=True,
        ymin=1e0,
    )


# END def ue_throughput_vs_number_test


def plot(
    x,
    y,
    z=None,
    figurefnbase="",
    xlabel="",
    ylabel="",
    title="",
    ymin=0.0,
    legend_ylabel="",
    legend_zlabel="",
    log_yscale=False,
    log_xscale=False,
):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "figure.autolayout": True})
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 1, 1)
    ax0.grid(linewidth=0.5, alpha=0.5)
    try:
        ax0.set_xlim(x[0][0], x[-1][0])
        ax0.set_ylim(min(ymin, min(y[0][0], y[-1][0])), max(y[0][0], y[-1][0]))
    except:
        ax0.set_xlim(x[0], x[-1])
        ax0.set_ylim(min(ymin, min(y[0], y[-1])), max(y[0], y[-1]))
    if title:
        ax0.set_title(title)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    if log_xscale:
        ax0.set_xscale("log")
    if log_yscale:
        ax0.set_yscale("log")
    for xi, yi in zip(x, y):
        ax0.scatter(xi, yi, color="blue", s=1, label=legend_ylabel)
    if z is not None:
        ax0.plot([xi[0] for xi in x], z, color="red", lw=0.5, label=legend_zlabel)
    if legend_ylabel:
        # don't repeat labels in legend...
        handles, labels = ax0.get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        ax0.legend(handles, labels, loc="upper right", shadow=True, framealpha=0.5)
    fig.tight_layout()
    fig_timestamp(fig, author="Keith Briggs")
    pngfn = figurefnbase + ".png"
    fig.savefig(pngfn, dpi=200)
    print("eog %s &" % pngfn)


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=500, suppress=False)
    ue_throughput_vs_distance_test()
    ue_throughput_vs_number_test(equidistant=True)
    ue_throughput_vs_number_test(equidistant=False)
