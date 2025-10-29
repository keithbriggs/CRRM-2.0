# Keith Briggs 2025-09-24
# Ibrahim Nur 2025-09-22
# A minimal demonstration of fading on a cell-edge user.

from sys import stdout

import numpy as np
import matplotlib.pyplot as plt

from CRRM import Simulator, Parameters


def go():
    n_runs = 100
    distances = np.linspace(100, 1000, 100)
    tps_no_fading = []
    tps_shadow_fading = []
    tps_rayleigh_fading = []
    tps_both_fading = []
    print(f"distance = ", end="")
    for d in distances:
        print(f"{d:.0f}m...", end="")
        ue_loc = [[d, 0.0, 1.8]]
        cell_loc = [[0.0, 0.0, 0.0]]
        crrm_parameters_0 = Parameters(
            n_ues=1,
            n_cell_locations=1,
            cell_locations=cell_loc,
            ue_initial_locations=ue_loc,
            shadow_fading=False,
            rayleigh_fading=False,
        )
        crrm_0 = Simulator(crrm_parameters_0)
        tps_no_fading.append(crrm_0.get_UE_throughputs(0, 0))
        tps_shadow_fading.append(
            np.mean(
                [
                    Simulator(
                        Parameters(
                            n_ues=1,
                            n_cell_locations=1,
                            cell_locations=cell_loc,
                            ue_initial_locations=ue_loc,
                            shadow_fading=True,
                            rayleigh_fading=False,
                            rng_seeds=s,
                        )
                    ).get_UE_throughputs(0, 0)
                    for s in range(n_runs)
                ]
            )
        )
        tps_rayleigh_fading.append(
            np.mean(
                [
                    Simulator(
                        Parameters(
                            n_ues=1,
                            n_cell_locations=1,
                            cell_locations=cell_loc,
                            ue_initial_locations=ue_loc,
                            shadow_fading=False,
                            rayleigh_fading=True,
                            rng_seeds=s,
                        )
                    ).get_UE_throughputs(0, 0)
                    for s in range(n_runs)
                ]
            )
        )
        tps_both_fading.append(
            np.mean(
                [
                    Simulator(
                        Parameters(
                            n_ues=1,
                            n_cell_locations=1,
                            cell_locations=cell_loc,
                            ue_initial_locations=ue_loc,
                            shadow_fading=True,
                            rayleigh_fading=True,
                            rng_seeds=s,
                        )
                    ).get_UE_throughputs(0, 0)
                    for s in range(n_runs)
                ]
            )
        )
        stdout.flush()
    print()
    plt.plot(
        distances,
        tps_no_fading,
        label="No fading",
        color="black",
        linestyle="--",
        linewidth=2,
        zorder=5,
    )
    plt.plot(distances, tps_shadow_fading, label="Shadow fading")
    plt.plot(distances, tps_rayleigh_fading, label="Rayleigh fading")
    plt.plot(distances, tps_both_fading, label="Both")
    plt.xlabel("Distance (m)")
    plt.ylabel("Throughput (Mb/s)")
    plt.grid(True)
    plt.legend()
    plt.xlim(100, 1000)
    plt.ylim(bottom=0)
    pngfn = "img/CRRM_example_14_fading_tests.png"
    plt.savefig(pngfn, dpi=200)
    print("eog %s &" % pngfn)


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=300, suppress=True)
    go()
