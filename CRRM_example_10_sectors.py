# Keith Briggs 2025-09-17

import numpy as np
from sys import path

import CRRM


def go(n_it=100):
    crrm_parameters = CRRM.Parameters(
        n_cell_locations=7,
        n_ues=100,
        n_sectors=1,
        author="Keith Briggs",
        layout_plot_fnbase="img/CRRM_example_10_sectors_layout",
        label_ues_in_layout_plot=False,
    )
    # 1 sector...
    throughput_ave = 0.0
    for i in range(n_it):
        crrm = CRRM.Simulator(crrm_parameters)
        crrm.set_rng_seeds(i)
        throughputs = crrm.get_UE_throughputs()
        throughput_ave += (np.average(throughputs) - throughput_ave) / (1.0 + i)
    print(
        f"n_sectors={crrm_parameters.n_sectors} average UE throughput over {n_it} UE PPP layouts = {throughput_ave:.2f} Mb/s"
    )
    # 3 sectors...
    crrm.set_n_sectors(3)
    throughput_ave = 0.0
    for i in range(n_it):
        crrm = CRRM.Simulator(crrm_parameters)
        crrm.set_rng_seeds(i)
        throughputs = crrm.get_UE_throughputs()
        throughput_ave += (np.average(throughputs) - throughput_ave) / (1.0 + i)
    print(
        f"n_sectors={crrm_parameters.n_sectors} average UE throughput over {n_it} UE PPP layouts = {throughput_ave:.2f} Mb/s"
    )
    crrm.update()  # get current attachment vector
    crrm.layout_plot(show_attachment_type="attachment", no_ticks=True)


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=300, suppress=True)
    go()
