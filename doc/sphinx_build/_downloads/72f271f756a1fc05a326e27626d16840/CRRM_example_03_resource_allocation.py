# Keith Briggs 2025-09-16
# p > 1: favours weak users (e.g., p=2 gives T ∝ 1/S).
# p = 1: results in equal throughput for all users on the cell.
# p < 1: favours strong users (e.g., p=0 gives T ∝ S).
# p = 0: proportional fair scheduling.
# p large and negative: maximize total throughout at the expense of weak users

import numpy as np
from sys import path
from CRRM import Parameters, Simulator, Logger


def go(n_ues=10, pmin=-1.5, pmax=3.0):
    crrm_parameters = Parameters(
        n_cell_locations=7,
        n_ues=n_ues,
        n_subbands=1,
        resource_allocation_fairness=pmin,
        layout_plot_fnbase="img/CRRM_example_03_resource_allocation_layout",
        label_ues_in_layout_plot=True,
        author="Keith Briggs",
        display_flat=True,
    )
    print(crrm_parameters)
    crrm = Simulator(crrm_parameters)
    crrm.scale_ue_locations("all", 0.5)
    crrm.layout_plot(show_pathloss_circles=True, show_attachment_type="attachment")
    crrm_log = Logger(crrm, captures=("tp",), ues=tuple(range(crrm_parameters.n_ues)))
    pmin, pmax = -1.5, 3.0
    ps = np.linspace(pmin, pmax, crrm_parameters.n_moves)
    for p in ps:
        crrm.set_resource_allocation_fairness(p)
        crrm_log.capture()  # this updates the data
    crrm_log.plot(
        fields=("tp",),
        fnbase="img/CRRM_example_03_resource_allocation",
        title=f"CRRM resource allocation n_cells={crrm_parameters.n_cells} n_ues={crrm_parameters.n_ues} $p=-1.5,\dots,3.0$",
        averages=("tp",),
        smooth_averages=True,
        x_axis=ps,
        xlabel="resource allocation fairness",
        linewidth=3,
        legend_fontsize=12,
        label_fontsize=14,
    )


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=300, suppress=True)
    go()
