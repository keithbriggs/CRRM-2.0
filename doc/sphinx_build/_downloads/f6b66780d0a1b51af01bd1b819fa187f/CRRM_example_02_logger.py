# Keith Briggs 2025-09-16

from sys import path, exit
import numpy as np

from CRRM import Parameters, Simulator, Logger


def go():
    crrm_parameters = Parameters(
        n_cell_locations=3,
        ue_initial_locations=np.array([[200.0, 0.0, 1.5], [0.0, 100.0, 1.5]]),
        n_subbands=2,
        n_moves=1000,
        move_stdev=10.0,
        author="Keith Briggs",
        layout_plot_fnbase="img/CRRM_example_02_logger_layout",
        power_matrix=[[100.0, 10.0], [100.0, 50.0], [100.0, 100.0]],
        display_flat=True,
    )
    print(crrm_parameters)
    crrm = Simulator(crrm_parameters)
    print(f"power_matrix=\n{crrm.get_power_matrix()}")
    crrm.layout_plot()
    logger = Logger(
        crrm,
        captures=("sinr", "cqi", "mcs", "se_Shannon", "se_from_mcs", "tp"),
        ues=(0, 1),
    )
    n_ues = crrm.params.n_ues
    move_indices = list(range(n_ues))  # move all UEs
    for i in range(crrm_parameters.n_moves):
        logger.capture()
        deltas = crrm.params.move_stdev * crrm.rngs[0].standard_normal(size=(n_ues, 2))
        crrm.move_ue_locations(move_indices, deltas)
        crrm.update()
    logger.plot(
        fields=("cqi", "mcs", "sinr", "se_Shannon", "tp"),
        fnbase="img/CRRM_example_02_logger",
        title=f"CRRM logger n_cells={crrm_parameters.n_cells} n_ues={crrm_parameters.n_ues}",
        averages=("tp",),
        smooth_averages=True,
    )


if __name__ == "__main__":
    go()
