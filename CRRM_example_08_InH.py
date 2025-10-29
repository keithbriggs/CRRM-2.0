# Ibrahim Nur 2025-09-15

from numpy import abs as np_abs
from CRRM import Simulator, Parameters


def go():
    # One UE at midpoint between two cells
    crrm_parameters = Parameters(
        n_ues=1,
        pathloss_model_name="InH",
        ue_initial_locations=[[100.0, 0.0, 1.5]],
        cell_locations=[[0.0, 0.0, 20.0], [200, 0.0, 20.0]],
        p_W=2,
    )
    crrm = Simulator(crrm_parameters)
    crrm.update()
    ue_throughput = crrm.get_UE_throughputs()
    distance = np_abs(
        crrm_parameters.cell_locations[1, 0] - crrm_parameters.cell_locations[0, 0]
    )
    print(
        f"throughput for one UE at midpoint between two cells spaced "
        f"{distance:.0f}m "
        f"from each other using InH is {ue_throughput[0,0]} Mb/s"
    )


if __name__ == "__main__":
    go()
