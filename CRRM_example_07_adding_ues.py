# Keith Briggs 2025-09-16

from time import strftime, localtime
import numpy as np
from sys import path
from CRRM import Simulator, Parameters


def adding_ues_test(nmax=20):
    crrm_parameters = Parameters(
        cell_locations=np.array([[0.0, 0.0, 20.0]]),
        ue_initial_locations=np.array([[1e3, 0.0, 2.0]]),
    )
    print(crrm_parameters)
    crrm = Simulator(crrm_parameters)
    for i in range(2, 2 + nmax):
        crrm.update()
        print(f"crrm.a.data={crrm.a.data}")
        ue_throughputs = crrm.get_UE_throughputs(subbands=0)
        print(f"i={i} ue_throughput={ue_throughputs}")
        crrm.add_ue(np.array([[10.0 * i, 0.0, 2.0]]))


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=500, suppress=False)
    adding_ues_test()
