# Keith Briggs 2025-09-09 use _set methods
# Keith Briggs 2025-09-06 adapt for CRRM-1.2
# Keith Briggs 2025-09-02 adapt for CRRM-1.1
# Keith Briggs 2025-08-27 v01
# Ibrahim Nur  2025-08-27 v00
# python3.11 CRRM_mini-example_00.py

from sys import path

path.append("../CRRM-1.2/src/")
from CRRM_core_07 import CRRM, CRRM_parameters


def go():
    crrm_parameters = CRRM_parameters(n_ues=1)
    crrm = CRRM(crrm_parameters)
    crrm.set_cell_locations([[0.0, 0.0, 20.0]])
    crrm_parameters.set_ue_initial_locations([[2500.0, 0.0, 1.6]])
    crrm.update()
    ue_throughput = crrm.get_UE_throughputs()
    print(f"throughput for one UE = {ue_throughput} Mb/s")


if __name__ == "__main__":
    go()
