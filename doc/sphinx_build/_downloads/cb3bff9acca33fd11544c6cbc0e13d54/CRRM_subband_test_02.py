# Ibrahim Nur 2025-09-12
# python3 CRRM_subband_test_02.py
# Demo of subbanding

from sys import path
import numpy as np

from CRRM import Sim, Params, to_dB


def go():
    # We define a simple scenario: 2 cells and 1 UE exactly between them
    cell_locs = np.array([[-250.0, 0.0, 20.0], [250.0, 0.0, 20.0]])
    ue_loc = np.array([[0.0, 0.0, 1.6]])

    print("Case 1: n_subbands=1")
    params_case1 = Params(
        cell_locations=cell_locs,
        ue_initial_locations=ue_loc,
        n_subbands=1,
        # Both cells transmit 100W on the same band...
        power_matrix=[[100.0], [100.0]],
        display_flat=True,
    )
    crrm_case1 = Sim(params_case1)
    crrm_case1.update()

    sinr_case1 = crrm_case1.sinr.data[0, 0]  # UE 0, SB 0
    tp_case1 = crrm_case1.get_UE_throughputs(subbands=[0]).squeeze()  # shape=(1, 1)
    print(f"UE SINR:    {to_dB(sinr_case1):.2f} dB")
    print(f"UE Throughput:   {tp_case1:.2f} Mb/s")

    print("Case 2: n_subbands=2")
    params_case2 = Params(
        cell_locations=cell_locs,
        ue_initial_locations=ue_loc,
        n_subbands=2,
        # Cell 0 transmits on subband 0 only
        # Cell 1 transmits on subband 1 only
        power_matrix=[[100.0, 1.0], [1.0, 100.0]],
        display_flat=True,
    )
    crrm_case2 = Sim(params_case2)
    crrm_case2.update()

    sinrs_case2 = crrm_case2.sinr.data[0]  # UE 0, both subbands
    tps_case2 = crrm_case2.get_UE_throughputs(subbands=[0, 1])  # shape=(1, 2)

    print(
        f"UE SINR (on subband 0): {to_dB(sinrs_case2[0]):.2f} dB (note: high signal, no interference)"
    )
    print(
        f"UE SINR (on subband 1): {to_dB(sinrs_case2[1]):.2f} dB (note: minimal signal)"
    )
    print(f"UE Throughput: {np.sum(tps_case2):.2f} Mbps (note: from SB 0 only)")


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=300, suppress=True)
    go()
