# Keith Briggs 2025-09-24
# bash use_most_recent_python.sh CRRM_large_system_timing_tests.py

from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from CRRM import Simulator, Parameters, fig_timestamp


def go(params):
    sim = Simulator(params)
    fraction_to_move = 0.1
    rng = sim.rngs[0]
    # pick the indices from this...
    choice_tuple = tuple(range(params.n_ues))
    move_size = int(params.move_fraction * params.n_ues)
    # we don't really need to move, only mark the UEs as if they had moved...
    locations = np.zeros((move_size, 3))
    t0 = perf_counter()
    for i in range(params.n_moves):
        indices = rng.choice(choice_tuple, size=move_size, replace=False).tolist()
        sim.move_ue_locations(indices, locations)
        sim.update()
    return perf_counter() - t0


def plot(n_uess, n_cellss, d, title="", fnbase="img/CRRM_large_system_timing_tests"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    ax.grid(color="gray", lw=0.5, alpha=0.5)
    for i, n_cells in enumerate(n_cellss):
        x = d[0, i][0]
        y = [d[j, i][1] for j, n_ues in enumerate(n_uess)]
        ax.scatter(x, y, marker="o", s=10, label=f"{n_cells:4d}")
        ax.plot(x, y, linestyle="dotted", lw=1, alpha=0.5)
    if title:
        ax.set_title(title)
    ax.set_xlim(0, 2000)
    ax.set_xlabel("number of UEs")
    ax.set_ylabel("computation time (seconds)")
    ax.legend(title="number of cells")
    fig.tight_layout()
    fig_timestamp(fig, rotation=0, fontsize=6, author="Keith Briggs")
    if fnbase:
        fig.savefig(f"{fnbase}.png", dpi=300)
        print(f"eog {fnbase}.png &")
        fig.savefig(f"{fnbase}.pdf")
        print(f"evince {fnbase}.pdf &")
    else:
        plt.show()


def main(max_n_ues=2000, move_fraction=0.1):
    n_cellss = (
        1000,
        500,
        200,
        100,
    )
    d = {}
    for i, n_cells in enumerate(n_cellss):
        n_uess = np.linspace(1, max_n_ues, num=20, endpoint=True)
        for j, n_ues in enumerate(n_uess):
            print(f"n_cells={n_cells:4} n_ues={int(n_ues):5d}: ", end="")
            params = Parameters(
                n_cell_locations=n_cells,
                n_ues=int(n_ues),
                move_fraction=move_fraction,
                n_moves=100,
            )
            dt = go(params)
            print(f"dt={dt:.2f}")
            d[j, i] = (
                n_uess,
                dt,
            )
    plot(
        n_uess,
        n_cellss,
        d,
        title=f"CRRM: time for {params.n_moves} moves of {100.0*params.move_fraction:.0f}% of UEs",
    )


if __name__ == "__main__":
    main()
