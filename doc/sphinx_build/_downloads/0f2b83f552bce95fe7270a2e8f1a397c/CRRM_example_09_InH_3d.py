# python3 InH_test_00.py
# Test script for InH (Indoor-Hotspot) pathloss model.

from sys import path

import numpy as np
import matplotlib.pyplot as plt

from CRRM import Simulator, Parameters

path.append("./CRRM/")
from geometry_3d import Building, block, draw_building_3d


def run_inh_house_simulation():
    n_ues = 5
    n_steps = 200
    Δt = 0.05
    parameters = Parameters(
        n_cell_locations=2,
        n_ues=n_ues,
        p_W=1.0,
        fc_GHz=3.5,
        pathloss_model_name="InH",
        LOS=True,
        cell_locations=np.array([[1, 45, 20], [45, 1, 20]]),
        ue_initial_locations=np.array(
            [
                [1.0, 2.0, 2],
                [2.0, 3.0, 2],
                [3.0, 2.0, 2],
                [4.0, 3.0, 2],
                [5.0, 2.0, 2],
            ]
        ),
    )
    crrm = Simulator(parameters)
    house_bounds_x = [0.0, 50.0]
    house_bounds_y = [0.0, 50.0]
    floor_heights_z = [2, 10]
    v = np.zeros((n_ues, 3))
    v[:, 0] = crrm.get_rngs(0).uniform(-10, 10, size=(n_ues))
    v[:, 1] = crrm.get_rngs(0).uniform(-10, 10, size=(n_ues))
    ue_floor_i = np.zeros(n_ues, dtype=int)
    house_geom = Building(
        block(
            (house_bounds_x[0], house_bounds_y[0], 0.0),
            (house_bounds_x[1], house_bounds_y[1], 12.0),
        )
    )
    plot_limits_3d = [
        (house_bounds_x[0] - 2, house_bounds_x[1] + 2),
        (house_bounds_y[0] - 2, house_bounds_y[1] + 2),
        (0, 14),
    ]
    ue_move_events = {}
    for step in range(n_steps):
        ue_move_events.clear()
        s = np.copy(crrm.ue_locations.data)
        v[:, 0] += crrm.get_rngs(0).normal(scale=0.5, size=(n_ues))
        v[:, 1] += crrm.get_rngs(0).normal(scale=0.5, size=(n_ues))
        Δs = v * Δt
        s_plus_Δs = s + Δs
        for i in range(n_ues):
            if not (house_bounds_x[0] < s_plus_Δs[i, 0] < house_bounds_x[1]):
                v[i, 0] *= -1.0
            if not (house_bounds_y[0] < s_plus_Δs[i, 1] < house_bounds_y[1]):
                v[i, 1] *= -1.0
            if crrm.get_rngs(0).random() < 0.05:
                current_floor_i = ue_floor_i[i]
                target_floor_i = 1 - current_floor_i
                ue_floor_i[i] = target_floor_i
                s_plus_Δs[i, 2] = floor_heights_z[target_floor_i]
                ue_move_events[i] = {"start": s[i], "end": s_plus_Δs[i]}
                print(
                    f"At time t={(step+1)*Δt:.1f}, UE {i} decided to move {'upstairs' if target_floor_i > current_floor_i else 'downstairs'}"
                )
        Δs[:, 2] = s_plus_Δs[:, 2] - s[:, 2]
        crrm.move_ue_locations(range(n_ues), Δs)
        crrm.update()
        throughputs = crrm.get_UE_throughputs(subbands=0)
        draw_building_3d(
            house_geom,
            dots=crrm.ue_locations.data,
            limits=plot_limits_3d,
            show=False,
            pngfn="",
        )
        fig = plt.gcf()
        ax = plt.gca()
        cells = crrm.cell_locations.data
        ax.plot(
            cells[:, 0],
            cells[:, 1],
            cells[:, 2],
            "r^",
            markersize=12,
            label="Cells",
            zorder=899,
        )
        ue_locations = crrm.ue_locations.data
        for i, ue_loc in enumerate(ue_locations):
            floor = 1 if ue_loc[2] == floor_heights_z[0] else 2
            label = f"F{floor}\n{throughputs[i]:.1f} Mb/s"
            ax.text(
                ue_loc[0],
                ue_loc[1],
                ue_loc[2] + 0.5,
                label,
                ha="center",
                fontsize=7,
                color=("blue" if floor == 1 else "red"),
                zorder=900,
            )
        for i, event in ue_move_events.items():
            start = event["start"]
            end = event["end"]
            ax.quiver(
                start[0],
                start[1],
                start[2],
                end[0] - start[0],
                end[1] - start[1],
                end[2] - start[2],
                color=("blue" if floor == 1 else "red"),
                arrow_length_ratio=0.6,
                linewidth=1.5,
                linestyle="solid",
                zorder=1000,
            )
        fig.savefig(f"ani/frame_{step:04d}.png")
        plt.close()


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=300, suppress=True)
    run_inh_house_simulation()
    print(
        f"ffmpeg -y -loglevel quiet -framerate 15 -pattern_type glob -i 'ani/*.png' -c:v libx264 -crf 18 -pix_fmt yuv420p mp4/InH_test_ani.mp4 && xdg-open mp4/InH_test_ani.mp4"
    )
