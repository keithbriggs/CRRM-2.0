# Keith Briggs 2025-09-17
# Runs several tests, including the making of an amimation.

from sys import path, stdout, exit, argv, version as python_version
from os.path import basename
from array import array as array_array
from time import perf_counter
import numpy as np

from CRRM import (
    Simulator,
    Parameters,
    Logger,
    cyan,
    blue,
    green,
    red,
    bright_yellow,
    to_dB,
    move_ues_Gaussian,
)


def test_00(crrm_parameters):
    crrm = Simulator(crrm_parameters)
    power_matrix = crrm.get_power_matrix()
    crrm.set_power_matrix(
        power_matrix * np.power(0.5, np.arange(crrm_parameters.n_subbands))
    )
    print(crrm_parameters)
    crrm.layout_plot(show_pathloss_circles=True)
    if crrm_parameters.verbose:
        print(blue(f"before moves:"))
        print(blue(f"sinr=      {to_dB(crrm.sinr.data[:20])} dB ..."))
        print(blue(f"se_Shannon={crrm.se_Shannon.data[:20]} ..."))
    # set up the data capture...
    logger = Logger(crrm, captures=("cqi", "mcs", "sinr", "se_Shannon", "tp"), ues=(0,))
    # we will pick from size_tuple to decide how many UEs to move...
    size_tuple = tuple(
        range(1, int(crrm_parameters.move_fraction * crrm_parameters.n_ues))
    )
    if size_tuple == ():
        size_tuple = (1,)
    # pick the indices from this...
    choice_tuple = tuple(range(crrm_parameters.n_ues))
    # now move some UEs...
    print(bright_yellow(f"moves: "), end="")
    t0 = perf_counter()
    if crrm_parameters.frame_interval:
        n_frames = crrm_parameters.n_moves // crrm_parameters.frame_interval
        print(f"animation will do {n_frames} frames")
    for j in range(crrm_parameters.n_moves):
        crrm.j = j
        if j % 1000 == 0:
            mean_ue_distance = crrm.check_ue_locations()
            print(bright_yellow(f"{j} ({mean_ue_distance:.0f}m)"), end="... ")
            stdout.flush()
        move_ues_Gaussian(crrm, size_tuple, choice_tuple)
        crrm.update()
        logger.capture()
        if crrm_parameters.frame_interval and j % crrm_parameters.frame_interval == 0:
            jf = j // crrm_parameters.frame_interval
            percent_done = 100.0 * jf / n_frames
            print(f"frame {jf} ({percent_done:.1f}%)", end=" ")
            stdout.flush()
            crrm_parameters.layout_plot_fnbase = f"ani/CRRM_layout_animation_{jf:05d}"
            crrm_parameters.author = f"{jf:03d}"
            crrm.layout_plot(
                show_voronoi=True,
                show_kilometres=False,
                show_attachment_type="attachment",
                no_ticks=True,
                show_pathloss_circles=False,
                fmt=("png",),
                dpi=100,
                quiet=True,
                figsize=(6, 6),
            )
    if crrm_parameters.frame_interval:
        print(f"animation done.")
    assert crrm.d.check(crrm.ue_locations, crrm.cell_locations)
    assert crrm.g.check()
    print(bright_yellow(f"{j}"))
    stdout.flush()
    dt = perf_counter() - t0
    print(
        green(
            f"n_cell_locations={crrm_parameters.n_cell_locations} n_ues={crrm_parameters.n_ues}: {crrm_parameters.n_moves} UE moves done in {dt:.2f} seconds."
        )
    )
    logger.plot(
        fields=("cqi", "mcs", "sinr", "se_Shannon", "tp"),
        fnbase=f"img/CRRM_example_13_timing_tests_job{crrm_parameters.job}",
        title=f"CRRM test job={crrm_parameters.job}",
    )
    if crrm_parameters.frame_interval:
        print(
            "ffmpeg -y -loglevel quiet -framerate 15 -pattern_type glob -i 'ani/*.png' -c:v libx264 -crf 18 -pix_fmt yuv420p crrm_animation.mp4 && xdg-open crrm_animation.mp4"
        )
    print(blue(f"after moves:"))
    subband_to_print = 0
    sinr = to_dB(crrm.sinr.data[:, subband_to_print])
    se_Shannon = crrm.se_Shannon.data[:, subband_to_print]
    print(blue(f"sinr for subband[{subband_to_print}]=      {sinr[:20]} dB"))
    print(blue(f"se_Shannon for subband[{subband_to_print}]={se_Shannon[:20]}"))
    return dt, sinr, se_Shannon


# END def test_00


def main(job, rng_seed):
    layout_plot_fnbase = "img/CRRM_example_13_timing_tests_layout"
    crrm_parameters = Parameters(
        n_cell_locations=19,
        n_ues=100,
        n_subbands=1,
        n_sectors=1,
        UE_layout="ppp",
        pathloss_model_name="RMa_discretised",
        p_W=1.0e2,
        h_UT_default=1.8,
        h_BS_default=20.0,
        author="Keith Briggs",
        rng_seeds=rng_seed,
        n_moves=5000,
        smart_update=True,
        verbose=False,
        move_fraction=0.05,
        layout_plot_fnbase=layout_plot_fnbase,
        move_mean=0.0,
        move_stdev=50.0,
        frame_interval=0,
        job=job,
    )
    if job == "0":  # standard test example - checks smart vs. not smart
        dt0, sinr0, se_Shannon0 = test_00(crrm_parameters)
        crrm_parameters.smart_update = False
        dt1, sinr1, se_Shannon1 = test_00(crrm_parameters)
        if np.allclose(sinr0, sinr1) and np.allclose(se_Shannon0, se_Shannon1):
            print(
                green(
                    f"The smart and the non-smart results match! 😁 speed-up factor={dt1/dt0:.2f}"
                )
            )
        else:
            print(red(f"smart and non-smart results do not match! 😭"))
    elif job == "a":  # animation test
        crrm_parameters.frame_interval = 20
        crrm_parameters.n_moves = 1000
        dt1, sinr1, se_Shannon1 = test_00(crrm_parameters)
    elif job == "1":  # small standard test example
        crrm_parameters.n_moves = 10000
        test_00(crrm_parameters)
    elif job == "1s":  # small standard test example with sectors
        crrm_parameters.n_moves = 10000
        crrm_parameters.n_sectors = 3
        test_00(crrm_parameters)
    elif job == "2":  # large test example
        crrm_parameters.n_cell_locations = 37
        crrm_parameters.n_ues = 500
        crrm_parameters.n_moves = 10000
        crrm_parameters.layout_plot_fnbase = f"{layout_plot_fnbase}_large"
        test_00(crrm_parameters)
    elif job == "3":  # huge test example
        crrm_parameters.n_cell_locations = 61
        crrm_parameters.n_ues = 1000
        crrm_parameters.n_moves = 10000
        crrm_parameters.layout_plot_fnbase = f"{layout_plot_fnbase}_huge"
        test_00(crrm_parameters)
    elif job == "4":  # HUGE test example
        crrm_parameters.n_cell_locations = 2000
        crrm_parameters.n_ues = 20000
        crrm_parameters.layout_plot_fnbase = f"{layout_plot_fnbase}_HUGE"
        test_00(crrm_parameters)
    else:  # tiny debug test
        crrm_parameters.n_cell_locations = 7
        crrm_parameters.n_ues = 15
        crrm_parameters.n_moves = 3
        crrm_parameters.layout_plot_fnbase = f"{layout_plot_fnbase}_tiny"
        test_00(crrm_parameters)


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=500, suppress=False)
    print(
        f'python_version={python_version[:python_version.index(" ")]} numpy version={np.__version__}'
    )
    job = "0"
    if len(argv) > 1:
        job = argv[1]
    main(job, rng_seed=12345)
