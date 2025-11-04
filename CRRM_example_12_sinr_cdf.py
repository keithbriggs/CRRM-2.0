# Keith Briggs 2025-09-17
# check against slide 10 of ...
#   evince ~/Poisson_point_process/tex/Briggs_SIR_distributions.pdf &
# Reference: Haenggi, Stochastic Geometry for Wireless Networks, CUP 2013, section 5.2.

from time import strftime, localtime
from sys import path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta, beta, hyp2f1

from CRRM import (
    Parameters,
    Simulator,
    Logger,
    default_UE_locations,
    fig_timestamp,
    to_dB,
    from_dB,
)


def go():
    rng = np.random.default_rng(12345)
    crrm_parameters = Parameters(
        pathloss_model_name="power-law",
        pathloss_exponent=3.5,
        n_cell_locations=5000,
        n_ues=1000,
        σ2=0.0,
        rayleigh_fading=True,
    )
    crrm_parameters.cell_locations = default_UE_locations(
        rng,
        crrm_parameters.n_cell_locations,
        crrm_parameters.h_UT_default,
        system_area=1e4,
    )
    crrm_parameters.ue_initial_locations = default_UE_locations(
        rng, crrm_parameters.n_ues, crrm_parameters.h_UT_default, system_area=1e2
    )
    crrm_parameters.ue_locations = crrm_parameters.ue_initial_locations
    crrm_parameters.power_matrix = np.array([[crrm_parameters.p_W]])
    print(crrm_parameters)
    crrm = Simulator(crrm_parameters)
    crrm.sinr.update()
    ue_sinrs_dB = to_dB(np.squeeze(crrm.sinr.data))
    print(f"ue_sinrs_dB: {ue_sinrs_dB[:10]}")
    plot_cdf(
        ue_sinrs_dB,
        figurefnbase="img/CRRM_example_12_sinr_cdf",
        xlabel="$x$",
        ylabel="log$_{10}$Prob[SIR$>x$]",
        title="SIR distributions as computed by CRRM",
    )


def plot_cdf(x, figurefnbase, xlabel="", ylabel="", title=""):
    plt.rcParams.update({"font.size": 8, "figure.autolayout": True})
    x[::-1].sort()
    n = len(x)
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 1, 1)
    ax0.grid(linewidth=0.5, alpha=0.5)
    ax0.set_ylim(-3.0, 0.0)
    ax0.set_xlim(-15.0, 50.0)
    if title:
        ax0.set_title(title)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    y = np.log10(np.arange(1, 1 + n) / n)
    ax0.scatter(x, y, color="blue", s=1, label="simulation\n(with fading)")
    delta = 2.0 / 3.5
    x_dB = np.linspace(-30, 50, 41)
    y = np.log10(np.array([exact_SIR_ccdf(from_dB(q), delta) for q in x_dB]))
    ax0.plot(x_dB, y, "r-", lw=1, label="exact theory\n(with fading)")
    ax0.legend()
    fig.tight_layout()
    fig_timestamp(fig, author="Keith Briggs")
    pngfn = figurefnbase + ".png"
    fig.savefig(pngfn, dpi=200)
    print("eog %s &" % pngfn)
    pdffn = figurefnbase + ".pdf"
    fig.savefig(pdffn)
    print("evince %s &" % pdffn)


def exact_SIR_ccdf(theta, delta):
    # from ~/Poisson_point_process/SIR_distributions_2023_04.py
    # Prob[SIR>theta] for pathloss exponent gamma=2/delta and Rayleigh fading
    # Ganti & Haenggi SIR asymptotics (2015) eqn. (3)
    if delta == 0.5:
        return 1.0 / (1.0 + sqrt(theta) * atan(sqrt(theta)))
    if abs(theta) < 1.0:
        return 1.0 / hyp2f1(1.0, -delta, 1.0 - delta, -theta)
    z = -theta
    return (1.0 - z) / hyp2f1(1.0, 1.0, 1.0 - delta, z / (z - 1.0))  # A&S 15.3.4


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=500, suppress=False)
    go()
