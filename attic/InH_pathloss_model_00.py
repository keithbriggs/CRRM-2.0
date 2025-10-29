# Keith Briggs 2025-09-07
# Ibrahim Nur 2025-09-05

import numpy as np
from sys import path
from utilities import from_dB, to_dB


class InH_pathloss:
    """
    Indoor Hotspot (InH) pathloss model from 3GPP TR 38.901, Table 7.4.1-1.

    This model covers the cases 3D-InH LOS and NLOS.

    References:
    - 3GPP TR 38.901: https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173

    Parameters
    ----------
    fc_GHz : float
        Centre frequency in gigahertz.
    h_UT : float, optional
        Default User Terminal height for plotting function (not used in calculation).
    h_BS : float, optional
        Default Base Station height for plotting function (not used in calculation).
    LOS : bool, optional
        Whether the line-of-sight model is to be used (default is True).

    """

    def __init__(s, fc_GHz=3.5, h_UT=1.5, h_BS=3.0, LOS=True, **args):
        """
        Initialise a pathloss model instance.
        """
        s.fc_GHz = fc_GHz
        s.LOS = LOS
        s.default_h_UT = h_UT
        s.default_h_BS = h_BS
        s.los_const = (10**3.24) * np.power(s.fc_GHz, 2.0)
        s.nlos_const = (10**1.73) * np.power(s.fc_GHz, 2.49)

    def get_pathloss_dB(s, d2D_m, d3D_m, U, C):
        return to_dB(s.get_pathloss(d2D_m, d3D_m, U, C))  # retained for compatibility

    def get_pathloss(s, d2D_m, d3D_m, U, C):
        if np.any(d3D_m < 1.0) or np.any(d3D_m > 150.0):
            raise ValueError(
                f"At least one d3D_m value is outside the valid InH range [1.0, 150.0]m"
            )
        pl_los_linear = s.los_const * np.power(d3D_m, 1.73)
        if s.LOS:
            return pl_los_linear
        else:
            pl_nlos_prime_linear = s.nlos_const * np.power(d3D_m, 3.83)
            return np.maximum(pl_los_linear, pl_nlos_prime_linear)

    def get_pathgain(s, d2D_m, d3D_m, U, C):
        return 1.0 / s.get_pathloss(d2D_m, d3D_m, U, C)

    def _get_approximate_pathloss_dB_for_layout_plot(s, d: float):
        # FIXME
        d_flat = d.flatten()
        num_pts = len(d_flat)
        U = np.column_stack(
            (d_flat, np.zeros(num_pts), np.full(num_pts, s.default_h_UT))
        )
        C = np.array([[0.0, 0.0, s.default_h_BS]])
        d3D_m = np.linalg.norm(U[:, np.newaxis] - C[np.newaxis], axis=2)
        d2D_m = d3D_m  # this value isn't used

        return s.get_pathloss_dB(d2D_m, d3D_m, U, C).squeeze()


# END class InH_pathloss


def plot_InH_pathloss_or_pathgain(
    plot_type="pathloss",
    fc_GHz=3.5,
    h_UT=1.5,
    h_BS=3.0,
    zoom_box=False,
    print_10m_pl=False,
    author=" ",
    x_min=1.0,
    x_max=120.0,
):
    """
    Plot the 3GPP InH pathloss or pathgain model predictions as a self-test.

    This function generates a plot of the 3GPP InH pathloss or pathgain models
    for both LOS (Line-of-Sight) and NLOS (Non-Line-of-Sight) scenarios. It
    also includes a free-space pathloss reference and an optional zoomed-in
    view of the plot.

    Parameters
    ----------
    plot_type : str, optional
      Type of plot to generate. Options are:
      - 'pathloss': Plot the pathloss in dB (default).
      - 'pathgain': Plot the pathgain.
    fc_GHz : float, optional
      Carrier frequency in GHz (default is 3.5 GHz).
    h_UT : float, optional
      Height of the User Terminal (UE) in meters (default is 1.5 m).
    h_BS : float, optional
      Height of the Base Station (BS) in meters (default is 3.0 m).
    zoom_box : bool, optional
      If True, include a zoomed-in view of the plot (default is False).
    print_10m_pl : bool, optional
      If True, print the pathloss values at 10 meters for LOS, NLOS, and
      free-space scenarios (default is False).
    author : str, optional
      Author name to include in the plot timestamp (default is an empty string).
    x_min : float, optional
      Minimum x-axis value for the plot, representing the minimum distance
      in meters (default is 1.0 m).
    x_max : float, optional
      Maximum x-axis value for the plot, representing the maximum distance
      in meters (default is 120.0 m).

    Raises
    ------
    ImportError
      If required modules (e.g., matplotlib) are not installed.

    Notes
    -----
    - The function uses the `InH_pathloss` class to compute pathloss and
      pathgain values for LOS and NLOS scenarios.
    - The free-space pathloss is included as a reference.
    - The zoomed-in view highlights a specific region of the plot for better
      visualisation of details.
    """
    try:
        import matplotlib.pyplot as plt
        from utilities import fig_timestamp
    except ImportError as e:
        print(f"Error importing modules: {e}")
        raise

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(color="gray", alpha=0.5, lw=0.5)

    x = np.linspace(x_min, x_max, 500)

    PL_nlos = InH_pathloss(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS, LOS=False)
    PL_NLOS_dB = PL_nlos._get_approximate_pathloss_dB_for_layout_plot(x)

    if plot_type == "pathloss":
        ax.set_title(f"3GPP InH Pathloss Model")
        ax.set_ylabel("Pathloss (dB)")
        line = ax.plot(x, PL_NLOS_dB, lw=2, label=r"NLOS ($\sigma=8.03$)", color="blue")
        ax.fill_between(
            x,
            PL_NLOS_dB - 8.03,
            PL_NLOS_dB + 8.03,
            color=line[0].get_color(),
            alpha=0.2,
        )
    else:
        ax.set_title(f"3GPP InH Pathgain Model")
        ax.set_ylabel("Pathgain")
        PG_NLOS = from_dB(-PL_NLOS_dB)
        ax.plot(x, PG_NLOS, lw=2, label="NLOS Pathgain")

    PL_los = InH_pathloss(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS, LOS=True)
    PL_LOS_dB = PL_los._get_approximate_pathloss_dB_for_layout_plot(x)

    if plot_type == "pathloss":
        line = ax.plot(x, PL_LOS_dB, lw=2, label=r"LOS ($\sigma=3$)", color="orange")
        ax.fill_between(
            x, PL_LOS_dB - 3.0, PL_LOS_dB + 3.0, color=line[0].get_color(), alpha=0.2
        )
    else:
        PG_LOS = from_dB(-PL_LOS_dB)
        ax.plot(x, PG_LOS, lw=2, label="LOS Pathgain")

    d3D_m = np.hypot(x, h_BS - h_UT)
    fs_pathloss_dB = 20 * np.log10(d3D_m) + 20 * np.log10(fc_GHz * 1e9) - 147.55
    if plot_type == "pathloss":
        ax.plot(
            x,
            fs_pathloss_dB,
            lw=2,
            label="Free-space Pathloss",
            color="red",
            linestyle="--",
        )
    else:
        fs_pathgain = from_dB(-fs_pathloss_dB)
        ax.plot(
            x,
            fs_pathgain,
            lw=2,
            label="Free-space Pathgain",
            color="red",
            linestyle="--",
        )

    if zoom_box and plot_type == "pathloss":
        x1, x2, y1, y2 = 0, 10, 45, 65
        axins = ax.inset_axes([0.65, 0.05, 0.33, 0.33])
        axins.set_facecolor("oldlace")
        axins.plot(x, PL_NLOS_dB, color="blue")
        axins.plot(x, PL_LOS_dB, color="orange")
        axins.plot(x, fs_pathloss_dB, color="red", linestyle="--")
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.grid(color="gray", alpha=0.7, lw=0.5)
        ax.indicate_inset_zoom(axins, edgecolor="gray")

    ax.set_xlabel("3D Distance (metres)")
    ax.legend(framealpha=1.0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=40) if plot_type == "pathloss" else ax.set_ylim(bottom=0)
    fig.tight_layout()

    if print_10m_pl:
        # Find index closest to 10m for accurate reporting
        idx_10m = np.searchsorted(x, 10.0)
        BLUE, ORANGE, RED, RESET = (
            "\033[38;5;027m",
            "\033[38;5;202m",
            "\033[38;5;196m",
            "\033[0m",
        )
        print(f"\nPathloss at 10 metres:")
        print("----------------------")
        print(f"{BLUE}InH-NLOS:       {PL_NLOS_dB[idx_10m]:.2f} dB")
        print(f"{ORANGE}InH-LOS:        {PL_LOS_dB[idx_10m]:.2f} dB")
        print(f"{RED}Free-space:     {fs_pathloss_dB[idx_10m]:.2f} dB{RESET}\n")

    fnbase = "InH_pathloss_model" if plot_type == "pathloss" else "InH_pathgain_model"
    fig_timestamp(fig, rotation=0, fontsize=6, author=author)
    fig.savefig(f"{fnbase}.png")
    fig.savefig(f"{fnbase}.pdf")
    print(f"eog {fnbase}.png &")
    print(f"evince --page-label=1  {fnbase}.pdf &")


if __name__ == "__main__":
    plot_InH_pathloss_or_pathgain(
        plot_type="pathloss",
        zoom_box=True,
        print_10m_pl=True,
        author="Keith Briggs, Kishan Sthankiya and Ibrahim Nur",
    )
