# Keith Briggs 2025-09-23
# Ibrahim Nur 2025-09-05

from sys import path

import numpy as np


def to_dB(x):
    return 10.0 * np.log10(x)


def from_dB(x):
    return np.power(10.0, x / 10.0)


class InH_pathloss:
    """
    Indoor Hotspot (InH) pathloss model from 3GPP TR 38.901.

    Calculates pathloss for indoor office environments, supporting
    both Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) scenarios.

    Attributes
    ----------
    fc_GHz : float
        Centre frequency in gigahertz.
    LOS : bool
        Flag indicating if the Line-of-Sight model is active.

    References
    ----------
    - 3GPP TR 38.901: https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173
    """

    def __init__(self, fc_GHz=3.5, LOS=True, **args):
        """
        Initialises the InH_pathloss model instance.

        Parameters
        ----------
        fc_GHz : float, optional
            Centre frequency in gigahertz. Defaults to 3.5.
        h_UT : float, optional
            Default User Terminal height in metres. Defaults to 1.5.
        h_BS : float, optional
            Default Base Station height in metres. Defaults to 3.0.
        LOS : bool, optional
            Specifies whether the Line-of-Sight (LOS) model should be
            used. Defaults to True.
        **args
            Catches unused keyword arguments for API compatibility.
        """
        # default_h_UT : float
        #     Default User Terminal height for the plotting function.
        # default_h_BS : float
        #     Default Base Station height for the plotting function.
        # los_const : float
        #     Pre-calculated linear constant for the LOS formula.
        # nlos_const : float
        #     Pre-calculated linear constant for the NLOS formula.
        self.fc_GHz = fc_GHz
        self.LOS = LOS
        self.los_const = np.power(10, 3.24) * np.power(self.fc_GHz, 2.0)
        self.nlos_const = np.power(10, 1.73) * np.power(self.fc_GHz, 2.49)

    def get_pathloss_dB(self, d2D_m, d3D_m, U, C):
        """
        Calculates the pathloss in decibels (dB).

        This is a convenience wrapper around :meth:`get_pathloss` that converts
        the linear pathloss value to the dB scale.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix (metres). Not used in calculation.
        d3D_m : numpy.ndarray
            3D distance matrix in metres between UEs and cells.
        U : numpy.ndarray
            Array of UE locations. Not used in calculation.
        C : numpy.ndarray
            Array of cell locations. Not used in calculation.

        Returns
        -------
        numpy.ndarray
            The pathloss in decibels.
        """
        return to_dB(
            self.get_pathloss(d2D_m, d3D_m, U, C)
        )  # retained for compatibility

    def get_pathloss(self, d2D_m, d3D_m, U, C):
        """
        Calculates the linear pathloss based on the 3GPP InH model.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix (metres). Not used in calculation.
        d3D_m : numpy.ndarray
            3D distance matrix in metres between UEs and cells.
        U : numpy.ndarray
            Array of UE locations. Not used in calculation.
        C : numpy.ndarray
            Array of cell locations. Not used in calculation.

        Raises
        ------
        ValueError
            If any d3D_m value is outside the valid range of [1.0, 150.0] metres.

        Returns
        -------
        numpy.ndarray
            The calculated linear pathloss for each UE-cell link.
        """
        if np.any(d3D_m < 1.0) or np.any(d3D_m > 150.0):
            raise ValueError(
                f"At least one d3D_m value is outside the valid InH range [1.0, 150.0]m"
            )
        pl_los_linear = self.los_const * np.power(d3D_m, 1.73)
        if self.LOS:
            return pl_los_linear
        else:
            pl_nlos_prime_linear = self.nlos_const * np.power(d3D_m, 3.83)
            return np.maximum(pl_los_linear, pl_nlos_prime_linear)

    def get_pathgain(self, d2D_m, d3D_m, U, C):
        """
        Calculates the linear pathgain.

        Pathgain is the reciprocal of the linear pathloss.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix in metres.
        d3D_m : numpy.ndarray
            3D distance matrix in metres.
        U : numpy.ndarray
            Array of UE locations.
        C : numpy.ndarray
            Array of cell locations.

        Returns
        -------
        numpy.ndarray
            The linear pathgain for each UE-cell link.
        """
        return 1.0 / self.get_pathloss(d2D_m, d3D_m, U, C)

    def _get_approximate_pathloss_dB_for_layout_plot(self, d: float):
        U = np.array([[d, 0.0, 1.5]])
        C = np.array([[0.0, 0.0, 3.0]])
        d2D_m = np.array([[d]])
        d3D_m = np.linalg.norm(d - np.array([[0.0, 0.0, 1.5]]))
        d3D_m = min(d3D_m, 149.9)
        return self.get_pathloss_dB(d2D_m, d3D_m, U, C).squeeze()


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
    Plots 3GPP InH pathloss or pathgain model predictions as a self-test.

    This function generates a plot of the 3GPP InH pathloss or pathgain models
    for both Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) scenarios. It
    also includes a free-space pathloss reference and an optional zoomed-in
    view of the plot for detailed analysis.

    Parameters
    ----------
    plot_type : str, optional
        Type of plot to generate. Options are 'pathloss' (default) to plot
        pathloss in dB, or 'pathgain' to plot linear pathgain.
    fc_GHz : float, optional
        Carrier frequency in GHz, defaults to 3.5.
    h_UT : float, optional
        Height of the User Terminal (UE) in metres, defaults to 1.5.
    h_BS : float, optional
        Height of the Base Station (BS) in metres, defaults to 3.0.
    zoom_box : bool, optional
        If True, includes a zoomed-in inset on the plot. Defaults to False.
    print_10m_pl : bool, optional
        If True, prints pathloss values at 10 metres to the console for
        LOS, NLOS, and free-space scenarios. Defaults to False.
    author : str, optional
        Author name to include in the plot timestamp. Defaults to ' '.
    x_min : float, optional
        Minimum distance for the plot's x-axis in metres. Defaults to 1.0.
    x_max : float, optional
        Maximum distance for the plot's x-axis in metres. Defaults to 120.0.

    Raises
    ------
    ImportError
        If required plotting modules (e.g., matplotlib) are not installed.

    Notes
    -----
    The function uses the :class:`InH_pathloss` class to compute the pathloss
    and pathgain values for the different scenarios.
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

    xyz_cell = np.array([[0.0, 0.0, h_BS]])
    xyz_ues = np.column_stack((x, np.zeros_like(x), np.full_like(x, h_UT)))

    d2D_m = np.linalg.norm(
        xyz_ues[:, np.newaxis, :2] - xyz_cell[np.newaxis, :, :2], axis=2
    )
    d3D_m = np.linalg.norm(xyz_ues[:, np.newaxis, :] - xyz_cell[np.newaxis], axis=2)

    PL_nlos = InH_pathloss(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS, LOS=False)
    PL_NLOS_dB = PL_nlos.get_pathloss_dB(d2D_m, d3D_m, xyz_ues, xyz_cell).squeeze()

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
    PL_LOS_dB = PL_los.get_pathloss_dB(d2D_m, d3D_m, xyz_ues, xyz_cell).squeeze()

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

    fnbase = (
        "img/InH_pathloss_model" if plot_type == "pathloss" else "InH_pathgain_model"
    )
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
