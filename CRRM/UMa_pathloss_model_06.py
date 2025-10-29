# Keith Briggs 2025-09-24
# Ibrahim Nur 2025-09-04

from sys import path

import numpy as np


def to_dB(x):
    return 10.0 * np.log10(x)


def from_dB(x):
    return np.power(10.0, x / 10.0)


class UMa_pathloss:
    """
    Urban Macrocell (UMa) pathloss model from 3GPP TR 38.901.

    Calculates pathloss for urban macrocell environments where UE and BS heights
    can vary. This model includes the stochastic calculation of the effective
    environment height (h_E).

    Attributes
    ----------
    fc_GHz : float
        Centre frequency in gigahertz.
    LOS : bool
        Indicates if the Line-of-Sight model is active.

    References
    ----------
    - 3GPP TR 38.901: https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173
    """

    def __init__(self, fc_GHz=3.5, LOS=True, **args):
        """
        Initialises the UMa_pathloss model instance.

        Parameters
        ----------
        fc_GHz : float, optional
            Centre frequency in gigahertz, by default 3.5.
        LOS : bool, optional
            Specifies whether the Line-of-Sight (LOS) model should be
            used, by default True.
        **args
            Catches unused keyword arguments for API compatibility.
        """
        self.fc_GHz = fc_GHz
        self.LOS = LOS
        self.rng = np.random.default_rng()
        fc_GHz_sq = self.fc_GHz**2.0
        self.pl1_const = (10**2.8) * fc_GHz_sq
        self.nlos_const = (10**1.354) * fc_GHz_sq

    def get_pathloss_dB(self, d2D_m, d3D_m, U, C):
        """Calculates the pathloss in decibels (dB).

        This is a convenience wrapper around :meth:`get_pathloss` that converts
        the linear pathloss value to the dB scale.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix in metres between UEs and cells.
        d3D_m : numpy.ndarray
            3D distance matrix in metres between UEs and cells.
        U : numpy.ndarray
            Array of UE locations, shape (n_ues, 3).
        C : numpy.ndarray
            Array of cell locations, shape (n_cells, 3).

        Returns
        -------
        numpy.ndarray
            The pathloss in decibels.
        """
        return to_dB(
            self.get_pathloss(d2D_m, d3D_m, U, C)
        )  # retained for compatibility

    def get_pathloss(self, d2D_m, d3D_m, U, C):
        """Calculates the linear pathloss based on the 3GPP UMa model.

        Implements the full UMa logic, including the stochastic calculation of the
        effective environment height h_E and the breakpoint distance d_BP, which
        depends on the effective antenna heights.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix in metres between UEs and cells.
        d3D_m : numpy.ndarray
            3D distance matrix in metres between UEs and cells.
        U : numpy.ndarray
            Array of UE locations, shape (n_ues, 3). The 3rd column (height) must
            be in the range [1.5, 22.5].
        C : numpy.ndarray
            Array of cell locations, shape (n_cells, 3).

        Raises
        ------
        ValueError
            If any h_UT value is outside its valid range.

        Returns
        -------
        numpy.ndarray
            The calculated linear pathloss for each UE-cell link.
        """
        h_UT = U[:, 2:]
        h_BS = C[:, 2:].T
        if np.any(h_UT < 1.5) or np.any(h_UT > 22.5):
            raise ValueError(
                f"At least one h_UT value is outside the valid range [1.5, 22.5]m"
            )
        g_d2D = np.where(
            d2D_m <= 18.0,
            0.0,
            (5.0 / 4.0) * np.power((d2D_m / 100.0), 3.0) * np.exp(-d2D_m / 150.0),
        )
        C_d2D_hUT = np.where(
            h_UT < 13.0,
            0.0,
            np.power(np.maximum(0.0, (h_UT - 13.0) / 10.0), 1.5) * g_d2D,
        )
        # h_E is randomly chosen as follows:
        prob_hE_is_1 = 1.0 / (1.0 + C_d2D_hUT)
        uniform_dist = self.rng.random(size=d2D_m.shape)
        # For the case where h_E is not 1m, choose from a discrete uniform distribution
        # The values are 12, 15, ..., up to h_UT - 1.5 as per Table 7.4.1-1 note 1
        num_choices_float = np.floor((h_UT - 1.5 - 12.0) / 3.0) + 1.0
        num_choices = np.maximum(1, num_choices_float.astype(int))
        h_E_dist = 12.0 + 3.0 * self.rng.integers(0, num_choices, size=d2D_m.shape)
        h_E = np.where(uniform_dist < prob_hE_is_1, 1.0, h_E_dist)
        h_BS_eff = np.maximum(0.01, h_BS - h_E)
        h_UT_eff = np.maximum(0.01, h_UT - h_E)
        d_BP = 4.0 * h_BS_eff * h_UT_eff * (self.fc_GHz * 1e9) / 3.0e8
        PL1_linear = self.pl1_const * np.power(d3D_m, 2.2)
        PL2_linear = (
            self.pl1_const
            * np.power(d3D_m, 4.0)
            * np.power(np.power(d_BP, 2.0) + np.power(h_BS - h_UT, 2.0), -0.9)
        )
        pl_los_linear = np.where(d2D_m <= d_BP, PL1_linear, PL2_linear)
        if self.LOS:
            return pl_los_linear
        else:
            pl_nlos_prime_linear = (
                self.nlos_const
                * np.power(d3D_m, 3.908)
                * np.power(10.0, -0.06 * (h_UT - 1.5))
            )
            return np.maximum(pl_los_linear, pl_nlos_prime_linear)

    def get_pathgain(self, d2D_m, d3D_m, U, C):
        """Calculates the linear pathgain.

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

    def _get_approximate_pathloss_dB_for_layout_plot(self, d):
        """
        Calculates an approximate pathloss for visualization purposes.

        This internal helper method, present in all pathloss models,
        is used to generate pathloss values for a range of simple 1D
        distances. This is used when creating 2D circular pathloss
        plots.

        Parameters
        ----------
        d : float
            A 1D array of distances in metres.

        Returns
        -------
        numpy.ndarray
            The calculated pathloss in dB for each distance.
        """
        U = np.array([[d, 0.0, 1.5]])
        C = np.array([[0.0, 0.0, 25.0]])
        d2D_m = np.array([[d]])
        d3D_m = np.linalg.norm(d - np.array([[0.0, 0.0, 1.5]]))
        return self.get_pathloss_dB(d2D_m, d3D_m, U, C)


# END class UMa_pathloss


def plot_UMa_pathloss_or_pathgain(
    plot_type="pathloss",
    fc_GHz=3.5,
    h_UT=1.5,
    h_BS=35.5,
    zoom_box=False,
    print_10m_pl=False,
    author=" ",
    x_min=35.0,
    x_max=5000.0,
):
    """
    Plot 3GPP UMa pathloss or pathgain model predictions as a self-test.

    Generates a plot comparing the UMa pathloss or pathgain models for both
    Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) scenarios. It includes
    a free-space pathloss reference and an optional zoomed-in view for
    detailed analysis.

    Parameters
    ----------
    plot_type : str, optional
        Type of plot to generate. Options are 'pathloss' (default) to plot
        pathloss in dB, or 'pathgain' to plot linear pathgain.
    fc_GHz : float, optional
        Carrier frequency in GHz, by default 3.5.
    h_UT : float, optional
        Height of the User Terminal (UE) in metres, by default 1.5.
    h_BS : float, optional
        Height of the Base Station (BS) in metres, by default 35.5.
    zoom_box : bool, optional
        If True, includes a zoomed-in inset on the plot. Defaults to False.
    print_10m_pl : bool, optional
        If True, prints pathloss values at 10 metres to the console.
        Defaults to False.
    author : str, optional
        Author name to include in the plot timestamp, by default ' '.
    x_min : float, optional
        Minimum distance for the plot's x-axis in metres, by default 35.0.
    x_max : float, optional
        Maximum distance for the plot's x-axis in metres, by default 5000.0.

    Raises
    ------
    ImportError
        If required plotting modules (e.g., matplotlib) are not installed.
    """
    try:
        import matplotlib.pyplot as plt
        from utilities import fig_timestamp
    except ImportError as e:
        print(f"Error importing modules: {e}")
        raise

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(color="gray", alpha=0.5, lw=0.5)

    # Define coordinates for cells and UEs
    xyz_cells = np.array([[0.0, 0.0, h_BS]])
    x = np.linspace(x_min, x_max, 4990)
    xyz_ues = np.column_stack((x, np.zeros_like(x), np.full_like(x, h_UT)))

    # Calculate distances
    d2D_m = np.linalg.norm(
        xyz_ues[:, np.newaxis, :2] - xyz_cells[np.newaxis, :, :2], axis=2
    )
    d3D_m = np.linalg.norm(xyz_ues[:, np.newaxis, :] - xyz_cells[np.newaxis], axis=2)

    h_E = 1.0  # simplified for visualisation. The actual code adopts an exact implementation of h_E.
    h_BS_eff = h_BS - h_E
    h_UT_eff = h_UT - h_E
    dBP_NLOS = dBP_LOS = 4 * h_BS_eff * h_UT_eff * (fc_GHz * 1e9) / 3.0e8

    # Plot NLOS
    PL = UMa_pathloss(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS, LOS=False)
    dBP_NLOS_index = np.searchsorted(x, dBP_NLOS)
    PL_NLOS_dB = PL.get_pathloss_dB(d2D_m, d3D_m, xyz_ues, xyz_cells).squeeze()
    PG_NLOS = 1.0 / np.power(10.0, PL_NLOS_dB / 10.0)

    if plot_type == "pathloss":
        ax.set_title(f"3GPP UMa pathloss models (dBP={dBP_NLOS:.0f}m)")
        ax.set_ylabel("pathloss (dB)")
        line = ax.plot(
            x, PL_NLOS_dB, lw=2, label=r"NLOS exact ($\sigma=6$)", color="blue"
        )
        line_color = line[0].get_color()
        ax.vlines(dBP_NLOS, 0, PL_NLOS_dB[dBP_NLOS_index], line_color, "dotted", lw=2)
        ax.fill_between(x, PL_NLOS_dB - 6, PL_NLOS_dB + 6, color=line_color, alpha=0.2)
        ax.set_ylim(50)
    else:
        ax.set_title(f"3GPP UMa pathgain models (dBP={dBP_NLOS:.0f}m)")
        ax.set_ylabel("pathgain")
        ax.plot(x, PG_NLOS, lw=2, label="NLOS pathgain")

    # Plot LOS
    PL = UMa_pathloss(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS, LOS=True)
    dBP_LOS_index = np.searchsorted(x, dBP_LOS)
    PL_LOS_dB = PL.get_pathloss_dB(d2D_m, d3D_m, xyz_ues, xyz_cells).squeeze()
    PG_LOS = 1.0 / np.power(10.0, PL_LOS_dB / 10.0)

    if plot_type == "pathloss":
        line = ax.plot(x, PL_LOS_dB, lw=2, label=r"LOS ($\sigma=4$)", color="orange")
        line_color = line[0].get_color()
        sigma = np.where(np.less_equal(x, dBP_LOS), 4.0, 4.0)
        ax.vlines(dBP_LOS, 0, PL_LOS_dB[dBP_LOS_index], line_color, "dotted", lw=2)
        ax.fill_between(
            x, PL_LOS_dB - sigma, PL_LOS_dB + sigma, color=line_color, alpha=0.2
        )
        ax.set_xlim(0, np.max(x))
        fnbase = "img/UMa_pathloss_model"
    else:
        ax.plot(x, PG_LOS, lw=2, label="LOS pathgain")
        ax.set_ylim(0)
        ax.set_xlim(0, 1000)
        fnbase = "img/UMa_pathgain_model"

    # Plot the Free-space pathloss as a reference
    fs_pathloss_dB = (
        20 * np.log10(d3D_m) + 20 * np.log10(fc_GHz * 1e9) - 147.55
    ).squeeze()
    fs_pathloss = np.power(10.0, fs_pathloss_dB / 10.0)
    fs_pathgain = 1.0 / fs_pathloss
    if plot_type == "pathloss":
        ax.plot(x, fs_pathloss_dB, lw=2, label="Free-space pathloss", color="red")
    else:
        ax.plot(x, fs_pathgain, lw=2, label="Free-space pathloss", color="red")

    # Add zoom box at lower left of plot
    if zoom_box and plot_type == "pathloss":

        # Define the area you want to zoom in on
        x1, x2, y1, y2 = 30, 90, 76, 86

        # Define where you want the zoom box to be placed
        axins = ax.inset_axes([0.4, 0.1, 0.2, 0.33])
        axins.set_facecolor("oldlace")

        # Plot the zoomed area
        axins.plot(x, PL_NLOS_dB, lw=2, label="NLOS pathloss", color="blue")
        axins.plot(x, PL_LOS_dB, lw=2, label="LOS pathloss", color="orange")
        axins.plot(x, fs_pathloss_dB, lw=2, label="Free-space pathloss", color="red")
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks(np.arange(x1, x2, 10))
        axins.set_yticks(np.arange(y1, y2, 1))
        axins.tick_params(axis="both", direction="in")
        axins.grid(color="gray", alpha=0.7, lw=0.5)
        ax.indicate_inset_zoom(axins, edgecolor="gray")

    # Final plot adjustments
    ax.set_xlabel("distance (metres)")
    ax.legend(framealpha=1.0)
    fig.tight_layout()

    # Print the pathloss at 10 metres
    if print_10m_pl:
        BLUE = "\033[38;5;027m"
        ORANGE = "\033[38;5;202m"
        RED = "\033[38;5;196m"
        RESET = "\033[0m"
        print(f"\nPathloss at 10 metres:")
        print("----------------------")
        print(f"{BLUE}UMa-NLOS:       {PL_NLOS_dB[0]:.2f} dB")
        print(f"{ORANGE}UMa-LOS:        {PL_LOS_dB[0]:.2f} dB")
        print(f"{RED}Free-space:     {fs_pathloss_dB[0]:.2f} dB{RESET}\n")

    # Add timestamp and save figures
    fig_timestamp(fig, rotation=0, fontsize=6, author=author)
    fig.savefig(f"{fnbase}.png")
    print(f"eog {fnbase}.png &")
    fig.savefig(f"{fnbase}.pdf")
    print(f"evince --page-label=1  {fnbase}.pdf &")


if __name__ == "__main__":
    plot_UMa_pathloss_or_pathgain(
        plot_type="pathloss",
        zoom_box=True,
        print_10m_pl=True,
        author="Keith Briggs, Kishan Sthankiya and Ibrahim Nur",
    )
