# Ibrahim Nur  2025-09-16 v02
# Keith Briggs 2025-08-27 v01
# Ibrahim Nur  2025-08-27 v00

import numpy as np


def to_dB(x):
    return 10.0 * np.log10(np.maximum(x, 1e-15))


def from_dB(x):
    return np.power(10.0, x / 10.0)


class Power_law_pathloss:
    """
    Simple power-law (or log-distance) pathloss model.

    Calculates pathloss using the formula:
    PL(d) [dB] = PL(d0) + 10 * n * log10(d / d0)

    This implementation fixes the reference distance d0 = 1 metre and calculates
    the reference pathloss PL(d0) using the free-space formula at 1m.

    Attributes
    ----------
    fc_GHz : float
        Centre frequency in gigahertz (used to calculate PL at 1m).
    exponent : float
        The pathloss exponent 'n'.
    L_0 : float
        The linear pathloss at the 1-metre reference distance.
    L_0_dB : float
        The pathloss in dB at the 1-metre reference distance.

    References
    ----------
    - L. W. Barclay (Ed.), "Propagation of Radiowaves," 2nd ed., IET, 2003.
    """

    def __init__(self, fc_GHz, exponent, **args):
        """
        Initialises the Power_law_pathloss model instance.

        Parameters
        ----------
        fc_GHz : float
            Centre frequency in gigahertz.
        exponent : float
            The pathloss exponent 'n'.
        **args
            Catches unused keyword arguments for API compatibility.
        """
        self.fc_GHz = fc_GHz
        self.exponent = exponent
        λ = 3e8 / (fc_GHz * 1e9)
        self.L_0 = (4.0 * np.pi / λ) ** 2
        self.L_0_dB = 20.0 * np.log10(4.0 * np.pi / λ)

    def get_pathloss_dB(self, d2D_m, d3D_m, **args):
        """
        Calculates the pathloss in decibels (dB).

        This is a convenience wrapper around :meth:`get_pathloss` that converts
        the linear pathloss value to the dB scale.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix (metres). Unused by this model, but
            included for API compatibility.
        d3D_m : numpy.ndarray
            3D distance matrix in metres between UEs and cells.
        **args
            Catches unused keyword arguments.

        Returns
        -------
        numpy.ndarray
            The pathloss in decibels.
        """
        return to_dB(self.get_pathloss(d2D_m, d3D_m))

    def get_pathloss(self, d2D_m, d3D_m, **args):
        """
        Calculates the linear pathloss.

        Implements the formula: PL_linear = L_0 * (d ^ exponent)
        where L_0 is the free-space pathloss at 1 metre. This model
        only uses the 3D distance.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix (metres). Unused by this model.
        d3D_m : numpy.ndarray
            3D distance matrix in metres between UEs and cells.
        **args
            Catches unused keyword arguments.

        Returns
        -------
        numpy.ndarray
            The calculated linear pathloss for each UE-cell link.
        """
        return self.L_0 * np.power(d3D_m, self.exponent)

    def get_pathgain(self, d2D_m, d3D_m, U, C, **args):
        """
        Calculates the linear pathgain.

        Pathgain is the reciprocal of the linear pathloss.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix (metres). Unused by this model.
        d3D_m : numpy.ndarray
            3D distance matrix in metres.
        U : numpy.ndarray
            Array of UE locations, shape (n_ues, 3). Unused.
        C : numpy.ndarray
            Array of cell locations, shape (n_cells, 3). Unused
        **args
            Catches unused keyword arguments.

        Returns
        -------
        numpy.ndarray
            The linear pathgain for each UE-cell link.
        """
        return 1.0 / self.get_pathloss(d2D_m, d3D_m)

    def _get_approximate_pathloss_dB_for_layout_plot(self, d):
        """
        Calculates an approximate pathloss for visualization purposes.

        This internal helper method, present in all pathloss models,
        is used to generate pathloss values for a range of simple 1D
        distances. This is used when creating 2D circular pathloss
        plots.

        This implementation assumes a default UE height of 1.5m and BS
        height of 25m to calculate the 3D distance.

        Parameters
        ----------
        d : float
            A 1D distance in metres.

        Returns
        -------
        numpy.ndarray
            The calculated pathloss in dB for the given distance.
        """
        U = np.array([[d, 0.0, 1.5]])
        C = np.array([[0.0, 0.0, 25]])
        d2D_m = np.array([[d]])
        d3D_m = np.array([[np.linalg.norm(U - C)]])
        return self.get_pathloss_dB(d2D_m, d3D_m)


# END class Power_law_pathloss


def plot_power_law_pathloss_or_pathgain(
    plot_type="pathloss",
    fc_GHz=3.5,
    exponent=3.0,
    print_10m_pl=False,
    author="",
    x_min=35.0,
    x_max=5000.0,
):
    """
    Plot 3GPP power_law pathloss or pathgain model predictions as a self-test.

    Generates a plot comparing the power_law pathloss or pathgain models.
    It includes a free-space pathloss reference and an optional zoomed-in
    view for detailed analysis.

    Parameters
    ----------
    plot_type : str, optional
        Type of plot to generate. Options are 'pathloss' (default) to plot
        pathloss in dB, or 'pathgain' to plot linear pathgain.
    fc_GHz : float, optional
        Carrier frequency in GHz, by default 3.5.
    exponent : float, optional
        Exponent used in the power-law model, by default 3.0.
    print_10m_pl : bool, optional
        If True, prints pathloss values at 10 metres to the console.
        Defaults to False.
    author : str, optional
        Author name to include in the plot timestamp, empty by default.
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
    xyz_cells = np.array([[0.0, 0.0, 25.0]])
    x = np.linspace(x_min, x_max, 4990)
    xyz_ues = np.column_stack((x, np.zeros_like(x), np.full_like(x, 1.5)))

    # Calculate distances
    d3D_m = np.linalg.norm(xyz_ues[:, np.newaxis, :] - xyz_cells[np.newaxis], axis=2)

    PL = Power_law_pathloss(fc_GHz=fc_GHz, exponent=exponent)
    PL_dB = PL.get_pathloss_dB(0, d3D_m).squeeze()
    PG_LOS = 1.0 / np.power(10.0, PL_dB / 10.0)

    # Plot the Free-space pathloss as a reference
    fs_pathloss_dB = (
        20 * np.log10(d3D_m) + 20 * np.log10(fc_GHz * 1e9) - 147.55
    ).squeeze()
    fs_pathloss = np.power(10.0, fs_pathloss_dB / 10.0)
    fs_pathgain = 1.0 / fs_pathloss
    if plot_type == "pathloss":
        ax.plot(x, fs_pathloss_dB, lw=2, label="Free-space pathloss", color="red")
        ax.plot(x, PL_dB, lw=2, label=r"Power-law pathloss", color="orange")
    else:
        ax.plot(x, fs_pathgain, lw=2, label="Free-space pathloss", color="red")
        ax.plot(x, PG_LOS, lw=2, label=r"Power-law pathloss", color="orange")

    # Final plot adjustments
    ax.set_xlim(x_min, x_max)
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
        print(f"{ORANGE}Power-law:     {PL_dB[0]:.2f} dB{RESET}\n")
        print(f"{RED}Free-space:     {fs_pathloss_dB[0]:.2f} dB{RESET}\n")

    fnbase = f"img/power_law_{plot_type}_model"
    fig_timestamp(fig, rotation=0, fontsize=6, author=author)
    fig.savefig(f"{fnbase}.png")
    print(f"eog {fnbase}.png &")
    fig.savefig(f"{fnbase}.pdf")
    print(f"evince --page-label=1  {fnbase}.pdf &")


if __name__ == "__main__":
    plot_power_law_pathloss_or_pathgain(
        plot_type="pathloss",
        print_10m_pl=True,
        author="Keith Briggs, Kishan Sthankiya, Ibrahim Nur",
    )
