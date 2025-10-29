# Keith Briggs 2025-09-23
# Ibrahim Nur 2025-09-10

from sys import path

import numpy as np


def to_dB(x):
    return 10.0 * np.log10(x)


def from_dB(x):
    return np.power(10.0, x / 10.0)


class RMa_pathloss_constant_height:
    """
    Rural Macrocell (RMa) pathloss model assuming constant antenna heights.

    Simplified version of the 3GPP TR 38.901 RMa model where the heights of the
    User Terminal (UT) and Base Station (BS) are constant for all links, leading
    to a performance gain.

    References
    ----------
    - 3GPP TR 38.901: https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173
    """

    def __init__(self, fc_GHz=3.5, h_UT=1.5, h_BS=35.0, LOS=True, **args):
        """
        Initialises the RMa_pathloss_constant_height model instance.

        Parameters
        ----------
        fc_GHz : float, optional
            Centre frequency in GHz. Must be <= 7 GHz. Defaults to 3.5.
        h_UT : float, optional
            Height of the User Terminal in metres. Must be in [1.0, 10.0].
            Defaults to 1.5.
        h_BS : float, optional
            Height of the Base Station in metres. Must be in [10.0, 150.0].
            Defaults to 35.0.
        LOS : bool, optional
            Specifies whether to use the Line-of-Sight model.
            Defaults to True.
        **args
            Catches unused keyword arguments for API compatibility.

        Raises
        ------
        ValueError
            If `h_UT` or `h_BS` are outside their valid ranges.
        """
        if not (1.0 <= h_UT <= 10.0):
            raise ValueError(f"h_UT={h_UT} is outside the valid range [1.0, 10.0]m")
        if not (10.0 <= h_BS <= 150.0):
            raise ValueError(f"h_BS={h_BS} is outside the valid range [10.0, 150.0]m")
        self.fc_GHz = fc_GHz
        self.h_BS = h_BS
        self.h_UT = h_UT
        self.LOS = LOS
        self.d_BP = 2 * np.pi * h_BS * h_UT * (fc_GHz * 1e9) / 3.0e8
        h = 5.0  # building height = 5m
        log_h_BS = np.log10(h_BS)
        # LOS terms are calculated linearly
        self.los_term1 = 2.0 + np.minimum(0.03 * h**1.72, 10) / 10.0
        self.los_term2 = (40 * np.pi * self.fc_GHz / 3) ** 2 * from_dB(
            -np.minimum(0.044 * h**1.72, 14.77)
        )
        self.los_term3 = 0.002 * np.log10(h) / 10.0
        # Constants in dB are separated from constants multiplied by a function of distance
        # i.e. PL_NLOS_dB = A_dB + B_dB*log10(d)
        A_dB = (
            161.04
            - 7.1 * np.log10(20.0)
            + 7.5 * np.log10(h)
            - (24.37 - 3.7 * (h / self.h_BS) ** 2) * log_h_BS
            + (43.42 - 3.1 * log_h_BS) * (-3)
            + 20.0 * np.log10(self.fc_GHz)
            - (3.2 * (np.log10(11.75 * self.h_UT)) ** 2 - 4.97)
        )
        B_dB = 43.42 - 3.1 * log_h_BS
        # A simple conversion to linear is then possible: 10^(A_dB/10) * d^(B_dB/10) in get_pathloss
        # The code below calculates the linear pathloss at breakpoint distance, which is used in PL_2
        self.nlos_term_A = from_dB(A_dB)
        self.nlos_term_B = B_dB / 10.0
        self.PL_1_at_d_BP = (
            self.los_term2
            * (self.d_BP**self.los_term1)
            * (10 ** (self.los_term3 * self.d_BP))
        ) / (self.d_BP**4)

    def get_pathloss_dB(self, d2D_m, d3D_m):
        """
        Calculates the pathloss in decibels (dB).

        This is a convenience wrapper around :meth:`get_pathloss` that converts
        the linear pathloss value to the dB scale.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix in metres between UEs and cells. Does not take into account height.
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
        return to_dB(self.get_pathloss(d2D_m, d3D_m))  # retained for compatibility

    def get_pathloss(self, d2D_m, d3D_m):
        """
        Calculates the linear pathloss.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix in metres between UEs and cells.
        d3D_m : numpy.ndarray
            3D distance matrix in metres between UEs and cells.

        Returns
        -------
        numpy.ndarray
            The calculated linear pathloss for each UE-cell link.
        """
        pl1_linear = (
            self.los_term2 * (d3D_m**self.los_term1) * (10 ** (self.los_term3 * d3D_m))
        )
        pl2_linear = self.PL_1_at_d_BP * (d3D_m**4)
        pl_los_linear = np.where(d2D_m <= self.d_BP, pl1_linear, pl2_linear)
        if self.LOS:
            return pl_los_linear
        else:
            pl_nlos_linear = self.nlos_term_A * (d3D_m**self.nlos_term_B)
            return np.maximum(pl_los_linear, pl_nlos_linear)

    def get_pathgain(self, d2D_m, d3D_m):
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
        return 1.0 / self.get_pathloss(d2D_m, d3D_m)

    def _get_approximate_pathloss_dB_for_layout_plot(self, d: float):
        U = np.array([[d, 0.0, 1.5]])
        C = np.array([[0.0, 0.0, 35.0]])
        d2D_m = np.array([[d]])
        d3D_m = np.linalg.norm(d - np.array([[0.0, 0.0, 1.5]]))
        return self.get_pathloss_dB(d2D_m, d3D_m, U, C)


# END class RMa_pathloss_constant_height


class RMa_pathloss:
    """
    The complete Rural Macrocell (RMa) pathloss model from 3GPP TR 38.901.

    Implements the full RMa model where antenna heights can vary for each
    link, calculating pathloss coefficients dynamically.

    References
    ----------
    - 3GPP TR 38.901: https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173
    """

    def __init__(self, fc_GHz=3.5, LOS=True, **args):
        """
        Initialises the RMa_pathloss model instance.

        Parameters
        ----------
        fc_GHz : float, optional
            Centre frequency in GHz. Must be <= 7 GHz. Defaults to 3.5.
        LOS : bool, optional
            Specifies whether to use the Line-of-Sight model.
            Defaults to True.
        **args
            Catches unused keyword arguments for API compatibility.
        """
        self.fc_GHz = fc_GHz
        self.LOS = LOS

    def get_pathloss_dB(self, d2D_m, d3D_m, U, C):
        """
        Calculates the pathloss in decibels (dB).

        This is a convenience wrapper around :meth:`get_pathloss` that converts
        the linear pathloss value to the dB scale.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix in metres between UEs and cells. Does not take into account height.
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
        Calculates the linear pathloss based on the general 3GPP RMa model.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix in metres between UEs and cells.
        d3D_m : numpy.ndarray
            3D distance matrix in metres between UEs and cells.
        U : numpy.ndarray
            Array of UE locations, shape (n_ues, 3). The 3rd column (height)
            must be in the range [1.0, 10.0].
        C : numpy.ndarray
            Array of cell locations, shape (n_cells, 3). The 3rd column (height)
            must be in the range [10.0, 150.0].

        Raises
        ------
        ValueError
            If any `h_UT` or `h_BS` value is outside its valid range.

        Returns
        -------
        numpy.ndarray
            The calculated linear pathloss for each UE-cell link.
        """
        h_UT = U[:, 2:]
        h_BS = C[:, 2:].T
        if np.any(h_UT < 1.0) or np.any(h_UT > 10.0):
            raise ValueError(
                f"At least one h_UT value is outside the valid range [1.0, 10.0]m"
            )
        if np.any(h_BS < 10.0) or np.any(h_BS > 150.0):
            raise ValueError(
                f"At least one h_BS value value is outside the valid range [10.0, 150.0]m"
            )
        d_BP = 2 * np.pi * h_BS * h_UT * (self.fc_GHz * 1e9) / 3.0e8
        h = 5.0  # building height = 5m
        log_h_BS = np.log10(h_BS)
        # LOS terms are calculated linearly
        los_term1 = 2.0 + np.minimum(0.03 * np.power(h, 1.72), 10) / 10.0
        los_term2 = np.power((40 * np.pi * self.fc_GHz / 3), 2) * from_dB(
            -np.minimum(0.044 * np.power(h, 1.72), 14.77)
        )
        los_term3 = 0.002 * np.log10(h) / 10.0
        # Constants in dB are separated from constants multiplied by a function of distance
        # i.e. PL_NLOS_dB = A_dB + B_dB*log10(d)
        A_dB = (
            161.04
            - 7.1 * np.log10(20.0)
            + 7.5 * np.log10(h)
            - (24.37 - 3.7 * np.power((h / h_BS), 2)) * log_h_BS
            + (43.42 - 3.1 * log_h_BS) * (-3)
            + 20 * np.log10(self.fc_GHz)
            - (3.2 * np.power(np.log10(11.75 * h_UT), 2) - 4.97)
        )
        B_dB = 43.42 - 3.1 * log_h_BS
        # A simple conversion to linear is then possible: 10^(A_dB/10) * d^(B_dB/10) in get_pathloss
        # The code below calculates the linear pathloss at breakpoint distance, which is used in PL_2
        nlos_term_A = from_dB(A_dB)
        nlos_term_B = B_dB / 10.0
        pl1_linear = (
            los_term2 * np.power(d3D_m, los_term1) * np.power(10, (los_term3 * d3D_m))
        )
        PL_1_at_d_BP_linear = (
            los_term2 * np.power(d_BP, los_term1) * np.power(10, (los_term3 * d_BP))
        )
        pl2_linear = (PL_1_at_d_BP_linear / np.power(d_BP, 4)) * np.power(d3D_m, 4)
        pl_los_linear = np.where(d2D_m <= d_BP, pl1_linear, pl2_linear)
        if self.LOS:
            return pl_los_linear
        else:
            pl_nlos_linear = nlos_term_A * np.power(d3D_m, nlos_term_B)
            return np.maximum(pl_los_linear, pl_nlos_linear)

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

    def _get_approximate_pathloss_dB_for_layout_plot(self, d):
        U = np.array([[d, 0.0, 1.5]])
        C = np.array([[0.0, 0.0, 35.0]])
        d2D_m = np.array([[d]])
        d3D_m = np.linalg.norm(d - np.array([[0.0, 0.0, 1.5]]))
        return self.get_pathloss_dB(d2D_m, d3D_m, U, C)


# END class RMa_pathloss


class RMa_pathloss_discretised:
    """
    Rural Macrocell (RMa) pathloss model with discretised heights.

    Pre-calculates pathloss coefficients for a grid of discrete UE and BS
    heights. At runtime, it rounds the actual heights to the nearest grid point
    to look up coefficients, trading accuracy for computational speed.

    The accuracy trade-off is minimal. This can be shown by running the
    plot_RMa_pathloss_runtime_comparison program.

    References
    ----------
    - 3GPP TR 38.901: https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173
    """

    def __init__(self, fc_GHz=3.5, LOS=True, h_ut_res=0.5, h_bs_res=1.0, **args):
        """
        Initialises the RMa_pathloss_discretised model instance.

        Parameters
        ----------
        fc_GHz : float, optional
            Centre frequency in GHz. Must be <= 7 GHz. Defaults to 3.5.
        LOS : bool, optional
            Specifies whether to use the Line-of-Sight model.
            Defaults to True.
        h_ut_res : float, optional
            Resolution of the discrete UT height grid in metres. Defaults to 0.5.
        h_bs_res : float, optional
            Resolution of the discrete BS height grid in metres. Defaults to 1.0.
        **args
            Catches unused keyword arguments for API compatibility.
        """
        self.fc_GHz = fc_GHz
        self.LOS = LOS
        self.h_ut_res = h_ut_res
        self.h_bs_res = h_bs_res
        self.h_UT = np.arange(1.0, 10.0 + h_ut_res, h_ut_res)[:, np.newaxis]
        self.h_BS = np.arange(10.0, 150.0 + h_bs_res, h_bs_res)[np.newaxis]
        self.d_BP = 2 * np.pi * self.h_BS * self.h_UT * (self.fc_GHz * 1e9) / 3.0e8
        h = 5.0  # building height = 5m
        log_h_BS = np.log10(self.h_BS)

        # LOS terms are calculated linearly
        self.los_term1 = 2.0 + np.minimum(0.03 * np.power(h, 1.72), 10) / 10.0
        self.los_term2 = np.power((40 * np.pi * self.fc_GHz / 3), 2) * from_dB(
            -np.minimum(0.044 * np.power(h, 1.72), 14.77)
        )
        self.los_term3 = 0.002 * np.log10(h) / 10.0

        # Constants in dB are separated from constants multiplied by a function of distance
        # i.e. PL_NLOS_dB = A_dB + B_dB*log10(d)
        A_dB = (
            161.04
            - 7.1 * np.log10(20.0)
            + 7.5 * np.log10(h)
            - (24.37 - 3.7 * np.power((h / self.h_BS), 2)) * log_h_BS
            + (43.42 - 3.1 * log_h_BS) * (-3)
            + 20 * np.log10(self.fc_GHz)
            - (3.2 * np.power(np.log10(11.75 * self.h_UT), 2) - 4.97)
        )
        B_dB = (43.42 - 3.1 * log_h_BS) + np.zeros_like(self.h_UT)

        # A simple conversion to linear is then possible: 10^(A_dB/10) * d^(B_dB/10) in get_pathloss
        self.nlos_term_A = from_dB(A_dB)
        self.nlos_term_B = B_dB / 10.0

        # The code below calculates the linear pathloss at breakpoint distance, which is used in PL_2
        PL_1_at_d_BP_linear = (
            self.los_term2
            * np.power(self.d_BP, self.los_term1)
            * np.power(10, (self.los_term3 * self.d_BP))
        )
        self.PL_1_at_d_BP = PL_1_at_d_BP_linear / np.power(self.d_BP, 4)

    def get_pathloss_dB(self, d2D_m, d3D_m, U, C):
        """
        Calculates the pathloss in decibels (dB).

        This is a convenience wrapper around :meth:`get_pathloss` that converts
        the linear pathloss value to the dB scale.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix in metres between UEs and cells. Does not take into account height.
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
        Calculates the linear pathloss based on the general 3GPP RMa model.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix in metres between UEs and cells.
        d3D_m : numpy.ndarray
            3D distance matrix in metres between UEs and cells.
        U : numpy.ndarray
            Array of UE locations, shape (n_ues, 3). The 3rd column (height)
            must be in the range [1.0, 10.0].
        C : numpy.ndarray
            Array of cell locations, shape (n_cells, 3). The 3rd column (height)
            must be in the range [10.0, 150.0].

        Raises
        ------
        ValueError
            If any `h_UT` or `h_BS` value is outside its valid range.

        Returns
        -------
        numpy.ndarray
            The calculated linear pathloss for each UE-cell link.
        """
        h_UT = U[:, 2:]
        h_BS = C[:, 2:].T

        if np.any(h_UT < 1.0) or np.any(h_UT > 10.0):
            raise ValueError(
                f"At least one h_UT value is outside the valid range [1.0, 10.0]m"
            )
        if np.any(h_BS < 10.0) or np.any(h_BS > 150.0):
            raise ValueError(
                f"At least one h_BS value is outside the valid range [10.0, 150.0]m"
            )

        h_UT_i = np.round((h_UT - 1.0) / self.h_ut_res).astype(int)
        h_BS_j = np.round((h_BS - 10.0) / self.h_bs_res).astype(int)

        d_BP = self.d_BP[h_UT_i, h_BS_j]
        PL_1_at_d_BP = self.PL_1_at_d_BP[h_UT_i, h_BS_j]
        nlos_term_A = self.nlos_term_A[h_UT_i, h_BS_j]
        nlos_term_B = self.nlos_term_B[h_UT_i, h_BS_j]

        pl1_linear = (
            self.los_term2
            * np.power(d3D_m, self.los_term1)
            * np.power(10, (self.los_term3 * d3D_m))
        )
        pl2_linear = PL_1_at_d_BP * np.power(d3D_m, 4)
        pl_los_linear = np.where(d2D_m <= d_BP, pl1_linear, pl2_linear)

        if self.LOS:
            return pl_los_linear
        else:
            pl_nlos_linear = nlos_term_A * np.power(d3D_m, nlos_term_B)
            return np.maximum(pl_los_linear, pl_nlos_linear)

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

    def _get_approximate_pathloss_dB_for_layout_plot(self, d):
        U = np.array([[d, 0.0, 1.5]])
        C = np.array([[0.0, 0.0, 35.0]])
        d2D_m = np.array([[d]])
        d3D_m = np.linalg.norm(d - np.array([[0.0, 0.0, 1.5]]))
        return self.get_pathloss_dB(d2D_m, d3D_m, U, C)


# END class RMa_pathloss_discretised


def plot_RMa_pathloss_or_pathgain(
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
    Plot 3GPP RMa pathloss or pathgain model predictions as a self-test.

    This function generates a plot comparing various 3GPP RMa pathloss or
    pathgain models for both Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS)
    scenarios. It includes a free-space pathloss reference and an optional
    zoomed-in view for detailed analysis.

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
        Height of the Base Station (BS) in metres, defaults to 35.5.
    zoom_box : bool, optional
        If True, includes a zoomed-in inset on the plot. Defaults to False.
    print_10m_pl : bool, optional
        If True, prints pathloss values at 10 metres to the console for
        LOS, NLOS, and free-space scenarios. Defaults to False.
    author : str, optional
        Author name to include in the plot timestamp. Defaults to ' '.
    x_min : float, optional
        Minimum distance for the plot's x-axis in metres. Defaults to 35.0.
    x_max : float, optional
        Maximum distance for the plot's x-axis in metres. Defaults to 5000.0.

    Raises
    ------
    ImportError
        If required plotting modules (e.g., matplotlib) are not installed.

    Notes
    -----
    The function uses the :class:`RMa_pathloss`,
    :class:`RMa_pathloss_constant_height`, and
    :class:`RMa_pathloss_discretised` classes to compute the values for
    the different scenarios.
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
    d3D_m = np.linalg.norm(
        xyz_ues[:, np.newaxis, :] - xyz_cells[np.newaxis, :, :], axis=2
    )

    # Plot NLOS
    PL = RMa_pathloss(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS, LOS=False)
    dBP_NLOS = 2 * np.pi * h_BS * h_UT * (fc_GHz * 1e9) / 3.0e8
    dBP_NLOS_index = np.searchsorted(x, dBP_NLOS)
    PL_NLOS_dB = PL.get_pathloss_dB(d2D_m, d3D_m, xyz_ues, xyz_cells).squeeze()
    PG_NLOS = 1.0 / np.power(10.0, PL_NLOS_dB / 10.0)

    if plot_type == "pathloss":
        ax.set_title(f"3GPP RMa pathloss models (dBP={dBP_NLOS:.0f}m)")
        ax.set_ylabel("pathloss (dB)")
        line = ax.plot(
            x, PL_NLOS_dB, lw=2, label=r"NLOS exact ($\sigma=8$)", color="blue"
        )
        line_color = line[0].get_color()
        PL_NLOS_dB_const = (
            RMa_pathloss_constant_height(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS, LOS=False)
            .get_pathloss_dB(d2D_m, d3D_m)
            .squeeze()
        )
        ax.plot(
            x,
            PL_NLOS_dB_const,
            lw=1.5,
            linestyle="--",
            label="NLOS (constant height)",
            color="#abe6e2",
        )

        PL_NLOS_dB_disc = (
            RMa_pathloss_discretised(fc_GHz=fc_GHz, LOS=False)
            .get_pathloss_dB(d2D_m, d3D_m, xyz_ues, xyz_cells)
            .squeeze()
        )
        ax.plot(
            x,
            PL_NLOS_dB_disc,
            lw=1.5,
            linestyle=":",
            label="NLOS (discretised heights)",
            color="hotpink",
        )

        ax.vlines(dBP_NLOS, 0, PL_NLOS_dB[dBP_NLOS_index], line_color, "dotted", lw=2)
        ax.fill_between(x, PL_NLOS_dB - 8, PL_NLOS_dB + 8, color=line_color, alpha=0.2)
        ax.set_ylim(50)
    else:
        ax.set_title(f"3GPP RMa pathgain models (dBP={dBP_NLOS:.0f}m)")
        ax.set_ylabel("pathgain")
        ax.plot(x, PG_NLOS, lw=2, label="NLOS pathgain")

    # Plot LOS
    PL = RMa_pathloss(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS, LOS=True)
    dBP_LOS = 2 * np.pi * h_BS * h_UT * (fc_GHz * 1e9) / 3.0e8
    dBP_LOS_index = np.searchsorted(x, dBP_LOS)
    PL_LOS_dB = PL.get_pathloss_dB(d2D_m, d3D_m, xyz_ues, xyz_cells).squeeze()
    PG_LOS = 1.0 / np.power(10.0, PL_LOS_dB / 10.0)

    if plot_type == "pathloss":
        line = ax.plot(x, PL_LOS_dB, lw=2, label=r"LOS ($\sigma=4$)", color="orange")
        line_color = line[0].get_color()
        PL_LOS_dB_const = (
            RMa_pathloss_constant_height(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS, LOS=True)
            .get_pathloss_dB(d2D_m, d3D_m)
            .squeeze()
        )
        ax.plot(
            x,
            PL_LOS_dB_const,
            lw=1.5,
            linestyle="--",
            label="LOS (constant height)",
            color="yellow",
        )
        PL_LOS_dB_disc = (
            RMa_pathloss_discretised(fc_GHz=fc_GHz, LOS=True)
            .get_pathloss_dB(d2D_m, d3D_m, xyz_ues, xyz_cells)
            .squeeze()
        )
        ax.plot(
            x,
            PL_LOS_dB_disc,
            lw=1.5,
            linestyle=":",
            label="LOS (discretised heights)",
            color="orangered",
        )
        sigma = np.where(np.less_equal(x, dBP_LOS), 4.0, 6.0)
        ax.vlines(dBP_LOS, 0, PL_LOS_dB[dBP_LOS_index], line_color, "dotted", lw=2)
        ax.fill_between(
            x, PL_LOS_dB - sigma, PL_LOS_dB + sigma, color=line_color, alpha=0.2
        )
        ax.set_xlim(0, np.max(x))
        fnbase = "img/RMa_pathloss_model"
    else:
        ax.plot(x, PG_LOS, lw=2, label="LOS pathgain")
        ax.set_ylim(0)
        ax.set_xlim(0, 1000)
        fnbase = "img/RMa_pathgain_model"

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
        axins.plot(x, PL_NLOS_dB_const, lw=1.5, linestyle="--", color="#abe6e2")
        axins.plot(x, PL_NLOS_dB_disc, lw=1.5, linestyle=":", color="hotpink")
        axins.plot(x, PL_LOS_dB_const, lw=1.5, linestyle="--", color="yellow")
        axins.plot(x, PL_LOS_dB_disc, lw=1.5, linestyle=":", color="orangered")
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks(np.arange(x1, x2, 10))
        axins.set_yticks(np.arange(y1, y2, 1))
        axins.tick_params(axis="both", direction="in")
        axins.grid(color="gray", alpha=0.7, lw=0.5)
        ax.indicate_inset_zoom(axins, edgecolor="gray")

    disc_model_ref = RMa_pathloss_discretised()
    h_bs_res = disc_model_ref.h_bs_res
    h_ut_res = disc_model_ref.h_ut_res
    discrete_h_bs = np.arange(10.0, 150.0 + h_bs_res, h_bs_res)
    worst_case_h_bs = (discrete_h_bs[:-1] + discrete_h_bs[1:]) / 2.0
    # ^^ above code adds two consecutive heights and divides by two to find midpoint
    discrete_h_ut = np.arange(1.0, 10.0 + h_ut_res, h_ut_res)
    worst_case_h_ut = (discrete_h_ut[:-1] + discrete_h_ut[1:]) / 2.0
    error_table_nlos, error_table_los = [], []
    disc_nlos_sweep = RMa_pathloss_discretised(
        fc_GHz=fc_GHz, LOS=False, h_ut_res=h_ut_res, h_bs_res=h_bs_res
    )
    disc_los_sweep = RMa_pathloss_discretised(
        fc_GHz=fc_GHz, LOS=True, h_ut_res=h_ut_res, h_bs_res=h_bs_res
    )
    d2D_m_w = x[:, np.newaxis]
    for h_bs_w in worst_case_h_bs:
        for h_ut_w in worst_case_h_ut:
            xyz_cells_w = np.array([[0.0, 0.0, h_bs_w]])
            xyz_ues_w = np.column_stack((x, np.zeros_like(x), np.full_like(x, h_ut_w)))
            d3D_m_w = np.hypot(x, h_bs_w - h_ut_w)[:, np.newaxis]
            pl_exact_nlos = RMa_pathloss(
                fc_GHz=fc_GHz, h_UT=h_ut_w, h_BS=h_bs_w, LOS=False
            ).get_pathloss_dB(d2D_m_w, d3D_m_w, xyz_ues_w, xyz_cells_w)
            pl_disc_nlos = disc_nlos_sweep.get_pathloss_dB(
                d2D_m_w, d3D_m_w, xyz_ues_w, xyz_cells_w
            )
            error_table_nlos.append(np.max(np.abs(pl_exact_nlos - pl_disc_nlos)))
            pl_exact_los = RMa_pathloss(
                fc_GHz=fc_GHz, h_UT=h_ut_w, h_BS=h_bs_w, LOS=True
            ).get_pathloss_dB(d2D_m_w, d3D_m_w, xyz_ues_w, xyz_cells_w)
            pl_disc_los = disc_los_sweep.get_pathloss_dB(
                d2D_m_w, d3D_m_w, xyz_ues_w, xyz_cells_w
            )
            error_table_los.append(np.max(np.abs(pl_exact_los - pl_disc_los)))

    max_err_nlos = np.max(error_table_nlos)
    max_err_los = np.max(error_table_los)
    rmse_nlos = np.sqrt(np.mean((PL_NLOS_dB - PL_NLOS_dB_disc) ** 2))
    rmse_los = np.sqrt(np.mean((PL_LOS_dB - PL_LOS_dB_disc) ** 2))
    error_text = (
        f"RMSE (discretised vs exact):\n"
        f"NLOS: {rmse_nlos:.2g} dB\n"
        f"LOS:  {rmse_los:.2g} dB\n"
        f"Max error:                            \n"
        f"NLOS: {max_err_nlos:.2g} dB\n"
        f"LOS:  {max_err_los:.2g} dB"
    )
    ax.text(
        0.27,
        0.98,
        error_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

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
        print(f"{BLUE}RMa-NLOS:       {PL_NLOS_dB[0]:.2f} dB")
        print(f"{ORANGE}RMa-LOS:        {PL_LOS_dB[0]:.2f} dB")
        print(f"{RED}Free-space:     {fs_pathloss_dB[0]:.2f} dB{RESET}\n")

    # Add timestamp and save figures
    fig_timestamp(fig, rotation=0, fontsize=6, author=author)
    fig.savefig(f"{fnbase}.png")
    print(f"eog {fnbase}.png &")
    fig.savefig(f"{fnbase}.pdf")
    print(f"evince --page-label=1  {fnbase}.pdf &")


def plot_RMa_pathloss_runtime_comparison(author=""):
    """
    Compare RMa pathloss model runtimes.

    This function provides a lightweight performance benchmark for three distinct
    implementations of the RMa pathloss model:

    1. :class:`~RMa_pathloss_model_07.RMa_pathloss`: The full model with dynamic height calculations.
    2. :class:`~RMa_pathloss_model_07.RMa_pathloss_constant_height`: A simplified model assuming fixed antenna heights.
    3. :class:`~RMa_pathloss_model_07.RMa_pathloss_discretised`: A model that uses pre-calculated coefficients
        based on a discrete grid of antenna heights.

    It directly measures the execution time required to calculate pathloss for an
    increasing number of UEs, thereby isolating the computational cost of each
    model from the overhead of the full CRRM simulation framework. The primary
    purpose is to visually demonstrate the performance gains achieved by the
    simplified and discretised models.

    Parameters
    ----------
    author : str, optional
        The name of the author to be included in the plot's timestamp metadata.
        If not provided, the author field will be empty.

    Returns
    -------
    None
        This function does not return any values. Its primary outputs are plot
        files (`.png`, `.pdf`) saved to the local directory and status messages
        printed to the console.

    See Also
    --------
    :class:`~RMa_pathloss_model_07.RMa_pathloss`
    :class:`~RMa_pathloss_model_07.RMa_pathloss_constant_height`
    :class:`~RMa_pathloss_model_07.RMa_pathloss_discretised`

    Notes
    -----
    The comparison methodology involves the following steps for each specified UE count:
    1.  Randomly generate locations for UEs and a fixed set of cell sites.
    2.  Calculate the 2D and 3D distance matrices between every UE and cell.
    3.  Time the `get_pathloss` method for each of the three model variants using these distance matrices.

    The resulting plot displays two sets of data:
    - The absolute runtime in milliseconds for each model on the primary y-axis.
    - The runtime ratio of the exact model to the optimised models on a
    secondary y-axis, quantifying the speed-up factor.

    Examples
    --------
    To run the comparison and generate the output plots, simply call the function
    from a script where the necessary modules are in the Python path.

    >>> plot_RMa_pathloss_runtime_comparison(author='Ibrahim Nur')

    This will produce 'plot_RMa_pathloss_runtime_comparison.png' and
    'plot_RMa_pathloss_runtime_comparison.pdf' in the current directory.
    """
    try:
        import matplotlib.pyplot as plt
        import time
        from utilities import fig_timestamp
    except ImportError as e:
        print(f"Error importing modules: {e}")
    h_BS, h_UT, fc_GHz, n_cells = 35.0, 1.5, 3.5, 10
    ue_counts = np.linspace(100, 7500, 50, dtype=int)
    times_exact, times_const, times_disc = [], [], []
    rng_cells = np.random.default_rng(42)
    cell_locations = np.zeros((n_cells, 3))
    cell_locations[:, 0] = rng_cells.uniform(-2500, 2500, n_cells)
    cell_locations[:, 1] = rng_cells.uniform(-2500, 2500, n_cells)
    cell_locations[:, 2] = h_BS
    model_exact = RMa_pathloss(fc_GHz=fc_GHz)
    model_const = RMa_pathloss_constant_height(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS)
    model_disc = RMa_pathloss_discretised(fc_GHz=fc_GHz)
    for n_ues in ue_counts:
        rng = np.random.default_rng(69420)
        ue_locations = np.zeros((n_ues, 3))
        ue_locations[:, 0] = rng.uniform(10, 5000, n_ues)
        ue_locations[:, 1] = rng.uniform(-100, 100, n_ues)
        ue_locations[:, 2] = h_UT
        d2D_m = np.linalg.norm(
            ue_locations[:, np.newaxis, :2] - cell_locations[np.newaxis, :, :2], axis=2
        )
        d3D_m = np.linalg.norm(
            ue_locations[:, np.newaxis] - cell_locations[np.newaxis], axis=2
        )
        start_time = time.perf_counter()
        model_exact.get_pathloss(d2D_m, d3D_m, ue_locations, cell_locations)
        times_exact.append((time.perf_counter() - start_time) * 1000)
        start_time = time.perf_counter()
        model_const.get_pathloss(d2D_m, d3D_m)
        times_const.append((time.perf_counter() - start_time) * 1000)
        start_time = time.perf_counter()
        model_disc.get_pathloss(d2D_m, d3D_m, ue_locations, cell_locations)
        times_disc.append((time.perf_counter() - start_time) * 1000)
    times_exact, times_const, times_disc = (
        np.array(times_exact),
        np.array(times_const),
        np.array(times_disc),
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ue_counts, times_exact, "o-", color="blue", label="exact")
    ax.plot(ue_counts, times_const, "s--", color="#abe6e2", label="constant height")
    ax.plot(ue_counts, times_disc, "x--", color="hotpink", label="discretised")
    info = (
        f"RMa (exact) is {(np.mean(times_exact/times_const)-1)*100:.0f}% slower than constant height.\n"
        f"RMa (discretised) is only {(np.mean(times_disc/times_const)-1)*100:.0f}% slower than constant height.\n"
        f"The discrepancy in accuracy can be found by running\n"
        f"plot_RMa_pathloss_or_pathgain(),\n"
        f"and viewing its results in RMa_pathloss_model.pdf."
    )
    ax.text(
        0.05,
        0.98,
        info,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_xlabel("Number of UEs")
    ax.set_ylabel("Runtime (milliseconds)")
    ax.set_title("RMa pathloss model runtime comparison")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.legend(loc="lower right")
    ax.grid(color="gray", alpha=0.7, lw=0.5)
    fig.tight_layout()
    fnbase = "img/RMa_pathloss_runtime_comparison"
    fig_timestamp(fig, author="Ibrahim Nur")
    fig.savefig(f"{fnbase}.png")
    print(f"eog {fnbase}.png &")
    fig.savefig(f"{fnbase}.pdf")
    print(f"evince --page-label=1 {fnbase}.pdf &")


if __name__ == "__main__":
    plot_RMa_pathloss_or_pathgain(
        plot_type="pathloss",
        zoom_box=True,
        print_10m_pl=True,
        author="Kishan Sthankiya & Ibrahim Nur",
    )
    plot_RMa_pathloss_runtime_comparison()
