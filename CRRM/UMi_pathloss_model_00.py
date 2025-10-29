# Keith Briggs 2025-09-24
# Ibrahim Nur 2025-09-11

from sys import path

import numpy as np


def to_dB(x):
    return 10.0 * np.log10(x)


def from_dB(x):
    return np.power(10.0, x / 10.0)


class UMi_pathloss_constant_height:
    """
    Urban Microcell (UMi) pathloss model assuming constant antenna heights.

    This class implements a simplified version of the 3GPP TR 38.901 UMi model.
    By assuming that the heights of the User Terminal and Base Station are
    constant for all links, it can pre-calculate several coefficients during
    initialisation. This leads to a significant performance gain during runtime
    compared to the full model, which must compute these values dynamically for
    each link.

    Attributes
    ----------
    fc_GHz : float
        Centre frequency in GHz.
    h_BS : float
        Height of the base station in metres.
    h_UT : float
        Height of the user terminal in metres.
    LOS : bool
        Indicates if the Line-of-Sight (True) or Non-Line-of-Sight (False)
        model is active.

    References
    ----------
    - 3GPP TR 38.901, Section 7.4.1, Table 7.4.1-1.
    """

    def __init__(self, fc_GHz=3.5, h_UT=1.5, h_BS=10.0, LOS=True, **args):
        """
        Initialise the UMi_pathloss_constant_height model instance.

        This constructor sets up the model's parameters and pre-calculates
        the constant terms used in the pathloss formulas, based on the fixed
        antenna heights.

        Parameters
        ----------
        fc_GHz : float, optional
            Centre frequency in GHz, by default 3.5.
        h_UT : float, optional
            Height of the User Terminal in metres. Must be in [1.5, 22.5].
            Defaults to 1.5.
        h_BS : float, optional
            Height of the Base Station in metres, by default 10.0.
        LOS : bool, optional
            Specifies whether to use the Line-of-Sight model. Defaults to True.
        **args
            Catches unused keyword arguments for API compatibility.

        Raises
        ------
        ValueError
            If `h_UT` is outside its valid range of [1.5, 22.5]m.
        """
        if not (1.5 <= h_UT <= 22.5):
            raise ValueError(f"h_UT={h_UT} is outside the valid range [1.5, 22.5]m")
        self.fc_GHz = fc_GHz
        self.h_BS = h_BS
        self.h_UT = h_UT
        self.LOS = LOS
        h_E = 1.0
        h_BS_eff = self.h_BS - h_E
        h_UT_eff = self.h_UT - h_E
        self.d_BP = 4 * h_BS_eff * h_UT_eff * (self.fc_GHz * 1e9) / 3.0e8
        fc_p2 = np.power(self.fc_GHz, 2.0)
        fc_p2_13 = np.power(self.fc_GHz, 2.13)
        self.los1_const = np.power(10.0, 3.24) * fc_p2
        los2_base = np.power(10.0, 3.24) * fc_p2
        los2_scale = np.power(
            self.d_BP**2 + np.power(self.h_BS - self.h_UT, 2.0), -0.95
        )
        self.los2_const = los2_base * los2_scale
        self.nlos_const = np.power(10.0, 2.24) * fc_p2_13
        self.nlos_h_term = np.power(10.0, -0.03 * (self.h_UT - 1.5))

    def get_pathloss_dB(self, d2D_m, d3D_m):
        """
        Calculate the pathloss in decibels (dB).

        This method is a convenience wrapper around :meth:`get_pathloss` that
        converts the linear pathloss value to the dB scale.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix in metres between UEs and cells.
        d3D_m : numpy.ndarray
            3D distance matrix in metres between UEs and cells.

        Returns
        -------
        numpy.ndarray
            The pathloss in decibels.
        """
        return to_dB(self.get_pathloss(d2D_m, d3D_m))  # retained for compatibility

    def get_pathloss(self, d2D_m, d3D_m):
        """
        Calculate the linear pathloss for each UE-cell link.

        This method computes the pathloss using pre-calculated coefficients.
        For Line-of-Sight (LOS) conditions, it applies a dual-slope model based
        on the breakpoint distance `d_BP`. For Non-Line-of-Sight (NLOS), it
        calculates the NLOS pathloss and returns the maximum of the two.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            A matrix of 2D distances (ground distances) in metres between UEs
            and cells. Shape: (n_ues, n_cells).
        d3D_m : numpy.ndarray
            A matrix of 3D distances (direct line-of-sight distances) in metres
            between UEs and cells. Shape: (n_ues, n_cells).

        Returns
        -------
        numpy.ndarray
            The calculated linear pathloss for each UE-cell link.
        """
        pl1_linear = self.los1_const * np.power(d3D_m, 2.1)
        pl2_linear = self.los2_const * np.power(d3D_m, 4.0)
        pl_los_linear = np.where(d2D_m <= self.d_BP, pl1_linear, pl2_linear)
        if self.LOS:
            return pl_los_linear
        else:
            pl_nlos_prime_linear = (
                self.nlos_const * np.power(d3D_m, 3.53) * self.nlos_h_term
            )
            return np.maximum(pl_los_linear, pl_nlos_prime_linear)

    def get_pathgain(self, d2D_m, d3D_m):
        """
        Calculate the linear pathgain.

        Pathgain is defined as the reciprocal of the linear pathloss.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix in metres.
        d3D_m : numpy.ndarray
            3D distance matrix in metres.

        Returns
        -------
        numpy.ndarray
            The linear pathgain for each UE-cell link.
        """
        return 1.0 / self.get_pathloss(d2D_m, d3D_m)

    def _get_approximate_pathloss_dB_for_layout_plot(self, d):
        """
        Calculate approximate pathloss for plotting purposes.

        This internal helper method is designed for generating layout plots. It
        takes a 1D array of distances and calculates the corresponding pathloss
        values in dB, assuming the default fixed heights set during the object's
        initialisation.

        Parameters
        ----------
        d : numpy.ndarray
            A 1D array of distances in metres for which to calculate pathloss.

        Returns
        -------
        numpy.ndarray
            The calculated pathloss in decibels for each distance in `d`.
        """
        U = np.array([[d, 0.0, 1.5]])
        C = np.array([[0.0, 0.0, 10.0]])
        d2D_m = np.array([[d]])
        d3D_m = np.linalg.norm(d - np.array([[0.0, 0.0, 1.5]]))
        return self.get_pathloss_dB(d2D_m, d3D_m, U, C)


class UMi_pathloss:
    """
    The complete Urban Microcell (UMi) pathloss model from 3GPP TR 38.901.

    This class implements the full UMi model where antenna heights can vary for
    each link. Unlike :class:`UMi_pathloss_constant_height`, it calculates all
    pathloss coefficients dynamically at runtime based on the specific heights
    of the UE and BS involved in each link. This provides maximum accuracy at
    the cost of computational performance.

    See Also
    --------
    :class:`UMi_pathloss_constant_height` : Simplified model with fixed heights.
    :class:`UMi_pathloss_discretised` : Performance-oriented model with quantised heights.

    References
    ----------
    - 3GPP TR 38.901, Section 7.4.1, Table 7.4.1-1.
    """

    def __init__(self, fc_GHz=3.5, h_UT=1.5, h_BS=10.0, LOS=True, **args):
        """
        Initialise the UMi_pathloss model instance.

        Parameters
        ----------
        fc_GHz : float, optional
            Centre frequency in GHz, by default 3.5.
        h_UT : float, optional
            Default height of the User Terminal in metres, used for plotting.
            Defaults to 1.5.
        h_BS : float, optional
            Default height of the Base Station in metres, used for plotting.
            Defaults to 10.0.
        LOS : bool, optional
            Specifies whether to use the Line-of-Sight model. Defaults to True.
        **args
            Catches unused keyword arguments for API compatibility.
        """
        self.fc_GHz = fc_GHz
        self.LOS = LOS
        self.default_h_UT = h_UT
        self.default_h_BS = h_BS
        fc_p2 = np.power(self.fc_GHz, 2.0)
        fc_p2_13 = np.power(self.fc_GHz, 2.13)
        self.los1_const = np.power(10.0, 3.24) * fc_p2
        self.nlos_const = np.power(10.0, 2.24) * fc_p2_13

    def get_pathloss_dB(self, d2D_m, d3D_m, U, C):
        """
        Calculate the pathloss in decibels (dB).

        This method is a convenience wrapper around :meth:`get_pathloss` that
        converts the linear pathloss value to the dB scale.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix in metres.
        d3D_m : numpy.ndarray
            3D distance matrix in metres.
        U : numpy.ndarray
            Array of UE locations and heights, shape (n_ues, 3).
        C : numpy.ndarray
            Array of cell locations and heights, shape (n_cells, 3).

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
        Calculate linear pathloss with dynamic antenna heights.

        This method computes pathloss by dynamically calculating coefficients
        for each UE-cell link based on their specific heights, extracted from
        the `U` and `C` arrays.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            A matrix of 2D distances in metres. Shape: (n_ues, n_cells).
        d3D_m : numpy.ndarray
            A matrix of 3D distances in metres. Shape: (n_ues, n_cells).
        U : numpy.ndarray
            Array of UE locations, shape (n_ues, 3). The 3rd column (height)
            must be in the range [1.5, 22.5].
        C : numpy.ndarray
            Array of cell locations, shape (n_cells, 3).

        Returns
        -------
        numpy.ndarray
            The calculated linear pathloss for each UE-cell link.

        Raises
        ------
        ValueError
            If any `h_UT` value is outside its valid range of [1.5, 22.5]m.
        """
        h_UT = U[:, 2:]
        h_BS = C[:, 2:].T
        if np.any(h_UT < 1.5) or np.any(h_UT > 22.5):
            raise ValueError(
                f"At least one h_UT value is outside the valid range [1.5, 22.5]m"
            )
        h_E = 1.0
        h_BS_eff = h_BS - h_E
        h_UT_eff = h_UT - h_E
        d_BP = 4 * h_BS_eff * h_UT_eff * (self.fc_GHz * 1e9) / 3.0e8
        pl1_linear = self.los1_const * np.power(d3D_m, 2.1)
        los2_scale = np.power(np.power(d_BP, 2.0) + np.power(h_BS - h_UT, 2.0), -0.95)
        pl2_linear = self.los1_const * los2_scale * np.power(d3D_m, 4.0)
        pl_los_linear = np.where(d2D_m <= d_BP, pl1_linear, pl2_linear)
        if self.LOS:
            return pl_los_linear
        else:
            nlos_h_term = np.power(10.0, -0.03 * (h_UT - 1.5))
            pl_nlos_prime_linear = self.nlos_const * np.power(d3D_m, 3.53) * nlos_h_term
            return np.maximum(pl_los_linear, pl_nlos_prime_linear)

    def get_pathgain(self, d2D_m, d3D_m, U, C):
        """
        Calculate the linear pathgain.

        Pathgain is defined as the reciprocal of the linear pathloss.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix in metres.
        d3D_m : numpy.ndarray
            3D distance matrix in metres.
        U : numpy.ndarray
            Array of UE locations and heights.
        C : numpy.ndarray
            Array of cell locations and heights.

        Returns
        -------
        numpy.ndarray
            The linear pathgain for each UE-cell link.
        """
        return 1.0 / self.get_pathloss(d2D_m, d3D_m, U, C)

    def _get_approximate_pathloss_dB_for_layout_plot(self, d):
        """
        Calculate approximate pathloss for plotting purposes.

        This internal helper method is designed for generating layout plots. It
        takes a 1D array of distances and calculates the corresponding pathloss
        values in dB, using the default fixed heights set during the object's
        initialisation.

        Parameters
        ----------
        d : numpy.ndarray
            A 1D array of distances in metres for which to calculate pathloss.

        Returns
        -------
        numpy.ndarray
            The calculated pathloss in decibels for each distance in `d`.
        """
        U = np.array([[d, 0.0, 1.5]])
        C = np.array([[0.0, 0.0, 10.0]])
        d2D_m = np.array([[d]])
        d3D_m = np.linalg.norm(d - np.array([[0.0, 0.0, 1.5]]))
        return self.get_pathloss_dB(d2D_m, d3D_m, U, C)


class UMi_pathloss_discretised:
    """
    Urban Microcell (UMi) pathloss model with discretised heights.

    This class implements a performance-optimised version of the UMi model. It
    pre-calculates pathloss coefficients for a grid of discrete UE and BS
    heights. At runtime, it rounds the actual antenna heights to the nearest
    grid point and uses the corresponding pre-calculated values. This approach
    trades a small amount of accuracy for a significant gain in computational
    speed, making it suitable for large-scale simulations.

    See Also
    --------
    :class:`UMi_pathloss` : The full, exact UMi pathloss model implementation.
    :class:`UMi_pathloss_constant_height` : Simplified model with fixed heights.

    References
    ----------
    - 3GPP TR 38.901, Section 7.4.1, Table 7.4.1-1.
    """

    def __init__(
        self,
        fc_GHz=3.5,
        h_ut_res=0.5,
        h_bs_res=1.0,
        h_bs_min=10.0,
        h_bs_max=25.0,
        LOS=True,
        **args,
    ):
        """
        Initialise the UMi_pathloss_discretised model instance.

        This constructor defines the discrete height grids for UEs and BSs and
        pre-calculates the pathloss coefficient matrices for every combination
        of heights on those grids.

        Parameters
        ----------
        fc_GHz : float, optional
            Centre frequency in GHz, by default 3.5.
        h_ut_res : float, optional
            Resolution of the discrete UT height grid in metres, by default 0.5.
        h_bs_res : float, optional
            Resolution of the discrete BS height grid in metres, by default 1.0.
        h_bs_min : float, optional
            The minimum BS height for the discretisation grid. Defaults to 10.0.
            3GPP does not specify a constraint on BS height, hence the user might
            want to change this value to suit their scenario.
        h_bs_max : float, optional
            The maximum BS height for the discretisation grid. Defaults to 10.0.
            3GPP does not specify a constraint on BS height, hence the user might
            want to change this value to suit their scenario.
        LOS : bool, optional
            Specifies whether to use the Line-of-Sight model. Defaults to True.
        **args
            Catches unused keyword arguments for API compatibility.
        """
        self.fc_GHz = fc_GHz
        self.h_ut_res = h_ut_res
        self.h_bs_res = h_bs_res
        self.LOS = LOS
        self.h_UT_min, self.h_UT_max = 1.5, 22.5
        self.h_BS_min, self.h_BS_max = (
            h_bs_min,
            h_bs_max,
        )  # No constraint provided in document.
        self.h_UT_grid = np.arange(
            self.h_UT_min, self.h_UT_max + self.h_ut_res, self.h_ut_res
        )[:, np.newaxis]
        self.h_BS_grid = np.arange(
            self.h_BS_min, self.h_BS_max + self.h_bs_res, self.h_bs_res
        )[np.newaxis]
        fc_p2 = np.power(self.fc_GHz, 2.0)
        fc_p2_13 = np.power(self.fc_GHz, 2.13)
        self.los1_const = np.power(10.0, 3.24) * fc_p2
        self.nlos_const = np.power(10.0, 2.24) * fc_p2_13
        h_E = 1.0
        h_BS_eff = self.h_BS_grid - h_E
        h_UT_eff = self.h_UT_grid - h_E
        self.d_BP_matrix = 4 * h_BS_eff * h_UT_eff * (self.fc_GHz * 1e9) / 3.0e8
        h_diff_sq_matrix = np.power(self.h_BS_grid - self.h_UT_grid, 2.0)
        self.los2_scaling_matrix = np.power(
            np.power(self.d_BP_matrix, 2.0) + h_diff_sq_matrix, -0.95
        )
        self.nlos_h_UT_term_matrix = np.power(10.0, -0.03 * (self.h_UT_grid - 1.5))

    def get_pathloss_dB(self, d2D_m, d3D_m, U, C):
        """
        Calculate the pathloss in decibels (dB).

        This method is a convenience wrapper around :meth:`get_pathloss` that
        converts the linear pathloss value to the dB scale.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix in metres.
        d3D_m : numpy.ndarray
            3D distance matrix in metres.
        U : numpy.ndarray
            Array of UE locations and heights, shape (n_ues, 3).
        C : numpy.ndarray
            Array of cell locations and heights, shape (n_cells, 3).

        Returns
        -------
        numpy.ndarray
            The pathloss in decibels.
        """
        return to_dB(self.get_pathloss(d2D_m, d3D_m, U, C))

    def get_pathloss(self, d2D_m, d3D_m, U, C):
        """
        Calculate linear pathloss using discretised height lookups.

        This method determines the appropriate indices in the pre-calculated
        coefficient matrices by rounding the actual UE and BS heights. It then
        uses these indexed values to compute the pathloss.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            A matrix of 2D distances in metres. Shape: (n_ues, n_cells).
        d3D_m : numpy.ndarray
            A matrix of 3D distances in metres. Shape: (n_ues, n_cells).
        U : numpy.ndarray
            Array of UE locations and heights. The 3rd column must be within
            the defined grid range.
        C : numpy.ndarray
            Array of cell locations and heights. The 3rd column must be within
            the defined grid range.

        Returns
        -------
        numpy.ndarray
            The calculated linear pathloss for each UE-cell link.

        Raises
        ------
        ValueError
            If any `h_UT` or `h_BS` value is outside its valid grid range.
        """
        h_UT = U[:, 2:]
        h_BS = C[:, 2:].T
        if np.any(h_UT < self.h_UT_min) or np.any(h_UT > self.h_UT_max):
            raise ValueError(
                f"At least one h_UT value is outside the valid range [{self.h_UT_min}, {self.h_UT_max}]m"
            )
        if np.any(h_BS < self.h_BS_min) or np.any(h_BS > self.h_BS_max):
            raise ValueError(
                f"At least one h_BS value is outside the valid range [{self.h_BS_min}, {self.h_BS_max}]m"
            )
        h_UT_i = np.round((h_UT - self.h_UT_min) / self.h_ut_res).astype(int)
        h_BS_j = np.round((h_BS - self.h_BS_min) / self.h_bs_res).astype(int)
        d_BP = self.d_BP_matrix[h_UT_i, h_BS_j]
        los2_scaling = self.los2_scaling_matrix[h_UT_i, h_BS_j]
        pl1_linear = self.los1_const * np.power(d3D_m, 2.1)
        pl2_linear = self.los1_const * los2_scaling * np.power(d3D_m, 4.0)
        pl_los_linear = np.where(d2D_m <= d_BP, pl1_linear, pl2_linear)
        if self.LOS:
            return pl_los_linear
        else:
            nlos_h_UT_term = self.nlos_h_UT_term_matrix[h_UT_i, 0]
            pl_nlos_prime_linear = (
                self.nlos_const * np.power(d3D_m, 3.53) * nlos_h_UT_term
            )
            return np.maximum(pl_los_linear, pl_nlos_prime_linear)

    def get_pathgain(self, d2D_m, d3D_m, U, C):
        """
        Calculate the linear pathgain.

        Pathgain is defined as the reciprocal of the linear pathloss.

        Parameters
        ----------
        d2D_m : numpy.ndarray
            2D distance matrix in metres.
        d3D_m : numpy.ndarray
            3D distance matrix in metres.
        U : numpy.ndarray
            Array of UE locations and heights.
        C : numpy.ndarray
            Array of cell locations and heights.

        Returns
        -------
        numpy.ndarray
            The linear pathgain for each UE-cell link.
        """
        return 1.0 / self.get_pathloss(d2D_m, d3D_m, U, C)

    def _get_approximate_pathloss_dB_for_layout_plot(self, d):
        """
        Calculate approximate pathloss for plotting purposes.

        This internal helper method is designed for generating layout plots. It
        takes a 1D array of distances and calculates the corresponding pathloss
        values in dB, using the default fixed heights set during the object's
        initialisation.

        Parameters
        ----------
        d : numpy.ndarray
            A 1D array of distances in metres for which to calculate pathloss.

        Returns
        -------
        numpy.ndarray
            The calculated pathloss in decibels for each distance in `d`.
        """
        U = np.array([[d, 0.0, 1.5]])
        C = np.array([[0.0, 0.0, 10.0]])
        d2D_m = np.array([[d]])
        d3D_m = np.linalg.norm(d - np.array([[0.0, 0.0, 1.5]]))
        return self.get_pathloss_dB(d2D_m, d3D_m, U, C)


def plot_UMi_pathloss_or_pathgain(
    plot_type="pathloss",
    fc_GHz=3.5,
    h_UT=1.5,
    h_BS=10.0,
    zoom_box=False,
    print_10m_pl=False,
    author=" ",
    x_min=35.0,
    x_max=5000.0,
):
    """
    Plot 3GPP UMi pathloss or pathgain model predictions as a self-test.

    This function generates a plot comparing various 3GPP UMi pathloss or
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
    The function uses the :class:`UMi_pathloss`,
    :class:`UMi_pathloss_constant_height`, and
    :class:`UMi_pathloss_discretised` classes to compute the values for
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
    d3D_m = np.linalg.norm(xyz_ues[:, np.newaxis] - xyz_cells[np.newaxis], axis=2)

    # Plot NLOS
    PL_nlos_model = UMi_pathloss(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS, LOS=False)
    h_E = 1.0  # Effective environment height
    dBP = 4 * (h_BS - h_E) * (h_UT - h_E) * (fc_GHz * 1e9) / 3.0e8
    dBP_index = np.clip(np.searchsorted(x, dBP) - 1, 0, len(x) - 1)
    PL_NLOS_dB = PL_nlos_model.get_pathloss_dB(
        d2D_m, d3D_m, xyz_ues, xyz_cells
    ).squeeze()
    PG_NLOS = 1.0 / np.power(10.0, PL_NLOS_dB / 10.0)

    if plot_type == "pathloss":
        ax.set_title(f"3GPP UMi pathloss models (dBP={dBP:.0f}m)")
        ax.set_ylabel("pathloss (dB)")
        line = ax.plot(
            x, PL_NLOS_dB, lw=2, label=r"NLOS exact ($\sigma=7.82$)", color="blue"
        )
        line_color = line[0].get_color()
        PL_NLOS_dB_const = (
            UMi_pathloss_constant_height(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS, LOS=False)
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
            UMi_pathloss_discretised(fc_GHz=fc_GHz, LOS=False)
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

        ax.vlines(dBP, 0, PL_NLOS_dB[dBP_index], line_color, "dotted", lw=2)
        ax.fill_between(
            x, PL_NLOS_dB - 7.82, PL_NLOS_dB + 7.82, color=line_color, alpha=0.2
        )
        ax.set_ylim(50)
    else:
        ax.set_title(f"3GPP UMi pathgain models (dBP={dBP:.0f}m)")
        ax.set_ylabel("pathgain")
        ax.plot(x, PG_NLOS, lw=2, label="NLOS pathgain")

    # Plot LOS
    PL_los_model = UMi_pathloss(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS, LOS=True)
    PL_LOS_dB = PL_los_model.get_pathloss_dB(d2D_m, d3D_m, xyz_ues, xyz_cells).squeeze()
    PG_LOS = 1.0 / np.power(10.0, PL_LOS_dB / 10.0)

    if plot_type == "pathloss":
        line = ax.plot(x, PL_LOS_dB, lw=2, label=r"LOS ($\sigma=4$)", color="orange")
        line_color = line[0].get_color()
        PL_LOS_dB_const = (
            UMi_pathloss_constant_height(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS, LOS=True)
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
            UMi_pathloss_discretised(fc_GHz=fc_GHz, LOS=True)
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
        sigma = np.where(np.less_equal(x, dBP), 4.0, 4.0)
        ax.vlines(dBP, 0, PL_LOS_dB[dBP_index], line_color, "dotted", lw=2)
        ax.fill_between(
            x, PL_LOS_dB - sigma, PL_LOS_dB + sigma, color=line_color, alpha=0.2
        )
        ax.set_xlim(0, np.max(x))
        fnbase = "img/UMi_pathloss_model"
    else:
        ax.plot(x, PG_LOS, lw=2, label="LOS pathgain")
        ax.set_ylim(0)
        ax.set_xlim(0, 1000)
        fnbase = "img/UMi_pathgain_model"

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
        x1, x2, y1, y2 = 30, 90, 75, 100

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
        axins.set_yticks(np.arange(y1, y2, 5))
        axins.tick_params(axis="both", direction="in")
        axins.grid(color="gray", alpha=0.7, lw=0.5)
        ax.indicate_inset_zoom(axins, edgecolor="gray")

    disc_model_ref = UMi_pathloss_discretised()
    h_bs_res = disc_model_ref.h_bs_res
    h_ut_res = disc_model_ref.h_ut_res
    discrete_h_bs = np.arange(10.0, 25.0 + h_bs_res, h_bs_res)
    worst_case_h_bs = (discrete_h_bs[:-1] + discrete_h_bs[1:]) / 2.0
    # ^^ above code adds two consecutive heights and divides by two to find midpoint
    discrete_h_ut = np.arange(1.5, 10.0 + h_ut_res, h_ut_res)
    worst_case_h_ut = (discrete_h_ut[:-1] + discrete_h_ut[1:]) / 2.0
    error_table_nlos, error_table_los = [], []
    disc_nlos_sweep = UMi_pathloss_discretised(
        fc_GHz=fc_GHz, LOS=False, h_ut_res=h_ut_res, h_bs_res=h_bs_res
    )
    disc_los_sweep = UMi_pathloss_discretised(
        fc_GHz=fc_GHz, LOS=True, h_ut_res=h_ut_res, h_bs_res=h_bs_res
    )
    d2D_m_w = x[:, np.newaxis]
    for h_bs_w in worst_case_h_bs:
        for h_ut_w in worst_case_h_ut:
            xyz_cells_w = np.array([[0.0, 0.0, h_bs_w]])
            xyz_ues_w = np.column_stack((x, np.zeros_like(x), np.full_like(x, h_ut_w)))
            d3D_m_w = np.hypot(x, h_bs_w - h_ut_w)[:, np.newaxis]
            pl_exact_nlos = UMi_pathloss(
                fc_GHz=fc_GHz, h_UT=h_ut_w, h_BS=h_bs_w, LOS=False
            ).get_pathloss_dB(d2D_m_w, d3D_m_w, xyz_ues_w, xyz_cells_w)
            pl_disc_nlos = disc_nlos_sweep.get_pathloss_dB(
                d2D_m_w, d3D_m_w, xyz_ues_w, xyz_cells_w
            )
            error_table_nlos.append(np.max(np.abs(pl_exact_nlos - pl_disc_nlos)))
            pl_exact_los = UMi_pathloss(
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
        print(f"{BLUE}UMi-NLOS:       {PL_NLOS_dB[0]:.2f} dB")
        print(f"{ORANGE}UMi-LOS:        {PL_LOS_dB[0]:.2f} dB")
        print(f"{RED}Free-space:     {fs_pathloss_dB[0]:.2f} dB{RESET}\n")

    # Add timestamp and save figures
    fig_timestamp(fig, rotation=0, fontsize=6, author=author)
    fig.savefig(f"{fnbase}.png")
    print(f"eog {fnbase}.png &")
    fig.savefig(f"{fnbase}.pdf")
    print(f"evince --page-label=1  {fnbase}.pdf &")


def plot_UMi_pathloss_runtime_comparison(author=""):
    """
    Compare UMi pathloss model runtimes.

    This function provides a lightweight performance benchmark for three distinct
    implementations of the UMi pathloss model:

    1. :class:`~UMi_pathloss_model_00.UMi_pathloss`: The full model with dynamic height calculations.
    2. :class:`~UMi_pathloss_model_00.UMi_pathloss_constant_height`: A simplified model assuming fixed antenna heights.
    3. :class:`~UMi_pathloss_model_00.UMi_pathloss_discretised`: A model that uses pre-calculated coefficients
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
    :class:`~UMi_pathloss_model_00.UMi_pathloss`
    :class:`~UMi_pathloss_model_00.UMi_pathloss_constant_height`
    :class:`~UMi_pathloss_model_00.UMi_pathloss_discretised`

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

    >>> plot_UMi_pathloss_runtime_comparison(author='Ibrahim Nur')

    This will produce 'plot_UMi_pathloss_runtime_comparison.png' and
    'plot_UMi_pathloss_runtime_comparison.pdf' in the current directory.
    """
    try:
        import matplotlib.pyplot as plt
        import time
        from utilities import fig_timestamp
    except ImportError as e:
        print(f"Error importing modules: {e}")
    h_BS, h_UT, fc_GHz, n_cells = 10.0, 1.5, 3.5, 10
    ue_counts = np.linspace(100, 7500, 50, dtype=int)
    times_exact, times_const, times_disc = [], [], []
    rng_cells = np.random.default_rng(42)
    cell_locations = np.zeros((n_cells, 3))
    cell_locations[:, 0] = rng_cells.uniform(-2500, 2500, n_cells)
    cell_locations[:, 1] = rng_cells.uniform(-2500, 2500, n_cells)
    cell_locations[:, 2] = h_BS
    model_exact = UMi_pathloss(fc_GHz=fc_GHz)
    model_const = UMi_pathloss_constant_height(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS)
    model_disc = UMi_pathloss_discretised(fc_GHz=fc_GHz)
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
        f"UMi (exact) is {(np.mean(times_exact/times_const)-1)*100:.0f}% slower than constant height.\n"
        f"UMi (discretised) is only {(np.mean(times_disc/times_const)-1)*100:.0f}% slower than constant height.\n"
        f"The discrepancy in accuracy can be found by running\n"
        f"plot_UMi_pathloss_or_pathgain(),\n"
        f"and viewing its results in UMi_pathloss_model.pdf."
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
    ax.set_title("UMi pathloss model runtime comparison")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.legend(loc="lower right")
    ax.grid(color="gray", alpha=0.7, lw=0.5)
    fig.tight_layout()
    fnbase = "img/UMi_pathloss_runtime_comparison"
    fig_timestamp(fig, author="Ibrahim Nur")
    fig.savefig(f"{fnbase}.png")
    print(f"eog {fnbase}.png &")
    fig.savefig(f"{fnbase}.pdf")
    print(f"evince --page-label=1 {fnbase}.pdf &")


if __name__ == "__main__":
    plot_UMi_pathloss_or_pathgain(
        plot_type="pathloss",
        zoom_box=True,
        print_10m_pl=True,
        author="Kishan Sthankiya & Ibrahim Nur",
    )
    plot_UMi_pathloss_runtime_comparison()
