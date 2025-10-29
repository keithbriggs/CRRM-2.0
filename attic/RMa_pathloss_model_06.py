# Keith Briggs 2025-09-07
# Ibrahim Nur 2025-09-01

from sys import path

import numpy as np

from utilities import from_dB, to_dB


class RMa_pathloss_constant_height:
    """
    Rural Macrocell (RMa) pathloss model from 3GPP TR 38.901, Table 7.4.1-1.
    This model assumes constant UE and base station heights.

    This model covers the cases 3D-UMa LOS and NLOS:
    - 3D-RMa: Three-dimensional rural macrocell model.
    - LOS   : Line-of-sight.
    - NLOS  : Non-line-of-sight.

    References:
    - 3GPP TR 38.901: https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173

    Parameters
    ----------
    fc_GHz : float
        Centre frequency in gigahertz. The RMa model is valid for
        frequencies up to 7 GHz.
    h_BS : float
        Height of the Base Station (BS) antenna in metres.
        Must be between 10 m and 150 m.
    h_UT : float
        Height of the User Terminal (UT) antenna in metres.
        Must be between 1 m and 10 m.
    LOS : bool, optional
        Whether the line-of-sight model is to be used (default is True).

    Attributes
    ----------
    d_BP : float
        The 2D breakpoint distance in metres. For the LOS model, the
        pathloss exponent changes at this distance.
    PL_1_at_d_BP : float
        The pre-calculated LOS pathloss value at the breakpoint distance.
    nlos_term_A : float
        The pre-calculated constant part of the NLOS pathloss formula.
    nlos_term_B : float
        The pre-calculated distance-dependent coefficient for the NLOS formula.'''
    """

    def __init__(s, fc_GHz=3.5, h_UT=1.5, h_BS=35.0, LOS=True, **args):
        """
        Initialise a pathloss model instance.

        Raises
        ------
        ValueError
            If `h_UT` is not between 1 m and 10 m.
            If `h_BS` is not between 10 m and 150 m.

        Note: all formulae in log space have been transformed to linear space.
        """
        _attributes_doc = """
      Attributes
      ----------
      d_BP : float
          The 2D breakpoint distance in metres. For the LOS model, the
          pathloss exponent changes at this distance.
      PL_1_at_d_BP : float
          The pre-calculated LOS pathloss value at the breakpoint distance.
      nlos_term_A : float
          The pre-calculated constant part of the NLOS pathloss formula.
      nlos_term_B : float
          The pre-calculated distance-dependent coefficient for the NLOS formula."""
        if not (1.0 <= h_UT <= 10.0):
            raise ValueError(f"h_UT={h_UT} is outside the valid range [1.0, 10.0]m")
        if not (10.0 <= h_BS <= 150.0):
            raise ValueError(f"h_BS={h_BS} is outside the valid range [10.0, 150.0]m")
        s.fc_GHz = fc_GHz
        s.h_BS = h_BS
        s.h_UT = h_UT
        s.LOS = LOS
        s.d_BP = 2 * np.pi * h_BS * h_UT * (fc_GHz * 1e9) / 3.0e8
        h = 5.0  # building height = 5m
        log_h_BS = np.log10(h_BS)
        # LOS terms are calculated linearly
        s.los_term1 = 2.0 + np.minimum(0.03 * h**1.72, 10) / 10.0
        s.los_term2 = (40 * np.pi * s.fc_GHz / 3) ** 2 * from_dB(
            -np.minimum(0.044 * h**1.72, 14.77)
        )
        s.los_term3 = 0.002 * np.log10(h) / 10.0
        # Constants in dB are separated from constants multiplied by a function of distance
        # i.e. PL_NLOS_dB = A_dB + B_dB*log10(d)
        A_dB = (
            161.04
            - 7.1 * np.log10(20.0)
            + 7.5 * np.log10(h)
            - (24.37 - 3.7 * (h / s.h_BS) ** 2) * log_h_BS
            + (43.42 - 3.1 * log_h_BS) * (-3)
            + 20.0 * np.log10(s.fc_GHz)
            - (3.2 * (np.log10(11.75 * s.h_UT)) ** 2 - 4.97)
        )
        B_dB = 43.42 - 3.1 * log_h_BS
        # A simple conversion to linear is then possible: 10^(A_dB/10) * d^(B_dB/10) in get_pathloss
        # The code below calculates the linear pathloss at breakpoint distance, which is used in PL_2
        s.nlos_term_A = from_dB(A_dB)
        s.nlos_term_B = B_dB / 10.0
        s.PL_1_at_d_BP = (
            s.los_term2 * (s.d_BP**s.los_term1) * (10 ** (s.los_term3 * s.d_BP))
        ) / (s.d_BP**4)

    def get_pathloss_dB(s, d2D_m, d3D_m):
        return to_dB(s.get_pathloss(d2D_m, d3D_m))  # retained for compatibility

    def get_pathloss(s, d2D_m, d3D_m):
        pl1_linear = s.los_term2 * (d3D_m**s.los_term1) * (10 ** (s.los_term3 * d3D_m))
        pl2_linear = s.PL_1_at_d_BP * (d3D_m**4)
        pl_los_linear = np.where(d2D_m <= s.d_BP, pl1_linear, pl2_linear)
        if s.LOS:
            return pl_los_linear
        else:
            pl_nlos_linear = s.nlos_term_A * (d3D_m**s.nlos_term_B)
            return np.maximum(pl_los_linear, pl_nlos_linear)

    def _get_approximate_pathloss_dB_for_layout_plot(s, d):
        # FIXME
        d2D_m = d.flatten()
        d3D_m = np.hypot(d2D_m, s.h_BS - s.h_UT)
        return s.get_pathloss_dB(d2D_m, d3D_m)

    def get_pathgain(s, d2D_m, d3D_m):
        return 1.0 / s.get_pathloss(d2D_m, d3D_m)


# END class RMa_pathloss_constant_height


class RMa_pathloss:
    """
    Rural Macrocell (RMa) pathloss model from 3GPP TR 38.901, Table 7.4.1-1.
    This model does not assume constant UE and base station heights.

    This model covers the cases 3D-UMa LOS and NLOS:
    - 3D-RMa: Three-dimensional rural macrocell model.
    - LOS   : Line-of-sight.
    - NLOS  : Non-line-of-sight.

    References:
    - 3GPP TR 38.901: https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173

    Parameters
    ----------
    fc_GHz : float
        Centre frequency in gigahertz. The RMa model is valid for
        frequencies up to 7 GHz.
    h_BS : float
        Height of the Base Station (BS) antenna in metres.
        Must be between 10 m and 150 m.
    h_UT : float
        Height of the User Terminal (UT) antenna in metres.
        Must be between 1 m and 10 m.
    LOS : bool, optional
        Whether the line-of-sight model is to be used (default is True).

    Attributes
    ----------
    fc_GHz : float
        Centre frequency in GHz.
    h_BS : float
        Height of the base station in metres.
    h_UT : float
        Height of the user terminal in metres.
    LOS : bool
        Indicates if the LOS model is used.
    d_BP : float
        The 2D breakpoint distance in metres. For the LOS model, the
        pathloss exponent changes at this distance.
    PL_1_at_d_BP : float
        The pre-calculated LOS pathloss value at the breakpoint distance.
    nlos_term_A : float
        The pre-calculated constant part of the NLOS pathloss formula.
    nlos_term_B : float
        The pre-calculated distance-dependent coefficient for the NLOS formula.
    """

    def __init__(s, fc_GHz=3.5, LOS=True, **args):
        """
        Initialise a pathloss model instance.
        """
        s.fc_GHz = fc_GHz
        s.LOS = LOS

    def get_pathloss_dB(s, d2D_m, d3D_m, U, C):
        return to_dB(s.get_pathloss(d2D_m, d3D_m, U, C))  # retained for compatibility

    def get_pathloss(s, d2D_m, d3D_m, U, C):
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
        d_BP = 2 * np.pi * h_BS * h_UT * (s.fc_GHz * 1e9) / 3.0e8
        h = 5.0  # building height = 5m
        log_h_BS = np.log10(h_BS)
        # LOS terms are calculated linearly
        los_term1 = 2.0 + np.minimum(0.03 * np.power(h, 1.72), 10) / 10.0
        los_term2 = np.power((40 * np.pi * s.fc_GHz / 3), 2) * from_dB(
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
            + 20 * np.log10(s.fc_GHz)
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
        if s.LOS:
            return pl_los_linear
        else:
            pl_nlos_linear = nlos_term_A * np.power(d3D_m, nlos_term_B)
            return np.maximum(pl_los_linear, pl_nlos_linear)

    def _get_approximate_pathloss_dB_for_layout_plot(s, d):
        d_flat = d.flatten()
        num_pts = len(d_flat)
        U = np.column_stack((d_flat, np.zeros(num_pts), np.full(num_pts, 1.5)))
        C = np.array([[0.0, 0.0, 35.0]])
        d2D_m = np.linalg.norm(U[:, np.newaxis, :2] - C[np.newaxis, :, :2], axis=2)
        d3D_m = np.linalg.norm(U[:, np.newaxis] - C[np.newaxis], axis=2)
        return s.get_pathloss_dB(d2D_m, d3D_m, U, C).squeeze()

    def get_pathgain(s, d2D_m, d3D_m, U, C):
        return 1.0 / s.get_pathloss(d2D_m, d3D_m, U, C)


# END class RMa_pathloss


class RMa_pathloss_discretised:
    """
    Rural Macrocell (RMa) pathloss model from 3GPP TR 38.901, Table 7.4.1-1.
    This model does not assume constant UE and base station heights.
    Coefficients used in the formulae use discretised UE and base station heights.
    The precision is controlled separately by h_ut_res (m) and h_bs_res (m).

    This model covers the cases 3D-UMa LOS and NLOS:
    - 3D-RMa: Three-dimensional rural macrocell model.
    - LOS   : Line-of-sight.
    - NLOS  : Non-line-of-sight.

    References:
    - 3GPP TR 38.901: https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173

    Parameters
    ----------
    fc_GHz : float
        Centre frequency in gigahertz. The RMa model is valid for
        frequencies up to 7 GHz.
    h_BS : float
        Height of the Base Station (BS) antenna in metres.
        Must be between 10 m and 150 m.
    h_UT : float
        Height of the User Terminal (UT) antenna in metres.
        Must be between 1 m and 10 m.
    LOS : bool, optional
        Whether the line-of-sight model is to be used (default is True).

    Attributes
    ----------
    fc_GHz : float
        Centre frequency in GHz.
    h_BS : float
        Height of the base station in metres.
    h_UT : float
        Height of the user terminal in metres.
    LOS : bool
        Indicates if the LOS model is used.
    d_BP : float
        The 2D breakpoint distance in metres. For the LOS model, the
        pathloss exponent changes at this distance.
    PL_1_at_d_BP : float
        The pre-calculated LOS pathloss value at the breakpoint distance.
    nlos_term_A : float
        The pre-calculated constant part of the NLOS pathloss formula.
    nlos_term_B : float
        The pre-calculated distance-dependent coefficient for the NLOS formula.
    """

    def __init__(s, fc_GHz=3.5, LOS=True, h_ut_res=0.5, h_bs_res=1.0, **args):
        """
        Initialize a pathloss model instance with antenna height grids and breakpoint distance.  This method sets up internal arrays for user terminal (UT) and base station (BS) heights and computes the corresponding 2D breakpoint distance matrix `d_BP` based on carrier frequency.

        Parameters
        ----------
        fc_GHz : float, optional
            Carrier frequency in gigahertz. Default is 3.5 GHz.
        LOS : bool, optional
            Whether line-of-sight conditions are assumed. Default is True.
        h_ut_res : float, optional
            Resolution (step size) for UT height grid in meters. Default is 0.5 m.
        h_bs_res : float, optional
            Resolution (step size) for BS height grid in meters. Default is 1.0 m.
        **args
            Additional keyword arguments (unused, accepted for extensibility).

        """
        s.fc_GHz = fc_GHz
        s.LOS = LOS
        s.h_ut_res = h_ut_res
        s.h_bs_res = h_bs_res
        s.h_UT = np.arange(1.0, 10.0 + h_ut_res, h_ut_res)[:, np.newaxis]
        s.h_BS = np.arange(10.0, 150.0 + h_bs_res, h_bs_res)[np.newaxis]
        s.d_BP = 2 * np.pi * s.h_BS * s.h_UT * (s.fc_GHz * 1e9) / 3.0e8
        h = 5.0  # building height = 5m
        log_h_BS = np.log10(s.h_BS)

        # LOS terms are calculated linearly
        s.los_term1 = 2.0 + np.minimum(0.03 * np.power(h, 1.72), 10) / 10.0
        s.los_term2 = np.power((40 * np.pi * s.fc_GHz / 3), 2) * from_dB(
            -np.minimum(0.044 * np.power(h, 1.72), 14.77)
        )
        s.los_term3 = 0.002 * np.log10(h) / 10.0

        # Constants in dB are separated from constants multiplied by a function of distance
        # i.e. PL_NLOS_dB = A_dB + B_dB*log10(d)
        A_dB = (
            161.04
            - 7.1 * np.log10(20.0)
            + 7.5 * np.log10(h)
            - (24.37 - 3.7 * np.power((h / s.h_BS), 2)) * log_h_BS
            + (43.42 - 3.1 * log_h_BS) * (-3)
            + 20 * np.log10(s.fc_GHz)
            - (3.2 * np.power(np.log10(11.75 * s.h_UT), 2) - 4.97)
        )
        B_dB = (43.42 - 3.1 * log_h_BS) + np.zeros_like(s.h_UT)

        # A simple conversion to linear is then possible: 10^(A_dB/10) * d^(B_dB/10) in get_pathloss
        s.nlos_term_A = from_dB(A_dB)
        s.nlos_term_B = B_dB / 10.0

        # The code below calculates the linear pathloss at breakpoint distance, which is used in PL_2
        PL_1_at_d_BP_linear = (
            s.los_term2
            * np.power(s.d_BP, s.los_term1)
            * np.power(10, (s.los_term3 * s.d_BP))
        )
        s.PL_1_at_d_BP = PL_1_at_d_BP_linear / np.power(s.d_BP, 4)

    def get_pathloss_dB(s, d2D_m, d3D_m, U, C):
        return to_dB(s.get_pathloss(d2D_m, d3D_m, U, C))  # retained for compatibility

    def get_pathloss(s, d2D_m, d3D_m, U, C):
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

        h_UT_i = np.round((h_UT - 1.0) / s.h_ut_res).astype(int)
        h_BS_j = np.round((h_BS - 10.0) / s.h_bs_res).astype(int)

        d_BP = s.d_BP[h_UT_i, h_BS_j]
        PL_1_at_d_BP = s.PL_1_at_d_BP[h_UT_i, h_BS_j]
        nlos_term_A = s.nlos_term_A[h_UT_i, h_BS_j]
        nlos_term_B = s.nlos_term_B[h_UT_i, h_BS_j]

        pl1_linear = (
            s.los_term2
            * np.power(d3D_m, s.los_term1)
            * np.power(10, (s.los_term3 * d3D_m))
        )
        pl2_linear = PL_1_at_d_BP * np.power(d3D_m, 4)
        pl_los_linear = np.where(d2D_m <= d_BP, pl1_linear, pl2_linear)

        if s.LOS:
            return pl_los_linear
        else:
            pl_nlos_linear = nlos_term_A * np.power(d3D_m, nlos_term_B)
            return np.maximum(pl_los_linear, pl_nlos_linear)

    def _get_approximate_pathloss_dB_for_layout_plot(s, d):
        d_flat = d.flatten()
        num_pts = len(d_flat)
        U = np.column_stack((d_flat, np.zeros(num_pts), np.full(num_pts, 1.5)))
        C = np.array([[0.0, 0.0, 35.0]])
        d2D_m = np.linalg.norm(U[:, np.newaxis, :2] - C[np.newaxis, :, :2], axis=2)
        d3D_m = np.linalg.norm(U[:, np.newaxis] - C[np.newaxis], axis=2)
        return s.get_pathloss_dB(d2D_m, d3D_m, U, C).squeeze()

    def get_pathgain(s, d2D_m, d3D_m, U, C):
        return 1.0 / s.get_pathloss(d2D_m, d3D_m, U, C)


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
    Plot the 3GPP RMa pathloss or pathgain model predictions as a self-test.

    This function generates a plot of the 3GPP RMa pathloss or pathgain models
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
      Height of the User Terminal (UE) in meters (default is 2.0 m).
    h_BS : float, optional
      Height of the Base Station (BS) in meters (default is 25.0 m).
    zoom_box : bool, optional
      If True, include a zoomed-in view of the plot (default is False).
    print_10m_pl : bool, optional
      If True, print the pathloss values at 10 meters for LOS, NLOS, and
      free-space scenarios (default is False).
    author : str, optional
      Author name to include in the plot timestamp (default is an empty string).
    x_min : float, optional
      Minimum x-axis value for the plot, representing the minimum distance
      in meters (default is 35.0 m).
    x_max : float, optional
      Maximum x-axis value for the plot, representing the maximum distance
      in meters (default is 5000.0 m).

    Raises
    ------
    ImportError
      If required modules (e.g., matplotlib) are not installed.

    Notes
    -----
    - The function uses the `RMa_pathloss` class to compute pathloss and
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
        sigma = np.where(np.less_equal(x, dBP_LOS), 4.0, 4.0)
        ax.vlines(dBP_LOS, 0, PL_LOS_dB[dBP_LOS_index], line_color, "dotted", lw=2)
        ax.fill_between(
            x, PL_LOS_dB - sigma, PL_LOS_dB + sigma, color=line_color, alpha=0.2
        )
        ax.set_xlim(0, np.max(x))
        fnbase = "RMa_pathloss_model"
    else:
        ax.plot(x, PG_LOS, lw=2, label="LOS pathgain")
        ax.set_ylim(0)
        ax.set_xlim(0, 1000)
        fnbase = "RMa_pathgain_model"

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
        f"Max error:\n"
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
    fig.savefig(f"{fnbase}.pdf")
    print(f"eog {fnbase}.png &")
    print(f"evince --page-label=1  {fnbase}.pdf &")


if __name__ == "__main__":
    plot_RMa_pathloss_or_pathgain(
        plot_type="pathloss",
        zoom_box=True,
        print_10m_pl=True,
        author="Keith Briggs, Kishan Sthankiya, Ibrahim Nur",
    )
