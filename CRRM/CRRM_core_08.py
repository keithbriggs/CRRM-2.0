# Keith Briggs 2025-11-07 v08
# Keith Briggs 2025-09-29
# CRRM main classes

from sys import path, stdout, exit, argv
from time import strftime, localtime
from os.path import basename
from dataclasses import dataclass, asdict, fields
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

from .utilities import bright_yellow, blue, cyan, green, red, to_dB, from_dB
from .hexagon_lattice_generator_02 import hexagon_lattice_generator
from .poisson_point_process_generator import poisson_point_process_generator
from .power_law_pathloss_model_02 import Power_law_pathloss
from .UMa_pathloss_model_06 import UMa_pathloss
from .UMi_pathloss_model_00 import (
    UMi_pathloss,
    UMi_pathloss_constant_height,
    UMi_pathloss_discretised,
)
from .RMa_pathloss_model_08 import (
    RMa_pathloss_constant_height,
    RMa_pathloss,
    RMa_pathloss_discretised,
)
from .InH_pathloss_model_01 import InH_pathloss
from .MCS_lookup_tables import MCS_index_tables
from .CRRM_layout_plot_03 import CRRM_layout_plot
from .MIMO_channel_splines import MIMO_channel

@dataclass(frozen=False)
class CRRM_parameters:
    """
    Container for physical, behavioral, and visualization parameters of a CRRM
    (Cellular Radio Reference Model) simulation.  This class encapsulates
    metadata, system geometry, physical layer parameters, pathloss modeling
    options, UE mobility control, animation settings, and display preferences.

    **Important**: the parameters class is used to set the *initial* conditions for the simulation. If these are changed during a simulation run, it is not safe to assume that the new values will be used by the simulation kernel. For only a few parameters is it meaningful to change them during a run, and for these methods with names like  :any:`CRRM.Simulator.set_power_matrix()` methods are provided. See the documentation on all available such methods under :any:`Simulator` below.

    Metadata parameters
    -------------------
    job : int = 0
        User-defined job ID.
    description : str = ''
        User-defined problem description.
    author : str = ''
        Simulation author.
    date_time : str (= current local date and time)
        Timestamp of configuration creation.

    Simulation control parameters
    -----------------------------
    rng_seeds : int or (int,int,int) = 0
        Seeds for the three random-number generators, used for
        UE movements, shadow fading, and Rayleigh fading respectively.
    smart_update : bool = True
        If True, enable optimized internal updates.
    verbose : bool = False
        If True, print general simulation debug information.
    do_asserts : bool = False
        If True, enable runtime assertions for debugging.
    display_flat : bool = False
        If True, display parameters in a single-line format.
    display_colored : bool = True
        If True, use colored output when printing parameters.
    omit_from_display : tuple = ('cell_locations','ue_locations','power_matrix')
        Fields to exclude from parameter display output.

    System geometry and technology parameters
    -----------------------------------------
    n_cell_locations : int = 7
        Number of base station (cell) locations.
    n_ues : int = 50
        Number of user equipment (UEs).
    ue_initial_locations : None
        Placeholder for UE positions (to be populated later).
    cell_locations : None
        Placeholder for cell positions (to be populated later).
    system_area : float
        Area of the system (computed at initialization).
    h_UT_default : float = 1.8
        Default height of a user terminal (meters).
    h_BS_default : float = 20.0
        Default height of a base station (meters).
    fc_GHz : float = 3.5
        Carrier frequency in GHz.
    bw_MHz : float = 20.0
        Channel bandwidth in MHz.
    σ2 : float = 2.0e-20 W/Hz
        Noise power spectral density
    n_subbands : int = 1
        Number of subbands.
    n_sectors : int = 1
        Number of sectors per base station.
    distance_scale : float = 1e3
        Spatial scaling factor (e.g., 1000 for km to meters).
    p_W : float = 100.0
        Transmit power in Watts.
    power_matrix : NDArray
        Matrix of transmit powers across subbands (populated during initialization).
    MIMO : None|tuple[int]
        Order (t,r) of the MIMO scheme, where t is the number of transmit antennas, and r the number of receive antennas. CRRM 2.0 implements MIMO only for computing channel spectral efficiency; the UE throughput computations based on MCS selection are unaffected by this setting.
    MCS_table_number : int = 2
        Index of the modulation and coding scheme table (most users will not need to change this).
    resource_allocation_fairness : float = 0.0
        Fairness factor (0.0 = proportional fair scheduling).

    Pathloss model parameters
    -------------------------
    pathloss_model_name : str = 'UMa'
        Name of the 3GPP pathloss model: options are 'UMa' (Urban Macrocell), 'UMi' (Urban Microcell), 'RMa_discretised' (Rural Macrocell, the fastest implementation, recommended for general rural use), 'RMa' (slower), 'RMa_constant_height' (for use when all cell and all UEs have the same height above ground), 'InH' (Indoor Hotspot), and 'power-law' (simple model for comparison purposes only).
    pathloss_model : None or str
        Pathloss model instance (assigned during set-up, not by the user).
    pathgain_function : Callable[[NDArray, NDArray], float]
        Function to compute path gain between transmitter and receiver (assigned during set-up, not by the user).
    pathloss_exponent : float = 3.0
        Pathloss exponent, used only for the power-law pathloss model.
    LOS : bool = True
        Whether line-of-sight conditions are assumed.
    shadow_fading : bool = False
        If true, shadow fading is used in the pathloss models.
    rayleigh_fading : bool = False
        If true, Rayleigh fading is used in the pathloss models.

    UE mobility parameters
    ----------------------
    n_moves : int = 1000
        Number of UE movement steps in the simulation.
    move_fraction : float = 0.1
        Fraction of UEs that move during each step.
    move_mean : float = 0.0
        Mean movement distance (meters).
    move_stdev : float = 10.0
        Standard deviation of movement (meters).

    Layout plotting parameters
    --------------------------
    layout_plot_fnbase : str = 'img/CRRM_layout_plot'
        Base filename for layout plot image output.
    label_ues_in_layout_plot : bool = False
        Whether to label UEs in the layout plot.
    frame_interval : int = 0
        Time between animation frames (0 disables animation).
    """

    # metaparameters...
    job: int = 0  # field for user-defined data
    description: str = ""  # field for user-defined data
    author: str = ""
    date_time: str = strftime("%Y-%m-%d %H:%M", localtime())
    rng_seeds: int | tuple[int] = 0
    # system geometry parameters...
    n_cell_locations: int = 7
    n_ues: int = 50
    ue_initial_locations: None = None
    cell_locations: None = None
    system_area: float = np.nan  # will be filled in at initialization time
    cell_layout: str = "hex"  # not used?
    UE_layout: str = "ppp"  # not used?
    h_UT_default: float = 1.8
    h_BS_default: float = 20.0
    # physical parameters governing system performance...
    fc_GHz: float = 3.5
    bw_MHz: float = 20.0
    σ2: float = 2.0e-20
    noise_power: float = np.nan  # computed during initialization
    n_subbands: int = 1
    n_sectors: int = 1
    distance_scale: float = 1.0e3
    p_W: float = 1.0e2
    power_matrix: NDArray = None  # will be assiged upon initialization
    MIMO: tuple[int]|None = None  # only partly implemented
    MCS_table_number: int = 2
    resource_allocation_fairness: float = 0.0  # 0 -> proportional fair scheduling
    # control of pathloss model...
    pathloss_model_name: str = "UMa"  # the name of the model we want to use
    pathloss_model: None | str = None  # the instance will go here
    pathgain_function: Callable[[NDArray, NDArray], float] = None  # will be filled in
    pathloss_exponent: float = 3.0  # only used for power-law lathloss model
    LOS: bool = True
    shadow_fading: bool = False
    rayleigh_fading: bool = False
    # control of the layout plot...
    layout_plot_fnbase: str = "img/CRRM_layout_plot"
    label_ues_in_layout_plot: bool = (False,)
    # control of UE moves...
    n_moves: int = 1000
    move_fraction: float = 0.1
    move_mean: float = 0.0
    move_stdev: float = 10.0
    # control of animation generation...
    frame_interval: int = 0
    # control of CRRM internal behaviour...
    smart_update: bool = True
    verbose: bool = False
    verbose_sinr: bool = False
    do_asserts: bool = False
    # control of display of these parameters ...
    display_flat: bool = False
    display_colored: bool = True
    omit_from_display = ("cell_locations", "ue_locations", "power_matrix")

    def __post_init__(self):
        if type(self.rng_seeds) is int:
            self.rng_seeds = (self.rng_seeds, self.rng_seeds + 1, self.rng_seeds + 2)
        if self.cell_locations is not None:
            self.cell_locations = np.array(self.cell_locations, dtype=np.float64)
            self.n_cell_locations = len(self.cell_locations)
        if self.ue_initial_locations is not None:
            self.ue_initial_locations = np.array(
                self.ue_initial_locations, dtype=np.float64
            )
            self.n_ues = len(self.ue_initial_locations)
        # scale noise power spectral density to actual channel bandwidth...
        self.noise_power = self.σ2 * (1e6 * self.bw_MHz) / self.n_subbands
        # TODO more sanity checks here

    def __str__(self):
        return _parameter_str(
            self,
            name="CRRM_parameters",
            color=self.display_colored,
            display_flat=self.display_flat,
        )
# END class CRRM_parameters


def _parameter_str(p, name="parameters:", color=True, display_flat=False):
    # Keith Briggs 2025-09-23
    # create a string (optionally colored) displaying the parameter settings
    # of a dataclass
    y, b, c, g, r = bright_yellow, blue, cyan, green, red if color else lambda x: x
    TFcolor = lambda x: (
        g("True")
        if x is True
        else (
            r("False")
            if x is False
            else (
                f"'{x}'"
                if type(x) is str
                else f"{x:g}" if type(x) in (float, np.float64) else b(x)
            )
        )
    )
    if display_flat:
        d = ", ".join(
            [
                f"{c(k)}={TFcolor(v)}"
                for k, v in asdict(p).items()
                if k not in p.omit_from_display
            ]
        )
    else:
        d = "  " + ",\n  ".join(
            [
                f"{c(k):<38s} = {TFcolor(v)}"
                for k, v in asdict(p).items()
                if k not in p.omit_from_display
            ]
        )
    return f"{y(name)}: " + d if display_flat else f"{y(name)}:\n" + d


def default_cell_locations(n_cell_locations, h_BS, distance_scale=1.0):
    hx = hexagon_lattice_generator()
    cells = np.empty((n_cell_locations, 3))
    for i in range(n_cell_locations):
        cells[i] = (*next(hx), h_BS)
    cell_origin = np.mean(cells, axis=0)
    cells[:, :2] -= cell_origin[:2]  # set mean cell position at (0,0)
    cells[:, :2] *= distance_scale
    x_range = np.max(cells[:, 0]) - np.min(cells[:, 0])
    y_range = np.max(cells[:, 1]) - np.min(cells[:, 1])
    system_area = 0.5 * x_range * y_range
    return cells, system_area


def default_UE_locations(rng, n_ues, h_UT, system_area, cells=None, UE_layout="ppp"):
    if UE_layout == "ppp":
        # we don't need ues*=distance_scale in this case, as system_area
        # already compensates correctly...
        λ = n_ues / system_area
        ppp = poisson_point_process_generator(rng.random, λ)
        rs = np.empty(n_ues)
        θs = np.empty(n_ues)
        for i in range(n_ues):
            rs[i], θs[i] = next(ppp)
        rs = rs.reshape((n_ues, 1))
        θs = θs.reshape((n_ues, 1))
        ues = np.hstack([rs * np.cos(θs), rs * np.sin(θs), h_UT * np.ones((n_ues, 1))])
    elif UE_layout == "uniform":  # uniform random UE locations
        ues = np.hstack(
            [distance_scale * rng.uniform(size=(n_ues, 3)), h_UT * np.ones((n_ues, 1))]
        )
    else:  # close-to-cell UE locations
        ues = np.empty((n_ues, 3))
        for i, row in enumerate(ues):
            row[:2] = cells[i % n_cell_locations, :2] + 100.0 * rng.standard_normal(
                size=2
            )
            row[2] = h_UT
    return ues


# here start the main classes for all the computational blocks...


class _Node:
    """Generic base class for smart updating of node data.
    Not for direct use by a normal user"""

    def __init__(self, name="", smart_update=True, root_node=False):
        self.name = name if name else "_Node"
        self.watchees = []
        self.watchers = []
        self.up_to_date = False
        self.data = None
        self.changed = []
        self.root_node = root_node
        self.smart_update = smart_update
        self.n_updates=0
        self.callback = None

    def __getitem__(self, i):
        """Special syntax: "None" means that there is no index at all,
        and we want to return the *whole* self.data.
        """
        return self.data if i is None else self.data[i]

    def __setitem__(self, i, x):
        """Special syntax: "None" means that there is no index at all,
        and we want to assign the *whole* self.data.
        """
        if i is None:
            self.data = x
        else:
            self.data[i] = x

    def set_callback(self, f):
        " Register user function f to be called with self.data as argument at every update. Inherited by all subclasses. "
        self.callback = f

    def __str__(self):
        watchees = [x.name for x in self.watchees]
        watchers = [x.name for x in self.watchers]
        self = f"{self.name}(watchees={watchees},watchers={watchers},up_to_date={self.up_to_date})"
        return self.replace("'", "").replace(" ", "")

    def watches(self, *watchees):
        "self watches w, w is watched by self"
        for w in watchees:
            self.watchees.append(w)
            w.watchers.append(self)

    def set_data(self, data):
        """use this for root _Nodes (those with no watchees);
        other _Nodes can directly set self.data"""
        self.data = data
        self.up_to_date = True
        self.flood_out_of_date()

    def flood_out_of_date(self):
        "recursively set all watchers to the out-of-date state"
        for w in self.watchers:
            w.up_to_date = False
            w.flood_out_of_date()

    def update(self):
        "update self - this is the really clever bit 😁"
        if not self.up_to_date:
            for w in self.watchees:
                if not w.up_to_date:
                    w.update()
                    w.up_to_date = True
        self.update_data()
        self.flood_out_of_date()
        self.up_to_date = True
        self.n_updates += 1
        if self.callback is not None: self.callback(self.data)

    def update_data(self):
        "This will be over-ridden by specific methods for each sub-class"
        pass

    def add_ue(self):
        # Keith Briggs 2025-08-29
        # add a row to internal arrays
        if type(self.data) is list:
            for i, x in enumerate(self.data):
                self.data[i] = np.vstack([x, np.empty_like(x[0])])
        else:  # self.data is an np.array
            dim = len(self.data.shape)
            if dim == 1:
                self.data = np.append(self.data, 0)
            elif dim == 2:
                new_row = np.empty((1, self.data.shape[1]))
                self.data = np.vstack([self.data, new_row])
            elif dim == 3:
                new_row = np.empty_like(self.data[0])[np.newaxis]
                self.data = np.vstack([self.data, new_row])
        self.flood_out_of_date()
        self.up_to_date = False  # new row is uninitialized

    def get_n_updates(self):
        return self.n_updates
# END class _Node


class Power(_Node):
    "transmit power"

    def __init__(self, p=None, name="P", root_node=True):
        super().__init__(name)
        self.changed = []  # indices of p last changed
        if p is not None:
            self.data = p
        self.up_to_date = True
        self.flood_out_of_date()


class Cell_locations(_Node):
    "cell locations"

    def __init__(
        self, cell_locations=None, n_sectors=1, name="Cell_locations", root_node=True
    ):
        super().__init__(name)
        if n_sectors > 1:
            self.data = np.repeat(cell_locations, n_sectors, axis=0)
        else:
            self.data = cell_locations
        self.changed = []  # indices of cells last moved
        self.up_to_date = True
        self.flood_out_of_date()


class UE_locations(_Node):
    "UE locations"

    def __init__(self, ue_locations=None, name="UE_locations", root_node=True):
        super().__init__(name)
        if ue_locations is not None:
            self.data = ue_locations
        self.changed = []  # indices of UEs last moved

    def _set_locations(self, indices, locations):
        "Internal use only. Use CRRM.set_ue_locations()."
        self.data[indices] = locations
        self.changed = indices
        self.up_to_date = True
        self.flood_out_of_date()

    def _move(self, indices, deltas):
        "Internal use only. Use CRRM.move_ue_locations()."
        d = np.atleast_2d(deltas)
        self.data[indices, : d.shape[1]] += d
        self.changed = indices
        self.up_to_date = True
        self.flood_out_of_date()


class Distance_matrix(_Node):
    """
    distance matrices and angles.
    d2d = sqrt(x^2+y^2)     = hypot(x,y)   = norm(x-y)
    d3d = sqrt(x^2+y^2+z^2) = hypot(d2d,z) = norm(d2d-y)
    theta = angles from cells to UEs
    """

    def __init__(self, name="D", smart_update=True):
        super().__init__(name, smart_update=smart_update)

    def update_data(self):
        i = (
            None
            if self.data is None or not self.smart_update
            else self.watchees[0].changed
        )
        uxyz = self.watchees[0][i]  # UE   3d positions, gets either all or some UEs
        cxyz = self.watchees[1].data  # cell 3d positions, we always need all cells
        v = uxyz[:, np.newaxis] - cxyz[np.newaxis]
        d2d = np.linalg.norm(v[..., :2], axis=2)
        d3d = np.hypot(d2d, v[..., 2])
        angles = np.arctan2(v[..., 1], v[..., 0])
        # we need two cases here because self.watchees is a list...
        if i is None:  # full update
            self.data = [d2d, d3d, angles]
        else:  # smart update
            # print(f'Distance_matrix.update_data: self.data[0].shape={self.data[0].shape}')
            self.data[0][i] = d2d
            self.data[1][i] = d3d
            self.data[2][i] = angles
        self.changed = (i, i, i)

    def check(self, u, c):
        v = u[:, :2][:, np.newaxis] - c[:, :2][np.newaxis]
        d2d = np.linalg.norm(v, axis=2)
        d3d = np.linalg.norm(u[:, np.newaxis] - c[np.newaxis], axis=2)
        angles = np.arctan2(v[..., 1], v[..., 0])
        return (
            np.allclose(self.data[0], d2d)
            and np.allclose(self.data[1], d3d)
            and np.allclose(self.data[2], angles)
        )
# END class Distance_matrix


class Gain_matrix(_Node):
    # Ibrahim Nur 2025-09-22
    # Keith Briggs 2025-09-01
    def __init__(
        self,
        pathgain_function,
        name="Gain_matrix",
        antenna_gain=None,
        sf_values_db=None,
        smart_update=True,
    ):
        super().__init__(name, smart_update=smart_update)
        self.pathgain_function = pathgain_function
        self.antenna_gain = antenna_gain
        self.sf_values_db = sf_values_db

    def update_data(self):
        d, c, u = self.watchees  # distance_matrix,cell_locations,ue_locations
        i = None if self.data is None or not self.smart_update else d.changed[0]
        d2d, d3d, theta = d.data
        if i is not None:
            d2d, d3d, theta = (d2d[i], d3d[i], theta[i])
        pathgain = self.pathgain_function(d2d, d3d, u[i], c)
        if self.sf_values_db is not None:
            sf_slice = self.sf_values_db if i is None else self.sf_values_db[i]
            pathgain *= from_dB(-sf_slice[:, np.newaxis])
        self[i] = (
            pathgain
            if self.antenna_gain is None
            else self.antenna_gain(theta) * pathgain
        )
        self.changed = i

    def check(self):
        d, c, u = self.watchees  # distance_matrix,cell_locations,ue_locations
        d2d, d3d, theta = d.data
        pathgain = self.pathgain_function(d2d, d3d, u, c)
        x = self.antenna_gain(theta) * pathgain
        return np.allclose(self.data, x)


class RSRP(_Node):
    """
    This class handles subbands by treating the power data as a 2D matrix
    and the gain data as another 2D matrix, then broadcasting them
    into a 3D result. The smart update logic operates on the gain
    matrix, which is indexed by UE.

    The broadcasting works as follows:
    g[i]:        Takes a slice of the gain matrix for the changed UEs.
                 Resultant shape: (n_ues, n_cells)
    g[i][...,np.newaxis]:
                 Adds a new axis for the subbands at the trailing end.
                 Resultant shape: (n_ues, n_cells, 1)
    p.data[np.newaxis]:
                 Adds a new axis to the power matrix at the leading end.
                 Shape: (1, n_cells, n_subbands)

    Result:      NumPy broadcasts the two arrays by scaling their
                 dimensions of length 1 to match the corresponding
                 dimensions of the other, resulting in a slice for
                 the changed UEs.
                 Shape: (n_ues, n_cells, n_subbands)
    """

    def __init__(self, params, name="RSRP", smart_update=True):
        super().__init__(name, smart_update=smart_update)
        self.params = params

    def update_data(self):
        p, g = self.watchees
        # if p (powers or subband masks) has changed, we trigger a full update...
        i = (
            None
            if any([self.data is None, not self.smart_update, p.changed])
            else g.changed
        )
        self[i] = g[i][..., np.newaxis] * p.data[np.newaxis]
        self.changed = i
# END class RSRP


class Attachment_vector(_Node):
    """attachment vector using RSRP matrix = self.watchees[0].
    self.data[i] is the cell to which UE is attached.
    """

    def __init__(self, name="Attachment_vector", smart_update=True):
        super().__init__(name, smart_update=smart_update)

    def update_data(self):
        rsrp = self.watchees[0]
        i = None if self.data is None or not self.smart_update else rsrp.changed
        # sum powers across all subbands...
        self[i] = np.argmax(np.sum(rsrp[i], axis=-1), axis=1)
        self.changed = i
# END class Attachment_vector


class SINR(_Node):
    # Ibrahim Nur 2025-09-18
    def __init__(self, params, name="SINR", rng=None, smart_update=True, verbose=False):
        super().__init__(name, smart_update=smart_update)
        self.params = params
        self.rng_exp = rng.exponential
        self.verbose = verbose
        self.changed = None

    def update_data(self):
        """
        watchees: Attachment_vector, RSRP
        a=attachment vector
        w=wanted signals
        u=unwanted interferences
        i=rows of which have changed
        """
        a, rsrp = self.watchees
        ir = None if self.data is None or not self.smart_update else rsrp.changed
        # separate indexing variable - set to range(len(a)) if ir is not none...
        i = ir if ir is not None else range(len(a.data))
        w = rsrp.data[i, a.data[i]]
        u = np.sum(rsrp.data[i], axis=1) - w
        if self.params.rayleigh_fading:
            w *= self.rng_exp(1.0, size=w.shape)
        self[ir] = w / (self.params.noise_power + u)
        self.changed = ir
# END class SINR(_Node)


class CQI(_Node):
    """
    Note: this class can handle subbands as its operations act on every
    element of the arrays that pass through them. Hence, the data's
    shape has no relevance to the class. The same applies to all
    downstream classes.
    """

    def __init__(self, name="CQI", smart_update=True):
        super().__init__(name, smart_update=smart_update)
        # Keith Briggs 2025-11-07.
        # Linearized (non-dB) version of SINR_to_CQI look-up table from
        # AIMM_simulator-2.0/src/AIMM_simulator/NR_5G_standard_functions.py
        self.SINR90pc_linear=np.array([0.00000000e+00, 6.47142616e-01, 8.27942164e-01, 1.24451461e+00, 1.97242274e+00, 3.09029543e+00, 5.48276965e+00, 7.74461798e+00, 1.26473635e+01, 2.14783047e+01, 2.93764965e+01, 4.59198013e+01, 7.78036551e+01, 1.43879858e+02, 3.07609681e+02, 8.55066713e+02, np.inf])

    def update_data(self):
        # Keith Briggs 2025-11-07. Don't use to_dB, in case SINR is zero.
        sinr = self.watchees[0]
        i = None if self.data is None or not self.smart_update else sinr.changed
        self[i] = np.searchsorted(self.SINR90pc_linear, sinr[i]) - 1
        self.changed = i


class MCS(_Node):
    def __init__(self, name="MCS", smart_update=True):
        super().__init__(name, smart_update=smart_update)
        # spread CQI range uniformly over MCS range...
        self.scale_factor = 28.0 / 15.0

    def update_data(self):
        cqi = self.watchees[0]
        i = None if self.data is None or not self.smart_update else cqi.changed
        # from CQI_to_64QAM_efficiency in NR_5G_standard_functions.py...
        # Keith Briggs 2025-11-06 handle SINR=0.0, CQI=-1 case ...
        self[i] = np.clip(np.int64(self.scale_factor * cqi[i]), a_min=-1, a_max=28)
        self.changed = i


class SE_Shannon(_Node):
    def __init__(self, MIMO=None, mimo_channel_ergodic_spectral_efficiency=None, name="SE_Shannon", smart_update=True):
        super().__init__(name, smart_update=smart_update)
        self.log2 = 0.6931471805599453094
        self.MIMO = MIMO
        self.mimo_channel_ergodic_spectral_efficiency = mimo_channel_ergodic_spectral_efficiency

    def update_data(self):
        # divide by log2 to get bits/s/Hz...
        sinr = self.watchees[0]
        i = None if self.data is None or not self.smart_update else sinr.changed
        if self.MIMO is None or self.MIMO == (1,1):
          self[i] = np.log1p(sinr[i]) / self.log2
        else:
          sinr_dB = np.clip(to_dB(sinr[i]),-50.0,50.0)
          self[i] = self.mimo_channel_ergodic_spectral_efficiency(sinr_dB)
        self.changed = i


class SE_from_MCS(_Node):
    def __init__(self, MCS_table_number, name="SE_from_MCS", smart_update=True):
        super().__init__(name, smart_update=smart_update)
        self.MCS_to_SE = MCS_index_tables[MCS_table_number][:, 3]

    def update_data(self):
        mcs = self.watchees[0]
        i = None if self.data is None or not self.smart_update else mcs.changed
        if i is None:  # Keith Briggs 2025-08-29
            self.data = self.MCS_to_SE[mcs[:]]
        else:
            self[i] = self.MCS_to_SE[mcs[i]]
        self.changed = i


class Antenna_gain:
    """
    Calculates the antenna gain (not in dB) based on the 3GPP standard horizontal pattern.

    This method applies a sectored antenna pattern as defined in 3GPP TR 38.901.
    The gain for each UE-cell link is determined by the absolute angular
    difference between the UE's direction and the sector's boresight. If the
    simulation is configured with only one sector per cell, a constant
    omnidirectional gain is applied instead.

    The horizontal attenuation (in dB) is given by the formula:

    .. math:: A_H(\\phi') = -\\min\\left(12 \\left(\\frac{\\phi'}{\\phi_\\text{3dB}}\\right)^2, A_\\text{max}\\right)

    where :math:`\\phi'` is the angular deviation from the boresight.

    Parameters
    ----------
    φ : numpy.ndarray
      An array of horizontal angles :math:`\\phi` (in radians) from each cell to each UE. Its shape must be broadcastable with the internal array of sector angles.

    Returns
    -------
      An array of linear (i.e., not in dB) gain values for each UE-cell link as type numpy.ndarray.

    Reference
    ---------

      - 3GPP TR 38.901, Table 7.3-1

    """

    # Ibrahim Nur 2025-08-22, Keith Briggs 2025-08-23
    def __init__(
        self,
        n_sectors,
        n_locations,
        max_gain=8.0,
        half_power_beamwidth=65.0,
        max_attenuation=30.0,
    ):
        "default parameters from 3GPP TR 38.901, Table 7.3-1"
        self.n_sectors = n_sectors
        self.sector_angles = np.tile(
            np.linspace(0, 2 * np.pi, n_sectors, endpoint=False), n_locations
        )
        self.max_gain = max_gain
        self.half_power_beamwidth = half_power_beamwidth
        self.max_attenuation = max_attenuation
        self.twopi = 2.0 * np.pi

    def __call__(self, φ):
        if self.n_sectors == 1:
            return from_dB(self.max_gain)
        Δφ = np.abs(φ - self.sector_angles)
        Δφ_prime = np.minimum(Δφ, self.twopi - Δφ)
        attenuation_dB = -np.minimum(
            12.0 * np.square(np.rad2deg(Δφ_prime) / self.half_power_beamwidth),
            self.max_attenuation,
        )
        return from_dB(self.max_gain + attenuation_dB)


class Throughput(_Node):
    """
    Calculate all UE throughputs, with a tunable resource allocation heuristic.
    Though this class is for internal use only, the internal details are of some interest.  Authors: Keith Briggs & Ibrahim Nur 2025-09-04.

    This model allocates a share of the total bandwidth to each user
    in proportion to an abstract notion of their cost, and then calculates
    the resulting throughput.

    The cost for user *i* is defined as :math:`1/S_i^p`, where :math:`S_i` is the
    user's spectral efficiency and :math:`p` is a tunable fairness parameter.

    This leads to the final throughput formula for user :math:`i`:

    .. math:: T_i = a S_i^{1-p}

    where :math:`a` is a proportionality constant calculated for the cell:

    :math:`a` = (total bandwidth) :math:`/` (sum of all user costs)

    The fairness parameter :math:`p` controls the distribution skew as follows:

    * :math:`p > 1`: favours weak users (e.g., :math:`p=2` gives :math:`T ∝ 1/S`).
    * :math:`p = 1`: results in equal throughput for all users on the cell.
    * :math:`p < 1`: favours strong users (e.g., :math:`p=0` gives :math:`T ∝ S`).
    * :math:`p = 0`: proportional fair scheduling.

    The parameter :math:`p` is set in the :class:`~CRRM_parameters` class, and can be changed during a run using the `CRRM.set_resource_allocation_fairness()` method.
    """

    # https://sphinx-rtd-trial.readthedocs.io/en/latest/ext/math.html
    def __init__(self, crrm_parameters, name="Throughput", smart_update=True):
        super().__init__(name, smart_update=smart_update)
        self.params = crrm_parameters
        self.subband_bw_MHz = self.params.bw_MHz / self.params.n_subbands

    def update_data(self):
        p = self.params.resource_allocation_fairness
        a, se_from_mcs = self.watchees
        ue_costs = np.power(se_from_mcs.data, -p)
        total_load_per_cell = np.zeros((self.params.n_cells, self.params.n_subbands))
        np.add.at(total_load_per_cell, a.data, ue_costs)
        a_per_cell = self.subband_bw_MHz / np.maximum(total_load_per_cell, 1e-9)
        self.data = a_per_cell[a.data] * se_from_mcs.data * ue_costs
# END class Throughput


class CRRM:
    """
    Main class representing a cellular system, and computing all
    internal performance data. It takes one argument of type CRRM_parameters.

    The simulator internally uses a list of three random number generators,
    each with their own seed.  This allows for fading to be switched on and off
    but not affect UE moves.

     - `rng[0]`: used for moving UEs
     - `rng[1]`: used for shadow fading
     - `rng[2]`: used for Rayleigh fading
    """

    def __init__(self, crrm_parameters: CRRM_parameters):
        self.params = crrm_parameters
        self.simulated_time = 0.0
        self.rngs = [
            np.random.default_rng(self.params.rng_seeds[0]),
            np.random.default_rng(self.params.rng_seeds[1]),
            np.random.default_rng(self.params.rng_seeds[2]),
        ]
        # set up the cell and UE locations if not already provided...
        if self.params.cell_locations is None:
            self.params.cell_locations, self.params.system_area = (
                default_cell_locations(
                    self.params.n_cell_locations,
                    self.params.h_BS_default,
                    self.params.distance_scale,
                )
            )
        self.cell_locations = Cell_locations(
            self.params.cell_locations, self.params.n_sectors
        )
        self.params.n_cells = len(self.cell_locations.data)
        self.layout_rmax = max(
            1.0, np.max(np.linalg.norm(self.params.cell_locations[:, :2], axis=1))
        )
        self.params.system_area = np.pi * self.layout_rmax**2
        if self.params.ue_initial_locations is None:
            ue_locations = default_UE_locations(
                self.rngs[0],
                self.params.n_ues,
                self.params.h_UT_default,
                self.params.system_area,
                self.params.cell_locations,
                self.params.UE_layout,
            )
        else:
            ue_locations = self.params.ue_initial_locations
        self.ue_locations = UE_locations(ue_locations)
        # Instantiate the pathloss model...
        if self.params.pathloss_model_name == "UMa":
            pathloss_model = UMa_pathloss(
                fc_GHz=self.params.fc_GHz,
                h_UT=self.params.h_UT_default,
                h_BS=self.params.h_BS_default,
                LOS=self.params.LOS,
            )
        elif self.params.pathloss_model_name == "UMi":
            pathloss_model = UMi_pathloss(
                fc_GHz=self.params.fc_GHz,
                h_UT=self.params.h_UT_default,
                h_BS=self.params.h_BS_default,
                LOS=self.params.LOS,
            )
        elif self.params.pathloss_model_name == "UMi_constant_height":
            pathloss_model = UMi_pathloss_constant_height(
                fc_GHz=self.params.fc_GHz,
                h_UT=self.params.h_UT_default,
                h_BS=self.params.h_BS_default,
                LOS=self.params.LOS,
            )
        elif self.params.pathloss_model_name == "UMi_discretised":
            pathloss_model = UMi_pathloss_discretised(
                fc_GHz=self.params.fc_GHz,
                h_UT=self.params.h_UT_default,
                h_BS=self.params.h_BS_default,
                LOS=self.params.LOS,
            )
        elif self.params.pathloss_model_name == "RMa":
            pathloss_model = RMa_pathloss(
                fc_GHz=self.params.fc_GHz, LOS=self.params.LOS
            )
        elif self.params.pathloss_model_name == "RMa_constant_height":
            pathloss_model = RMa_pathloss_constant_height(
                fc_GHz=self.params.fc_GHz,
                h_UT=self.params.h_UT_default,
                h_BS=self.params.h_BS_default,
                LOS=self.params.LOS,
            )
        elif self.params.pathloss_model_name == "RMa_discretised":
            pathloss_model = RMa_pathloss_discretised(
                fc_GHz=self.params.fc_GHz, LOS=self.params.LOS
            )
        elif self.params.pathloss_model_name == "InH":
            pathloss_model = InH_pathloss(
                fc_GHz=self.params.fc_GHz, LOS=self.params.LOS
            )
        elif self.params.pathloss_model_name == "power-law":
            pathloss_model = Power_law_pathloss(
                fc_GHz=self.params.fc_GHz, exponent=self.params.pathloss_exponent
            )
        else:
            print(
                red(
                    f"pathloss_model_name={self.params.pathloss_model_name} not available, quitting"
                )
            )
            exit(1)
        self.params.pathloss_model = pathloss_model
        self.params.pathgain_function = pathloss_model.get_pathgain
        # Instantiate the nodes...
        power_matrix = (
            self.params.p_W
            * np.ones((self.params.n_cell_locations, self.params.n_subbands))
            if self.params.power_matrix is None
            else self.params.power_matrix
        )
        self.p = Power(np.repeat(power_matrix, self.params.n_sectors, axis=0))
        self.d = Distance_matrix(smart_update=self.params.smart_update)
        self.a = Attachment_vector(smart_update=self.params.smart_update)
        self.g = Gain_matrix(
            self.params.pathgain_function,
            smart_update=self.params.smart_update,
            antenna_gain=Antenna_gain(
                self.params.n_sectors, self.params.n_cell_locations
            ),
            sf_values_db=self._generate_shadow_fading(),
        )
        self.rsrp = RSRP(self.params, smart_update=self.params.smart_update)
        self.sinr = SINR(
            self.params,
            smart_update=self.params.smart_update,
            rng=self.rngs[2],
            verbose=self.params.verbose_sinr,
        )
        self.cqi = CQI(smart_update=self.params.smart_update)
        self.mcs = MCS(smart_update=self.params.smart_update)
        MIMO=self.params.MIMO
        if MIMO is None or MIMO == (1, 1):
          self.se_Shannon = SE_Shannon(smart_update=self.params.smart_update)
        else:
          MIMO_se = MIMO_channel().ergodic_spectral_efficiency
          if MIMO not in MIMO_se.keys():
            print(red(f"{MIMO[0]}×{MIMO[1]} MIMO not implemented, quitting."))
            exit(1)
          self.se_Shannon = SE_Shannon(MIMO=MIMO, mimo_channel_ergodic_spectral_efficiency=MIMO_se[MIMO], smart_update=self.params.smart_update)
        self.se_from_mcs = SE_from_MCS(
            MCS_table_number=self.params.MCS_table_number,
            smart_update=self.params.smart_update,
        )
        self.tp = Throughput(self.params, smart_update=self.params.smart_update)
        # set up the dependencies of _Nodes on each other...
        self.a.watches(self.rsrp)
        self.d.watches(self.ue_locations, self.cell_locations)
        self.g.watches(self.d, self.cell_locations, self.ue_locations)
        self.rsrp.watches(self.p, self.g)
        self.sinr.watches(self.a, self.rsrp)
        self.cqi.watches(self.sinr)
        self.mcs.watches(self.cqi)
        self.se_Shannon.watches(self.sinr)
        self.se_from_mcs.watches(self.mcs)
        self.tp.watches(self.a, self.se_from_mcs)
    # END def CRRM.__init__

    def update(self):
        "Force an update of the whole computational stack. This is only needed before printing out any data. When using CRRR_data, the required updates are automatically done internally."
        self.se_Shannon.update()
        self.tp.update()

    def _generate_shadow_fading(self):
        if not self.params.shadow_fading:
            return None
        sigma_sf_map = {
            "UMa": (6.0, 4.0),
            "UMi": (7.82, 4.0),
            "RMa": (8.0, 4.0),
            "InH": (8.03, 3.0),
            "power-law": (8.0, 6.0),
        }
        sigma_sf = sigma_sf_map.get(self.params.pathloss_model_name, (0.0, 0.0))[
            self.params.LOS
        ]
        return self.rngs[1].normal(0.0, sigma_sf, size=self.params.n_ues)

    # CRRM setters and getters...

    def set_rng_seeds(self, seeds):
        "Re-set the rngs with a new seeds."
        if type(seeds) is int:
            seeds = (seeds, seeds + 1, seeds + 2)
        self.params.rng_seeds = seeds
        for seed, rng in zip(seeds, self.rngs):
            if seed is not None:
                rng = np.random.default_rng(seed)

    def get_rngs(self, i=slice(None)):
        "Get the current rngs, or some or one of them."
        return self.rngs[i]

    def set_n_sectors(self, n_sectors):
        "Set the number of sectors for all base stations."
        self.params.n_sectors = n_sectors
        self.params.n_cells = n_sectors * self.params.n_cell_locations

    def set_noise_power_spectral_density(self, σ2):
        "Set the noise power spectral density to a new value. The actual noise power in the channel or subbands will be computed and used in subsequent SIR calculations."
        self.params.σ2 = σ2
        self.params.noise_power = σ2 * (1e6 * self.bw_MHz) / self.n_subbands
        self.noise_power.up_to_date = False
        self.noise_power.flood_out_of_date()

    def set_resource_allocation_fairness(self, p):
        "Set the resource allocation fairness parameter."
        self.params.resource_allocation_fairness = p

    def get_resource_allocation_fairness(self):
        "Get the current resource allocation fairness parameter."
        return self.params.resource_allocation_fairness

    def set_ue_locations(self, indices, locations):
        "Set the UE locations to new values."
        self.ue_locations._set_locations(indices, locations)
        self.ue_locations.up_to_date = False
        self.ue_locations.flood_out_of_date()

    def get_ue_locations(self):
        "Get the current UE locations."
        return self.ue_locations.data

    def get_UE_throughputs(self, ues=slice(None), subbands=slice(None)):
        "Get the current UE throughputs for specified lists of ues  (default all UEs), and of subbands (default all subbands)."
        self.tp.update()
        return self.tp.data[ues, subbands]

    def get_spectral_efficiency(self, ues=slice(None), subbands=slice(None)):
        "Get the current UE spectral efficiencies for specified lists of ues  (default all UEs), and of subbands (default all subbands)."
        self.se_Shannon.update()
        return self.se_Shannon.data[ues, subbands]

    def get_power_matrix(self):
        "Return a reference to the current power matrix."
        return self.p.data

    def set_power_matrix(self, p):
        "Set the power matrix to a new value. The matrix must have shape (n_cell_locations,n_subbands). It will be repeated across sectors if there are more than 1."
        power_matrix = np.array(p, dtype=np.float64)
        if np.any(power_matrix<0.0):
            print(
                red(
                    f"set_power_matrix: negative values, quitting!"
                )
            )
            exit(1)
        shp = power_matrix.shape
        if shp[0] != self.params.n_cell_locations:
            print(
                red(
                    f"set_power_matrix: power_matrix.shape[0]={shp[0]} does not equal n_cell_locations={self.params.n_cell_locations}, quitting!"
                )
            )
            exit(1)
        if shp[1] != self.params.n_subbands:
            print(
                red(
                    f"set_power_matrix: power_matrix.shape[1]={shp[1]} does not equal n_cell_locations={self.params.n_cell_locations}, quitting!"
                )
            )
            exit(1)
        self.p.data = np.repeat(power_matrix, self.params.n_sectors, axis=0)
        self.p.up_to_date = False
        self.p.flood_out_of_date()

    def change_power_matrix(self, indices, deltas):
        """
        Shift some or all rows of the power matrix, by the given deltas.

        Parameters
        ----------
        indices : a list of cell indices, or 'all'
        deltas : float, or 2d array (one row for each index, one column per subband)
        """
        if indices in (
            "all",
            slice(None),
        ):
            self.p.data[:] += deltas
        else:
            self.p.data[indices] += deltas
        self.p.data=np.maximum(0.0,self.p.data) # don't allow negative powers
        self.p.up_to_date = False
        self.p.flood_out_of_date()


    def scale_ue_locations(self, indices="all", scale_factor=1.0):
        """
        Scale UEs with given indices, by the given factor.

        Parameters
        ----------
        indices : a list of UE indices, or 'all'
        scale_factor : float
        """
        if indices in (
            "all",
            slice(None),
        ):
            self.ue_locations.data[:, :2] *= scale_factor
        else:
            self.ue_locations.data[indices, :2] *= scale_factor
        self.ue_locations.up_to_date = False
        self.ue_locations.flood_out_of_date()

    def move_ue_locations(self, indices, deltas):
        """
        Move UEs with given indices, by the given deltas.

        Parameters
        ----------
        indices : a list of UE indices, or 'all'
        deltas : an array of shape (len(indices),3) representing the displacements
        """
        if indices in (
            "all",
            slice(None),
        ):
            self.ue_locations._move(slice(None), deltas)
        else:
            self.ue_locations._move(indices, deltas)
        self.ue_locations.up_to_date = False
        self.ue_locations.flood_out_of_date()

    # END CRRM setters and getters

    def add_ue(self, location):
        """
        Add one UE to the system, at the given location.

        Parameters
        ----------
        location : a 3-vector representing the (x,y,z) coordinates
        """
        # first add a row to all arrays in the _Nodes...
        for node in (
            self.ue_locations,
            self.a,
            self.d,
            self.g,
            self.rsrp,
            self.sinr,
            self.cqi,
            self.mcs,
            self.se_Shannon,
            self.se_from_mcs,
            self.tp,
        ):
            node.add_ue()
        self.ue_locations.data[-1] = location
        self.ue_locations.up_to_date = False
        self.ue_locations.changed = None

    def invert_attachment_vector(self):
        "Return a dict mapping cells to lists of attached UEs"
        return {i: np.where(self.a.data == i)[0] for i in range(self.n_cells)}

    def check_ue_locations(self):
        mean_ue_locations = np.mean(self.ue_locations.data[:, :2], axis=0)
        return np.hypot(mean_ue_locations[0], mean_ue_locations[1])

    def layout_plot(
        self,
        grid=False,
        title="",
        show_attachment_type="attachment",
        show_plot=False,
        show_voronoi=True,
        padding_factor=1.02,
        show_kilometres=True,
        show_system_rmax_circle=True,
        show_UE_radius_circle=True,
        show_pathloss_circles=True,
        cell_image=None,
        UE_image=None,
        cell_image_zoom=5e-2,
        UE_image_zoom=8e-2,
        return_figure=False,
        fmt=("png", "pdf"),
        no_ticks=False,
        dbg=False,
        quiet=False,
        dpi=200,
        figsize=(6, 6),
        label_ues=True,
    ):
        """
        Plot the cellular reference model layout with various optional overlays.

        Parameters
        ----------
        grid : bool, optional
          Whether to display grid lines on the plot.
        title : str, optional
          Title for the plot.
        fnbase : str, optional
          Base filename for saving the plot.
        show_attachment_type : str, optional
          If not empty, show UE-to-cell attachment lines.
        show_plot : bool, optional
          If True, display the plot interactively.
        show_voronoi : bool, optional
          If True, overlay Voronoi tessellation of cell locations.
        padding_factor : float, optional
          Factor to expand plot limits beyond system radius.
        show_kilometres : bool, optional
          If True, display axes in kilometres.
        show_system_rmax_circle : bool, optional
          If True, draw a circle at the system maximum radius.
        show_UE_radius_circle : bool, optional
          If True, draw a circle at the UE radius.
        show_pathloss_circles : bool, optional
          If True, overlay circles for specific pathloss values.
        cell_image : array-like or None, optional
          Image to use for cell markers.
        UE_image : array-like or None, optional
          Image to use for UE markers.
        cell_image_zoom : float, optional
          Zoom factor for cell images.
        UE_image_zoom : float, optional
          Zoom factor for UE images.
        return_figure : bool, optional
          If True, return the plot object instead of saving.
        fmt : list of str, optional
          List of file formats for saving the plot.
        dbg : bool, optional
          If True, print debug information.

        Returns
        -------
        plot : Plot_CRRM_layout or None
          The plot object if `return_figure` is True, otherwise None.
        """
        plot_axlim = (
            -padding_factor * self.layout_rmax,
            padding_factor * self.layout_rmax,
        )
        plot = CRRM_layout_plot(
            xlim=plot_axlim,
            ylim=plot_axlim,
            grid=grid,
            cell_image=cell_image,
            UE_image=UE_image,
            cell_image_zoom=cell_image_zoom,
            UE_image_zoom=UE_image_zoom,
            layout_plot_fnbase=self.params.layout_plot_fnbase,
            no_ticks=no_ticks,
            dpi=dpi,
            quiet=quiet,
            figsize=figsize,
            label_ues=self.params.label_ues_in_layout_plot,
            author=self.params.author,
        )
        plot.base(
            self.cell_locations, self.ue_locations, show_kilometres=show_kilometres
        )
        if show_voronoi and self.params.n_cells > 1:
            plot.voronoi(self.cell_locations[:, :2])
        if show_attachment_type:
            plot.attachment(
                cells=self.cell_locations,
                ues=self.ue_locations,
                show_attachment_type=show_attachment_type,
                attachment_vector=self.a.data,
                n_sectors=self.params.n_sectors,
            )
        UE_radius = 1.0
        if show_UE_radius_circle:
            UE_radius_circle = plt.Circle(
                (0.0, 0.0),
                UE_radius,
                color="grey",
                fill=False,
                lw=2,
                linestyle="dashed",
                zorder=8,
            )
            plot.ax.add_patch(UE_radius_circle)
        if show_system_rmax_circle:
            circle = plt.Circle(
                (0.0, 0.0),
                self.layout_rmax,
                color="black",
                fill=False,
                lw=1,
                linestyle="dotted",
                zorder=8,
            )
            plot.ax.add_patch(circle)
        # which circles to draw...
        if self.params.LOS:
            pathloss_circles_dB = (
                (100.0, 120.0, 140.0)
                if self.params.pathloss_model_name != "InH"
                else (65.0, 70.0, 75.0)
            )
        else:
            pathloss_circles_dB = (
                (140.0, 160.0, 170.0)
                if self.params.pathloss_model_name != "InH"
                else (80.0, 90.0, 100.0)
            )
        if show_pathloss_circles:
            if self.params.pathloss_model_name == "free-space":
                pathloss_dB = (
                    lambda d: 20.0 * (np.log10(d) + np.log10(1e9 * self.params.fc_GHz))
                    - 147.55
                )
            elif "power-law" == self.params.pathloss_model_name:
                pathloss_dB = lambda d: self.params.pathloss_model.get_pathloss_dB(
                    d, d + 20.0, None, None
                )  # d,d+20.0 good enough for approximate pathloss!
            # here we'll keep separate cases for potential future use...
            elif "RMa" in self.params.pathloss_model_name:
                pathloss_dB = self.params.pathloss_model._get_approximate_pathloss_dB_for_layout_plot
            elif "UMa" in self.params.pathloss_model_name:
                pathloss_dB = self.params.pathloss_model._get_approximate_pathloss_dB_for_layout_plot
            elif "UMi" in self.params.pathloss_model_name:
                pathloss_dB = self.params.pathloss_model._get_approximate_pathloss_dB_for_layout_plot
            elif "InH" == self.params.pathloss_model_name:
                pathloss_dB = self.params.pathloss_model._get_approximate_pathloss_dB_for_layout_plot
            else:
                return
            for pl_dB in pathloss_circles_dB:
                f = lambda x: pathloss_dB(x) - pl_dB
                root = root_scalar(f, bracket=(10.0, 1e4))
                d = root.root
                if dbg:
                    print(
                        f"{self.params.pathloss_model_name}: root_scalar finds d={d:g} for pathloss={pl_dB:.0f}dB"
                    )
                circle = plt.Circle((0.0, 0.0), d, color="purple", fill=False)
                plot.ax.add_patch(circle)
                theta = 0.25 * np.pi
                xy = (d * np.cos(theta), d * np.sin(theta))
                plot.ax.annotate(f"{pl_dB:.0f}dB", xy, color="purple", fontsize=8)
        if show_plot:
            plt.show()
        if title:
            plot.ax.set_title(title)
        plot.fig.tight_layout()
        if return_figure:
            return plot
        plot.savefig(timestamp=True, fmt=fmt, author=self.params.author)
    # END def plot_layout
# END class CRRM
