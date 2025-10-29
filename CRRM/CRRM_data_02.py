# Keith Briggs 2025-09-23
# class to capture data during a CRRM run, and plot it

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.interpolate import make_interp_spline, BSpline

from .utilities import fig_timestamp, to_dB


class CRRM_logger:
    """
    Create an object which will log data during a simulation run.

    Parameters
    ----------

    captures: tuple of strings. The allowed values are:

      * 'UE_location'
      * 'rsrp' (received signal reference power)
      * 'a' (attachment vector)
      * 'sinr' (signal to interference plus noise matrix in dB)
      * 'se_Shannon' (Shannon capacity in b/s/Hz)
      * 'cqi' (channel quality index)
      * 'mcs' (modulation and coding scheme index)
      * 'se_from_mcs' (actual spectral efficiency in b/s/Hz)
      * 'tp' (UE downlink throughput in Mb/s)

    ues: tuple of UE indices
    block_size: int, size of pre-allocated internal data. Not normally altered.
    """

    def __init__(self, crrm, captures=(), ues=(), block_size=1000):
        self.crrm = crrm
        self.ues = ues
        self.n_ues = len(ues)
        self.bs = block_size
        self.n_subbands = self.crrm.params.n_subbands
        self.traces = {
            "UE_location": "UE location",
            "rsrp": "RSRP",
            "a": "attachment",
            "sinr": "SINR (dB)",
            "se_Shannon": "capacity\n(b/s/Hz)",
            "cqi": "CQI",
            "mcs": "MCS",
            "se_from_mcs": "spectral efficiency\n(b/s/Hz)",
            "tp": "throughput\n(Mb/s)",
        }
        # only allow captures which are in traces.keys()...
        self.captures = tuple(set(captures) & set(self.traces.keys()))
        self.data = {c: {} for c in self.captures}
        ns = self.n_subbands
        self.data["UE_location"] = {
            i: np.empty((self.bs, 3)) for i in range(self.n_ues)
        }
        self.data["rsrp"] = {
            i: np.empty((self.bs, self.crrm.params.n_cells, ns))
            for i in range(self.n_ues)
        }
        self.data["sinr"] = {i: np.empty((self.bs, ns)) for i in range(self.n_ues)}
        self.data["se_Shannon"] = {
            i: np.empty((self.bs, ns)) for i in range(self.n_ues)
        }
        self.data["se_from_mcs"] = {
            i: np.empty((self.bs, ns)) for i in range(self.n_ues)
        }
        self.data["tp"] = {i: np.empty((self.bs, ns)) for i in range(self.n_ues)}
        self.data["a"] = {
            i: np.empty((self.bs, 1), dtype=np.int64) for i in range(self.n_ues)
        }
        self.data["cqi"] = {
            i: np.empty((self.bs, ns), dtype=np.int64) for i in range(self.n_ues)
        }
        self.data["mcs"] = {
            i: np.empty((self.bs, ns), dtype=np.int64) for i in range(self.n_ues)
        }
        self.n_blocks_allocated = 1
        self.n_rows_filled = 0

    def capture(self, verbose=False):
        self.crrm.update()
        for c in self.captures:
            if verbose:
                print(f"c={c}")
            field = getattr(self.crrm, c)
            for i in range(self.n_ues):
                self.data[c][i][self.n_rows_filled] = field.data[i]
        self.n_rows_filled += 1
        if (
            self.n_rows_filled >= self.bs * self.n_blocks_allocated
        ):  # full, allocate new blocks
            if verbose:
                print(f"capture: n_rows_filled={self.n_rows_filled}, reallocating")
            for c in self.captures:
                for i in range(self.n_ues):
                    new_block = np.empty_like(self.data[c][i][: self.bs])
                    self.data[c][i] = np.vstack([self.data[c][i], new_block])
            self.n_blocks_allocated += 1

    def get_data(self, c, i):
        "get trace c for UE i"
        return self.data[c][i][: self.bs]

    def dump(self, fields="all", precision=3, linewidth=500, suppress=False):
        np.set_printoptions(precision=precision, linewidth=linewidth, suppress=suppress)
        print(f"CRRM_logger.dump, n_rows_filled={self.n_rows_filled}:")
        cs = self.captures if fields == "all" else fields
        cs = tuple(set(cs) & set(self.captures))
        for c in cs:
            print(f"{c}:")
            for i in self.ues:
                print(f"  UE[{i}]:")
                if fields == "all" or c in fields:
                    data = self.data[c][i][: self.n_rows_filled]
                    print("    " + str(data).replace("\n", "\n    "))

    def plot(
        self,
        fields="all",
        averages=[],
        fnbase="img/CRRM_logger",
        image_formats=("png", "pdf"),
        title="",
        xlabel="time",
        linewidth=1.5,
        markersize=0.3,
        colormap="tab20b",
        legend_fontsize=10,
        label_fontsize=12,
        title_fontsize=14,
        smooth_averages=True,
        x_axis=None,
    ):
        if x_axis is None:
            x_axis = np.arange(0, self.crrm.params.n_moves)
        cs = self.captures if fields == "all" else fields
        n_axes = len(cs)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.subplots(n_axes, 1)
        if n_axes == 1:
            ax = [ax]
        if title:
            ax[0].set_title(title, fontsize=title_fontsize)
        if xlabel:
            ax[-1].set_xlabel(xlabel, fontsize=label_fontsize)
        ax[-1].set_xlim(x_axis[0], x_axis[-1])
        xticks = ax[-1].get_xticks()
        capture_to_axis_index_map = {}
        for j, c in enumerate(cs):
            capture_to_axis_index_map[c] = j
            ax[j].set_xlim(x_axis[0], x_axis[-1])
            if j < n_axes - 1:
                ax[j].set_xticks(xticks, [""] * len(xticks))
            ax[j].grid(color="gray", linewidth=0.5, alpha=0.5)
            for i, trace0 in self.data[c].items():  # for each UE
                trace = trace0[: self.n_rows_filled]
                label = f"UE[{i}]"
                if c in ("a",):  # discrete data, no subbands
                    color = self.get_color(i, 0, colormap)
                    ax[j].scatter(x_axis, trace, label=label, s=markersize, color=color)
                elif c in ("cqi", "mcs"):  # discrete data, show each subband
                    if c == "cqi":
                        ax[j].set_ylim(0, 15)
                    elif c == "mcs":
                        ax[j].set_ylim(0, 28)
                    for k, subband in enumerate(trace.T):
                        color = self.get_color(i, k, colormap)
                        plot_line_with_y_jumps(
                            ax[j],
                            x_axis,
                            subband,
                            label=label + f" sb[{k}]",
                            color=color,
                            lw=linewidth,
                        )
                elif c in ("UE_location",):  # continuous data, no subbands
                    color = self.get_color(i, 0, colormap)
                    ax[j].plot(x_axis, trace, label=label, lw=linewidth, color=color)
                elif c in ("sinr"):  # continuous data, show each subband in dB
                    for k, subband in enumerate(trace.T):
                        color = self.get_color(i, k, colormap)
                        ax[j].plot(
                            x_axis,
                            to_dB(subband),
                            label=label + f" sb[{k}]",
                            lw=linewidth,
                            color=color,
                        )
                elif c in (
                    "se_Shannon"
                ):  # continuous data, show each subband not in dB
                    for k, subband in enumerate(trace.T):
                        color = self.get_color(i, k, colormap)
                        ax[j].plot(
                            x_axis,
                            subband,
                            label=label + f" sb[{k}]",
                            lw=linewidth,
                            color=color,
                        )
                elif c in ("rsrp",):  # continuous data, show each subband not in dB
                    # it doesn't really make sense to plot RSRP - too many dimensions!
                    print(f"rsrp plots not yet implemented")
                elif c in ("tp",):  # continuous data, sum over subbands
                    color = self.get_color(i, 0, colormap)
                    y = np.sum(trace, axis=-1)
                    ax[j].plot(x_axis, y, label=label, lw=linewidth, color=color)
        # TODO implement averages for other captures...
        if "tp" in averages and "tp" in capture_to_axis_index_map:
            tp_data = np.array(
                [
                    np.sum(self.data["tp"][i][: self.n_rows_filled], axis=-1)
                    for i in self.ues
                ]
            )
            y = np.average(tp_data, axis=0)
            j = capture_to_axis_index_map["tp"]
            if smooth_averages:
                n_x = len(x_axis)
                x_axis_new = np.linspace(np.min(x_axis), np.max(x_axis), n_x // 10)
                spline = make_interp_spline(x_axis, y, k=3)
                y = spline(x_axis_new)
                ax[j].plot(
                    x_axis_new, y, label="average", lw=1.5 * linewidth, color="green"
                )
            else:  # don't smooth
                ax[j].plot(
                    x_axis, y, label="average", lw=1.5 * linewidth, color="green"
                )
        # configure the legends...
        for j, c in enumerate(cs):
            ax[j].set_ylabel(self.traces[c], fontsize=label_fontsize)
            handles, labels = ax[j].get_legend_handles_labels()
            # don't repeat labels in legend...
            labels, ids = np.unique(labels, return_index=True)
            handles = [handles[l] for l in ids]
            legend = ax[j].legend(
                handles,
                labels,
                loc="upper right",
                shadow=True,
                framealpha=0.75,
                fontsize=legend_fontsize,
            )
            # bigger dots in legend...
            for handle in legend.legend_handles:
                handle._sizes = [20]
            # change the line width for the legend
            for line in legend.get_lines():
                line.set_linewidth(2.0)
        fig_timestamp(fig, author=self.crrm.params.author, fontsize=10)
        fig.tight_layout()
        if "png" in image_formats:
            fig.savefig(fnbase + ".png", dpi=300)
            print(f"eog {fnbase}.png &")
        if "pdf" in image_formats:
            fig.savefig(fnbase + ".pdf")
            print(f"evince --page-label=1 {fnbase}.pdf &")
        if not image_formats or "show" in image_formats:
            plt.show()

    def get_color(self, i, j, colormap="tab20b"):
        """Get a color for UE[i] subband [j].
        For colormap in ['tab20a','tab20b','tab20c',],
        it will be unique if less than 5 UEs and 4 subbands are plotted.
        https://matplotlib.org/stable/users/explain/colors/colormaps.html
        """
        if "tab20" in colormap:
            block = (3 * i) % 5  # which of the 5 colors block we want
            grad = (3 * j) % 4  # one of the four gradients in that color block
            return colormaps[colormap](0.25 * block + grad / 20.0)
        # else fall back on a crude scheme...
        discrete_colors = (
            "red",
            "green",
            "blue",
            "orange",
            "cyan",
            "violet",
            "yellow",
        )
        n_discrete_colors = len(discrete_colors)
        if colormap == "discrete":
            return discrete_colors[i % n_discrete_colors]


# END class CRRM_data


def plot_line_with_x_gaps(ax, x, y, *args, **kwargs):
    # Keith Briggs 2025-07-28
    # function to plot a broken line: detect x gaps, split into separate lines
    # not useful here, as it splits at gaps in x, not jumps in y!
    xdiff = np.diff(x)
    min_xdiff = np.min(xdiff)
    split_points = np.where(xdiff > min_xdiff)[0] + 1
    xs = np.split(x, split_points)
    ys = np.split(y, split_points)
    for x, y in zip(xs, ys):
        ax.plot(x, y, *args, **kwargs)


def plot_line_with_y_jumps(ax, x, y, *args, **kwargs):
    # Keith Briggs 2025-09-04
    # function to plot a broken line: detect y jumps, split into separate lines
    ydiff = np.diff(y)
    split_points = np.where(ydiff != 0.0)[0] + 1
    xs = np.split(x, split_points)
    ys = np.split(y, split_points)
    for x, y in zip(xs, ys):
        ax.plot(x, y, *args, **kwargs)
