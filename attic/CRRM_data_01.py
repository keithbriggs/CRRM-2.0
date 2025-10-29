# Keith Briggs 2025-09-04
# class to capture data during a CRRM run, and plot it

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.interpolate import make_interp_spline, BSpline

from utilities import fig_timestamp, to_dB


class CRRM_data:
    def __init__(s, crrm, captures=[], ues=[], n_captures=None):
        s.crrm = crrm
        s.ues = ues
        s.ylabels = {
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
        # only allow captures which are in ylabels.keys()...
        s.captures = tuple(set(captures) & set(s.ylabels.keys()))
        if n_captures is None:
            s.data = {c: {} for c in s.captures}
        else:
            s.data = {
                c: {} for c in s.captures
            }  # TODO pre-allocate n_captures (if nto None) rows for all arrays

    def capture(s):
        s.crrm.update()
        for c in s.captures:
            data = getattr(s.crrm, c).data
            if len(data.shape) == 3:  # FIXME do we have to have a special case?
                for i in s.ues:
                    if i in s.data[c]:
                        s.data[c][i] = np.vstack([s.data[c][i], data[i, np.newaxis]])
                    else:
                        s.data[c][i] = data[i, np.newaxis]
            else:
                for i in s.ues:
                    if i in s.data[c]:
                        s.data[c][i] = np.vstack([s.data[c][i], data[i]])
                    else:
                        s.data[c][i] = data[i]

    def dump(s, fields="all", precision=3, linewidth=500, suppress=False):
        np.set_printoptions(precision=precision, linewidth=linewidth, suppress=suppress)
        print(f"CRRM_data.dump:")
        cs = s.captures if fields == "all" else fields
        cs = tuple(set(cs) & set(s.captures.keys()))
        for c in cs:
            print(f"{c}:")
            for i in s.ues:
                print(f"  UE[{i}]:")
                if fields == ("all",) or c in fields:
                    print("    " + str(s.data[c][i]).replace("\n", "\n    "))

    def plot(
        s,
        fields="all",
        averages=[],
        fnbase="img/CRRM_data",
        image_formats=("png", "pdf"),
        title="",
        xlabel="time",
        linewidth=1.5,
        markersize=0.3,
        colormap="tab20b",
        legend_fontsize=8,
        smooth_averages=True,
        x_axis=None,
    ):
        if x_axis is None:
            x_axis = np.arange(0, s.crrm.params.n_moves)
        cs = s.captures if fields == "all" else fields
        n_axes = len(cs)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.subplots(n_axes, 1)
        if n_axes == 1:
            ax = [ax]
        if title:
            ax[0].set_title(title)
        if xlabel:
            ax[-1].set_xlabel(xlabel)
        ax[-1].set_xlim(x_axis[0], x_axis[-1])
        xticks = ax[-1].get_xticks()
        capture_to_axis_index_map = {}
        for j, c in enumerate(cs):
            capture_to_axis_index_map[c] = j
            ax[j].set_xlim(x_axis[0], x_axis[-1])
            if j < n_axes - 1:
                ax[j].set_xticks(xticks, [""] * len(xticks))
            ax[j].grid(color="gray", linewidth=0.5, alpha=0.5)
            for i, trace in s.data[c].items():  # for each UE
                label = f"UE[{i}]"
                if c in ("a",):  # discrete data, no subbands
                    color = s.get_color(i, 0, colormap)
                    ax[j].scatter(x_axis, trace, label=label, s=markersize, color=color)
                elif c in ("cqi", "mcs"):  # discrete data, show each subband
                    if c == "cqi":
                        ax[j].set_ylim(0, 15)
                    elif c == "mcs":
                        ax[j].set_ylim(0, 28)
                    for k, subband in enumerate(trace.T):
                        color = s.get_color(i, k, colormap)
                        plot_line_with_y_jumps(
                            ax[j],
                            x_axis,
                            subband,
                            label=label + f" sb[{k}]",
                            color=color,
                            lw=linewidth,
                        )
                elif c in ("UE_location",):  # continuous data, no subbands
                    color = s.get_color(i, 0, colormap)
                    ax[j].plot(x_axis, trace, label=label, lw=linewidth, color=color)
                elif c in ("sinr"):  # continuous data, show each subband in dB
                    for k, subband in enumerate(trace.T):
                        color = s.get_color(i, k, colormap)
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
                        color = s.get_color(i, k, colormap)
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
                    color = s.get_color(i, 0, colormap)
                    y = np.sum(trace, axis=-1)
                    ax[j].plot(x_axis, y, label=label, lw=linewidth, color=color)
        # TODO implement averages for other captures...
        if "tp" in averages and "tp" in capture_to_axis_index_map:
            tp_data = np.array([np.sum(s.data["tp"][i], axis=-1) for i in s.ues])
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
            ax[j].set_ylabel(s.ylabels[c])
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
        fig_timestamp(fig, author=s.crrm.params.author, fontsize=10)
        fig.tight_layout()
        if "png" in image_formats:
            fig.savefig(fnbase + ".png", dpi=300)
            print(f"eog {fnbase}.png &")
        if "pdf" in image_formats:
            fig.savefig(fnbase + ".pdf")
            print(f"evince --page-label=1 {fnbase}.pdf &")
        if not image_formats or "show" in image_formats:
            plt.show()

    def get_color(s, i, j, colormap="tab20b"):
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
