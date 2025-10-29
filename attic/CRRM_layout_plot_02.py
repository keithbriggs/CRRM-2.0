# Keith Briggs 2025-09-07 renamed from CRRM_plot_02.py
# Keith Briggs 2025-09-05 v02, uses new _get_approximate_pathloss_dB_for_layout_plot
# Keith Briggs 2025-08-08, from https://github.com.mcas.ms/apw804/CellularReferenceModel/blob/main/src/cellular_reference_model_10d.py

from platform import system as platform_system

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import ticker

from utilities import fig_timestamp, bright_yellow, red


class CRRM_layout_plot:
    """
    Class for visualizing cellular reference model layouts, including cells, user equipment (UE), distances, and Voronoi diagrams.

    Parameters
    ----------
    fnbase : str, optional
      Base filename for saving figures (default is '').
    xlim : tuple of float, optional
      X-axis limits for the plot (default is (-4.5, 4.5)).
    ylim : tuple of float, optional
      Y-axis limits for the plot (default is (-4.5, 4.5)).
    grid : bool, optional
      Whether to display a grid on the plot (default is False).
    cell_image : str or None, optional
      Path to image file for representing cells (default is None).
    UE_image : str or None, optional
      Path to image file for representing UEs (default is None).
    cell_image_zoom : float, optional
      Zoom factor for cell images (default is 5e-2).
    UE_image_zoom : float, optional
      Zoom factor for UE images (default is 8e-2).

    Attributes
    ----------
    fig : matplotlib.figure.Figure
      The matplotlib figure object.
    ax : matplotlib.axes.Axes
      The axes object for plotting.
    cell_image : str or None
      Path to cell image file.
    cell_image_zoom : float
      Zoom factor for cell images.
    UE_image : str or None
      Path to UE image file.
    UE_image_zoom : float
      Zoom factor for UE images.
    fnbase : str
      Base filename for saving figures.
    """

    def __init__(
        self,
        layout_plot_fnbase,
        author="",
        xlim=(-4.5, 4.5),
        ylim=(-4.5, 4.5),
        grid=False,
        cell_image=None,
        UE_image=None,
        cell_image_zoom=2e-2,
        UE_image_zoom=8e-2,
        no_ticks=True,
        quiet=False,
        dpi=200,
        figsize=(6, 6),
        label_ues=False,
    ):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot()
        if grid:
            self.ax.grid(color="gray", lw=0.5, alpha=0.5)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.cell_image = cell_image
        self.cell_image_zoom = cell_image_zoom
        self.UE_image = UE_image
        self.no_ticks = no_ticks
        self.UE_image_zoom = UE_image_zoom
        self.quiet = quiet
        self.dpi = dpi
        self.label_ues = label_ues
        self.fnbase = layout_plot_fnbase
        self.author = author

    def getImage(self, path, zoom=1):
        # https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points?
        return OffsetImage(plt.imread(path), zoom=zoom)

    def base(self, cells, ues, show_kilometres=False):
        """
        Plots the base layout of cells and user equipments (UEs) on the current axes.

        Parameters
        ----------
        cells : np.ndarray
          Array of cell coordinates with shape (N, 2).
        ues : np.ndarray
          Array of UE coordinates with shape (M, 2).
        show_kilometres : bool, optional
          If True, axis labels and ticks are shown in kilometres. Default is False.

        Notes
        -----
        If image files for cells or UEs are specified and can be loaded, they are used as markers; otherwise, default scatter markers are used.
        """
        if self.cell_image is None:
            self.ax.scatter(
                cells[:, 0],
                cells[:, 1],
                marker="o",
                s=50,
                color="red",
                alpha=1.0,
                zorder=3,
            )
        else:
            image_ok = True
            try:
                cell_image = self.getImage(self.cell_image, zoom=self.cell_image_zoom)
            except:
                print(red(f"Could not open image file {self.cell_image}!"))
                image_ok = False
            if image_ok:
                for x, y in zip(cells[:, 0], cells[:, 1]):
                    self.ax.add_artist(
                        AnnotationBbox(cell_image, (x, y), frameon=False)
                    )
        if self.UE_image is None:
            self.ax.scatter(
                ues[:, 0],
                ues[:, 1],
                marker=".",
                s=max(30 - 0.02 * len(ues[:]), 5),
                color="blue",
                zorder=3,
            )
            if self.label_ues:  # Keith Briggs 2025-09-05
                for ue_i, ue in enumerate(ues):
                    self.ax.annotate(
                        f"{ue_i}", (ue[0], ue[1]), fontsize=8, color="blue"
                    )
        else:
            image_ok = True
            try:
                UE_image = self.getImage(self.UE_image, zoom=self.UE_image_zoom)
            except:
                print(red(f"Could not open image file {self.UE_image}!"))
                image_ok = False
            if image_ok:
                for x, y in zip(ues[:, 0], ues[:, 1]):
                    self.ax.add_artist(AnnotationBbox(UE_image, (x, y), frameon=False))
        if show_kilometres:
            self.ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{x/1000:.1f}")
            )
            self.ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda y, pos: f"{y/1000:.1f}")
            )
            if not self.no_ticks:
                self.ax.set_xlabel("km")
                self.ax.set_ylabel("km")

    def distance(
        self,
        cells=None,
        ues=None,
        plot_type="nearest",
        label_distance=False,
        attachment_vector=None,
        n_sectors=1,
    ):  # 2025-08-20
        """
        Plots distances between cells and user equipments (UEs) on the current axes.

        Parameters
        ----------
        cells : array-like, optional
          Coordinates of the cells. If None, uses the first scatter plot data from `self.ax`.
        ues : array-like, optional
          Coordinates of the UEs. If None, uses the second scatter plot data from `self.ax`.
        plot_type : {'all', 'actual_attachment', 'nearest'}, default='all'
          Type of distance visualization:
          - 'all': Draws lines between every cell and every UE.
          - 'actual_attachment': (Not implemented) Intended for actual associations.
          - 'nearest': Draws lines between each UE and its nearest cell.
        label_distance : bool, default=False
          If True, annotates the plotted lines with the computed distances.

        Returns
        -------
        None
        """
        if cells is None:
            cells = self.ax.collections[0].get_offsets()
        if ues is None:
            ues = self.ax.collections[1].get_offsets()
        if plot_type == "all":  # draw a line between each cell and each ue
            for cell, ue in product(cells, ues):
                self.ax.plot(
                    [cell[0], ue[0]],
                    [cell[1], ue[1]],
                    color="black",
                    lw=0.5,
                    alpha=0.5,
                    zorder=1,
                )
                dist = np.linalg.norm(cell - ue)
                self.ax.text(
                    (cell[0] + ue[0]) / 2,
                    (cell[1] + ue[1]) / 2,
                    f"{dist:.2f}",
                    fontsize=6,
                    color="black",
                    alpha=0.7,
                    zorder=1,
                )
        if plot_type == "actual_attachment":
            print(red("actual_attachment not implemented"))
        if plot_type == "nearest":  # 2025-08-20
            if attachment_vector is not None and n_sectors > 1:
                colors = ["gray", "black", "brown", "navy", "olive", "maroon"]
                for i, ue in enumerate(ues):
                    cell_index = attachment_vector[i]
                    cell_location = cells[cell_index]
                    sector_index = cell_index % n_sectors
                    color = colors[sector_index]
                    self.ax.plot(
                        [cell_location[0], ue[0]],
                        [cell_location[1], ue[1]],
                        color=color,
                        lw=0.75,
                        alpha=1,
                    )
            else:
                distances = np.linalg.norm(
                    cells[:, np.newaxis, :] - ues[np.newaxis, :, :], axis=2
                )
                nearest_cell_index = np.argmin(distances, axis=0)
                for i, ue in enumerate(ues):
                    self.ax.plot(
                        [cells[nearest_cell_index[i], 0], ue[0]],
                        [cells[nearest_cell_index[i], 1], ue[1]],
                        color="gray",
                        lw=0.5,
                        alpha=0.5,
                    )
                    if label_distance:
                        d = np.linalg.norm(cells[nearest_cell_index[i]] - ue)
                        self.ax.text(
                            (cells[nearest_cell_index[i], 0] + ue[0]) / 2,
                            (cells[nearest_cell_index[i], 1] + ue[1]) / 2,
                            f"{d:.2f}",
                            fontsize=6,
                            color="black",
                            alpha=0.7,
                        )

    def voronoi(self, cells=None):
        """
        Plots the Voronoi diagram for a given set of 2D cell coordinates.

        Parameters
        ----------
        cells : np.ndarray, optional
          A 2D numpy array of shape (n, 2) representing the coordinates of the cells.
          If None, raises a ValueError.

        Raises
        ------
        ValueError
          If `cells` is not a non-empty 2D numpy array with shape (n, 2).

        Notes
        -----
        The axis limits are preserved after plotting the Voronoi diagram.
        """
        if (
            cells is None
            or not isinstance(cells, np.ndarray)
            or cells.ndim != 2
            or cells.shape[1] != 2
        ):
            raise ValueError(
                "cells must be a non-empty 2D numpy array with shape (n, 2)"
            )
        try:
            vor = Voronoi(cells)
        except Exception as e:
            print(red(f"Error computing Voronoi diagram: {e}"))
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        voronoi_plot_2d(
            vor,
            ax=self.ax,
            show_vertices=False,
            show_points=False,
            line_colors="green",
            line_width=1.0,
            line_alpha=0.5,
        )
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

    def savefig(self, fn=None, timestamp=True, fmt=("png", "pdf"), author=""):
        """
        Save the current figure to file(s) in specified formats, optionally adding a timestamp and author.

        Parameters
        ----------
        fn : str or None, optional
          Base filename for saving the figure. If None, uses the existing filename base.
        timestamp : bool, default=True
          Whether to add a timestamp and author annotation to the figure.
        fmt : list of str, default=['png', 'pdf']
          List of file formats to save the figure in.
        author : str, optional
          Name of the author to include in the timestamp annotation.

        Notes
        -----
        Prints commands to open the saved files depend on the operating system.
        """
        if self.no_ticks:  # 2025-08-21
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        if fn is not None:
            self.fnbase = fn
        if timestamp:
            fig_timestamp(self.fig, author=author, fontsize=6)
        self.fig.tight_layout()
        commands = {"Darwin": "open", "Linux": "eog", "Windows": ""}
        system = platform_system()
        command = commands.get(system, "")
        for ext in fmt:
            if ext == "pdf":
                self.fig.savefig(f"{self.fnbase}.{ext}", pad_inches=0.1)
            else:
                self.fig.savefig(f"{self.fnbase}.{ext}", pad_inches=0.1, dpi=self.dpi)
            if command:
                if system == "Linux":
                    if not self.quiet and ext == "png":
                        print(f"eog {self.fnbase}.{ext} &")
                    if not self.quiet and ext == "pdf":
                        print(f"evince --page-label=1 {self.fnbase}.{ext} &")
                else:
                    if not self.quiet:
                        print(f"{command} {self.fnbase}.{ext}")
            else:
                if not self.quiet:
                    print(f"{self.fnbase}.{ext} written.")
        plt.close(self.fig)
