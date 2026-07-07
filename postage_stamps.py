import warnings
from pathlib import Path
from typing import Union

import arrow
import cartopy.crs as ccrs
import cf_units
import iris
import iris.cube
import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from iris.analysis.cartography import unrotate_pole
from mpl_toolkits.axes_grid1 import AxesGrid

warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")


class BasePlot:
    """
    Contains base plotting routines for individual postage stamps. Can also be run on its own to
    plot a single map.
    """

    def __init__(self, rasterized=False) -> None:
        """
        Args:
            rasterized (bool, optional):
                Whether to rasterize the plot (i.e., don't plot as vectors to reduce space).
                Defaults to False.
        """
        self.rasterized = rasterized
        # plt.rcParams.update({"figure.figsize": [8, 11]})

    @staticmethod
    def get_projection(cube: iris.cube.Cube) -> ccrs.Projection:
        """
        Gets the cartopy projection from the cube. If this is not possible, defaults to PlateCarree

        Args:
            cube (iris.cube.Cube):
                Input cube to find projection from

        Returns:
            ccrs.Projection
        """
        # Find name of the x coordinate in cube
        coord_var_names = [coord.var_name for coord in cube.coords()]
        coord_std_names = [coord.standard_name for coord in cube.coords()]
        coord_names = coord_std_names + coord_var_names
        matching_coord_names = [
            "grid_longitude",
            "longitude",
            "projection_x_coordinate",
        ]
        for coord_name in matching_coord_names:
            if coord_name in coord_names:
                crs_coord_name = coord_name

        # Use this x coordinate to try using the crs from the cube.
        crs_from_cube = cube.coord(crs_coord_name).coord_system.as_cartopy_crs()
        if isinstance(crs_from_cube, ccrs.Projection):
            projection = crs_from_cube
        else:
            projection = ccrs.PlateCarree()

        return projection

    def get_contourf_levels_and_cmap(self, data_type: str):
        if data_type == "features":
            clevels = [0, 1, 2]
            cmap = ["w", "#31a354"]
        elif data_type == "probabilities":
            clevels = np.arange(0, 1.1, 0.1)
            # Just to catch any rounding errors. Does not show up on cbar
            clevels[-1] = 1.01
            cmap = plt.get_cmap("GnBu")(clevels)
        else:
            clevels = [0.0, 0.01, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
            cmap = (
                "w",
                "#6baed6",
                "#3182bd",
                "#08519c",
                "#31a354",
                "y",
                "#fd8d3c",
                "#e31a1c",
                "#df65b0",
                "w",
            )

        return clevels, cmap

    def get_contour_levels(self, cube_name: str):
        precip_cubenames = ["rainfall", "precipitation", "accumulation"]
        if "pressure" in cube_name:
            clevels = np.arange(900, 1050, 5)
        elif any(precip_name in cube_name for precip_name in precip_cubenames):
            clevels = [0.0, 0.01, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        else:
            clevels = [0.0, 0.01, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

        return clevels

    def plot_precip_cube(
        self,
        cube: iris.cube.Cube,
        show_mask: bool = True,
        show_cbar: bool = True,
        show_title: bool = True,
        title_info: str = None,
        data_type: str = "data",
        projection=None,
    ):
        """
        Plots the input cube as a precip map. Can also plot the mask if flagged, and can change the title
        info based on the input. Can also change the colour scheme based on the data type
        (e.g., just features, probabilities or actual precip values).

        Args:
            cube (iris.cube.Cube):
                Input cube to plot
            show_mask (bool, optional):
                Whether to plot the mask. Defaults to True.
            show_cbar (bool, optional):
                Whether to plot the colorbar. Defaults to True.
            show_title (bool, optional):
                Whether to plot the title. Defaults to True.
            title_info (str, optional):
                Information to include in the title. Defaults to None.
            data_type (str, optional):
                The type of data to plot. Defaults to "data".
            projection (_type_, optional):
                The cartopy projection to use. Defaults to None, i.e., uses the cube's projection.


        Returns:
            tuple: (fig, ax, cf) where fig is the figure object, ax is the axis object and cf is the contourf object
        """
        if projection is None:
            ax = plt.axes(projection=self.get_projection(cube))
        else:
            ax = plt.axes(projection=projection)
        cf = self.make_precip_plot(
            cube,
            ax=ax,
            show_mask=show_mask,
            show_cbar=show_cbar,
            cmap_type=data_type,
        )

        if show_title:
            plt.gcf().suptitle(self.make_title(cube, title_info))

        plt.tight_layout()
        return plt.gcf(), ax, cf

    def plot_wind_cube(
        self,
        cube: iris.cube.Cube,
        show_mask: bool = True,
        show_cbar: bool = True,
        show_title: bool = True,
        title_info: str = None,
        data_type: str = "data",
        projection=None,
    ):
        """
        Plots the input cube as a precip map. Can also plot the mask if flagged, and can change the title
        info based on the input. Can also change the colour scheme based on the data type
        (e.g., just features, probabilities or actual precip values).

        Args:
            cube (iris.cube.Cube):
                Input cube to plot
            show_mask (bool, optional):
                Whether to plot the mask. Defaults to True.
            show_cbar (bool, optional):
                Whether to plot the colorbar. Defaults to True.
            show_title (bool, optional):
                Whether to plot the title. Defaults to True.
            title_info (str, optional):
                Information to include in the title. Defaults to None.
            data_type (str, optional):
                The type of data to plot. Defaults to "data".
            projection (_type_, optional):
                The cartopy projection to use. Defaults to None, i.e., uses the cube's projection.


        Returns:
            tuple: (fig, ax, cf) where fig is the figure object, ax is the axis object and cf is the contourf object
        """
        if projection is None:
            ax = plt.axes(projection=self.get_projection(cube))
        else:
            ax = plt.axes(projection=projection)
        cf = self.make_vector_plot(
            cube,
            ax=ax,
            show_mask=show_mask,
            show_cbar=show_cbar,
            cmap_type=data_type,
        )

        if show_title:
            plt.gcf().suptitle(self.make_title(cube, title_info))

        plt.tight_layout()
        return plt.gcf(), ax, cf

    def make_precip_plot(
        self,
        cube,
        ax,
        show_mask=True,
        show_cbar=True,
        cmap_type="data",
        plotting_func=iplt.contourf,
    ):
        if cmap_type == "features":
            cbar_label = "Features"
        elif cmap_type == "probabilities":
            cbar_label = "Probability of Features"
        else:
            cbar_label = "Accumulation (mm)"

        self.clevels, cmap = self.get_contourf_levels_and_cmap(cmap_type)

        if plotting_func not in [iplt.contourf, iplt.pcolormesh]:
            raise ValueError(
                f"plotting_func must be either iplt.contourf or iplt.pcolormesh, got {plotting_func}"
            )

        if show_mask and np.ma.is_masked(cube.data):
            mask_cube = self.__get_mask__(cube)
            cf_mask = plotting_func(
                mask_cube,
                colors=["grey", "w"],
                levels=[0, 2, 4],
                alpha=0.7,
                axes=ax,
                rasterized=self.rasterized,
            )
        else:
            cube = cube.copy(data=np.array(np.ma.getdata(cube.data)))

        cf = plotting_func(
            cube,
            levels=self.clevels,
            colors=cmap,
            axes=ax,
            rasterized=self.rasterized,
        )

        if show_cbar:
            cbar = plt.colorbar(
                cf, orientation="horizontal", ticks=self.clevels, pad=0.01
            )
            cbar.set_label(cbar_label)

        ax.coastlines(resolution="10m")
        return cf

    def make_vector_plot(
        self, cube, ax, show_mask=True, show_cbar=True, cmap_type="data"
    ):
        # TODO: test this code works
        # Get u and v components from cube
        u_cube = cube.extract(iris.Constraint(name="eastward_wind"))
        v_cube = cube.extract(iris.Constraint(name="northward_wind"))

        # Get the x and y coordinates for the quiver plot
        x_coord_name = None
        y_coord_name = None
        for coord in cube.coords():
            if "longitude" in coord.name().lower() or "x" in coord.name().lower():
                x_coord_name = coord.name()
            elif "latitude" in coord.name().lower() or "y" in coord.name().lower():
                y_coord_name = coord.name()

        x_coords = cube.coord(x_coord_name).points
        y_coords = cube.coord(y_coord_name).points

        # Create the quiver plot
        q = ax.quiver(
            x_coords,
            y_coords,
            u_cube.data,
            v_cube.data,
            scale=400,
            regrid_shape=20,
            transform=self.get_projection(cube),
        )

        if show_cbar:
            cbar = plt.colorbar(q, orientation="horizontal", pad=0.01)
            cbar.set_label("Wind Speed (m/s)")

        ax.coastlines(resolution="10m")
        return q

    def overplot_contours(
        self,
        cube,
        ax,
        levels=None,
        colours="k",
        linewidths=1,
        linestyles="solid",
        contour_labels=False,
    ):
        if levels is None:
            levels = self.get_contour_levels(cube.name())

        contours = iplt.contour(
            cube,
            levels=levels,
            colors=colours,
            linewidths=linewidths,
            linestyles=linestyles,
            axes=ax,
        )
        if contour_labels:
            plt.clabel(contours, inline=True, fontsize=8, fmt="%1.0f")
        return contours

    def make_title(
        self,
        cube: iris.cube.Cube,
        extra_title_info: str = None,
        accumulation_window: int = None,
    ) -> str:
        """
        Construct title for each plot/postage stamp. This is based on the cycle time
        and leadtime of the precip, as well as any extra info passed in.
        """
        fcst_coord = cube.coord("forecast_reference_time")
        cycle_datetime = cf_units.num2date(
            fcst_coord.points[0],
            unit=fcst_coord.units.name,
            calendar=fcst_coord.units.calendar,
        )
        cycle_datetime = arrow.Arrow.fromdatetime(cycle_datetime)

        if "leadtime" in [coord.long_name for coord in cube.coords()]:
            lt_in_middle_of_window = False
            lt = cube.coord("leadtime").points
            if len(lt) > 1:
                raise ValueError(
                    f"Expected only a single leadtime coordinate, got: {lt}"
                )
            lt = int(lt[0])

            if accumulation_window is not None:
                lt_str = f"T+{lt - accumulation_window}-{lt}h"
            else:
                lt_str = f"T+{lt}h"

        elif "forecast_period" in [coord.var_name for coord in cube.coords()]:
            lt_in_middle_of_window = False
            fp_coord = cube.coord("forecast_period")
            fp_coord.convert_units("hours")
            fp = fp_coord.points
            if len(fp) > 1:
                raise ValueError(
                    f"Expected only a single forecast_period coordinate, got: {fp_coord.points}"
                )
            lt = fp[0]

            if accumulation_window is not None:
                lt_str = f"T+{lt - accumulation_window}-{lt}h"
            else:
                lt_str = f"T+{lt}h"

        else:
            lt_str = ""
            lt = None

        title = f"Cycle: {cycle_datetime.format('YYYY-MM-DD HH:mm')}Z {lt_str}\n"
        if lt is not None:
            if accumulation_window is not None and not lt_in_middle_of_window:
                title += f"Valid: {cycle_datetime.shift(hours=lt - accumulation_window).format('YYYY-MM-DD HH:mm')}Z "
                title += (
                    f"to {cycle_datetime.shift(hours=lt).format('YYYY-MM-DD HH:mm')}Z\n"
                )
            elif accumulation_window is not None and lt_in_middle_of_window:
                title += f"Valid: {cycle_datetime.shift(hours=lt - accumulation_window / 2).format('YYYY-MM-DD HH:mm')}Z "
                title += f"to {cycle_datetime.shift(hours=lt + accumulation_window / 2).format('YYYY-MM-DD HH:mm')}Z\n"
            else:
                title += f"Valid: {cycle_datetime.shift(hours=lt).format('YYYY-MM-DD HH:mm')}Z\n"
        if extra_title_info is not None:
            title += extra_title_info
        return title

    def get_y_pos_for_title(self, grid, pad_points=8.0, min_gap_points=4.0, max_y=0.98):
        # Use plotted/visible axes to estimate a stable top edge for the title.
        active_axes = [ax for ax in grid if ax.get_visible() and ax.axison]
        if len(active_axes) == 0:
            active_axes = list(grid)

        fig = active_axes[0].figure
        fig_height_points = fig.get_figheight() * 72.0
        pad = pad_points / fig_height_points
        min_gap = min_gap_points / fig_height_points

        top_of_axes = max(ax.get_position().ymax for ax in active_axes)
        y_pos = top_of_axes + pad

        # Keep a small gap above the plots and avoid pushing text off the figure.
        y_pos = max(y_pos, top_of_axes + min_gap)
        y_pos = min(y_pos, max_y)
        return y_pos

    def enforce_ax_bounds_from_data(self, cube: iris.cube.Cube, ax: plt.axes) -> None:
        """
        Sometimes when using GridSpec, the bounds for each plot default to global for some reason.
        Here, take the x and y bounds and re-enforceaxis bounds (taking into account rotated
        latlon grid where necessary)
        """
        coord_var_names = [coord.var_name for coord in cube.coords()]
        coord_std_names = [coord.standard_name for coord in cube.coords()]
        coord_names = coord_std_names + coord_var_names
        matching_x_coord_names = [
            "grid_longitude",
            "longitude",
            "projection_x_coordinate",
        ]
        matching_y_coord_names = [
            "grid_latitude",
            "latitude",
            "projection_y_coordinate",
        ]
        for coord_name in matching_x_coord_names:
            if coord_name in coord_names:
                x_coord_name = coord_name

        for coord_name in matching_y_coord_names:
            if coord_name in coord_names:
                y_coord_name = coord_name

        x_coords = cube.coord(x_coord_name).points
        y_coords = cube.coord(y_coord_name).points

        # Unrotate pole
        if "grid" in x_coord_name:
            cube_cs = cube.coord_system()
            lons, lats = np.meshgrid(x_coords, y_coords)
            x_coords, y_coords = unrotate_pole(
                lons,
                lats,
                cube_cs.grid_north_pole_longitude,
                cube_cs.grid_north_pole_latitude,
            )
        else:
            # Only wrap longitudes when coordinates are on a 0..360 domain.
            # Subtracting 360 unconditionally can push already negative longitudes
            # far outside the map domain and break Cartopy clipping.
            x_coords = np.asarray(x_coords)
            if np.nanmax(x_coords) > 180:
                x_coords = x_coords - 360

        x_flat = np.asarray(x_coords).flatten()
        y_flat = np.asarray(y_coords).flatten()
        x_bounds = [np.nanmin(x_flat), np.nanmax(x_flat)]
        y_bounds = [np.nanmin(y_flat), np.nanmax(y_flat)]

        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)

    def __get_mask__(self, cube):
        """
        Returns a cube where the data is given a value of 3 if the mask is False and 0 if the maks is True

        Args:
            cube (iris.cube.Cube): Cube with mask
        """
        mask = np.ma.getmask(cube.data)
        mask_idxs = np.where(mask == False)
        mask_data = np.zeros(np.shape(cube.data))
        mask_data[mask_idxs] = 3
        mask_cube = cube.copy(data=mask_data)
        return mask_cube

    def __plot_mask__(self, cube):
        if not np.ma.is_masked(cube.data):
            return
        mask_cube = self.__get_mask__(cube)

        qplt.pcolormesh(mask_cube, vmin=0, vmax=20, cmap="CMRmap_r")
        plt.gca().coastlines(resolution="10m")

        return mask_cube


class GridArranger:
    def __init__(
        self,
        arrangement: str = "horizontal",
        max_axes_per_lead_dim: int = 6,
        projection="default",
    ) -> None:
        self.max_axes_per_lead_dim = max_axes_per_lead_dim
        self.nrows_ncols = ()
        self.ax_idxs = {}
        self.projection = projection

        vtcl_keywords = ["vertical", "v"]
        hzntl_keywords = ["horizontal", "h"]

        if arrangement in hzntl_keywords:
            self.arrangement = "horizontal"
            # self.figsize = (30, 20)
            self.cbar_location = "right"
            self.cbar_pad = 0.2

        elif arrangement in vtcl_keywords:
            self.arrangement = "vertical"
            # self.figsize = (20, 30)
            self.cbar_location = "bottom"
            self.cbar_pad = 0.05

        else:
            msg = "arrangement arg must be either 'horizontal' or 'vertical'"
            raise Exception(msg)

    def from_cube(
        self, cube: iris.cube.Cube, hide_axs_without_data: bool = True
    ) -> Union[plt.figure, plt.grid, dict]:
        self.set_nrows_ncols(cube)
        # # ax_idxs are just in numerical order if no member ordering is used
        real_iter = cube.coord("realization").points
        self.ax_idxs = {real: real_idx for real_idx, real in enumerate(real_iter)}

        fig = plt.figure(figsize=self.get_optimal_figsize())
        grid = self.setup_grid(cube, fig)

        if hide_axs_without_data:
            self.hide_axs_without_data(grid, self.ax_idxs)

        return fig, grid, self.ax_idxs

    def from_enforced_order_list(
        self, cube: iris.cube.Cube, order: list, hide_axs_without_data: bool = True
    ) -> Union[plt.figure, plt.grid, dict]:
        # Set the number of rows and cols to use based on memebr order
        self.set_ordered_nrows_ncols(order)

        # Get axis idxs dict for each member from the member_order arg, if used
        self.ax_idxs = self.get_ax_idxs_from_member_order(order)

        fig = plt.figure(figsize=self.get_optimal_figsize())
        grid = self.setup_grid(cube, fig)

        if hide_axs_without_data:
            self.hide_axs_without_data(grid, self.ax_idxs)

        return fig, grid, self.ax_idxs

    def set_nrows_ncols(self, cube: iris.cube.Cube, max_length=6) -> None:
        # The larger side will ether be the max side length, or the total number
        # of members, whichever is smaller
        # The smaller side length has to be big enough to contain the rest
        # of the members
        no_membs = len(cube.coord("realization").points)
        large_side_length = min(no_membs, max_length)
        small_side_length = int(np.ceil(no_membs / large_side_length))

        if self.arrangement == "horizontal":
            self.nrows_ncols = (small_side_length, large_side_length)
        else:
            self.nrows_ncols = (large_side_length, small_side_length)

    def set_ordered_nrows_ncols(self, clusters: list) -> None:
        # As an initial guess, large side length is the length of the longest cluster
        # But, also need to check if this exceeds the max axes per lead dim
        # small_side length is just the number f clusters
        cluster_sizes = [len(clust) for clust in clusters]
        large_side_length = min(max(cluster_sizes), self.max_axes_per_lead_dim)
        small_side_length = len(clusters)

        # Used to keep track of cluster ax idxs which can't fit on one line
        # This dict determines which row/column idx each cluster belongs to
        # By default, to start with, set this equal to the cluster number
        self.cluster_ax_idx_dims = {idx: [idx] for idx in range(len(clusters))}

        # Now, check large_side_length is not larger than each cluster size
        # For each overflowing cluster, add 1 to the small side length and
        # to the start idx of the next cluster
        for cluster_idx, cluster in enumerate(cluster_sizes):
            # -1 From the cluster size since we only want a new dim if there is an extra
            # axis to place on that dim. E.g., for 12 clusters and max_axes of 6,
            # Just doing floor(clusters/max_axes) = 2, even though only 1 extra row is needed.
            extra_small_side_lengths = int(
                np.floor((cluster - 1) / self.max_axes_per_lead_dim)
            )
            # If there are more cluster members than max axes, the floor will be > 0
            small_side_length += extra_small_side_lengths

            # Add extra dims to the current cluster ax idx
            current_ax_idx = self.cluster_ax_idx_dims[cluster_idx][0]
            new_ax_range = list(
                range(current_ax_idx, current_ax_idx + extra_small_side_lengths + 1)
            )
            self.cluster_ax_idx_dims[cluster_idx] = new_ax_range

            # Also add this as an offset to all of the next cluster ax idxs
            # Do this by finding all of the cluster labels greater than the
            # current one and adding the offset to all values in their respective lists
            for cluster_label, cluster_ax_idxs in self.cluster_ax_idx_dims.items():
                if cluster_label > cluster_idx:
                    self.cluster_ax_idx_dims[cluster_label] = [
                        c_ax + extra_small_side_lengths for c_ax in cluster_ax_idxs
                    ]

        if self.arrangement == "horizontal":
            self.nrows_ncols = (small_side_length, large_side_length)
        else:
            self.nrows_ncols = (large_side_length, small_side_length)

    def get_optimal_figsize(self) -> tuple:
        # As a simple way of getting a dynamic figsize which keeps each postage stamp
        # the same size, just multiply nrows and ncols by given parameters
        # This also makes the fig suptitle behaviour more controllable with just a simple
        # pad/offset to the y value. Otherwise, it gets a bit crazy
        stamp_x_size = 3
        stamp_y_size = 3
        figsize = (
            stamp_x_size * self.nrows_ncols[1],
            stamp_y_size * self.nrows_ncols[0],
        )
        return figsize

    def setup_grid(self, cube: iris.cube.Cube, fig: plt.figure) -> plt.grid:
        # Define the AxesGrid which holds the axes and cbar
        if self.projection == "default":
            axes_class = (GeoAxes, {"map_projection": BasePlot.get_projection(cube)})
        else:
            axes_class = (GeoAxes, {"map_projection": self.projection})

        grid = AxesGrid(
            fig,
            111,
            axes_class=axes_class,
            nrows_ncols=self.nrows_ncols,
            axes_pad=0.5,
            share_all=True,
            cbar_location=self.cbar_location,
            cbar_mode="single",
            cbar_size="5%",
            cbar_pad=self.cbar_pad,
            label_mode="1",  # Empty value necessary
        )
        return grid

    def get_full_grid_ax_idxs(self, nrows_ncols: tuple, arrangement: str) -> list:
        n_axs = nrows_ncols[0] * nrows_ncols[1]

        # To get the idx of every ax in the grid, reshape a range list into the correct shape
        # For horizontal arrangement, we want ascending order for each row
        # For vertical arrangement, we want ascending order for each column
        # However, ax_idxs are always row_ordered.
        # So, for a useful output for vertical arrangement, need to get the range in
        # nrow_ncol format but transpose this so that each row is the ax_idxs for a given cluster
        if arrangement == "horizontal":
            idxs = np.arange(n_axs).reshape(nrows_ncols)
        elif arrangement == "vertical":
            idxs = np.arange(n_axs).reshape(nrows_ncols).transpose()

        # For our purposes, we want to return this as a list so we can pop the elements
        return idxs.tolist()

    def get_ax_idxs_from_member_order(self, member_order: list) -> dict:
        """
        Returns a dictionary with the member as the key and ax_idx as value

        Args:
            member_order (list): list of lists defining member ordering

        Returns:
            dict: member-ax_idx pairs
        """
        ax_idxs = {}

        # Get the grid of ax_idxs using nrows_ncols and arrangement attributes
        full_grid_ax_idxs = self.get_full_grid_ax_idxs(
            self.nrows_ncols, self.arrangement
        )

        # Using the previously defined cluster_ax_idx_dims method, which defines
        # which rows/columns are spanned by each cluster, to merge lists of ax_idxs
        # into sublists for each cluster
        merged_cluster_ax_idxs = []
        for cluster_ax_idxs in self.cluster_ax_idx_dims.values():
            merged_idxs = [
                full_grid_ax_idxs[sublist_idx] for sublist_idx in cluster_ax_idxs
            ]
            # flatten new_list
            merged_idxs = [x for xs in merged_idxs for x in xs]
            merged_cluster_ax_idxs.append(merged_idxs)

        # For each member, find the nested list which contains that member
        # The index of that list within the list of lists is the cluster number
        # Pop the corresponding ax_idx from the full_grid_ax_idxs
        membs = [x for xs in member_order for x in xs]
        for memb in membs:
            for memb_list_idx, memb_list in enumerate(member_order):
                if memb in memb_list:
                    ax_idxs[memb] = merged_cluster_ax_idxs[memb_list_idx].pop(0)

        return ax_idxs

    def hide_axs_without_data(self, grid: plt.grid, ax_idxs: dict) -> None:
        # Get list of axs which have not been used for plotting
        all_ax_idxs = np.arange(self.nrows_ncols[0] * self.nrows_ncols[1])
        axs_with_data = ax_idxs.values()

        # Find axes which have not been used for plotting and turn frame off
        for ax_idx in all_ax_idxs:
            if ax_idx not in axs_with_data:
                grid[ax_idx].axis("off")

    def get_ax_idxs_of_first_membs(self):
        # Get the idxs of the first members from clustered grid

        # Using the previously defined cluster_ax_idx_dims method, which defines
        # which rows/columns are spanned by each cluster, to get the first ax idxs
        if self.arrangement == "vertical":
            # For vertical arrangement, this is just the first idx in each list
            first_ax_idxs = [
                idx_list[0] for idx_list in self.cluster_ax_idx_dims.values()
            ]
        elif self.arrangement == "horizontal":
            # For horizontal arrangement, this is just the first idx in each list
            # multiplied by number of axes in each rows
            first_ax_idxs = [
                idx_list[0] * self.nrows_ncols[1]
                for idx_list in self.cluster_ax_idx_dims.values()
            ]

        return first_ax_idxs

    def get_ax_idxs(self):
        return self.ax_idxs


class PostageStamps(BasePlot):
    def __init__(self, rasterized=False) -> None:
        super().__init__(rasterized)

    def set_arrangement_attrs(
        self, arrangement: str, max_axes: int, projection
    ) -> None:
        vtcl_keywords = ["vertical", "v"]
        hzntl_keywords = ["horizontal", "h"]

        if arrangement in hzntl_keywords:
            self.arrangement = "horizontal"
            self.title_pad_points = 8.0
        elif arrangement in vtcl_keywords:
            self.arrangement = "vertical"
            self.title_pad_points = 6.0
        else:
            msg = "arrangement arg must be either 'horizontal' or 'vertical'"
            raise Exception(msg)
        self.arranger = GridArranger(self.arrangement, max_axes, projection)

    def plot_precip(
        self,
        cube: iris.cube.Cube,
        overplot_cube: iris.cube.Cube = None,
        show_mask: bool = True,
        show_title: bool = True,
        title_info: str = None,
        show_member_age_offset: bool = False,
        member_order: list = None,
        arrangement: str = "horizontal",
        data_type: str = "data",
        member_title: str = "Member",
        max_axes_per_lead_dim: int = 6,
        accumulation_window: int = None,
        projection="default",
    ):
        return self.plot(
            cube=cube,
            overplot_cube=overplot_cube,
            show_mask=show_mask,
            show_title=show_title,
            title_info=title_info,
            show_member_age_offset=show_member_age_offset,
            member_order=member_order,
            arrangement=arrangement,
            data_type=data_type,
            member_title=member_title,
            max_axes_per_lead_dim=max_axes_per_lead_dim,
            accumulation_window=accumulation_window,
            projection=projection,
            plotting_func="make_precip_plot",
        )

    def plot_wind(
        self,
        cube: iris.cube.Cube,
        overplot_cube: iris.cube.Cube = None,
        show_mask: bool = False,
        show_title: bool = True,
        title_info: str = None,
        show_member_age_offset: bool = False,
        member_order: list = None,
        arrangement: str = "horizontal",
        data_type: str = "data",
        member_title: str = "Member",
        max_axes_per_lead_dim: int = 6,
        accumulation_window: int = None,
        projection="default",
    ):
        return self.plot(
            cube=cube,
            overplot_cube=overplot_cube,
            show_mask=show_mask,
            show_title=show_title,
            title_info=title_info,
            show_member_age_offset=show_member_age_offset,
            member_order=member_order,
            arrangement=arrangement,
            data_type=data_type,
            member_title=member_title,
            max_axes_per_lead_dim=max_axes_per_lead_dim,
            accumulation_window=accumulation_window,
            projection=projection,
            plotting_func="make_vector_plot",
        )

    def plot(
        self,
        cube: iris.cube.Cube,
        overplot_cube: iris.cube.Cube = None,
        show_mask: bool = True,
        show_title: bool = True,
        title_info: str = None,
        show_member_age_offset: bool = False,
        member_order: list = None,
        arrangement: str = "horizontal",
        data_type: str = "data",
        member_title: str = "Member",
        max_axes_per_lead_dim: int = 6,
        accumulation_window: int = None,
        projection="default",
        plotting_func="make_precip_plot",
        overplot_contour_labels: bool = False,
    ):
        # Get the plotting func to use based on input string
        plotting_func = getattr(self, plotting_func)

        # Set figure properties depending on the arrangement
        self.set_arrangement_attrs(arrangement, max_axes_per_lead_dim, projection)
        real_iter = cube.coord("realization").points

        # Create dict of member ages for use later in axes titles
        if show_member_age_offset:
            member_age_dict = get_muk_member_ages(real_iter)
        else:
            member_age_dict = None

        # If clustering info is passed to the function, check that here.
        if member_order is not None:
            # Check that member_order contains the same members as in the input cube
            membs_in_memb_order = np.sort([x for xs in member_order for x in xs])
            if np.any(membs_in_memb_order != real_iter):
                print(f"Cube Members: {real_iter}")
                print(f"Member Ordering: {member_order}")
                raise ValueError("Different members in member_order and cube")

            # Setup fig and grid using GridArranger, cube data and cluster info
            # GridArranger also returns the ax_idxs for each member
            fig, grid, ax_idxs = self.arranger.from_enforced_order_list(
                cube, member_order
            )

        else:
            # Setup fig and grid using GridArranger and cube data only
            # GridArranger also returns the ax_idxs for each member, though
            # for this configuration, this is just in numeric order
            fig, grid, ax_idxs = self.arranger.from_cube(cube)

        # Loop over all members, plot in the defined axis
        for real in real_iter:
            # Get the axis for member from the predetermined dict of idxs
            ax = grid[ax_idxs[real]]
            precip = cube.extract(iris.Constraint(realization=real))

            cf = plotting_func(
                precip,
                ax=ax,
                show_mask=show_mask,
                show_cbar=False,
                cmap_type=data_type,
            )

            if overplot_cube is not None:
                overplot_data = overplot_cube.extract(iris.Constraint(realization=real))
                self.overplot_contours(
                    overplot_data, ax, contour_labels=overplot_contour_labels
                )

            # When using GridSpec without all axes filled, the bounds for each plot default
            # to global for some reason. Here, take the x and y bounds and re-enforce
            # axis bounds (taking into account rotated latlon grid where necessary)
            self.enforce_ax_bounds_from_data(cube, ax)
            ax_title = f"{member_title} {real}"
            if member_age_dict is not None:
                ax_title += f" (T{member_age_dict[real]})"
            ax.set_title(ax_title)

        # Add cbar
        # Change cbar label based on data type
        if data_type == "features":
            cbar_label = "Features"
        elif data_type == "probabilities":
            cbar_label = "Probability of Features"
        else:
            cbar_label = "Accumulation (mm)"
        cbar = grid.cbar_axes[0].colorbar(cf, ticks=self.clevels, drawedges=True)
        cbar.set_label(cbar_label)

        # Make and show figure sup title if flagged
        # Note: the position of the supn title is *very* finnicky, can only get consistent-ish
        # behaviour when the fig size changes to match max number of cols or rows
        if show_title:
            fig.suptitle(
                self.make_title(cube, title_info, accumulation_window),
                y=self.get_y_pos_for_title(grid, pad_points=self.title_pad_points),
                verticalalignment="bottom",
            )

        # Set spine frame of each ax to black
        for ax in grid:
            for spine in ax.spines.values():
                spine.set_edgecolor("k")
                spine.set_linewidth(1)

        cbar.outline.set_color("k")
        cbar.outline.set_linewidth(1)
        cbar.dividers.set_color("k")
        cbar.dividers.set_linewidth(1)

        return fig, grid, cbar


# Very hacky code below to get the relative age of each member wihtout having to pass in dates
def consecutive(data, stepsize=1):
    # From https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def get_latest_memb(membs: list):
    if 34 in membs:
        if 1 not in membs:
            latest_memb = 34
            return latest_memb

    consecutive_split = consecutive(membs)
    if len(consecutive_split) > 2:
        raise Exception(
            f"Expected maximum of 2 consecutive groups, got {len(consecutive_split)}"
        )

    # Find the max member in the control group. This will be the latest member
    for arr in consecutive_split:
        if 0 in arr:
            control_group = arr
    latest_memb = np.max(control_group)

    return latest_memb


def get_muk_member_ages(members: list):
    muk_model_init_times = {
        0: [5, 11],
        1: 5,
        2: 5,
        3: 6,
        4: 6,
        5: 6,
        6: 7,
        7: 7,
        8: 7,
        9: 8,
        10: 8,
        11: 8,
        12: 9,
        13: 9,
        14: 9,
        15: 10,
        16: 10,
        17: 10,
        18: 11,
        19: 11,
        20: 0,
        21: 0,
        22: 0,
        23: 1,
        24: 1,
        25: 1,
        26: 2,
        27: 2,
        28: 2,
        29: 3,
        30: 3,
        31: 3,
        32: 4,
        33: 4,
        34: 4,
    }

    latest_memb = get_latest_memb(members)
    ctrl_memb = 0

    if latest_memb == ctrl_memb:
        if 1 in members:
            reference_time = muk_model_init_times[ctrl_memb][0]
        elif 18 in members:
            reference_time = muk_model_init_times[ctrl_memb][1]
        else:
            msg = "Could not determine ref time for control member since neither member 1 or 18 are in ensemble"
            raise Exception(msg)
    else:
        reference_time = muk_model_init_times[latest_memb]

    # Now, take the reference time and use this to find relative age of each member
    member_ages = {}

    # Need to handle control member seperately, since it appears twice
    if 1 in members:
        memb_time = muk_model_init_times[ctrl_memb][0]
    elif 18 in members:
        memb_time = muk_model_init_times[ctrl_memb][1]
    else:
        msg = "Could not determine ref time for control member since neither member 1 or 18 are in ensemble"
        raise Exception(msg)

    ctrl_memb_age = memb_time - reference_time
    if ctrl_memb_age > 0:
        ctrl_memb_age -= 12
    member_ages[ctrl_memb] = ctrl_memb_age

    for memb in members:
        if memb == 0:
            # Already handled this above
            continue
        memb_time = muk_model_init_times[memb] - reference_time
        if memb_time > 0:
            memb_time -= 12
        member_ages[memb] = memb_time

    return member_ages
