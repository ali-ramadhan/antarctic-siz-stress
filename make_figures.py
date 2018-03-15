import os

import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec

import cartopy
import cartopy.crs as ccrs
import cmocean.cm

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig('logging.ini')
logger = logging.getLogger(__name__)


def make_five_box_climo_fig():
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from constants import figure_dir_path

    climo_filepath = get_netCDF_filepath(field_type='climo', year_start=1992, year_end=2015)
    JFM_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM', year_start=1992, year_end=2015)
    AMJ_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='AMJ', year_start=1992, year_end=2015)
    JAS_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS', year_start=1992, year_end=2015)
    OND_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='OND', year_start=1992, year_end=2015)

    lons, lats, climo_tau_x_field = get_field_from_netcdf(climo_filepath, 'tau_x')

    tau_x_fields = {
        'climo': climo_tau_x_field,
        'JFM': get_field_from_netcdf(JFM_filepath, 'tau_x')[2],
        'AMJ': get_field_from_netcdf(AMJ_filepath, 'tau_x')[2],
        'JAS': get_field_from_netcdf(JAS_filepath, 'tau_x')[2],
        'OND': get_field_from_netcdf(OND_filepath, 'tau_x')[2]
    }

    tau_y_fields = {
        'climo': get_field_from_netcdf(climo_filepath, 'tau_y')[2],
        'JFM': get_field_from_netcdf(JFM_filepath, 'tau_y')[2],
        'AMJ': get_field_from_netcdf(AMJ_filepath, 'tau_y')[2],
        'JAS': get_field_from_netcdf(JAS_filepath, 'tau_y')[2],
        'OND': get_field_from_netcdf(OND_filepath, 'tau_y')[2]
    }

    alpha_fields = {
        'climo': get_field_from_netcdf(climo_filepath, 'alpha')[2],
        'JFM': get_field_from_netcdf(JFM_filepath, 'alpha')[2],
        'AMJ': get_field_from_netcdf(AMJ_filepath, 'alpha')[2],
        'JAS': get_field_from_netcdf(JAS_filepath, 'alpha')[2],
        'OND': get_field_from_netcdf(OND_filepath, 'alpha')[2]
    }

    u_wind_fields = {
        'climo': get_field_from_netcdf(climo_filepath, 'wind_u')[2],
        'JFM': get_field_from_netcdf(JFM_filepath, 'wind_u')[2],
        'AMJ': get_field_from_netcdf(AMJ_filepath, 'wind_u')[2],
        'JAS': get_field_from_netcdf(JAS_filepath, 'wind_u')[2],
        'OND': get_field_from_netcdf(OND_filepath, 'wind_u')[2]
    }

    titles = {
        'climo': 'tau_y_climo',
        'JFM': 'tau_y_JFM',
        'AMJ': 'tau_y_AMJ',
        'JAS': 'tau_y_JAS',
        'OND': 'tau_y_OND'
    }

    gs_coords = {
        'climo': (slice(0, 2), slice(2, 4)),
        'JFM': (0, 0),
        'AMJ': (0, 1),
        'JAS': (1, 0),
        'OND': (1, 1)
    }

    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    vector_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 4)
    matplotlib.rcParams.update({'font.size': 10})

    for var in tau_x_fields.keys():
        ax = plt.subplot(gs[gs_coords[var]], projection=ccrs.SouthPolarStereo())
        ax.add_feature(land_50m)
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        ax.set_title(titles[var])

        im = ax.pcolormesh(lons, lats, tau_y_fields[var], transform=vector_crs, cmap=cm.get_cmap('seismic', 15),
                           vmin=-0.15, vmax=0.15)

        ax.quiver(lons[::10], lats[::10], figure_fields[var][::10, ::10], tau_y_fields[var][::10, ::10],
                  transform=vector_crs, units='width', width=0.002, scale=4)

        ax.contour(lons, lats, np.ma.array(figure_fields[var], mask=np.isnan(alpha_fields[var])), levels=[0],
                   colors='green', linewidths=2, transform=vector_crs)
        ax.contour(lons, lats, np.ma.array(u_wind_fields[var], mask=np.isnan(alpha_fields[var])), levels=[0],
                   colors='gold', linewidths=2, transform=vector_crs)
        ax.contour(lons, lats, np.ma.array(alpha_fields[var], mask=np.isnan(alpha_fields[var])), levels=[0.15],
                   colors='black', linewidths=2, transform=vector_crs)

        if var == 'climo':
            clb = fig.colorbar(im, ax=ax, extend='both', fraction=0.046, pad=0.04)
            clb.ax.set_title(r'N/m$^2$')

            zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
            zero_wind_line_patch = mpatches.Patch(color='gold', label='zero zonal wind line')
            ice_edge_patch = mpatches.Patch(color='black', label='15% ice edge')

            plt.legend(handles=[zero_stress_line_patch, zero_wind_line_patch, ice_edge_patch], loc='lower center',
                       bbox_to_anchor=(0, -0.05, 1, -0.05), ncol=3, mode='expand', borderaxespad=0)

    png_filepath = os.path.join(figure_dir_path, 'tau_y_seasonal_climo_figure.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    make_five_box_climo_fig()
