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


def make_melt_rate_diagnostic_fig():
    from SalinityDataset import SalinityDataset
    from utils import distance, get_netCDF_filepath, get_field_from_netcdf
    from constants import Omega, rho_0, figure_dir_path

    salinity_dataset = SalinityDataset(time_span='A5B2', avg_period='00', grid_size='04', field_type='an')

    climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2012)
    lons, lats, tau_x = get_field_from_netcdf(climo_filepath, 'tau_x')
    tau_y = get_field_from_netcdf(climo_filepath, 'tau_y')[2]
    alpha = get_field_from_netcdf(climo_filepath, 'alpha')[2]
    u_geo = get_field_from_netcdf(climo_filepath, 'geo_u')[2]

    salinity = np.zeros(tau_x.shape)
    dSdx = np.zeros(tau_x.shape)
    dSdy = np.zeros(tau_x.shape)
    zonal_melt = np.zeros(tau_x.shape)
    merid_melt = np.zeros(tau_x.shape)
    melt_rate = np.zeros(tau_x.shape)

    for i in range(len(lats)):
        for j in range(len(lons)):
            salinity[i][j] = salinity_dataset.salinity(lats[i], lons[j], 0)

    logger.info('Smoothing salinity field using Gaussian filter...')
    from scipy.ndimage import gaussian_filter
    salinity_smoothed = gaussian_filter(salinity, sigma=3)

    salinity[:] = salinity_smoothed[:]

    for i in range(1, len(lats)-1):
        lat = lats[i]
        f = 2 * Omega * np.sin(np.deg2rad(lat))  # Coriolis parameter [s^-1]

        progress_percent = 100 * i / (len(lats) - 2)
        logger.info('(melt_rate) lat = {:.2f}/{:.2f} ({:.1f}%)'.format(lat, lats[-1], progress_percent))

        dx = distance(lats[i-1], lons[0], lats[i+1], lons[0])
        dy = distance(lats[i], lons[0], lats[i], lons[2])

        for j in range(1, len(lons)-1):
            lon = lons[j]

            if not np.isnan(salinity[i][j-1]) and not np.isnan(salinity[i][j+1]):
                dSdx[i][j] = (salinity[i][j+1] - salinity[i][j-1]) / dx
                # dSdx = (salinity_ip1_j - salinity_im1_j) / dx
            else:
                dSdx[i][j] = np.nan

            if not np.isnan(salinity[i-1][j]) and not np.isnan(salinity[i+1][j]):
                dSdy[i][j] = (salinity[i+1][j] - salinity[i-1][j]) / dy
                # dSdy = (salinity_i_jp1 - salinity_i_jm1) / dy
            else:
                dSdy[i][j] = np.nan

            if not np.isnan(dSdx[i][j]) and not np.isnan(dSdy[i][j]):
                zonal_melt[i][j] = -(tau_x[i][j] / (rho_0 * f)) * (1 / salinity[i][j]) * dSdy[i][j]
                merid_melt[i][j] = (tau_y[i][j] / (rho_0 * f)) * (1 / salinity[i][j]) * dSdx[i][j]
                melt_rate[i][j] = zonal_melt[i][j] + merid_melt[i][j]
            else:
                zonal_melt[i][j] = np.nan
                merid_melt[i][j] = np.nan
                melt_rate[i][j] = np.nan

    plot_fields = {
        'salinity': salinity,
        'dSdx': dSdx,
        'dSdy': dSdy,
        'tau_x': tau_x,
        'u_geo': u_geo,
        'zonal_melt': zonal_melt,
        'merid_melt': merid_melt,
        'melt_rate': melt_rate
    }

    titles = {
        'salinity': 'salinity S',
        'dSdx': 'dS/dx',
        'dSdy': 'dS/dy',
        'tau_x': 'tau_x',
        'u_geo': 'u_geo',
        'zonal_melt': '-tau_x/(rho*f*S) * dS/dy',
        'merid_melt': 'tau_y/(rho*f*S) * dS/dx',
        'melt_rate': 'M-F (sum of left 2 plots)'
    }

    gs_coords = {
        'salinity': (0, 0),
        'dSdx': (0, 1),
        'dSdy': (0, 2),
        'tau_x': (0, 3),
        'u_geo': (1, 0),
        'zonal_melt': (1, 1),
        'merid_melt': (1, 2),
        'melt_rate': (1, 3)
    }

    cmaps = {
        'salinity': cmocean.cm.haline,
        'dSdx': 'seismic',
        'dSdy': 'seismic',
        'tau_x': 'seismic',
        'u_geo': 'seismic',
        'zonal_melt': 'seismic',
        'merid_melt': 'seismic',
        'melt_rate': 'seismic'
    }

    cmap_ranges = {
        'salinity': (33.75, 35),
        'dSdx': (-1, 1),
        'dSdy': (-1, 1),
        'tau_x': (-0.15, 0.15),
        'u_geo': (-5, 5),
        'zonal_melt': (-1, 1),
        'merid_melt': (-1, 1),
        'melt_rate': (-1, 1)
    }

    scale_factor = {
        'salinity': 1,
        'dSdx': 1e6,
        'dSdy': 1e6,
        'tau_x': 1,
        'u_geo': 100,
        'zonal_melt': 3600*24*365,
        'merid_melt': 3600*24*365,
        'melt_rate': 3600*24*365
    }

    colorbar_labels = {
        'salinity': 'psu',
        'dSdx': r'1/m (x10$^6$)',
        'dSdy': r'1/m (x10$^6$)',
        'tau_x': r'N/m$^2$',
        'u_geo': 'cm/s',
        'zonal_melt': 'm/year',
        'merid_melt': 'm/year',
        'melt_rate': 'm/year'
    }

    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    vector_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 4)
    matplotlib.rcParams.update({'font.size': 10})

    for var in plot_fields.keys():
        ax = plt.subplot(gs[gs_coords[var]], projection=ccrs.SouthPolarStereo())
        ax.add_feature(land_50m)
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        ax.set_title(titles[var])

        im = ax.pcolormesh(lons, lats, scale_factor[var] * plot_fields[var], transform=vector_crs,
                           cmap=cmaps[var], vmin=cmap_ranges[var][0], vmax=cmap_ranges[var][1])

        clb = fig.colorbar(im, ax=ax, extend='both', fraction=0.046, pad=0.04)
        clb.ax.set_title(colorbar_labels[var])

        ax.contour(lons, lats, np.ma.array(tau_x, mask=np.isnan(alpha)), levels=[0],
                   colors='green', linewidths=2, transform=vector_crs)
        ax.contour(lons, lats, np.ma.array(alpha, mask=np.isnan(alpha)), levels=[0.15],
                   colors='black', linewidths=2, transform=vector_crs)

    png_filepath = os.path.join(figure_dir_path, 'melt_rate_diagnostic_figure.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')


if __name__ == '__main__':
    # make_five_box_climo_fig()
    make_melt_rate_diagnostic_fig()
