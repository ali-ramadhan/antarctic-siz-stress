import os
import datetime
import calendar

import netCDF4
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec

import cartopy
import cartopy.util
import cartopy.crs as ccrs
import cmocean.cm

from SurfaceStressDataWriter import SurfaceStressDataWriter
from constants import figure_dir_path
from utils import date_range, get_netCDF_filepath, get_field_from_netcdf


# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig('logging.ini')
logger = logging.getLogger(__name__)


def make_five_box_climo_fig(var):
    climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)
    JFM_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM', year_start=2005, year_end=2015)
    AMJ_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='AMJ', year_start=2005, year_end=2015)
    JAS_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS', year_start=2005, year_end=2015)
    OND_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='OND', year_start=2005, year_end=2015)

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

    var_fields = {
        'climo': get_field_from_netcdf(climo_filepath, var)[2],
        'JFM': get_field_from_netcdf(JFM_filepath, var)[2],
        'AMJ': get_field_from_netcdf(AMJ_filepath, var)[2],
        'JAS': get_field_from_netcdf(JAS_filepath, var)[2],
        'OND': get_field_from_netcdf(OND_filepath, var)[2]
    }

    alpha_fields = {
        'climo': get_field_from_netcdf(climo_filepath, 'alpha')[2],
        'JFM': get_field_from_netcdf(JFM_filepath, 'alpha')[2],
        'AMJ': get_field_from_netcdf(AMJ_filepath, 'alpha')[2],
        'JAS': get_field_from_netcdf(JAS_filepath, 'alpha')[2],
        'OND': get_field_from_netcdf(OND_filepath, 'alpha')[2]
    }

    titles = {
        'climo': var + '_climo',
        'JFM': var + '_JFM',
        'AMJ': var + '_AMJ',
        'JAS': var + '_JAS',
        'OND': var + '_OND'
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

    for season in tau_x_fields.keys():
        ax = plt.subplot(gs[gs_coords[season]], projection=ccrs.SouthPolarStereo())
        ax.add_feature(land_50m)
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        ax.set_title(titles[season])

        # im = ax.contourf(lons, lats, var_fields[season], 14, transform=vector_crs, cmap='seismic',
        #                  vmin=-0.15, vmax=0.15)
        im = ax.pcolormesh(lons, lats, var_fields[season], transform=vector_crs, cmap=cm.get_cmap('seismic', 15),
                           vmin=-0.1, vmax=0.1)

        if var == 'tau_x' or var == 'tau_y':
            ax.quiver(lons[::10], lats[::10], tau_x_fields[season][::10, ::10], tau_y_fields[season][::10, ::10],
                      transform=vector_crs, units='width', width=0.002, scale=4)

        ax.contour(lons, lats, np.ma.array(tau_x_fields[season], mask=np.isnan(alpha_fields[season])), levels=[0],
                   colors='green', linewidths=2, transform=vector_crs)
        ax.contour(lons, lats, np.ma.array(alpha_fields[season], mask=np.isnan(alpha_fields[season])), levels=[0.15],
                   colors='black', linewidths=2, transform=vector_crs)

        if season == 'climo':
            clb = fig.colorbar(im, ax=ax, extend='both', fraction=0.046, pad=0.04)
            clb.ax.set_title(r'm/s')

            zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
            ice_edge_patch = mpatches.Patch(color='black', label='15% ice edge')

            plt.legend(handles=[zero_stress_line_patch, ice_edge_patch], loc='lower center',
                       bbox_to_anchor=(0, -0.05, 1, -0.05), ncol=3, mode='expand', borderaxespad=0)

    png_filepath = os.path.join(figure_dir_path, var + '_seasonal_climo_figure.png')

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

    climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)
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

    # logger.info('Smoothing salinity field using Gaussian filter...')
    # from scipy.ndimage import gaussian_filter
    # salinity_smoothed = gaussian_filter(salinity, sigma=3)
    #
    # salinity[:] = salinity_smoothed[:]

    for i in range(1, len(lats)-1):
        lat = lats[i]
        f = 2 * Omega * np.sin(np.deg2rad(lat))  # Coriolis parameter [s^-1]

        progress_percent = 100 * i / (len(lats) - 2)
        logger.info('(melt_rate_field) lat = {:.2f}/{:.2f} ({:.1f}%)'.format(lat, lats[-1], progress_percent))

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
        'melt_rate_field': melt_rate
    }

    titles = {
        'salinity': 'salinity S',
        'dSdx': 'dS/dx',
        'dSdy': 'dS/dy',
        'tau_x': 'tau_x',
        'u_geo': 'u_geo',
        'zonal_melt': '-tau_x/(rho*f*S) * dS/dy',
        'merid_melt': 'tau_y/(rho*f*S) * dS/dx',
        'melt_rate_field': 'M-F (sum of left 2 plots)'
    }

    gs_coords = {
        'salinity': (0, 0),
        'dSdx': (0, 1),
        'dSdy': (0, 2),
        'tau_x': (0, 3),
        'u_geo': (1, 0),
        'zonal_melt': (1, 1),
        'merid_melt': (1, 2),
        'melt_rate_field': (1, 3)
    }

    cmaps = {
        'salinity': cmocean.cm.haline,
        'dSdx': 'seismic',
        'dSdy': 'seismic',
        'tau_x': 'seismic',
        'u_geo': 'seismic',
        'zonal_melt': 'seismic',
        'merid_melt': 'seismic',
        'melt_rate_field': 'seismic'
    }

    cmap_ranges = {
        'salinity': (33.75, 35),
        'dSdx': (-1, 1),
        'dSdy': (-1, 1),
        'tau_x': (-0.15, 0.15),
        'u_geo': (-5, 5),
        'zonal_melt': (-1, 1),
        'merid_melt': (-1, 1),
        'melt_rate_field': (-1, 1)
    }

    scale_factor = {
        'salinity': 1,
        'dSdx': 1e6,
        'dSdy': 1e6,
        'tau_x': 1,
        'u_geo': 100,
        'zonal_melt': 3600*24*365,
        'merid_melt': 3600*24*365,
        'melt_rate_field': 3600*24*365
    }

    colorbar_labels = {
        'salinity': 'psu',
        'dSdx': r'1/m (x10$^6$)',
        'dSdy': r'1/m (x10$^6$)',
        'tau_x': r'N/m$^2$',
        'u_geo': 'cm/s',
        'zonal_melt': 'm/year',
        'merid_melt': 'm/year',
        'melt_rate_field': 'm/year'
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

    png_filepath = os.path.join(figure_dir_path, 'melt_rate_diagnostic_figure2.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')


def make_zonal_and_contour_averaged_plots():
    """ plot zonally averaged div(alpha*u_ice*h_ice) and \Psi_{-\delta}*1/S*dS/dy. """
    import datetime

    import matplotlib.pyplot as plt
    import cartopy
    import cartopy.crs as ccrs

    from utils import get_netCDF_filepath, get_field_from_netcdf
    from utils import get_northward_zero_zonal_stress_line, get_northward_ice_edge, get_coast_coordinates
    from constants import figure_dir_path, D_e, Omega

    tau_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)

    lons, lats, ice_div_field = get_field_from_netcdf(tau_filepath, 'ice_flux_div')
    melt_rate_field = get_field_from_netcdf(tau_filepath, 'melt_rate')[2]
    psi_delta_field = get_field_from_netcdf(tau_filepath, 'psi_delta')[2]
    salinity_field = get_field_from_netcdf(tau_filepath, 'salinity')[2]
    temperature_field = get_field_from_netcdf(tau_filepath, 'temperature')[2]

    climo_tau_x_field = get_field_from_netcdf(tau_filepath, 'tau_x')[2]
    climo_alpha_field = get_field_from_netcdf(tau_filepath, 'alpha')[2]

    U_Ekman_field = D_e * get_field_from_netcdf(tau_filepath, 'Ekman_u')[2]
    V_Ekman_field = D_e * get_field_from_netcdf(tau_filepath, 'Ekman_v')[2]
    dSdx_field = get_field_from_netcdf(tau_filepath, 'dSdx')[2]
    dSdy_field = get_field_from_netcdf(tau_filepath, 'dSdy')[2]

    melt_rate_field_v2 = np.zeros(melt_rate_field.shape)

    for i in range(1, len(lats) - 1):
        lat = lats[i]
        f = 2 * Omega * np.sin(np.deg2rad(lat))  # Coriolis parameter [s^-1]

        progress_percent = 100 * i / (len(lats) - 2)
        logger.info('(M-F) lat = {:.2f}/-40 ({:.1f}%)'.format(lat, progress_percent))
        for j in range(len(lons)):
            U_Ekman = U_Ekman_field[i][j]
            V_Ekman = V_Ekman_field[i][j]
            S = salinity_field[i][j]
            dSdx = dSdx_field[i][j]
            dSdy = dSdy_field[i][j]

            if not np.isnan(U_Ekman) and not np.isnan(V_Ekman) and not np.isnan(dSdx) and not np.isnan(dSdy):
                melt_rate_field_v2[i][j] = (U_Ekman * dSdx + V_Ekman * dSdy) / S

    """ Plot some fields """
    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    # Compute a circle in axes coordinates, which we can use as a boundary for the map. We can pan/zoom as much as we
    # like - the boundary will be permanently circular.
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(1, 2)
    matplotlib.rcParams.update({'font.size': 10})

    crs_sps = ccrs.SouthPolarStereo()
    crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

    ax1 = plt.subplot(121, projection=crs_sps)

    ax1.add_feature(land_50m)
    ax1.add_feature(ice_50m)
    ax1.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax1.set_boundary(circle, transform=ax1.transAxes)

    gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl1.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl1.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im1 = ax1.pcolormesh(lons, lats, 1e6 * dSdx_field, transform=vector_crs, cmap=cmocean.cm.balance,
                         vmin=-1, vmax=1)

    ax1.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    ax1.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=2, transform=vector_crs)

    ax1.text(0.50, 1.05, r'dS/dx', fontsize=14, va='bottom', ha='center', rotation='horizontal',
             rotation_mode='anchor', transform=ax1.transAxes)

    ax1.text(0.49, 1.01, '0°', transform=ax1.transAxes)
    ax1.text(1.0105, 0.49, '90°E', transform=ax1.transAxes)
    ax1.text(0.47, -0.03, '180°', transform=ax1.transAxes)
    ax1.text(-0.09, 0.49, '90°W', transform=ax1.transAxes)
    ax1.text(0.855, 0.895, '45°E', rotation=45, transform=ax1.transAxes)
    ax1.text(0.85, 0.125, '135°E', rotation=-45, transform=ax1.transAxes)
    ax1.text(0.07, 0.90, '45°W', rotation=-45, transform=ax1.transAxes)
    ax1.text(0.06, 0.13, '135°W', rotation=45, transform=ax1.transAxes)

    clb = fig.colorbar(im1, ax=ax1, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'psu/m ($\times 10^6$)')

    ax2 = plt.subplot(122, projection=crs_sps)

    ax2.add_feature(land_50m)
    ax2.add_feature(ice_50m)
    ax2.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax2.set_boundary(circle, transform=ax2.transAxes)

    gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl2.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl2.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im2 = ax2.pcolormesh(lons, lats, 1e6 * dSdy_field, transform=vector_crs, cmap=cmocean.cm.balance,
                         vmin=-1, vmax=1)

    ax2.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    ax2.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=2, transform=vector_crs)

    ax2.text(0.50, 1.05, r'dS/dy', fontsize=14, va='bottom', ha='center', rotation='horizontal',
             rotation_mode='anchor', transform=ax2.transAxes)

    ax2.text(0.49, 1.01, '0°', transform=ax2.transAxes)
    ax2.text(1.0105, 0.49, '90°E', transform=ax2.transAxes)
    ax2.text(0.47, -0.03, '180°', transform=ax2.transAxes)
    ax2.text(-0.09, 0.49, '90°W', transform=ax2.transAxes)
    ax2.text(0.855, 0.895, '45°E', rotation=45, transform=ax2.transAxes)
    ax2.text(0.85, 0.125, '135°E', rotation=-45, transform=ax2.transAxes)
    ax2.text(0.07, 0.90, '45°W', rotation=-45, transform=ax2.transAxes)
    ax2.text(0.06, 0.13, '135°W', rotation=45, transform=ax2.transAxes)

    clb = fig.colorbar(im2, ax=ax2, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'psu/m ($\times 10^6$)')

    png_filepath = os.path.join(figure_dir_path, 'salinity_derivatives.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')
    plt.close()

    ###

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(1, 2)
    matplotlib.rcParams.update({'font.size': 10})

    crs_sps = ccrs.SouthPolarStereo()
    crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

    ax1 = plt.subplot(121, projection=crs_sps)

    ax1.add_feature(land_50m)
    ax1.add_feature(ice_50m)
    ax1.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax1.set_boundary(circle, transform=ax1.transAxes)

    gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl1.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl1.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im1 = ax1.pcolormesh(lons, lats, 3600*24*365*np.divide(np.multiply(U_Ekman_field, dSdx_field), salinity_field),
                         transform=vector_crs, cmap=cmocean.cm.balance, vmin=-1, vmax=1)

    ax1.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    ax1.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=2, transform=vector_crs)

    ax1.text(0.50, 1.05, r'$\mathcal{U}_{Ek}$ * 1/S * dS/dx', fontsize=14, va='bottom', ha='center', rotation='horizontal',
             rotation_mode='anchor', transform=ax1.transAxes)

    ax1.text(0.49, 1.01, '0°', transform=ax1.transAxes)
    ax1.text(1.0105, 0.49, '90°E', transform=ax1.transAxes)
    ax1.text(0.47, -0.03, '180°', transform=ax1.transAxes)
    ax1.text(-0.09, 0.49, '90°W', transform=ax1.transAxes)
    ax1.text(0.855, 0.895, '45°E', rotation=45, transform=ax1.transAxes)
    ax1.text(0.85, 0.125, '135°E', rotation=-45, transform=ax1.transAxes)
    ax1.text(0.07, 0.90, '45°W', rotation=-45, transform=ax1.transAxes)
    ax1.text(0.06, 0.13, '135°W', rotation=45, transform=ax1.transAxes)

    clb = fig.colorbar(im1, ax=ax1, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'm/year')

    ax2 = plt.subplot(122, projection=crs_sps)

    ax2.add_feature(land_50m)
    ax2.add_feature(ice_50m)
    ax2.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax2.set_boundary(circle, transform=ax2.transAxes)

    gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl2.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl2.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im2 = ax2.pcolormesh(lons, lats, 3600*24*365*np.divide(np.multiply(V_Ekman_field, dSdy_field), salinity_field),
                         transform=vector_crs, cmap=cmocean.cm.balance, vmin=-1, vmax=1)

    ax2.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    ax2.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=2, transform=vector_crs)

    ax2.text(0.50, 1.05, r'$\mathcal{V}_{Ek}$ * 1/S * dS/dy', fontsize=14, va='bottom', ha='center', rotation='horizontal',
             rotation_mode='anchor', transform=ax2.transAxes)

    ax2.text(0.49, 1.01, '0°', transform=ax2.transAxes)
    ax2.text(1.0105, 0.49, '90°E', transform=ax2.transAxes)
    ax2.text(0.47, -0.03, '180°', transform=ax2.transAxes)
    ax2.text(-0.09, 0.49, '90°W', transform=ax2.transAxes)
    ax2.text(0.855, 0.895, '45°E', rotation=45, transform=ax2.transAxes)
    ax2.text(0.85, 0.125, '135°E', rotation=-45, transform=ax2.transAxes)
    ax2.text(0.07, 0.90, '45°W', rotation=-45, transform=ax2.transAxes)
    ax2.text(0.06, 0.13, '135°W', rotation=45, transform=ax2.transAxes)

    clb = fig.colorbar(im2, ax=ax2, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'm/year')

    png_filepath = os.path.join(figure_dir_path, 'U_Ekman_times_salinity_derivatives.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')
    plt.close()

    ###

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(1, 2)
    matplotlib.rcParams.update({'font.size': 10})

    crs_sps = ccrs.SouthPolarStereo()
    crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

    ax1 = plt.subplot(121, projection=crs_sps)

    ax1.add_feature(land_50m)
    ax1.add_feature(ice_50m)
    ax1.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax1.set_boundary(circle, transform=ax1.transAxes)

    gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl1.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl1.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im1 = ax1.pcolormesh(lons, lats, 3600*24*365*melt_rate_field,
                         transform=vector_crs, cmap=cmocean.cm.balance, vmin=-1, vmax=1)

    ax1.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    ax1.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=2, transform=vector_crs)

    ax1.text(0.50, 1.05, r'$M-F v1', fontsize=14, va='bottom', ha='center', rotation='horizontal',
             rotation_mode='anchor', transform=ax1.transAxes)

    ax1.text(0.49, 1.01, '0°', transform=ax1.transAxes)
    ax1.text(1.0105, 0.49, '90°E', transform=ax1.transAxes)
    ax1.text(0.47, -0.03, '180°', transform=ax1.transAxes)
    ax1.text(-0.09, 0.49, '90°W', transform=ax1.transAxes)
    ax1.text(0.855, 0.895, '45°E', rotation=45, transform=ax1.transAxes)
    ax1.text(0.85, 0.125, '135°E', rotation=-45, transform=ax1.transAxes)
    ax1.text(0.07, 0.90, '45°W', rotation=-45, transform=ax1.transAxes)
    ax1.text(0.06, 0.13, '135°W', rotation=45, transform=ax1.transAxes)

    clb = fig.colorbar(im1, ax=ax1, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'm/year')

    ax2 = plt.subplot(122, projection=crs_sps)

    ax2.add_feature(land_50m)
    ax2.add_feature(ice_50m)
    ax2.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax2.set_boundary(circle, transform=ax2.transAxes)

    gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl2.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl2.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im2 = ax2.pcolormesh(lons, lats,
                         3600*24*365*np.add(np.divide(np.multiply(U_Ekman_field, dSdx_field), salinity_field),
                                            np.divide(np.multiply(V_Ekman_field, dSdy_field), salinity_field)),
                         transform=vector_crs, cmap=cmocean.cm.balance, vmin=-1, vmax=1)

    ax2.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    ax2.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=2, transform=vector_crs)

    ax2.text(0.50, 1.05, r'M-F v2', fontsize=14, va='bottom', ha='center', rotation='horizontal',
             rotation_mode='anchor', transform=ax2.transAxes)

    ax2.text(0.49, 1.01, '0°', transform=ax2.transAxes)
    ax2.text(1.0105, 0.49, '90°E', transform=ax2.transAxes)
    ax2.text(0.47, -0.03, '180°', transform=ax2.transAxes)
    ax2.text(-0.09, 0.49, '90°W', transform=ax2.transAxes)
    ax2.text(0.855, 0.895, '45°E', rotation=45, transform=ax2.transAxes)
    ax2.text(0.85, 0.125, '135°E', rotation=-45, transform=ax2.transAxes)
    ax2.text(0.07, 0.90, '45°W', rotation=-45, transform=ax2.transAxes)
    ax2.text(0.06, 0.13, '135°W', rotation=45, transform=ax2.transAxes)

    clb = fig.colorbar(im2, ax=ax2, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'm/year')

    png_filepath = os.path.join(figure_dir_path, 'M-F.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')
    plt.close()

    """ Calculate streamwise coordinates """
    contour_coordinate = np.empty((len(lats), len(lons)))
    contour_coordinate[:] = np.nan

    tau_x_lons, tau_x_lats = get_northward_zero_zonal_stress_line(tau_filepath)
    alpha_lons, alpha_lats = get_northward_ice_edge(tau_filepath)
    coast_lons, coast_lats = get_coast_coordinates(tau_filepath)

    for i in range(len(lons)):
        lon = lons[i]

        if alpha_lats[i] > tau_x_lats[i] > coast_lats[i]:
            lat_0 = coast_lats[i]
            lat_h = tau_x_lats[i]  # lat_h ~ lat_half ~ lat_1/2
            lat_1 = alpha_lats[i]

            for j in range(len(lats)):
                lat = lats[j]

                if lat < lat_0 or lat > lat_1:
                    contour_coordinate[j][i] = np.nan
                elif lat_0 <= lat <= lat_h:
                    contour_coordinate[j][i] = (lat - lat_0) / (2 * (lat_h - lat_0))
                elif lat_h <= lat <= lat_1:
                    contour_coordinate[j][i] = 0.5 + ((lat - lat_h) / (2 * (lat_1 - lat_h)))

    # fig = plt.figure()
    #
    # ax = plt.axes(projection=ccrs.SouthPolarStereo())
    # land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face',
    #                                                facecolor='dimgray', linewidth=0)
    # ax.add_feature(land_50m)
    # ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    # vector_crs = ccrs.PlateCarree()
    #
    # im = ax.pcolormesh(lons, lats, contour_coordinate, transform=vector_crs, cmap='PuOr')
    # fig.colorbar(im, ax=ax)
    # plt.title('\"green line coordinates\" (0=coast, 0.5=zero stress line, 1=ice edge)')
    # plt.show()
    # plt.close()

    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure(figsize=(16, 9))
    # ax = fig.add_subplot(111)
    # ax.plot(tau_x_lons, tau_x_lats, linewidth=1, label='tau_x')
    # ax.plot(alpha_lons, alpha_lats, linewidth=1, label='alpha')
    # ax.plot(coast_lons, coast_lats, linewidth=1, label='coast')
    # ax.legend()
    # plt.show()

    c_bins = np.linspace(0, 1, 41)[:-1]
    delta_c = c_bins[1] - c_bins[0]
    c_bins = c_bins + (delta_c / 2)

    ice_div_cavg = np.zeros(c_bins.shape)
    melt_rate_cavg = np.zeros(c_bins.shape)
    melt_rate_v2_cavg = np.zeros(c_bins.shape)
    psi_delta_cavg = np.zeros(c_bins.shape)

    for i in range(len(c_bins)):
        c = c_bins[i]
        c_low = c - (delta_c / 2)
        c_high = c + (delta_c / 2)

        c_in_range = np.logical_and(contour_coordinate > c_low, contour_coordinate < c_high)

        ice_div_cavg[i] = np.nanmean(ice_div_field[c_in_range])
        melt_rate_cavg[i] = np.nanmean(melt_rate_field[c_in_range])
        melt_rate_v2_cavg[i] = np.nanmean(melt_rate_field_v2[c_in_range])
        psi_delta_cavg[i] = np.nanmean(psi_delta_field[c_in_range])

    fig = plt.figure(figsize=(20, 6))

    ax = fig.add_subplot(131)

    ax.plot(lats, np.nanmean(ice_div_field, axis=1) * 24 * 3600 * 365, label='div(α*h*u_ice)')
    ax.plot(lats, np.nanmean(-melt_rate_field_v2, axis=1) * 24 * 3600 * 365, label=r'M-F = ($\mathcal{U}_{Ek} \cdot \nabla S)/S$')
    ax.set_xlim(-80, -50)
    ax.set_ylim(-5, 15)
    ax.set_xlabel('latitude', fontsize='large')
    ax.set_ylabel('m/year', fontsize='large')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(linestyle='--')
    ax.legend(fontsize='large')

    ax1 = fig.add_subplot(132)
    ax1.plot(lats, np.nanmean(salinity_field, axis=1), label='salinity [g/kg]', color='b')
    ax1.set_ylabel('salinity [g/kg]', color='b', fontsize='large')
    ax1.tick_params('y', colors='b')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax2 = ax1.twinx()
    ax2.plot(lats, np.nanmean(temperature_field, axis=1), label='temperature [°C]', color='r')
    ax2.set_ylabel('temperature [°C]', color='r', fontsize='large')
    ax2.tick_params('y', colors='r')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax1.set_xlim(-80, -50)
    ax1.set_xlabel('latitude', fontsize='large')
    ax1.grid(linestyle='--')

    ax = fig.add_subplot(133)

    ax.plot(lats, np.nanmean(psi_delta_field, axis=1), label='Ekman transport streamfunction ψ(-δ) [Sv]')
    ax.set_xlim(-80, -50)
    ax.set_xlabel('latitude', fontsize='large')
    ax.set_ylabel('Sverdrups', fontsize='large')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(linestyle='--')
    ax.legend()
    ax.legend(fontsize='large')

    png_filepath = os.path.join(figure_dir_path, 'zonal_average_M-F.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(20, 6))

    ax = fig.add_subplot(131)
    ax.plot(c_bins, ice_div_cavg * 24 * 3600 * 365, label='div(α*h*u_ice)')
    ax.plot(c_bins, -melt_rate_v2_cavg * 24 * 3600 * 365, label=r'M-F = ($\mathcal{U}_{Ek} \cdot \nabla S)/S$')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], minor=False)
    ax.set_xlabel('\"green line coordinates\"', fontsize='large')
    ax.set_ylabel('m/year', fontsize='large')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(linestyle='--')
    ax.legend(fontsize='large')

    ax1 = fig.add_subplot(132)
    ax1.plot(lats, np.nanmean(salinity_field, axis=1), label='salinity [g/kg]', color='b')
    ax1.set_ylabel('salinity [g/kg]', color='b', fontsize='large')
    ax1.tick_params('y', colors='b')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax2 = ax1.twinx()
    ax2.plot(lats, np.nanmean(temperature_field, axis=1), label='temperature [°C]', color='r')
    ax2.set_ylabel('temperature [°C]', color='r', fontsize='large')
    ax2.tick_params('y', colors='r')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax1.set_xlim(-80, -50)
    ax1.set_xlabel('latitude', fontsize='large')
    ax1.grid(linestyle='--')

    ax = fig.add_subplot(133)
    ax.plot(c_bins, psi_delta_cavg, label='Ekman transport streamfunction ψ(-δ) [Sv]')
    ax.set_xlabel('\"green line coordinates\"', fontsize='large')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], minor=False)
    ax.set_ylabel('Sverdrups', fontsize='large')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(linestyle='--')
    ax.legend(fontsize='large')

    png_filepath = os.path.join(figure_dir_path, 'streamwise_average_M-F.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')


def plot_meridional_salinity_profiles(time_span, grid_size, field_type, lon, split_depth):
    import os

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    import cmocean.cm

    from SalinityDataset import SalinityDataset
    from constants import output_dir_path

    image_filepaths = []

    for avg_period in ['00', '13', '14', '15', '16']:
        salinity_dataset = SalinityDataset(time_span, avg_period, grid_size, field_type)
        lats, depths, salinity_profile = salinity_dataset.meridional_salinity_profile(lon=lon, lat_min=-80, lat_max=0)

        time_span_str = time_span
        if time_span == 'A5B2':
            time_span_str = '2005-12'
        elif time_span == '95A4':
            time_span_str = '1995-2004'

        avg_period_str = avg_period
        if avg_period == '00':
            avg_period_str = 'mean'
        elif avg_period == '13':
            avg_period_str = 'JFM-seasonal'
        elif avg_period == '14':
            avg_period_str = 'AMJ-seasonal'
        elif avg_period == '15':
            avg_period_str = 'JAS-seasonal'
        elif avg_period == '16':
            avg_period_str = 'OND-seasonal'

        title_str = time_span_str + '-' + avg_period_str + '-lon=' + str(int(lon))

        fig, (ax1, ax2) = plt.subplots(2)

        levels = np.linspace(33.8, 36, 21)
        idx_split_depth = np.abs(depths - split_depth).argmin()

        im1 = ax1.contourf(lats, depths[:idx_split_depth], salinity_profile[:idx_split_depth, :],
                             cmap=cmocean.cm.haline, colors=None, vmin=33.8, vmax=36, levels=levels, extend='both')
        im2 = ax2.contourf(lats, depths[idx_split_depth:], salinity_profile[idx_split_depth:, :],
                             cmap=cmocean.cm.haline, colors=None, vmin=33.8, vmax=36, levels=levels, extend='both')

        # plt.xticks(list(plt.xticks()[0]) + [split_depth])

        idx_40S = np.nanargmin(np.abs(lats - -40))
        idx_80S = np.nanargmin(np.abs(lats - -80))
        idx_min_salinity = np.nanargmin(salinity_profile[0, idx_80S:idx_40S]) + idx_80S
        lat_min_salinity = lats[idx_min_salinity]
        ax1.plot([lat_min_salinity, lat_min_salinity], [0, 50], 'red', lw=2)
        ax1.text(lat_min_salinity + 0.5, 30, '{:.1f}°'.format(lat_min_salinity), fontsize=10, color='red')

        ax1.set_title(title_str, y=1.15, fontsize=12)

        fig.subplots_adjust(left=0.10, bottom=0.20, right=0.95, top=0.9, hspace=0)
        cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
        clb = fig.colorbar(im1, cax=cbar_ax, extend='both', orientation='horizontal')
        clb.ax.set_title('salinity (g/kg)', fontsize=12)

        ax1.set_ylim(0, depths[idx_split_depth - 1])
        ax2.set_ylim(depths[idx_split_depth], 5000)
        ax1.set_xlim(-75, 0)
        ax2.set_xlim(-75, 0)
        ax1.invert_yaxis()
        ax2.invert_yaxis()

        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax2.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        ax1.xaxis.tick_top()
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%d°'))

        # plt.subplot_tool()
        # plt.show()

        png_filename = 'salinity_profile_woa13_' + time_span + '_' + avg_period + '_' + grid_size + '_' + \
                       'lon' + str(int(lon))
        png_filepath = os.path.join(output_dir_path, 'salinity_profiles', png_filename + '.png')

        image_filepaths.append(png_filepath)

        dir = os.path.dirname(png_filepath)
        if not os.path.exists(dir):
            logger.info('Creating directory: {:s}'.format(dir))
            os.makedirs(dir)

        logger.info('Saving salinity profile: {:s}'.format(png_filepath))
        plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')

    from PIL import Image

    images = []
    for fp in image_filepaths:
        images.append(Image.open(fp, 'r'))

    widths, heights = zip(*(i.size for i in images))

    w = widths[0]
    h = heights[0]

    new_im = Image.new('RGB', (3*w, 2*h), color=(255, 255, 255))

    new_im.paste(images[1], (0, 0))
    new_im.paste(images[2], (w, 0))
    new_im.paste(images[3], (0, h))
    new_im.paste(images[4], (w, h))
    new_im.paste(images[0], (2*w, int(np.ceil(0.5*h))))

    all_filename = 'salinity_profile_woa13_' + time_span + '_all_' + grid_size + '_' + 'lon' + str(int(lon))
    all_filepath = os.path.join(output_dir_path, 'salinity_profiles', all_filename + '.png')

    logger.info('Saving combined salinity profiles: {:s}'.format(all_filepath))
    new_im.save(all_filepath)


def plot_meridional_temperature_profiles(time_span, grid_size, field_type, lon, split_depth):
    import os

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    import cmocean.cm

    from TemperatureDataset import TemperatureDataset
    from constants import output_dir_path

    image_filepaths = []

    for avg_period in ['00', '13', '14', '15', '16']:
        temperature_dataset = TemperatureDataset(time_span, avg_period, grid_size, field_type)
        lats, depths, temperature_profile = temperature_dataset.meridional_temperature_profile(lon=lon, lat_min=-80, lat_max=0)

        time_span_str = time_span
        if time_span == 'A5B2':
            time_span_str = '2005-12'
        elif time_span == '95A4':
            time_span_str = '1995-2004'

        avg_period_str = avg_period
        if avg_period == '00':
            avg_period_str = 'mean'
        elif avg_period == '13':
            avg_period_str = 'JFM-seasonal'
        elif avg_period == '14':
            avg_period_str = 'AMJ-seasonal'
        elif avg_period == '15':
            avg_period_str = 'JAS-seasonal'
        elif avg_period == '16':
            avg_period_str = 'OND-seasonal'

        title_str = time_span_str + '-' + avg_period_str + '-lon=' + str(int(lon))

        fig, (ax1, ax2) = plt.subplots(2)

        idx_split_depth = np.abs(depths - split_depth).argmin()

        im1 = ax1.pcolormesh(lats, depths[:idx_split_depth], temperature_profile[:idx_split_depth, :],
                             cmap=cmocean.cm.thermal, vmin=-2, vmax=2)
        im2 = ax2.pcolormesh(lats, depths[idx_split_depth:], temperature_profile[idx_split_depth:, :],
                             cmap=cmocean.cm.thermal, vmin=-2, vmax=2)

        # idx_40S = np.abs(lats - -40).argmin()
        # idx_60S = np.abs(lats - -60).argmin()
        # idx_max_temperature = temperature_profile[0, idx_60S:idx_40S].argmin() + idx_60S
        # lat_max_temperature = lats[idx_max_temperature]
        # ax1.plot([lat_max_temperature, lat_max_temperature], [0, 50], 'red', lw=2)
        # ax1.text(lat_max_temperature + 0.5, 30, '{:.1f}°'.format(lat_max_temperature), fontsize=10, color='red')

        ax1.set_title(title_str, y=1.15, fontsize=12)

        fig.subplots_adjust(left=0.10, bottom=0.20, right=0.95, top=0.9, hspace=0)
        cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
        clb = fig.colorbar(im1, cax=cbar_ax, extend='both', orientation='horizontal')
        clb.ax.set_title('temperature (g/kg)', fontsize=12)

        ax1.set_ylim(0, depths[idx_split_depth - 1])
        ax2.set_ylim(depths[idx_split_depth], 5000)
        ax1.set_xlim(-70, 0)
        ax2.set_xlim(-70, 0)
        ax1.invert_yaxis()
        ax2.invert_yaxis()

        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax2.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        ax1.xaxis.tick_top()
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%d°'))

        # plt.subplot_tool()
        # plt.show()

        png_filename = 'temperature_profile_woa13_' + time_span + '_' + avg_period + '_' + grid_size + '_' + \
                       'lon' + str(int(lon))
        png_filepath = os.path.join(output_dir_path, 'temperature_profiles', png_filename + '.png')

        image_filepaths.append(png_filepath)

        dir = os.path.dirname(png_filepath)
        if not os.path.exists(dir):
            logger.info('Creating directory: {:s}'.format(dir))
            os.makedirs(dir)

        logger.info('Saving temperature profile: {:s}'.format(png_filepath))
        plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')

    from PIL import Image

    images = []
    for fp in image_filepaths:
        images.append(Image.open(fp, 'r'))

    widths, heights = zip(*(i.size for i in images))

    w = widths[0]
    h = heights[0]

    new_im = Image.new('RGB', (3*w, 2*h), color=(255, 255, 255))

    new_im.paste(images[1], (0, 0))
    new_im.paste(images[2], (w, 0))
    new_im.paste(images[3], (0, h))
    new_im.paste(images[4], (w, h))
    new_im.paste(images[0], (2*w, int(np.ceil(0.5*h))))

    all_filename = 'temperature_profile_woa13_' + time_span + '_all_' + grid_size + '_' + 'lon' + str(int(lon))
    all_filepath = os.path.join(output_dir_path, 'temperature_profiles', all_filename + '.png')

    logger.info('Saving combined temperature profiles: {:s}'.format(all_filepath))
    new_im.save(all_filepath)


def plot_meridional_gamma_profiles(time_span, grid_size, field_type, lon, split_depth):
    import os

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    import cmocean.cm

    from NeutralDensityDataset import NeutralDensityDataset
    from constants import output_dir_path

    image_filepaths = []

    for avg_period in ['00']:
        gamma_dataset = NeutralDensityDataset(time_span, avg_period, grid_size, field_type, depth_levels=np.arange(100))
        lats, depths, gamma_profile = gamma_dataset.meridional_gamma_profile(lon=lon, lat_min=-80, lat_max=-40)

        depths = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                  125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500,
                  550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400,
                  1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000,
                  2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700,
                  3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300]
        depths = np.array(depths)

        time_span_str = time_span
        if time_span == 'A5B2':
            time_span_str = '2005-12'
        elif time_span == '95A4':
            time_span_str = '1995-2004'

        avg_period_str = avg_period
        if avg_period == '00':
            avg_period_str = 'mean'
        elif avg_period == '13':
            avg_period_str = 'JFM-seasonal'
        elif avg_period == '14':
            avg_period_str = 'AMJ-seasonal'
        elif avg_period == '15':
            avg_period_str = 'JAS-seasonal'
        elif avg_period == '16':
            avg_period_str = 'OND-seasonal'

        title_str = time_span_str + '-' + avg_period_str + '-lon=' + str(int(lon))

        fig, (ax1, ax2) = plt.subplots(2)

        levels = np.linspace(26.6, 28, 21)
        idx_split_depth = np.abs(depths - split_depth).argmin()

        im1 = ax1.contourf(lats, depths[:idx_split_depth], gamma_profile[:idx_split_depth, :],
                             cmap=cmocean.cm.dense, colors=None, vmin=26.6, vmax=28, levels=levels, extend='both')
        im2 = ax2.contourf(lats, depths[idx_split_depth:], gamma_profile[idx_split_depth:, :],
                             cmap=cmocean.cm.dense, colors=None, vmin=26.6, vmax=28, levels=levels, extend='both')

        # plt.xticks(list(plt.xticks()[0]) + [split_depth])

        idx_40S = np.nanargmin(np.abs(lats - -40))
        idx_80S = np.nanargmin(np.abs(lats - -80))
        idx_min_gamma = np.nanargmax(gamma_profile[0, idx_80S:idx_40S]) + idx_80S
        lat_min_gamma = lats[idx_min_gamma]
        ax1.plot([lat_min_gamma, lat_min_gamma], [0, 50], 'red', lw=2)
        ax1.text(lat_min_gamma + 0.5, 30, '{:.1f}°'.format(lat_min_gamma), fontsize=10, color='red')

        ax1.set_title(title_str, y=1.15, fontsize=12)

        fig.subplots_adjust(left=0.10, bottom=0.20, right=0.95, top=0.9, hspace=0)
        cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
        clb = fig.colorbar(im1, cax=cbar_ax, extend='both', orientation='horizontal')
        clb.ax.set_title('gamma_n (kg/m$^3$)', fontsize=12)

        ax1.set_ylim(0, depths[idx_split_depth - 1])
        ax2.set_ylim(depths[idx_split_depth], 5000)
        ax1.set_xlim(-75, -40)
        ax2.set_xlim(-75, -40)
        ax1.invert_yaxis()
        ax2.invert_yaxis()

        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax2.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        ax1.xaxis.tick_top()
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%d°'))

        # plt.subplot_tool()
        # plt.show()

        png_filename = 'gamma_profile_woa13_' + time_span + '_' + avg_period + '_' + grid_size + '_' + \
                       'lon' + str(int(lon))
        png_filepath = os.path.join(output_dir_path, 'gamma_profiles', png_filename + '.png')

        image_filepaths.append(png_filepath)

        dir = os.path.dirname(png_filepath)
        if not os.path.exists(dir):
            logger.info('Creating directory: {:s}'.format(dir))
            os.makedirs(dir)

        logger.info('Saving gamma profile: {:s}'.format(png_filepath))
        plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')

    # from PIL import Image
    #
    # images = []
    # for fp in image_filepaths:
    #     images.append(Image.open(fp, 'r'))
    #
    # widths, heights = zip(*(i.size for i in images))
    #
    # w = widths[0]
    # h = heights[0]
    #
    # new_im = Image.new('RGB', (3*w, 2*h), color=(255, 255, 255))
    #
    # new_im.paste(images[1], (0, 0))
    # new_im.paste(images[2], (w, 0))
    # new_im.paste(images[3], (0, h))
    # new_im.paste(images[4], (w, h))
    # new_im.paste(images[0], (2*w, int(np.ceil(0.5*h))))
    #
    # all_filename = 'gamma_profile_woa13_' + time_span + '_all_' + grid_size + '_' + 'lon' + str(int(lon))
    # all_filepath = os.path.join(output_dir_path, 'gamma_profiles', all_filename + '.png')
    #
    # logger.info('Saving combined gamma profiles: {:s}'.format(all_filepath))
    # new_im.save(all_filepath)


def look_at_neutral_density_contours(year_start, year_end):
    # Just looking at the surface neutral density for A5B2.
    for avg_period in ['00']:  # '['13', '14', '15', '16', '00']:
        dates = []
        for year in range(year_start, year_end+1):
            if avg_period == '00' or avg_period == '13':
                dates = dates + date_range(datetime.date(year, 1, 1), datetime.date(year, 3, 31))
            if avg_period == '00' or avg_period == '14':
                dates = dates + date_range(datetime.date(year, 4, 1), datetime.date(year, 6, 30))
            if avg_period == '00' or avg_period == '15':
                dates = dates + date_range(datetime.date(year, 7, 1), datetime.date(year, 9, 30))
            if avg_period == '00' or avg_period == '16':
                dates = dates + date_range(datetime.date(year, 10, 1), datetime.date(year, 12, 31))

        custom_label = ''
        if avg_period == '00':
            custom_label = str(year_start) + '-' + str(year_end) + '_climo'
        elif avg_period == '13':
            custom_label = str(year_start) + '-' + str(year_end) + '_seasonal_JFM'
        elif avg_period == '14':
            custom_label = str(year_start) + '-' + str(year_end) + '_seasonal_AMJ'
        elif avg_period == '15':
            custom_label = str(year_start) + '-' + str(year_end) + '_seasonal_JAS'
        elif avg_period == '16':
            custom_label = str(year_start) + '-' + str(year_end) + '_seasonal_OND'

        surface_stress_dataset = SurfaceStressDataWriter(None)
        surface_stress_dataset.date = dates[-1]
        surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
        surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=custom_label, avg_period=avg_period)


def make_tau_climo_fig():
    from os import path
    import netCDF4
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from constants import figure_dir_path, data_dir_path

    # climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)
    climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2011, year_end=2016)

    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
    #                                      year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM',
    #                                      year_start=2005, year_end=2015)

    lons, lats, climo_tau_x_field = get_field_from_netcdf(climo_filepath, 'tau_x')
    climo_tau_y_field = get_field_from_netcdf(climo_filepath, 'tau_y')[2]
    climo_alpha_field = get_field_from_netcdf(climo_filepath, 'alpha')[2]

    # climo_tau_x_field, lons_tau = cartopy.util.add_cyclic_point(climo_tau_x_field, coord=lons)
    # climo_tau_y_field, lons_tau = cartopy.util.add_cyclic_point(climo_tau_y_field, coord=lons)

    import astropy.convolution
    kernel = astropy.convolution.Box2DKernel(10)
    # kernel = astropy.convolution.Gaussian2DKernel(2)
    climo_tau_x_field = astropy.convolution.convolve(climo_tau_x_field, kernel, boundary='wrap')
    climo_tau_y_field = astropy.convolution.convolve(climo_tau_y_field, kernel, boundary='wrap')

    # Load neutral density map from Pellichero et al. (2017, 2018).
    gamma_filepath = path.join(data_dir_path, 'Climatology_MLD_v2017.nc')
    logger.info('Loading gamma dataset: {}'.format(gamma_filepath))
    gamma_dataset = netCDF4.Dataset(gamma_filepath)

    lats_gamma = np.array(gamma_dataset.variables['lat'])
    lons_gamma = np.array(gamma_dataset.variables['lon'])
    gamma_data = np.array(gamma_dataset.variables['ML_Gamma'])

    gamma_avg = None

    for month in [6, 7, 8]:
        gamma_monthly = gamma_data[month]

        if gamma_avg is None:
            gamma_avg = np.nan_to_num(gamma_monthly)
            gamma_monthly[~np.isnan(gamma_monthly)] = 1
            gamma_monthly[np.isnan(gamma_monthly)] = 0
            gamma_days = gamma_monthly
        else:
            gamma_avg = gamma_avg + np.nan_to_num(gamma_monthly)
            gamma_monthly[~np.isnan(gamma_monthly)] = 1
            gamma_monthly[np.isnan(gamma_monthly)] = 0
            gamma_days = gamma_days + gamma_monthly

    gamma_avg = np.divide(gamma_avg, gamma_days)

    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    # Compute a circle in axes coordinates, which we can use as a boundary for the map. We can pan/zoom as much as we
    # like - the boundary will be permanently circular.
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(1, 2)
    matplotlib.rcParams.update({'font.size': 10})

    """ Plot tau_x """
    crs_sps = ccrs.SouthPolarStereo()
    crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

    ax1 = plt.subplot(121, projection=crs_sps)

    ax1.add_feature(land_50m)
    ax1.add_feature(ice_50m)
    ax1.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax1.set_boundary(circle, transform=ax1.transAxes)

    gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl1.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl1.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im1 = ax1.contourf(lons, lats, climo_tau_x_field, transform=vector_crs, cmap=cmocean.cm.balance,
                       vmin=-0.20, vmax=0.20, levels=np.linspace(-0.20, 0.20, 16))

    Q1 = ax1.quiver(lons[::10], lats[::10],
                    np.ma.array(climo_tau_x_field, mask=(climo_alpha_field > 0.15))[::10, ::10],
                    np.ma.array(climo_tau_y_field, mask=(climo_alpha_field > 0.15))[::10, ::10],
                    pivot='middle', transform=vector_crs, units='width', width=0.002, scale=4)
    Q2 = ax1.quiver(lons[::10], lats[::10],
                    np.ma.array(climo_tau_x_field, mask=(climo_alpha_field < 0.15))[::10, ::10],
                    np.ma.array(climo_tau_y_field, mask=(climo_alpha_field < 0.15))[::10, ::10],
                    pivot='middle', transform=vector_crs, units='width', width=0.002, scale=2)

    plt.quiverkey(Q2, 0.33, 0.88, 0.1, r'0.1 N/m$^2$ (inside ice zone)', labelpos='E', coordinates='figure',
                  fontproperties={'size': 11}, transform=ax1.transAxes)
    plt.quiverkey(Q1, 0.33, 0.85, 0.1, r'0.1 N/m$^2$ (outside ice zone)', labelpos='E', coordinates='figure',
                  fontproperties={'size': 11}, transform=ax1.transAxes)

    ax1.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    ax1.contour(lons_gamma, lats_gamma, gamma_avg, levels=[27.6], colors='red', linewidths=2, transform=vector_crs)
    ax1.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=2, transform=vector_crs)

    ax1.text(0.49,  1.01,  '0°',   transform=ax1.transAxes)
    ax1.text(1.01,  0.49,  '90°E', transform=ax1.transAxes)
    ax1.text(0.47,  -0.03, '180°', transform=ax1.transAxes)
    ax1.text(-0.09, 0.49,  '90°W', transform=ax1.transAxes)
    ax1.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax1.transAxes)
    ax1.text(0.85,  0.125, '135°E', rotation=-45, transform=ax1.transAxes)
    ax1.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax1.transAxes)
    ax1.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax1.transAxes)

    ax1.text(0.50, 1.05, r'Zonal component $\tau_x$', fontsize=14, va='bottom', ha='center', rotation='horizontal',
             rotation_mode='anchor', transform=ax1.transAxes)

    clb = fig.colorbar(im1, ax=ax1, extend='both', fraction=0.046, pad=0.1, ticks=np.linspace(-0.20, 0.20, 6))
    clb.ax.set_title(r'N/m$^2$')

    zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
    ice_edge_patch = mpatches.Patch(color='black', label=r'15% ice edge')
    gamma_patch = mpatches.Patch(color='red', label=r'$\gamma^n$ = 27.6 kg/m$^3$ contour')
    plt.legend(handles=[zero_stress_line_patch, gamma_patch, ice_edge_patch], loc='lower center',
               bbox_to_anchor=(0, -0.15, 1, -0.15), ncol=1, mode='expand', borderaxespad=0, framealpha=0)

    # plt.suptitle(r'Figure 2: Ocean surface stress $\mathbf{\tau}$ observations, winter (JAS) mean', fontsize=16)
    plt.suptitle(r'Figure 4: Ocean surface stress $\mathbf{\tau}$ observations, with geostrophic current, '
                 'winter (JAS) mean', fontsize=16)
    # plt.suptitle(r'Ocean surface stress $\mathbf{\tau}$ observations, with geostrophic current, annual mean',
    #              fontsize=16)

    """ Plot tau_y """
    crs_sps = ccrs.SouthPolarStereo()
    crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

    ax2 = plt.subplot(122, projection=crs_sps)

    ax2.add_feature(land_50m)
    ax2.add_feature(ice_50m)
    ax2.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax2.set_boundary(circle, transform=ax2.transAxes)

    gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl2.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl2.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im2 = ax2.contourf(lons, lats, climo_tau_y_field, transform=vector_crs, cmap=cmocean.cm.balance,
                       vmin=-0.20, vmax=0.20, levels=np.linspace(-0.20, 0.20, 16))

    ax2.quiver(lons[::10], lats[::10],
               np.ma.array(climo_tau_x_field, mask=(climo_alpha_field > 0.15))[::10, ::10],
               np.ma.array(climo_tau_y_field, mask=(climo_alpha_field > 0.15))[::10, ::10],
               pivot='middle', transform=vector_crs, units='width', width=0.002, scale=4)
    ax2.quiver(lons[::10], lats[::10],
               np.ma.array(climo_tau_x_field, mask=(climo_alpha_field < 0.15))[::10, ::10],
               np.ma.array(climo_tau_y_field, mask=(climo_alpha_field < 0.15))[::10, ::10],
               pivot='middle', transform=vector_crs, units='width', width=0.002, scale=2)

    ax2.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=2, transform=vector_crs)
    ax2.contour(lons_gamma, lats_gamma, gamma_avg, levels=[27.6], colors='red', linewidths=2, transform=vector_crs)
    ax2.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)

    ax2.text(0.49,  1.01,  '0°',   transform=ax2.transAxes)
    ax2.text(1.01,  0.49,  '90°E', transform=ax2.transAxes)
    ax2.text(0.47,  -0.03, '180°', transform=ax2.transAxes)
    ax2.text(-0.09, 0.49,  '90°W', transform=ax2.transAxes)
    ax2.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax2.transAxes)
    ax2.text(0.85,  0.125, '135°E', rotation=-45, transform=ax2.transAxes)
    ax2.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax2.transAxes)
    ax2.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax2.transAxes)

    ax2.text(0.50, 1.05, r'Meridional component $\tau_y$', fontsize=14, va='bottom', ha='center',
             rotation='horizontal', rotation_mode='anchor', transform=ax2.transAxes)

    clb = fig.colorbar(im2, ax=ax2, extend='both', fraction=0.046, pad=0.1, ticks=np.linspace(-0.20, 0.20, 6))
    clb.ax.set_title(r'N/m$^2$')

    png_filepath = os.path.join(figure_dir_path, 'tau_climo_figure.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False)


def make_uEk_climo_fig():
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from constants import figure_dir_path, D_e

    # climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
    #                                      year_start=2005, year_end=2015)
    climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
                                         year_start=2011, year_end=2016)

    # feb_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 2, 1),
    #                                          year_start=2005, year_end=2015)
    # sep_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 9, 1),
    #                                          year_start=2005, year_end=2015)

    lons, lats, climo_tau_x_field = get_field_from_netcdf(climo_filepath, 'tau_x')
    climo_tau_y_field = get_field_from_netcdf(climo_filepath, 'tau_y')[2]
    climo_alpha_field = get_field_from_netcdf(climo_filepath, 'alpha')[2]
    climo_u_Ekman_field = get_field_from_netcdf(climo_filepath, 'Ekman_u')[2]
    climo_v_Ekman_field = get_field_from_netcdf(climo_filepath, 'Ekman_v')[2]

    # feb_climo_alpha_field = get_field_from_netcdf(feb_climo_filepath, 'alpha')[2]
    # sep_climo_alpha_field = get_field_from_netcdf(sep_climo_filepath, 'alpha')[2]

    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(1, 2)
    matplotlib.rcParams.update({'font.size': 10})

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    """ Plot u_Ekman """
    crs_sps = ccrs.SouthPolarStereo()
    crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

    ax1 = plt.subplot(121, projection=crs_sps)

    ax1.set_boundary(circle, transform=ax1.transAxes)
    ax1.add_feature(land_50m)
    ax1.add_feature(ice_50m)
    ax1.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl1.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl1.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im1 = ax1.contourf(lons, lats, D_e * climo_u_Ekman_field, transform=vector_crs, cmap='BrBG',
                       vmin=-2, vmax=2, levels=np.linspace(-2.0, 2.0, 16))

    Q1 = ax1.quiver(lons[::10], lats[::10],
                    np.ma.array(D_e * climo_u_Ekman_field, mask=(climo_alpha_field > 0.15))[::10, ::10],
                    np.ma.array(D_e * climo_v_Ekman_field, mask=(climo_alpha_field > 0.15))[::10, ::10],
                    pivot='middle', transform=vector_crs, units='width', width=0.002, scale=30)
    Q2 = ax1.quiver(lons[::10], lats[::10],
                    np.ma.array(D_e * climo_u_Ekman_field, mask=(climo_alpha_field < 0.15))[::10, ::10],
                    np.ma.array(D_e * climo_v_Ekman_field, mask=(climo_alpha_field < 0.15))[::10, ::10],
                    pivot='middle', transform=vector_crs, units='width', width=0.002, scale=15)

    plt.quiverkey(Q2, 0.33, 0.88, 1, r'1 m$^2$/s (inside ice zone)', labelpos='E', coordinates='figure',
                  fontproperties={'size': 11}, transform=ax1.transAxes)
    plt.quiverkey(Q1, 0.33, 0.85, 1, r'1 m$^2$/s (outside ice zone)', labelpos='E', coordinates='figure',
                  fontproperties={'size': 11}, transform=ax1.transAxes)

    ax1.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=2, transform=vector_crs)
    ax1.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    # ax1.contour(lons, lats, np.ma.array(feb_climo_alpha_field, mask=np.isnan(feb_climo_alpha_field)),
    #             levels=[0.15], colors='black', linewidths=1, transform=vector_crs)
    # ax1.contour(lons, lats, np.ma.array(sep_climo_alpha_field, mask=np.isnan(sep_climo_alpha_field)),
    #             levels=[0.15], colors='black', linewidths=1, transform=vector_crs)

    ax1.text(0.49,  1.01,  '0°',   transform=ax1.transAxes)
    ax1.text(1.01,  0.49,  '90°E', transform=ax1.transAxes)
    ax1.text(0.47,  -0.03, '180°', transform=ax1.transAxes)
    ax1.text(-0.09, 0.49,  '90°W', transform=ax1.transAxes)
    ax1.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax1.transAxes)
    ax1.text(0.85,  0.125, '135°E', rotation=-45, transform=ax1.transAxes)
    ax1.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax1.transAxes)
    ax1.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax1.transAxes)

    ax1.text(0.50, 1.05, r'Zonal component $\mathcal{U}_{Ek}$', fontsize=14, va='bottom', ha='center', rotation='horizontal',
             rotation_mode='anchor', transform=ax1.transAxes)

    clb = fig.colorbar(im1, ax=ax1, extend='both', fraction=0.046, pad=0.1, ticks=np.linspace(-2, 2, 6))
    clb.ax.set_title(r'm$^2$/s')

    zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
    ice_edge_patch = mpatches.Patch(color='black', label=r'15% ice edge')
    plt.legend(handles=[zero_stress_line_patch, ice_edge_patch], loc='lower center',
               bbox_to_anchor=(0, -0.1, 1, -0.1), ncol=1, mode='expand', borderaxespad=0, framealpha=0)

    plt.suptitle(r'Figure 5: Ekman volume transport $\mathcal{U}_{Ek}$ observations, winter (JAS) mean', fontsize=16)

    """ Plot v_Ekman """
    crs_sps = ccrs.SouthPolarStereo()
    crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

    ax2 = plt.subplot(122, projection=crs_sps)

    gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl2.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl2.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    ax2.add_feature(land_50m)
    ax2.add_feature(ice_50m)
    ax2.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax2.set_boundary(circle, transform=ax2.transAxes)

    im2 = ax2.contourf(lons, lats, D_e * climo_v_Ekman_field, transform=vector_crs, cmap='BrBG',
                       vmin=-2, vmax=2, levels=np.linspace(-2.0, 2.0, 16))

    ax2.quiver(lons[::10], lats[::10],
               np.ma.array(D_e * climo_u_Ekman_field, mask=(climo_alpha_field > 0.15))[::10, ::10],
               np.ma.array(D_e * climo_v_Ekman_field, mask=(climo_alpha_field > 0.15))[::10, ::10],
               pivot='middle', transform=vector_crs, units='width', width=0.002, scale=30)
    ax2.quiver(lons[::10], lats[::10],
               np.ma.array(D_e * climo_u_Ekman_field, mask=(climo_alpha_field < 0.15))[::10, ::10],
               np.ma.array(D_e * climo_v_Ekman_field, mask=(climo_alpha_field < 0.15))[::10, ::10],
               pivot='middle', transform=vector_crs, units='width', width=0.002, scale=15)

    ax2.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=2, transform=vector_crs)
    ax2.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)

    ax2.text(0.49,  1.01,  '0°',   transform=ax2.transAxes)
    ax2.text(1.01,  0.49,  '90°E', transform=ax2.transAxes)
    ax2.text(0.47,  -0.03, '180°', transform=ax2.transAxes)
    ax2.text(-0.09, 0.49,  '90°W', transform=ax2.transAxes)
    ax2.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax2.transAxes)
    ax2.text(0.85,  0.125, '135°E', rotation=-45, transform=ax2.transAxes)
    ax2.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax2.transAxes)
    ax2.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax2.transAxes)

    ax2.text(0.50, 1.05, r'Meridional component $\mathcal{V}_{Ek}$', fontsize=14, va='bottom', ha='center',
             rotation='horizontal', rotation_mode='anchor', transform=ax2.transAxes)

    clb = fig.colorbar(im2, ax=ax2, extend='both', fraction=0.046, pad=0.1, ticks=np.linspace(-2, 2, 6))
    clb.ax.set_title(r'm$^2$/s')

    png_filepath = os.path.join(figure_dir_path, 'Ekman_transport_climo_figure.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False)


def make_curl_climo_fig():
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from constants import figure_dir_path

    # climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)
    climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
                                         year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM',
    #                                      year_start=2005, year_end=2015)

    feb_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 2, 1),
                                             year_start=2005, year_end=2015)
    sep_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 9, 1),
                                             year_start=2005, year_end=2015)

    lons, lats, climo_tau_x_field = get_field_from_netcdf(climo_filepath, 'tau_x')
    climo_tau_y_field = get_field_from_netcdf(climo_filepath, 'tau_y')[2]
    climo_alpha_field = get_field_from_netcdf(climo_filepath, 'alpha')[2]
    climo_curl_field = get_field_from_netcdf(climo_filepath, 'curl_stress')[2]
    climo_w_Ekman_field = get_field_from_netcdf(climo_filepath, 'Ekman_w')[2]

    feb_climo_alpha_field = get_field_from_netcdf(feb_climo_filepath, 'alpha')[2]
    sep_climo_alpha_field = get_field_from_netcdf(sep_climo_filepath, 'alpha')[2]

    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(1, 2)
    matplotlib.rcParams.update({'font.size': 10})

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    """ Plot wind stress curl """
    crs_sps = ccrs.SouthPolarStereo()
    crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

    ax1 = plt.subplot(121, projection=crs_sps)

    ax1.set_boundary(circle, transform=ax1.transAxes)
    ax1.add_feature(land_50m)
    ax1.add_feature(ice_50m)
    ax1.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl1.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl1.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im1 = ax1.contourf(lons, lats, climo_curl_field * 1e7, transform=vector_crs, cmap=cmocean.cm.curl,
                       vmin=-5, vmax=5, levels=np.linspace(-5, 5, 16), extend='both')

    ax1.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=2, transform=vector_crs)
    ax1.contour(lons, lats, np.ma.array(feb_climo_alpha_field, mask=np.isnan(feb_climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    ax1.contour(lons, lats, np.ma.array(sep_climo_alpha_field, mask=np.isnan(sep_climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)

    ax1.text(0.49,  1.01,  '0°',   transform=ax1.transAxes)
    ax1.text(1.01,  0.49,  '90°E', transform=ax1.transAxes)
    ax1.text(0.47,  -0.03, '180°', transform=ax1.transAxes)
    ax1.text(-0.09, 0.49,  '90°W', transform=ax1.transAxes)
    ax1.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax1.transAxes)
    ax1.text(0.85,  0.125, '135°E', rotation=-45, transform=ax1.transAxes)
    ax1.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax1.transAxes)
    ax1.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax1.transAxes)

    ax1.text(0.50, 1.05, r'Surface stress curl $\nabla \cdot (\mathbf{\tau} / \rho f)$', fontsize=14, va='bottom',
             ha='center', rotation='horizontal', rotation_mode='anchor', transform=ax1.transAxes)

    clb = fig.colorbar(im1, ax=ax1, extend='both', fraction=0.046, pad=0.1, ticks=np.linspace(-5, 5, 6))
    clb.ax.set_title(r'$10^{-7}$ N/m$^3$')

    zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
    ice_edge_patch = mpatches.Patch(color='black', label=r'15% ice edge')
    plt.legend(handles=[zero_stress_line_patch, ice_edge_patch], loc='lower center',
               bbox_to_anchor=(0, -0.1, 1, -0.1), ncol=1, mode='expand', borderaxespad=0, framealpha=0)

    plt.suptitle(r'Figure 6: Surface stress curl and Ekman pumping observations, winter (JAS) mean', fontsize=16)

    """ Plot Ekman pumping """
    crs_sps = ccrs.SouthPolarStereo()
    crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

    ax2 = plt.subplot(122, projection=crs_sps)

    ax2.set_boundary(circle, transform=ax2.transAxes)
    ax2.add_feature(land_50m)
    ax2.add_feature(ice_50m)
    ax2.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl2.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl2.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im2 = ax2.contourf(lons, lats, climo_w_Ekman_field * 365*24*3600, transform=vector_crs, cmap=cmocean.cm.balance,
                       vmin=-100, vmax=100, levels=np.linspace(-100, 100, 16), extend='both')

    ax2.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=2, transform=vector_crs)
    ax2.contour(lons, lats, np.ma.array(feb_climo_alpha_field, mask=np.isnan(feb_climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    ax2.contour(lons, lats, np.ma.array(sep_climo_alpha_field, mask=np.isnan(sep_climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)

    ax2.text(0.49,  1.01,  '0°',   transform=ax2.transAxes)
    ax2.text(1.01,  0.49,  '90°E', transform=ax2.transAxes)
    ax2.text(0.47,  -0.03, '180°', transform=ax2.transAxes)
    ax2.text(-0.09, 0.49,  '90°W', transform=ax2.transAxes)
    ax2.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax2.transAxes)
    ax2.text(0.85,  0.125, '135°E', rotation=-45, transform=ax2.transAxes)
    ax2.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax2.transAxes)
    ax2.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax2.transAxes)

    ax2.text(0.50, 1.05, r'Ekman pumping $w_{Ek}$', fontsize=14, va='bottom',
             ha='center', rotation='horizontal', rotation_mode='anchor', transform=ax2.transAxes)

    clb = fig.colorbar(im2, ax=ax2, extend='both', fraction=0.046, pad=0.1, ticks=np.linspace(-100, 100, 6))
    clb.ax.set_title(r'm/year')

    png_filepath = os.path.join(figure_dir_path, 'curl_climo.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False)  # , bbox_inches='tight')


def make_figure1():
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from constants import figure_dir_path

    # climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)
    climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
                                         year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM',
    #                                      year_start=2005, year_end=2015)

    feb_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 2, 1),
                                             year_start=2005, year_end=2015)
    sep_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 9, 1),
                                             year_start=2005, year_end=2015)

    JAS_climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
                                             year_start=2005, year_end=2015)
    JFM_climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM',
                                             year_start=2005, year_end=2015)

    lons, lats, climo_tau_x_field = get_field_from_netcdf(climo_filepath, 'tau_x')
    climo_tau_y_field = get_field_from_netcdf(climo_filepath, 'tau_y')[2]
    climo_alpha_field = get_field_from_netcdf(climo_filepath, 'alpha')[2]
    climo_u_wind_field = get_field_from_netcdf(climo_filepath, 'wind_u')[2]
    climo_v_wind_field = get_field_from_netcdf(climo_filepath, 'wind_v')[2]
    climo_u_ice_field = get_field_from_netcdf(climo_filepath, 'ice_u')[2]
    climo_v_ice_field = get_field_from_netcdf(climo_filepath, 'ice_v')[2]

    feb_climo_alpha_field = get_field_from_netcdf(feb_climo_filepath, 'alpha')[2]
    sep_climo_alpha_field = get_field_from_netcdf(sep_climo_filepath, 'alpha')[2]

    JAS_climo_alpha_field = get_field_from_netcdf(JAS_climo_filepath, 'alpha')[2]
    JFM_climo_alpha_field = get_field_from_netcdf(JFM_climo_filepath, 'alpha')[2]

    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 9))
    matplotlib.rcParams.update({'font.size': 10})

    crs_sps = ccrs.SouthPolarStereo()
    crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363
    ax = plt.subplot(111, projection=crs_sps)

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.add_feature(land_50m)
    ax.add_feature(ice_50m)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    im = ax.contourf(lons, lats, climo_alpha_field, transform=vector_crs, cmap='Blues', vmin=-0, vmax=1,
                     levels=np.linspace(0, 1, 11))

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    Q1 = ax.quiver(lons[::10], lats[::10],
                   np.ma.array(climo_u_wind_field, mask=(sep_climo_alpha_field > 0.15))[::10, ::10],
                   np.ma.array(climo_v_wind_field, mask=(sep_climo_alpha_field > 0.15))[::10, ::10],
                   pivot='middle', color='green', transform=vector_crs, units='width', width=0.002, scale=250)
    Q2 = ax.quiver(lons[::10], lats[::10],
                   np.ma.array(climo_u_wind_field, mask=(sep_climo_alpha_field < 0.15))[::10, ::10],
                   np.ma.array(climo_v_wind_field, mask=(sep_climo_alpha_field < 0.15))[::10, ::10],
                   pivot='middle', color='yellow', transform=vector_crs, units='width', width=0.002, scale=150)

    Q3 = ax.quiver(lons[::10], lats[::10],
                   np.ma.array(climo_u_ice_field, mask=(sep_climo_alpha_field < 0.15))[::10, ::10],
                   np.ma.array(climo_v_ice_field, mask=(sep_climo_alpha_field < 0.15))[::10, ::10],
                   pivot='middle', color='#E8000B', transform=vector_crs, units='width', width=0.002, scale=2)

    plt.quiverkey(Q2, 0.70, 0.86, 10, r'10 m/s (wind, inside ice zone)', labelpos='E', coordinates='figure',
                  fontproperties={'size': 11}, transform=ax.transAxes)
    plt.quiverkey(Q1, 0.70, 0.83, 10, r'10 m/s (wind, outside ice zone)', labelpos='E', coordinates='figure',
                  fontproperties={'size': 11}, transform=ax.transAxes)
    plt.quiverkey(Q3, 0.77, 0.80, 0.05, r'5 cm/s (ice drift)', labelpos='E', coordinates='figure',
                  fontproperties={'size': 11}, transform=ax.transAxes)

    # ax.contour(lons, lats, np.ma.array(feb_climo_alpha_field, mask=np.isnan(feb_climo_alpha_field)),
    #            levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    # ax.contour(lons, lats, np.ma.array(sep_climo_alpha_field, mask=np.isnan(sep_climo_alpha_field)),
    #            levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    ax.contour(lons, lats, np.ma.array(JAS_climo_alpha_field, mask=np.isnan(JAS_climo_alpha_field)),
               levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    ax.contour(lons, lats, np.ma.array(JFM_climo_alpha_field, mask=np.isnan(JFM_climo_alpha_field)),
               levels=[0.15], colors='black', linewidths=2, transform=vector_crs)

    ax.text(0.49,  1.01,  '0°',   transform=ax.transAxes)
    ax.text(1.01,  0.49,  '90°E', transform=ax.transAxes)
    ax.text(0.47,  -0.03, '180°', transform=ax.transAxes)
    ax.text(-0.09, 0.49,  '90°W', transform=ax.transAxes)
    ax.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax.transAxes)
    ax.text(0.85,  0.125, '135°E', rotation=-45, transform=ax.transAxes)
    ax.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax.transAxes)
    ax.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax.transAxes)

    clb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.1)
    clb.ax.set_title(r'sea ice fraction $\alpha$')

    # plt.gcf().text(0.5, 0.95, 'Figure 1(a): key observations, annual mean', fontsize=14)
    plt.gcf().text(0.5, 0.95, 'Figure 1(a): key observations, winter (JAS) mean', fontsize=14)
    # plt.gcf().text(0.5, 0.95, 'Figure 1(a): key observations, summer (JFM) mean', fontsize=14)

    # zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
    # gamma_contour_patch = mpatches.Patch(color='black', label=r'$\gamma$ = 27.6 kg/m$^3$ contour')
    # plt.legend(handles=[zero_stress_line_patch, gamma_contour_patch], loc='lower center',
    #            bbox_to_anchor=(0, -0.05, 1, -0.05), ncol=1, mode='expand', borderaxespad=0, framealpha=0)

    png_filepath = os.path.join(figure_dir_path, 'figure1.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')


def make_ugeo_uice_figure():
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from constants import figure_dir_path

    import constants
    constants.output_dir_path = 'E:\\output\\'

    # climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)
    climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
                                         year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM',
    #                                      year_start=2005, year_end=2015)

    lons, lats, climo_tau_x_field = get_field_from_netcdf(climo_filepath, 'tau_x')
    climo_tau_y_field = get_field_from_netcdf(climo_filepath, 'tau_y')[2]

    constants.output_dir_path = 'C:\\Users\\Ali\\Downloads\\output\\'

    climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2011, year_end=2016)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
    #                                      year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM',
    #                                      year_start=2005, year_end=2015)

    feb_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 2, 1),
                                             year_start=2005, year_end=2015)
    sep_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 9, 1),
                                             year_start=2005, year_end=2015)

    climo_alpha_field = get_field_from_netcdf(climo_filepath, 'alpha')[2]
    climo_u_ice_field = get_field_from_netcdf(climo_filepath, 'ice_u')[2]
    climo_v_ice_field = get_field_from_netcdf(climo_filepath, 'ice_v')[2]
    climo_u_geo_field = get_field_from_netcdf(climo_filepath, 'geo_u')[2]
    climo_v_geo_field = get_field_from_netcdf(climo_filepath, 'geo_v')[2]

    feb_climo_alpha_field = get_field_from_netcdf(feb_climo_filepath, 'alpha')[2]
    sep_climo_alpha_field = get_field_from_netcdf(sep_climo_filepath, 'alpha')[2]

    climo_u_geo_field, lons_geo = cartopy.util.add_cyclic_point(climo_u_geo_field, coord=lons)
    climo_v_geo_field, lons_geo = cartopy.util.add_cyclic_point(climo_v_geo_field, coord=lons)

    # Smooth out the u_geo fields
    import astropy.convolution
    kernel = astropy.convolution.Gaussian2DKernel(2)
    climo_u_geo_field = astropy.convolution.convolve(climo_u_geo_field, kernel, boundary='wrap')
    climo_v_geo_field = astropy.convolution.convolve(climo_v_geo_field, kernel, boundary='wrap')

    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(1, 2)
    matplotlib.rcParams.update({'font.size': 10})

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    """ Plot u_geo """
    ax1 = plt.subplot(221, projection=ccrs.SouthPolarStereo())

    ax1.set_boundary(circle, transform=ax1.transAxes)
    ax1.add_feature(land_50m)
    ax1.add_feature(ice_50m)
    ax1.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im1 = ax1.contourf(lons_geo, lats, climo_u_geo_field, transform=vector_crs, cmap='seismic',
                       vmin=-0.1, vmax=0.1, levels=np.linspace(-0.10, 0.10, 16), extend='both')

    ax1.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=1.5, transform=vector_crs)
    ax1.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=1.5, transform=vector_crs)

    ax1.text(0.485, 1.01,  '0°',   transform=ax1.transAxes)
    ax1.text(1.01,  0.49,  '90°E', transform=ax1.transAxes)
    ax1.text(0.46, -0.04,  '180°', transform=ax1.transAxes)
    ax1.text(-0.14, 0.49,  '90°W', transform=ax1.transAxes)
    ax1.text(0.84,  0.915, '45°E',  rotation=45,  transform=ax1.transAxes)
    ax1.text(0.845, 0.115, '135°E', rotation=-45, transform=ax1.transAxes)
    ax1.text(0.04,  0.925, '45°W',  rotation=-45, transform=ax1.transAxes)
    ax1.text(0.025, 0.115, '135°W', rotation=45,  transform=ax1.transAxes)

    ax1.text(0.505, 1.05, r'Zonal geostrophic velocity $u_g$', fontsize=14, va='bottom', ha='center', rotation='horizontal',
             rotation_mode='anchor', transform=ax1.transAxes)

    plt.suptitle(r'Figure 3: Geostrophic current $\mathbf{u}_g$ and ice drift $\mathbf{u}_i$ observations, '
                 'winter (JAS) mean', fontsize=16)

    """ Plot v_geo """
    ax2 = plt.subplot(222, projection=ccrs.SouthPolarStereo())

    ax2.set_boundary(circle, transform=ax2.transAxes)
    ax2.add_feature(land_50m)
    ax2.add_feature(ice_50m)
    ax2.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im2 = ax2.contourf(lons_geo, lats, climo_v_geo_field, transform=vector_crs, cmap='seismic',
                       vmin=-0.1, vmax=0.1, levels=np.linspace(-0.10, 0.10, 16), extend='both')

    ax2.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=1.5, transform=vector_crs)
    ax2.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=1.5, transform=vector_crs)
    # ax2.contour(lons, lats, np.ma.array(feb_climo_alpha_field, mask=np.isnan(feb_climo_alpha_field)),
    #             levels=[0.15], colors='black', linewidths=1, transform=vector_crs)
    # ax2.contour(lons, lats, np.ma.array(sep_climo_alpha_field, mask=np.isnan(sep_climo_alpha_field)),
    #             levels=[0.15], colors='black', linewidths=1, transform=vector_crs)

    ax2.text(0.485, 1.01,  '0°',   transform=ax2.transAxes)
    ax2.text(1.01,  0.49,  '90°E', transform=ax2.transAxes)
    ax2.text(0.46, -0.04,  '180°', transform=ax2.transAxes)
    ax2.text(-0.14, 0.49,  '90°W', transform=ax2.transAxes)
    ax2.text(0.84,  0.915, '45°E',  rotation=45,  transform=ax2.transAxes)
    ax2.text(0.845, 0.115, '135°E', rotation=-45, transform=ax2.transAxes)
    ax2.text(0.04,  0.925, '45°W',  rotation=-45, transform=ax2.transAxes)
    ax2.text(0.025, 0.115, '135°W', rotation=45,  transform=ax2.transAxes)

    ax2.text(0.50, 1.05, r'Meridional geostrophic velocity $v_g$', fontsize=14, va='bottom', ha='center',
             rotation='horizontal', rotation_mode='anchor', transform=ax2.transAxes)

    """ Plot u_ice """
    ax3 = plt.subplot(223, projection=ccrs.SouthPolarStereo())

    ax3.set_boundary(circle, transform=ax3.transAxes)
    ax3.add_feature(land_50m)
    ax3.add_feature(ice_50m)
    ax3.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    gl = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im3 = ax3.contourf(lons, lats, np.ma.array(climo_u_ice_field, mask=(sep_climo_alpha_field < 0.15)),
                       transform=vector_crs, cmap='seismic',
                       vmin=-0.1, vmax=0.1, levels=np.linspace(-0.10, 0.10, 16), extend='both')

    ax3.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=1.5, transform=vector_crs)
    ax3.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=1.5, transform=vector_crs)
    # ax3.contour(lons, lats, np.ma.array(feb_climo_alpha_field, mask=np.isnan(feb_climo_alpha_field)),
    #             levels=[0.15], colors='black', linewidths=1, transform=vector_crs)
    # ax3.contour(lons, lats, np.ma.array(sep_climo_alpha_field, mask=np.isnan(sep_climo_alpha_field)),
    #             levels=[0.15], colors='black', linewidths=1, transform=vector_crs)

    ax3.text(0.485, 1.01,  '0°',   transform=ax3.transAxes)
    ax3.text(1.01,  0.49,  '90°E', transform=ax3.transAxes)
    ax3.text(0.46, -0.04,  '180°', transform=ax3.transAxes)
    ax3.text(-0.14, 0.49,  '90°W', transform=ax3.transAxes)
    ax3.text(0.84,  0.915, '45°E',  rotation=45,  transform=ax3.transAxes)
    ax3.text(0.845, 0.115, '135°E', rotation=-45, transform=ax3.transAxes)
    ax3.text(0.04,  0.925, '45°W',  rotation=-45, transform=ax3.transAxes)
    ax3.text(0.025, 0.115, '135°W', rotation=45,  transform=ax3.transAxes)

    ax3.text(0.50, 1.05, r'Zonal ice drift $u_i$', fontsize=14, va='bottom', ha='center',
             rotation='horizontal', rotation_mode='anchor', transform=ax3.transAxes)

    ax4 = plt.subplot(224, projection=ccrs.SouthPolarStereo())

    ax4.set_boundary(circle, transform=ax4.transAxes)
    ax4.add_feature(land_50m)
    ax4.add_feature(ice_50m)
    ax4.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    gl = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im4 = ax4.contourf(lons, lats, np.ma.array(climo_v_ice_field, mask=(sep_climo_alpha_field < 0.15)),
                       transform=vector_crs, cmap='seismic',
                       vmin=-0.1, vmax=0.1, levels=np.linspace(-0.10, 0.10, 16), extend='both')

    ax4.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=1.5, transform=vector_crs)
    ax4.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=1.5, transform=vector_crs)
    # ax4.contour(lons, lats, np.ma.array(feb_climo_alpha_field, mask=np.isnan(feb_climo_alpha_field)),
    #             levels=[0.15], colors='black', linewidths=1, transform=vector_crs)
    # ax4.contour(lons, lats, np.ma.array(sep_climo_alpha_field, mask=np.isnan(sep_climo_alpha_field)),
    #             levels=[0.15], colors='black', linewidths=1, transform=vector_crs)

    ax4.text(0.485, 1.01,  '0°',   transform=ax4.transAxes)
    ax4.text(1.01,  0.49,  '90°E', transform=ax4.transAxes)
    ax4.text(0.46, -0.04,  '180°', transform=ax4.transAxes)
    ax4.text(-0.14, 0.49,  '90°W', transform=ax4.transAxes)
    ax4.text(0.84,  0.915, '45°E',  rotation=45,  transform=ax4.transAxes)
    ax4.text(0.845, 0.115, '135°E', rotation=-45, transform=ax4.transAxes)
    ax4.text(0.04,  0.925, '45°W',  rotation=-45, transform=ax4.transAxes)
    ax4.text(0.025, 0.115, '135°W', rotation=45,  transform=ax4.transAxes)

    ax4.text(0.50, 1.05, r'Meridional ice velocity $v_i$', fontsize=14, va='bottom', ha='center',
             rotation='horizontal', rotation_mode='anchor', transform=ax4.transAxes)

    zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
    ice_edge_patch = mpatches.Patch(color='black', label=r'15% ice edge')
    plt.legend(handles=[zero_stress_line_patch, ice_edge_patch], loc='lower center',
               bbox_to_anchor=(-0.5, -0.1, 0.5, -0.1), ncol=1, mode='expand', borderaxespad=0, framealpha=0)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    clb = fig.colorbar(im1, cax=cbar_ax, extend='both', ticks=np.linspace(-0.10, 0.10, 6))
    clb.ax.set_title(r'm/s')

    png_filepath = os.path.join(figure_dir_path, 'figure3.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False)


def make_salinity_figure():
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from constants import figure_dir_path

    # climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)
    climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
                                         year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM',
    #                                      year_start=2005, year_end=2015)

    feb_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 2, 1),
                                             year_start=2005, year_end=2015)
    sep_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 9, 1),
                                             year_start=2005, year_end=2015)

    lons, lats, climo_tau_x_field = get_field_from_netcdf(climo_filepath, 'tau_x')
    climo_alpha_field = get_field_from_netcdf(climo_filepath, 'alpha')[2]
    climo_salinity_field = get_field_from_netcdf(climo_filepath, 'temperature')[2]

    feb_climo_alpha_field = get_field_from_netcdf(feb_climo_filepath, 'alpha')[2]
    sep_climo_alpha_field = get_field_from_netcdf(sep_climo_filepath, 'alpha')[2]

    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 9))
    matplotlib.rcParams.update({'font.size': 10})

    ax = plt.subplot(111, projection=ccrs.SouthPolarStereo())

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.add_feature(land_50m)
    ax.add_feature(ice_50m)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    im = ax.pcolormesh(lons, lats, climo_salinity_field, transform=vector_crs, cmap=cmocean.cm.thermal,
                       vmin=-2, vmax=2)

    ax.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
               colors='green', linewidths=2, transform=vector_crs)
    ax.contour(lons, lats, np.ma.array(feb_climo_alpha_field, mask=np.isnan(feb_climo_alpha_field)),
               levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    ax.contour(lons, lats, np.ma.array(sep_climo_alpha_field, mask=np.isnan(sep_climo_alpha_field)),
               levels=[0.15], colors='black', linewidths=2, transform=vector_crs)

    ax.text(0.51, 1.01, '0°', va='bottom', ha='center', rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(1.05, 0.50, '90°E', va='bottom', ha='center', rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(0.50, -0.04, '180°', va='bottom', ha='center', rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(-0.05, 0.50, '90°W', va='bottom', ha='center', rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes)

    clb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.1, extend='both')
    clb.ax.set_title(r'T (°C)')

    plt.suptitle(r'Temperature (averaged over the Ekman layer), winter (JAS) mean', fontsize=16)

    # zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
    # ice_edge_patch = mpatches.Patch(color='black', label=r'15% ice edge')
    # plt.legend(handles=[zero_stress_line_patch, ice_edge_patch], loc='lower center',
    #            bbox_to_anchor=(-0.5, -0.1, 0.5, -0.1), ncol=1, mode='expand', borderaxespad=0, framealpha=0)

    png_filepath = os.path.join(figure_dir_path, 'salinity_figure.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')


def make_urel_figure():
    import constants
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from constants import figure_dir_path, n_lat, n_lon

    # climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)
    climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
                                         year_start=2011, year_end=2016)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM',
    #                                      year_start=2005, year_end=2015)

    lons, lats, climo_u_ice_field = get_field_from_netcdf(climo_filepath, 'ice_u')
    climo_v_ice_field = get_field_from_netcdf(climo_filepath, 'ice_v')[2]
    climo_u_Ekman_field = get_field_from_netcdf(climo_filepath, 'Ekman_u')[2]
    climo_v_Ekman_field = get_field_from_netcdf(climo_filepath, 'Ekman_v')[2]
    climo_alpha_field = get_field_from_netcdf(climo_filepath, 'alpha')[2]

    climo_u_geo_field = get_field_from_netcdf(climo_filepath, 'geo_u')[2]
    climo_v_geo_field = get_field_from_netcdf(climo_filepath, 'geo_v')[2]

    constants.output_dir_path = 'E:\\output\\'  # u_geo = 0
    climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
                                         year_start=2005, year_end=2015)
    climo_u_Ekman_nogeo_field = get_field_from_netcdf(climo_filepath, 'Ekman_u')[2]
    climo_v_Ekman_nogeo_field = get_field_from_netcdf(climo_filepath, 'Ekman_v')[2]

    constants.output_dir_path = 'C:\\Users\\Ali\\Downloads\\output\\'  # With u_geo
    climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
                                         year_start=2011, year_end=2016)
    climo_u_Ekman_total_field = get_field_from_netcdf(climo_filepath, 'Ekman_u')[2]
    climo_v_Ekman_total_field = get_field_from_netcdf(climo_filepath, 'Ekman_v')[2]

    climo_ice_mag_field = np.zeros((n_lat, n_lon))
    climo_geo_mag_field = np.zeros((n_lat, n_lon))
    climo_Ekman_mag_field = np.zeros((n_lat, n_lon))
    climo_rel_mag_field = np.zeros((n_lat, n_lon))
    climo_rel_ice_ratio_field = np.zeros((n_lat, n_lon))
    climo_Ek_ratio_field = np.zeros((n_lat, n_lon))

    for i in range(n_lat):
        for j in range(n_lon):
            climo_ice_mag_field[i][j] = np.sqrt(climo_u_ice_field[i][j]**2 + climo_v_ice_field[i][j]**2)
            climo_Ekman_mag_field[i][j] = np.sqrt(climo_u_Ekman_field[i][j] ** 2 + climo_v_Ekman_field[i][j] ** 2)

            climo_geo_mag_field[i][j] = np.sqrt(climo_u_geo_field[i][j] ** 2 + climo_v_geo_field[i][j] ** 2)

            u_rel = climo_u_ice_field[i][j] - (climo_u_Ekman_field[i][j] + climo_u_geo_field[i][j])
            v_rel = climo_v_ice_field[i][j] - (climo_v_Ekman_field[i][j] + climo_v_geo_field[i][j])
            climo_rel_mag_field[i][j] = np.sqrt(u_rel**2 + v_rel**2)

            climo_rel_ice_ratio_field[i][j] = climo_rel_mag_field[i][j] / climo_ice_mag_field[i][j]

            u_Ek_nogeo_mag = np.sqrt(climo_u_Ekman_nogeo_field[i][j]**2 + climo_v_Ekman_total_field[i][j]**2)
            u_Ek_total_mag = np.sqrt(climo_u_Ekman_total_field[i][j]**2 + climo_v_Ekman_total_field[i][j]**2)
            climo_Ek_ratio_field[i][j] = u_Ek_total_mag / u_Ek_nogeo_mag

    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 9))
    matplotlib.rcParams.update({'font.size': 10})

    crs_sps = ccrs.SouthPolarStereo()
    crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363
    ax = plt.subplot(111, projection=crs_sps)

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.add_feature(land_50m)
    ax.add_feature(ice_50m)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im = ax.contourf(lons, lats, climo_Ek_ratio_field, transform=vector_crs, cmap='seismic',
                     vmin=0, vmax=2, levels=np.linspace(0, 2, 16), extend='both')

    ax.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
               levels=[0.15], colors='black', linewidths=2, transform=vector_crs)

    ax.text(0.49,  1.01,  '0°',   transform=ax.transAxes)
    ax.text(1.01,  0.49,  '90°E', transform=ax.transAxes)
    ax.text(0.47,  -0.03, '180°', transform=ax.transAxes)
    ax.text(-0.09, 0.49,  '90°W', transform=ax.transAxes)
    ax.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax.transAxes)
    ax.text(0.85,  0.125, '135°E', rotation=-45, transform=ax.transAxes)
    ax.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax.transAxes)
    ax.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax.transAxes)

    clb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.1)
    clb.ax.set_title(r'm/s')

    plt.gcf().text(0.5, 0.95, r'$|\mathbf{u}_{Ek}(total)| / |\mathbf{u}_{Ek}(no geo)$, winter (JAS) mean', fontsize=14)

    png_filepath = os.path.join(figure_dir_path, 'urel_mag_climo.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')


def make_streamwise_coordinate_map():
    from constants import figure_dir_path
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from utils import get_northward_zero_zonal_stress_line, get_northward_ice_edge, get_coast_coordinates

    climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)
    lons, lats, tau_x_field = get_field_from_netcdf(climo_filepath, 'tau_x')

    contour_coordinate = np.empty((len(lats), len(lons)))
    contour_coordinate[:] = np.nan

    tau_x_lons, tau_x_lats = get_northward_zero_zonal_stress_line(climo_filepath)
    alpha_lons, alpha_lats = get_northward_ice_edge(climo_filepath)
    coast_lons, coast_lats = get_coast_coordinates(climo_filepath)

    for i in range(len(lons)):
        if alpha_lats[i] > tau_x_lats[i] > coast_lats[i]:
            lat_0 = coast_lats[i]
            lat_h = tau_x_lats[i]  # lat_h is short for lat_half ~ lat_1/2
            lat_1 = alpha_lats[i]

            for j in range(len(lats)):
                lat = lats[j]

                if lat < lat_0 or lat > lat_1:
                    contour_coordinate[j][i] = np.nan
                elif lat_0 <= lat <= lat_h:
                    contour_coordinate[j][i] = (lat - lat_0) / (2 * (lat_h - lat_0))
                elif lat_h <= lat <= lat_1:
                    contour_coordinate[j][i] = 0.5 + ((lat - lat_h) / (2 * (lat_1 - lat_h)))

    fig = plt.figure()

    ax = plt.axes(projection=ccrs.SouthPolarStereo())
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face',
                                                   facecolor='dimgray', linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)

    ax.add_feature(land_50m)
    ax.add_feature(ice_50m)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    vector_crs = ccrs.PlateCarree()

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    im = ax.pcolormesh(lons, lats, contour_coordinate, transform=vector_crs, cmap='PuOr')
    fig.colorbar(im, ax=ax)
    plt.title('streamwise coordinates (0=coast, 0.5=zero stress line, 1=ice edge)')

    png_filepath = os.path.join(figure_dir_path, 'streamwise_coordinate_map.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')

    # fig = plt.figure(figsize=(16, 9))
    # ax = fig.add_subplot(111)
    # ax.plot(tau_x_lons, tau_x_lats, linewidth=1, label='tau_x')
    # ax.plot(alpha_lons, alpha_lats, linewidth=1, label='alpha')
    # ax.plot(coast_lons, coast_lats, linewidth=1, label='coast')
    # ax.legend()
    # plt.show()


def make_streamwise_averaged_plots():
    from constants import figure_dir_path, D_e
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from utils import get_northward_zero_zonal_stress_line, get_northward_ice_edge, get_coast_coordinates

    # climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)
    climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
                                         year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM',
    #                                      year_start=20

    lons, lats, tau_x_field = get_field_from_netcdf(climo_filepath, 'tau_x')
    tau_y_field = get_field_from_netcdf(climo_filepath, 'tau_y')[2]
    u_Ekman_field = get_field_from_netcdf(climo_filepath, 'Ekman_u')[2]
    v_Ekman_field = get_field_from_netcdf(climo_filepath, 'Ekman_v')[2]
    w_Ekman_field = get_field_from_netcdf(climo_filepath, 'Ekman_w')[2]

    contour_coordinate = np.empty((len(lats), len(lons)))
    contour_coordinate[:] = np.nan

    tau_x_lons, tau_x_lats = get_northward_zero_zonal_stress_line(climo_filepath)
    alpha_lons, alpha_lats = get_northward_ice_edge(climo_filepath)
    coast_lons, coast_lats = get_coast_coordinates(climo_filepath)

    for i in range(len(lons)):
        if alpha_lats[i] > tau_x_lats[i] > coast_lats[i]:
            lat_0 = coast_lats[i]
            lat_h = tau_x_lats[i]  # lat_h is short for lat_half ~ lat_1/2
            lat_1 = alpha_lats[i]

            for j in range(len(lats)):
                lat = lats[j]

                if lat < lat_0 or lat > lat_1:
                    contour_coordinate[j][i] = np.nan
                elif lat_0 <= lat <= lat_h:
                    contour_coordinate[j][i] = (lat - lat_0) / (2 * (lat_h - lat_0))
                elif lat_h <= lat <= lat_1:
                    contour_coordinate[j][i] = 0.5 + ((lat - lat_h) / (2 * (lat_1 - lat_h)))

    c_bins = np.linspace(0, 1, 41)[:-1]
    delta_c = c_bins[1] - c_bins[0]
    c_bins = c_bins + (delta_c / 2)

    tau_x_cavg = np.zeros(c_bins.shape)
    tau_y_cavg = np.zeros(c_bins.shape)
    u_Ekman_cavg = np.zeros(c_bins.shape)
    v_Ekman_cavg = np.zeros(c_bins.shape)
    w_Ekman_cavg = np.zeros(c_bins.shape)

    for i in range(len(c_bins)):
        c = c_bins[i]
        c_low = c - (delta_c / 2)
        c_high = c + (delta_c / 2)

        c_in_range = np.logical_and(contour_coordinate > c_low, contour_coordinate < c_high)

        tau_x_cavg[i] = np.nanmean(tau_x_field[c_in_range])
        tau_y_cavg[i] = np.nanmean(tau_y_field[c_in_range])
        u_Ekman_cavg[i] = np.nanmean(u_Ekman_field[c_in_range])
        v_Ekman_cavg[i] = np.nanmean(v_Ekman_field[c_in_range])
        w_Ekman_cavg[i] = np.nanmean(w_Ekman_field[c_in_range])

    fig = plt.figure(figsize=(20, 6))

    ax = fig.add_subplot(131)
    ax.plot(c_bins, tau_x_cavg, label=r'$\tau_x$')
    ax.plot(c_bins, tau_y_cavg, label=r'$\tau_y$')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], minor=False)
    ax.set_xlabel('streamwise coordinate', fontsize='large')
    ax.set_ylabel(r'Surface stress (N/m$^2$)', fontsize='large')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(linestyle='--')
    ax.legend(fontsize='large')

    ax = fig.add_subplot(132)
    ax.plot(c_bins, w_Ekman_cavg * 3600 * 24 * 365, label=r'$w_{Ek}$')
    ax.set_xlabel('streamwise coordinate', fontsize='large')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], minor=False)
    ax.set_ylabel('Ekman pumping (m/year)', fontsize='large')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(linestyle='--')
    ax.legend(fontsize='large')

    ax = fig.add_subplot(133)
    ax.plot(c_bins, D_e * u_Ekman_cavg, label=r'$u_{Ek}$')
    ax.plot(c_bins, D_e * v_Ekman_cavg, label=r'$v_{Ek}$')
    ax.set_xlabel('streamwise coordinate', fontsize='large')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], minor=False)
    ax.set_ylabel('Ekman volume transport (m$^2$/s or Sv?)', fontsize='large')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(linestyle='--')
    ax.legend(fontsize='large')

    plt.suptitle(r'Streamwise averages, winter (JAS) mean', fontsize=16)

    png_filepath = os.path.join(figure_dir_path, 'streamwise_average.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')


def make_zonal_average_plots():
    from constants import figure_dir_path, D_e
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from utils import get_northward_zero_zonal_stress_line, get_northward_ice_edge, get_coast_coordinates

    climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
    #                                      year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM',
    #                                      year_start=20

    lons, lats, tau_x_field = get_field_from_netcdf(climo_filepath, 'tau_x')
    tau_y_field = get_field_from_netcdf(climo_filepath, 'tau_y')[2]
    u_Ekman_field = get_field_from_netcdf(climo_filepath, 'Ekman_u')[2]
    v_Ekman_field = get_field_from_netcdf(climo_filepath, 'Ekman_v')[2]
    w_Ekman_field = get_field_from_netcdf(climo_filepath, 'Ekman_w')[2]

    fig = plt.figure(figsize=(20, 6))

    ax1 = fig.add_subplot(131)

    ax1.plot(lats, np.nanmean(tau_x_field, axis=1), label=r'$\tau_x$')
    ax1.plot(lats, np.nanmean(tau_y_field, axis=1), label=r'$\tau_y$')
    ax1.set_xlim(-80, -40)
    ax1.set_xlabel('latitude', fontsize='large')
    ax1.set_ylabel('Surface stress (N/m$^2$)', fontsize='large')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.grid(linestyle='--')
    ax1.legend(fontsize='large')

    ax2 = fig.add_subplot(132)
    ax2.plot(lats, np.nanmean(w_Ekman_field * 365 * 24 * 3600, axis=1), label=r'$w_{Ek}$')
    ax2.set_xlim(-80, -40)
    ax2.set_xlabel('latitude', fontsize='large')
    ax2.set_ylabel('Ekman pumping (m/year)', fontsize='large')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.grid(linestyle='--')
    ax2.legend(fontsize='large')

    # [m^2/s] -> [Sv]
    u_Ekman_zonal_avg = np.nanmean(D_e * u_Ekman_field, axis=1)
    u_Ekman_zonal_avg = np.multiply(u_Ekman_zonal_avg, (2 * np.pi * 6371e3 * np.cos(np.deg2rad(lats))) / 1e6)
    v_Ekman_zonal_avg = np.nanmean(D_e * v_Ekman_field, axis=1)
    v_Ekman_zonal_avg = np.multiply(v_Ekman_zonal_avg, (2 * np.pi * 6371e3 * np.cos(np.deg2rad(lats))) / 1e6)

    ax3 = fig.add_subplot(133)
    ax3.plot(lats, u_Ekman_zonal_avg, label='$u_{Ek}$')
    ax3.plot(lats, v_Ekman_zonal_avg, label='$v_{Ek}$')
    ax3.set_xlim(-80, -40)
    ax3.set_xlabel('latitude', fontsize='large')
    ax3.set_ylabel('Ekman volume transport (m$^2$/s or Sv?)', fontsize='large')
    ax3.tick_params(axis='both', which='major', labelsize=10)
    ax3.grid(linestyle='--')
    ax3.legend()
    ax3.legend(fontsize='large')

    plt.suptitle(r'Zonal averages, annual mean', fontsize=16)

    png_filepath = os.path.join(figure_dir_path, 'zonal_averags.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')


def compare_zzsl_with_gamma_contour():
    import netCDF4
    from os import path
    from constants import data_dir_path, figure_dir_path
    from utils import log_netCDF_dataset_metadata, get_netCDF_filepath, get_field_from_netcdf

    year = 2005
    depth_levels = 0

    for year in range(2005, 2014):
        """ Load neutral density map. """
        gamma_filepath = path.join(data_dir_path, 'mapped_gamma_all_sources',
                                   'mapped_gamma_all_sources_' + str(year) + '.nc')
        logger.info('Loading gamma dataset: {}'.format(gamma_filepath))
        gamma_dataset = netCDF4.Dataset(gamma_filepath)
        # log_netCDF_dataset_metadata(gamma_dataset)

        lats_gamma = np.array(gamma_dataset.variables['lat'])
        lons_gamma = np.array(gamma_dataset.variables['lon'])
        depths_gamma = np.array(gamma_dataset.variables['depth'])

        gamma_data = np.array(gamma_dataset.variables['gamma'])

        if year == 2005:
            gamma_avg = np.nan_to_num(gamma_data)
            gamma_data[~np.isnan(gamma_data)] = 1
            gamma_data[np.isnan(gamma_data)] = 0
            gamma_days = gamma_data
        else:
            gamma_avg = gamma_avg + np.nan_to_num(gamma_data)
            gamma_data[~np.isnan(gamma_data)] = 1
            gamma_data[np.isnan(gamma_data)] = 0
            gamma_days = gamma_days + gamma_data

    gamma_avg = np.divide(gamma_avg, gamma_days)

    climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)
    lons, lats, tau_x_field = get_field_from_netcdf(climo_filepath, 'tau_x')
    alpha_field = get_field_from_netcdf(climo_filepath, 'alpha')[2]

    for year in range(2013, 2014):
        """ Load surface stress map. """
        # annual_avg_filepath = get_netCDF_filepath(field_type='annual', date=datetime.date(year, 1, 1))
        #
        # lons, lats, tau_x_field = get_field_from_netcdf(annual_avg_filepath, 'tau_x')
        # alpha_field = get_field_from_netcdf(annual_avg_filepath, 'alpha')[2]

        # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
        # the smaller plots.
        land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                       linewidth=0)
        ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m',
                                                      edgecolor='face', facecolor='darkgray', linewidth=0)
        vector_crs = ccrs.PlateCarree()

        # Compute a circle in axes coordinates, which we can use as a boundary
        # for the map. We can pan/zoom as much as we like - the boundary will be
        # permanently circular.
        import matplotlib.path as mpath
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)

        for depth_levels in range(0, 3):
            fig = plt.figure(figsize=(16, 9))
            matplotlib.rcParams.update({'font.size': 10})

            ax = plt.subplot(111, projection=ccrs.SouthPolarStereo())
            ax.set_boundary(circle, transform=ax.transAxes)

            ax.add_feature(land_50m)
            ax.add_feature(ice_50m)
            ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

            # im = ax.pcolormesh(lons_gamma, lats_gamma, gamma_data[depth_levels], transform=vector_crs,
            #                    cmap=cmocean.cm.haline, vmin=27.4, vmax=27.8)
            # ax.contour(lons_gamma, lats_gamma, gamma_data[depth_levels], levels=[27.6], colors='black', linewidths=2,
            #            transform=vector_crs)

            im = ax.pcolormesh(lons_gamma, lats_gamma, gamma_avg[depth_levels], transform=vector_crs,
                               cmap=cmocean.cm.haline, vmin=27.4, vmax=27.8)
            ax.contour(lons_gamma, lats_gamma, gamma_avg[depth_levels], levels=[27.6], colors='black', linewidths=2,
                       transform=vector_crs)
            ax.contour(lons, lats, np.ma.array(tau_x_field, mask=np.isnan(alpha_field)), levels=[0],
                       colors='green', linewidths=2, transform=vector_crs)

            clb = fig.colorbar(im, ax=ax, extend='both', fraction=0.046, pad=0.1)
            clb.ax.set_title(r'$\gamma$ (kg/m$^3$)')

            ax.set_title('annual mean, depth_level={:} m'.format(depths_gamma[depth_levels]))

            zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
            gamma_contour_patch = mpatches.Patch(color='black', label=r'$\gamma$ = 27.6 kg/m$^3$ contour')
            plt.legend(handles=[zero_stress_line_patch, gamma_contour_patch], loc='lower center',
                       bbox_to_anchor=(0, -0.05, 1, -0.05), ncol=3, mode='expand', borderaxespad=0, framealpha=0)

            # png_filepath = os.path.join(figure_dir_path, 'compare_zzsl_gamma',
            #                             'compare_zzsl_gamma_' + str(year) + '_' + str(depths_gamma[depth_levels]) + '.png')
            png_filepath = os.path.join(figure_dir_path, 'compare_zzsl_gamma',
                                        'compare_zzsl_gamma_' + str(depths_gamma[depth_levels]) + '.png')

            tau_dir = os.path.dirname(png_filepath)
            if not os.path.exists(tau_dir):
                logger.info('Creating directory: {:s}'.format(tau_dir))
                os.makedirs(tau_dir)

            logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
            plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')


def compare_zzsl_with_pellichero_gamma():
    import netCDF4
    from os import path
    from constants import data_dir_path, figure_dir_path
    from utils import log_netCDF_dataset_metadata, get_netCDF_filepath, get_field_from_netcdf

    # Load neutral density map from Pellichero et al. (2017, 2018).
    gamma_filepath = path.join(data_dir_path, 'Climatology_MLD_v2017.nc')
    logger.info('Loading gamma dataset: {}'.format(gamma_filepath))
    gamma_dataset = netCDF4.Dataset(gamma_filepath)

    lats_gamma = np.array(gamma_dataset.variables['lat'])
    lons_gamma = np.array(gamma_dataset.variables['lon'])
    gamma_data = np.array(gamma_dataset.variables['ML_Gamma'])

    gamma_avg = None

    for month in [3, 4, 5]:
        gamma_monthly = gamma_data[month]

        if gamma_avg is None:
            gamma_avg = np.nan_to_num(gamma_monthly)
            gamma_monthly[~np.isnan(gamma_monthly)] = 1
            gamma_monthly[np.isnan(gamma_monthly)] = 0
            gamma_days = gamma_monthly
        else:
            gamma_avg = gamma_avg + np.nan_to_num(gamma_monthly)
            gamma_monthly[~np.isnan(gamma_monthly)] = 1
            gamma_monthly[np.isnan(gamma_monthly)] = 0
            gamma_days = gamma_days + gamma_monthly

    gamma_avg = np.divide(gamma_avg, gamma_days)

    # Load appropriate surface stress map.
    # climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
    #                                      year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM',
    #                                      year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='AMJ',
    #                                      year_start=2005, year_end=2015)
    climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='AMJ',
                                         year_start=2005, year_end=2015)

    lons, lats, tau_x_field = get_field_from_netcdf(climo_filepath, 'tau_x')
    alpha_field = get_field_from_netcdf(climo_filepath, 'alpha')[2]

    gamma_avg, lons_gamma = cartopy.util.add_cyclic_point(gamma_avg, coord=lons_gamma)

    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m',
                                                  edgecolor='face', facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    import matplotlib.path as mpath
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    fig = plt.figure(figsize=(16, 9))
    matplotlib.rcParams.update({'font.size': 10})

    ax = plt.subplot(111, projection=ccrs.SouthPolarStereo())
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.add_feature(land_50m)
    ax.add_feature(ice_50m)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    im = ax.pcolormesh(lons_gamma, lats_gamma, gamma_avg, transform=vector_crs, cmap=cmocean.cm.haline,
                       vmin=27, vmax=28)
    ax.contour(lons_gamma, lats_gamma, gamma_avg, levels=[27.6], colors='red', linewidths=2, transform=vector_crs)
    ax.contour(lons, lats, np.ma.array(tau_x_field, mask=np.isnan(alpha_field)), levels=[0],
               colors='green', linewidths=2, transform=vector_crs)
    ax.contour(lons, lats, np.ma.array(alpha_field, mask=np.isnan(alpha_field)), levels=[0.15],
               colors='black', linewidths=2, transform=vector_crs)

    clb = fig.colorbar(im, ax=ax, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'$\gamma^n$ (kg/m$^3$)')

    zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
    gamma_contour_patch = mpatches.Patch(color='red', label=r'$\gamma^n$ = 27.6 kg/m$^3$ (Pellichero et al., 2018)')
    ice_edge_patch = mpatches.Patch(color='black', label='15% ice edge')
    plt.legend(handles=[zero_stress_line_patch, gamma_contour_patch, ice_edge_patch], loc='lower center',
               bbox_to_anchor=(0, -0.05, 1, -0.05), ncol=3, mode='expand', borderaxespad=0, framealpha=0)

    ax.set_title(r'Zero zonal stress line vs. $\gamma^n$ = 27.6 kg/m$^3$ contour, fall (AMJ) mean')

    png_filepath = os.path.join(figure_dir_path, 'compare_zzsl_gamma_pellichero.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')


def make_melt_rate_plots():
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from constants import figure_dir_path, data_dir_path

    climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='OND',
    #                                      year_start=2005, year_end=2015)

    lons, lats, ice_flux_div_field = get_field_from_netcdf(climo_filepath, 'ice_flux_div')
    melt_rate_field = get_field_from_netcdf(climo_filepath, 'melt_rate')[2]
    climo_tau_x_field = get_field_from_netcdf(climo_filepath, 'tau_x')[2]
    climo_alpha_field = get_field_from_netcdf(climo_filepath, 'alpha')[2]

    # climo_tau_x_field, lons_tau = cartopy.util.add_cyclic_point(climo_tau_x_field, coord=lons)
    # climo_tau_y_field, lons_tau = cartopy.util.add_cyclic_point(climo_tau_y_field, coord=lons)

    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    # Compute a circle in axes coordinates, which we can use as a boundary for the map. We can pan/zoom as much as we
    # like - the boundary will be permanently circular.
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(1, 2)
    matplotlib.rcParams.update({'font.size': 10})

    """ Plot ice_flux_div """
    crs_sps = ccrs.SouthPolarStereo()
    crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

    ax1 = plt.subplot(121, projection=crs_sps)

    ax1.add_feature(land_50m)
    ax1.add_feature(ice_50m)
    ax1.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax1.set_boundary(circle, transform=ax1.transAxes)

    gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl1.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl1.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im1 = ax1.pcolormesh(lons, lats, np.ma.array(3600*24*365*ice_flux_div_field, mask=(climo_alpha_field < 0.15)),
                         transform=vector_crs, cmap=cmocean.cm.balance, vmin=-2, vmax=2)

    ax1.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    ax1.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=2, transform=vector_crs)

    ax1.text(0.49,  1.01,  '0°',   transform=ax1.transAxes)
    ax1.text(1.01,  0.49,  '90°E', transform=ax1.transAxes)
    ax1.text(0.47,  -0.03, '180°', transform=ax1.transAxes)
    ax1.text(-0.09, 0.49,  '90°W', transform=ax1.transAxes)
    ax1.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax1.transAxes)
    ax1.text(0.85,  0.125, '135°E', rotation=-45, transform=ax1.transAxes)
    ax1.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax1.transAxes)
    ax1.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax1.transAxes)

    ax1.text(0.50, 1.05, r'$\nabla \cdot (\alpha h_{ice} \mathbf{u}_{ice})$',
             fontsize=14, ha='center', transform=ax1.transAxes)

    clb = fig.colorbar(im1, ax=ax1, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'm/year')

    zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
    ice_edge_patch = mpatches.Patch(color='black', label=r'15% ice edge')
    plt.legend(handles=[zero_stress_line_patch, ice_edge_patch], loc='lower center',
               bbox_to_anchor=(0, -0.15, 1, -0.15), ncol=1, mode='expand', borderaxespad=0, framealpha=0)

    plt.suptitle(r'Annual mean', fontsize=16)

    """ Plot melt_rate """
    crs_sps = ccrs.SouthPolarStereo()
    crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

    ax2 = plt.subplot(122, projection=crs_sps)

    ax2.add_feature(land_50m)
    ax2.add_feature(ice_50m)
    ax2.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax2.set_boundary(circle, transform=ax2.transAxes)

    gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl2.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl2.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im2 = ax2.pcolormesh(lons, lats, np.ma.array(3600*24*365*melt_rate_field, mask=(climo_alpha_field < 0.15)),
                         transform=vector_crs, cmap=cmocean.cm.balance, vmin=-1, vmax=1)

    ax2.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=2, transform=vector_crs)
    ax2.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)

    ax2.text(0.49,  1.01,  '0°',   transform=ax2.transAxes)
    ax2.text(1.01,  0.49,  '90°E', transform=ax2.transAxes)
    ax2.text(0.47,  -0.03, '180°', transform=ax2.transAxes)
    ax2.text(-0.09, 0.49,  '90°W', transform=ax2.transAxes)
    ax2.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax2.transAxes)
    ax2.text(0.85,  0.125, '135°E', rotation=-45, transform=ax2.transAxes)
    ax2.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax2.transAxes)
    ax2.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax2.transAxes)

    ax2.text(0.50, 1.05, r'$\frac{\mathcal{U}_{Ek} \cdot \nabla S}{S}$',
             fontsize=14, ha='center', transform=ax2.transAxes)

    clb = fig.colorbar(im2, ax=ax2, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'm/year')

    png_filepath = os.path.join(figure_dir_path, 'melt_rate_figure.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False)


def look_for_ice_ocean_governor():
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from constants import figure_dir_path

    import constants
    constants.output_dir_path = 'C:\\Users\\Ali\\Downloads\\output\\'

    climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2011, year_end=2016)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
    #                                      year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM',
    #                                      year_start=2005, year_end=2015)

    lons, lats, climo_tau_x_field_geo = get_field_from_netcdf(climo_filepath, 'tau_x')
    climo_w_Ekman_field_geo = get_field_from_netcdf(climo_filepath, 'Ekman_w')[2]

    constants.output_dir_path = 'D:\\output\\'

    climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
    #                                      year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM',
    #                                      year_start=2005, year_end=2015)

    feb_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 2, 1),
                                             year_start=2005, year_end=2015)
    sep_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 9, 1),
                                             year_start=2005, year_end=2015)

    lons, lats, climo_tau_x_field = get_field_from_netcdf(climo_filepath, 'tau_x')
    climo_alpha_field = get_field_from_netcdf(climo_filepath, 'alpha')[2]
    climo_w_Ekman_field = get_field_from_netcdf(climo_filepath, 'Ekman_w')[2]

    feb_climo_alpha_field = get_field_from_netcdf(feb_climo_filepath, 'alpha')[2]
    sep_climo_alpha_field = get_field_from_netcdf(sep_climo_filepath, 'alpha')[2]

    import astropy.convolution
    kernel = astropy.convolution.Box2DKernel(10)
    # kernel = astropy.convolution.Gaussian2DKernel(2)
    climo_w_Ekman_field_geo = astropy.convolution.convolve(climo_w_Ekman_field_geo, kernel, boundary='wrap')
    climo_tau_x_field_geo = astropy.convolution.convolve(climo_tau_x_field_geo, kernel, boundary='wrap')

    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(1, 2)
    matplotlib.rcParams.update({'font.size': 10})

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    """ Plot Ekman pumping (without u_geo) """
    ax1 = plt.subplot(121, projection=ccrs.SouthPolarStereo())

    ax1.add_feature(land_50m)
    ax1.add_feature(ice_50m)
    ax1.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax1.set_boundary(circle, transform=ax1.transAxes)

    gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl1.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl1.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im1 = ax1.pcolormesh(lons, lats, climo_w_Ekman_field * 365*24*3600, transform=vector_crs, cmap=cmocean.cm.balance,
                         vmin=-100, vmax=100)

    ax1.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=2, transform=vector_crs)
    ax1.contour(lons, lats, np.ma.array(feb_climo_alpha_field, mask=np.isnan(feb_climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    ax1.contour(lons, lats, np.ma.array(sep_climo_alpha_field, mask=np.isnan(sep_climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)

    ax1.text(0.49,  1.01,  '0°',   transform=ax1.transAxes)
    ax1.text(1.01,  0.49,  '90°E', transform=ax1.transAxes)
    ax1.text(0.47,  -0.03, '180°', transform=ax1.transAxes)
    ax1.text(-0.09, 0.49,  '90°W', transform=ax1.transAxes)
    ax1.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax1.transAxes)
    ax1.text(0.85,  0.125, '135°E', rotation=-45, transform=ax1.transAxes)
    ax1.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax1.transAxes)
    ax1.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax1.transAxes)

    ax1.text(0.50, 1.05, 'Ekman pumping (no u_geo)', fontsize=14, ha='center', transform=ax1.transAxes)

    clb = fig.colorbar(im1, ax=ax1, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title('m/year')

    zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
    ice_edge_patch = mpatches.Patch(color='black', label=r'15% ice edge')
    plt.legend(handles=[zero_stress_line_patch, ice_edge_patch], loc='lower center',
               bbox_to_anchor=(0, -0.1, 1, -0.1), ncol=1, mode='expand', borderaxespad=0, framealpha=0)

    # plt.suptitle(r'Figure 6: Surface stress curl and Ekman pumping observations, winter (JAS) mean', fontsize=16)

    """ Plot Ekman pumping (including u_geo) """
    ax2 = plt.subplot(122, projection=ccrs.SouthPolarStereo())

    ax2.add_feature(land_50m)
    ax2.add_feature(ice_50m)
    ax2.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax2.set_boundary(circle, transform=ax2.transAxes)

    gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl2.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl2.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    climo_w_Ekman_field_geo, lons_cyclic = cartopy.util.add_cyclic_point(climo_w_Ekman_field_geo, coord=lons)

    im2 = ax2.pcolormesh(lons_cyclic, lats, climo_w_Ekman_field_geo * 365*24*3600, transform=vector_crs,
                         cmap=cmocean.cm.balance, vmin=-100, vmax=100)

    ax2.contour(lons, lats, np.ma.array(climo_tau_x_field_geo, mask=np.isnan(climo_alpha_field)), levels=[0],
                colors='green', linewidths=2, transform=vector_crs)
    ax2.contour(lons, lats, np.ma.array(feb_climo_alpha_field, mask=np.isnan(feb_climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    ax2.contour(lons, lats, np.ma.array(sep_climo_alpha_field, mask=np.isnan(sep_climo_alpha_field)),
                levels=[0.15], colors='black', linewidths=2, transform=vector_crs)

    ax2.text(0.49,  1.01,  '0°',   transform=ax2.transAxes)
    ax2.text(1.01,  0.49,  '90°E', transform=ax2.transAxes)
    ax2.text(0.47,  -0.03, '180°', transform=ax2.transAxes)
    ax2.text(-0.09, 0.49,  '90°W', transform=ax2.transAxes)
    ax2.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax2.transAxes)
    ax2.text(0.85,  0.125, '135°E', rotation=-45, transform=ax2.transAxes)
    ax2.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax2.transAxes)
    ax2.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax2.transAxes)

    ax2.text(0.50, 1.05, 'Ekman pumping (including u_geo)', fontsize=14, ha='center', transform=ax2.transAxes)

    clb = fig.colorbar(im2, ax=ax2, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'm/year')

    png_filepath = os.path.join(figure_dir_path, 'ice_ocean_governor.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False)  # , bbox_inches='tight')


def plot_Wedell_Gyre_Ekman_pumping():
    import constants
    constants.output_dir_path = 'C:\\Users\\Ali\\Downloads\\output\\'

    climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2011, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
    #                                      year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM',
    #                                      year_start=2005, year_end=2015)

    lons, lats, climo_tau_x_field_geo = get_field_from_netcdf(climo_filepath, 'tau_x')
    climo_w_Ekman_field_geo = get_field_from_netcdf(climo_filepath, 'Ekman_w')[2]

    constants.output_dir_path = 'D:\\output\\'

    climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
    #                                      year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM',
    #                                      year_start=2005, year_end=2015)

    feb_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 2, 1),
                                             year_start=2005, year_end=2015)
    sep_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 9, 1),
                                             year_start=2005, year_end=2015)

    lons, lats, climo_tau_x_field = get_field_from_netcdf(climo_filepath, 'tau_x')
    climo_alpha_field = get_field_from_netcdf(climo_filepath, 'alpha')[2]
    climo_w_Ekman_field = get_field_from_netcdf(climo_filepath, 'Ekman_w')[2]

    feb_climo_alpha_field = get_field_from_netcdf(feb_climo_filepath, 'alpha')[2]
    sep_climo_alpha_field = get_field_from_netcdf(sep_climo_filepath, 'alpha')[2]

    import astropy.convolution
    kernel = astropy.convolution.Box2DKernel(10)
    # kernel = astropy.convolution.Gaussian2DKernel(2)
    climo_w_Ekman_field_geo = astropy.convolution.convolve(climo_w_Ekman_field_geo, kernel, boundary='wrap')

    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 9))

    crs_sps = ccrs.SouthPolarStereo(central_longitude=-15)
    crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

    ax = plt.subplot(111, projection=crs_sps)
    ax.add_feature(land_50m)
    ax.add_feature(ice_50m)
    ax.set_extent([-65, 32, -80, -53], ccrs.PlateCarree())

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-90, -70, -50, -30, -10, 10, 30, 50, 70])
    gl.ylocator = mticker.FixedLocator([-80, -70, -60, -50, -40])

    # im = ax.contourf(lons, lats, climo_w_Ekman_field * 365*24*3600, levels=np.arange(-100, 110, 10),
    #                  transform=vector_crs, cmap=cmocean.cm.balance, vmin=-100, vmax=100, extend='both')
    im = ax.contourf(lons, lats, climo_w_Ekman_field_geo * 365*24*3600, levels=np.arange(-100, 110, 10),
                     transform=vector_crs, cmap=cmocean.cm.balance, vmin=-100, vmax=100, extend='both')

    ax.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
               colors='green', linewidths=2, transform=vector_crs)

    ax.contour(lons, lats, np.ma.array(feb_climo_alpha_field, mask=np.isnan(feb_climo_alpha_field)),
               levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    ax.contour(lons, lats, np.ma.array(sep_climo_alpha_field, mask=np.isnan(sep_climo_alpha_field)),
               levels=[0.15], colors='black', linewidths=2, transform=vector_crs)

    ax.add_patch(mpatches.Rectangle(xy=[-60, -68], width=90, height=13, transform=ccrs.PlateCarree(),
                                    edgecolor='gold', linewidth=3, linestyle='solid', facecolor='none'))

    clb = fig.colorbar(im, ax=ax, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'm/year')

    plt.title(r'Annual mean Ekman pumping in the Weddell Gyre region (with $u_{geo}$)')

    zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
    ice_edge_patch = mpatches.Patch(color='black', label=r'15% ice edge')
    WGR_patch = mpatches.Patch(color='gold', label=r'boundary of the Weddell Gyre Region (WGR)')
    plt.legend(handles=[zero_stress_line_patch, ice_edge_patch, WGR_patch], loc='lower center',
               bbox_to_anchor=(0, -0.1, 1, -0.1), ncol=3, mode='expand', borderaxespad=0, framealpha=0)

    png_filepath = os.path.join(figure_dir_path, 'Ekman pumping in the Weddell Gyre region.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False)
    plt.close()


def plot_Weddell_Gyre_time_series():
    import pickle

    lat1_WGR, lat2_WGR = -68, -55
    lon1_WGR, lon2_WGR = -60, 30

    lat1_idx_WGR, lat2_idx_WGR = 48, 101
    lon1_idx_WGR, lon2_idx_WGR = 480, 841

    year_start = 2011
    year_end = 2015
    dates = date_range(datetime.date(year_start, 1, 1), datetime.date(year_end, 12, 31))

    n_days = len(dates)

    pickle_filepath = 'D:\\output\\WGR_time_series.pickle'

    pickle_found = False
    try:
        with open(pickle_filepath, 'rb') as f:
            WGR_time_series_dict = pickle.load(f)
            logger.info('Previous computation found. Loading {:s}...'.format(pickle_filepath))
            pickle_found = True
    except OSError:
        logger.info('Computing Weddell Gyre Region (WGR) time series...')

    if pickle_found:
        alpha_WGR = WGR_time_series_dict['alpha']
        u_wind_WGR = WGR_time_series_dict['u_wind']
        v_wind_WGR = WGR_time_series_dict['v_wind']
        wind_speed_WGR = WGR_time_series_dict['wind_speed']
        u_ice_WGR = WGR_time_series_dict['u_ice']
        v_ice_WGR = WGR_time_series_dict['v_ice']
        ice_speed_WGR = WGR_time_series_dict['ice_speed']
        u_geo_WGR = WGR_time_series_dict['u_geo']
        v_geo_WGR = WGR_time_series_dict['v_geo']
        geo_speed_WGR = WGR_time_series_dict['geo_speed']
        w_Ekman_WGR = WGR_time_series_dict['w_Ekman']
        w_Ekman_geo_WGR = WGR_time_series_dict['w_Ekman_geo']

    else:
        alpha_WGR = np.zeros(n_days)
        u_wind_WGR = np.zeros(n_days)
        v_wind_WGR = np.zeros(n_days)
        wind_speed_WGR = np.zeros(n_days)
        u_ice_WGR = np.zeros(n_days)
        v_ice_WGR = np.zeros(n_days)
        ice_speed_WGR = np.zeros(n_days)
        u_geo_WGR = np.zeros(n_days)
        v_geo_WGR = np.zeros(n_days)
        geo_speed_WGR = np.zeros(n_days)
        w_Ekman_WGR = np.zeros(n_days)
        w_Ekman_geo_WGR = np.zeros(n_days)

        for d, date in enumerate(dates):
            tau_filepath = get_netCDF_filepath(field_type='daily', date=date)

            import constants
            constants.output_dir_path = 'D:\\output\\'

            try:
                climo_filepath_no_geo = get_netCDF_filepath(field_type='daily', date=date)
                _, _, w_Ekman_field = get_field_from_netcdf(climo_filepath_no_geo, 'Ekman_w')
            except Exception as e:
                logger.error('{}'.format(e))
                logger.warning('{:s} not found. Proceeding without it...'.format(tau_filepath))
                n_days = n_days - 1  # Must account for lost day if no data available for that day.
                continue

            constants.output_dir_path = 'C:\\Users\\Ali\\Downloads\\output\\'

            try:
                tau_dataset = netCDF4.Dataset(tau_filepath)
            except OSError as e:
                logger.error('{}'.format(e))
                logger.warning('{:s} not found. Proceeding without it...'.format(tau_filepath))
                n_days = n_days - 1  # Must account for lost day if no data available for that day.
                continue

            logger.info('Averaging {:%b %d, %Y} ({:s})...'.format(date, tau_filepath))

            lats = np.array(tau_dataset.variables['lat'])[lat1_idx_WGR:lat2_idx_WGR]
            lons = np.array(tau_dataset.variables['lon'])[lon1_idx_WGR:lon2_idx_WGR]

            alpha_field = np.array(tau_dataset.variables['alpha'])[lat1_idx_WGR:lat2_idx_WGR, lon1_idx_WGR:lon2_idx_WGR]
            u_wind_field = np.array(tau_dataset.variables['wind_u'])[lat1_idx_WGR:lat2_idx_WGR, lon1_idx_WGR:lon2_idx_WGR]
            v_wind_field = np.array(tau_dataset.variables['wind_v'])[lat1_idx_WGR:lat2_idx_WGR, lon1_idx_WGR:lon2_idx_WGR]
            u_ice_field = np.array(tau_dataset.variables['ice_u'])[lat1_idx_WGR:lat2_idx_WGR, lon1_idx_WGR:lon2_idx_WGR]
            v_ice_field = np.array(tau_dataset.variables['ice_v'])[lat1_idx_WGR:lat2_idx_WGR, lon1_idx_WGR:lon2_idx_WGR]
            u_geo_field = np.array(tau_dataset.variables['geo_u'])[lat1_idx_WGR:lat2_idx_WGR, lon1_idx_WGR:lon2_idx_WGR]
            v_geo_field = np.array(tau_dataset.variables['geo_v'])[lat1_idx_WGR:lat2_idx_WGR, lon1_idx_WGR:lon2_idx_WGR]
            w_Ekman_geo_field = np.array(tau_dataset.variables['Ekman_w'])[lat1_idx_WGR:lat2_idx_WGR, lon1_idx_WGR:lon2_idx_WGR]

            w_Ekman_field = w_Ekman_field[lat1_idx_WGR:lat2_idx_WGR, lon1_idx_WGR:lon2_idx_WGR]

            import astropy.convolution
            kernel = astropy.convolution.Box2DKernel(10)
            # kernel = astropy.convolution.Gaussian2DKernel(2)
            w_Ekman_geo_field = astropy.convolution.convolve(w_Ekman_geo_field, kernel, boundary='wrap')

            alpha_WGR[d] = np.nanmean(alpha_field)
            u_wind_WGR[d] = np.nanmean(u_wind_field)
            v_wind_WGR[d] = np.nanmean(v_wind_field)
            u_ice_WGR[d] = np.nanmean(u_ice_field)
            v_ice_WGR[d] = np.nanmean(v_ice_field)
            u_geo_WGR[d] = np.nanmean(u_geo_field)
            v_geo_WGR[d] = np.nanmean(v_geo_field)
            w_Ekman_WGR[d] = np.nanmean(w_Ekman_field)
            w_Ekman_geo_WGR[d] = np.nanmean(w_Ekman_geo_field)

            wind_speed_field = np.zeros(alpha_field.shape)
            ice_speed_field = np.zeros(alpha_field.shape)
            geo_speed_field = np.zeros(alpha_field.shape)

            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    if not np.isnan(u_wind_field[i][j]) and not np.isnan(v_wind_field[i][j]):
                        wind_speed_field[i][j] = np.sqrt(u_wind_field[i][j]**2 + v_wind_field[i][j]**2)
                    else:
                        wind_speed_field[i][j] = np.nan

                    if not np.isnan(u_ice_field[i][j]) and not np.isnan(v_ice_field[i][j]):
                        ice_speed_field[i][j] = np.sqrt(u_ice_field[i][j]**2 + v_ice_field[i][j]**2)
                    else:
                        ice_speed_field[i][j] = np.nan

                    if not np.isnan(u_geo_field[i][j]) and not np.isnan(v_geo_field[i][j]):
                        geo_speed_field[i][j] = np.sqrt(u_geo_field[i][j]**2 + v_geo_field[i][j]**2)
                    else:
                        geo_speed_field[i][j] = np.nan

            wind_speed_WGR[d] = np.nanmean(wind_speed_field)
            ice_speed_WGR[d] = np.nanmean(ice_speed_field)
            geo_speed_WGR[d] = np.nanmean(geo_speed_field)

    with open(pickle_filepath, 'wb') as f:
        WGR_time_series_dict = {
            'alpha': alpha_WGR,
            'u_wind': u_wind_WGR,
            'v_wind': v_wind_WGR,
            'wind_speed': wind_speed_WGR,
            'u_ice': u_ice_WGR,
            'v_ice': v_ice_WGR,
            'ice_speed': ice_speed_WGR,
            'u_geo': u_geo_WGR,
            'v_geo': v_geo_WGR,
            'geo_speed': geo_speed_WGR,
            'w_Ekman': w_Ekman_WGR,
            'w_Ekman_geo': w_Ekman_geo_WGR
        }
        pickle.dump(WGR_time_series_dict, f, pickle.HIGHEST_PROTOCOL)

    import astropy.convolution

    kernel = astropy.convolution.Box1DKernel(30)

    alpha_WGR = astropy.convolution.convolve(alpha_WGR, kernel, boundary='extend')
    wind_speed_WGR = astropy.convolution.convolve(wind_speed_WGR, kernel, boundary='extend')
    ice_speed_WGR = astropy.convolution.convolve(ice_speed_WGR, kernel, boundary='extend')
    geo_speed_WGR = astropy.convolution.convolve(geo_speed_WGR, kernel, boundary='extend')

    w_Ekman_WGR = astropy.convolution.convolve(w_Ekman_WGR, kernel, boundary='extend')
    w_Ekman_geo_WGR = astropy.convolution.convolve(w_Ekman_geo_WGR, kernel, boundary='extend')

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig, host = plt.subplots(figsize=(16, 4.5))
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    # Offset the right spine of par2.  The ticks and label have already been placed on the right by twinx above.
    par2.spines['right'].set_position(('axes', 1.1))

    # Having been created by twinx, par2 has its frame off, so the line of its detached spine is invisible.  First,
    # activate the frame but make the patch and spines invisible.
    make_patch_spines_invisible(par2)

    # Second, show the right spine.
    par2.spines['right'].set_visible(True)

    host.fill_between(dates, 0, alpha_WGR, color='lightgray')

    p1, = host.plot(dates, alpha_WGR, 'lightgray')
    p2, = par1.plot(dates, wind_speed_WGR, 'red')
    p3, = par2.plot(dates, ice_speed_WGR, 'green')
    p4, = par2.plot(dates, geo_speed_WGR, 'dodgerblue')

    host.set_xlim(dates[0], dates[-1])
    host.set_ylim(0, 1)
    par1.set_ylim(0, 10)
    par2.set_ylim(0, 0.50)

    host.set_xlabel('Date')
    host.set_ylabel(r'Mean sea ice fraction $\alpha$')
    par1.set_ylabel('Mean wind speed (m/s)', color='red')
    par2.set_ylabel(r'Mean ice and geostrophic speed (m/s)')

    # host.yaxis.label.set_color(p1.get_color())
    # par1.yaxis.label.set_color(p2.get_color())
    # par2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    # host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    # par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    # lines = [p1, p2, p3]
    # host.legend(lines, [l.get_label() for l in lines])

    plt.title('30-day running mean of sea ice fraction, wind, ice, and geostrophic speeds in the Weddell Gyre Region')

    alpha_patch = mpatches.Patch(color='lightgray', label='Sea ice fraction')
    wind_speed_patch = mpatches.Patch(color='red', label='Wind speed')
    ice_speed_patch = mpatches.Patch(color='green', label='Ice speed')
    geostrophic_speed_patch = mpatches.Patch(color='dodgerblue', label='Geostrophic speed')

    plt.legend(handles=[alpha_patch, wind_speed_patch, ice_speed_patch, geostrophic_speed_patch], loc='lower center',
               bbox_to_anchor=(0, -0.2, 1, -0.2), ncol=4, mode='expand', borderaxespad=0, framealpha=0)

    png_filepath = os.path.join(figure_dir_path, 'WGR velocities time series.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')
    plt.close()

    """ Second figure """
    fig, host = plt.subplots(figsize=(16, 4.5))
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    # Offset the right spine of par2.  The ticks and label have already been placed on the right by twinx above.
    par2.spines['right'].set_position(('axes', 1.1))

    # Having been created by twinx, par2 has its frame off, so the line of its detached spine is invisible.  First,
    # activate the frame but make the patch and spines invisible.
    make_patch_spines_invisible(par2)

    # Second, show the right spine.
    par2.spines['right'].set_visible(True)

    host.fill_between(dates, 0, alpha_WGR, color='lightgray')

    p1, = host.plot(dates, alpha_WGR, 'lightgray')
    p2, = par1.plot(dates, 3600*24*365 * w_Ekman_WGR, 'red')
    p3, = par2.plot(dates, 3600*24*365 * w_Ekman_geo_WGR, 'black')

    par2.axhline(linewidth=1, color='black', linestyle='dashed')

    host.set_xlim(dates[0], dates[-1])
    host.set_ylim(0, 1)
    par1.set_ylim(-20, 100)
    par2.set_ylim(-20, 100)

    host.set_xlabel('Date')
    host.set_ylabel(r'Mean sea ice fraction $\alpha$')
    par1.set_ylabel(r'Mean Ekman pumping (no $u_{geo}$) (m/year)', color='red')
    par2.set_ylabel(r'Mean Ekman pumping (with $u_{geo}$) (m/year)')

    # host.yaxis.label.set_color(p1.get_color())
    # par1.yaxis.label.set_color(p2.get_color())
    # par2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    # host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    # lines = [p1, p2, p3]
    # host.legend(lines, [l.get_label() for l in lines])

    plt.title(r'30-day running mean of Ekman pumping (with and without $u_{geo}$) in the Weddell Gyre Region')

    alpha_patch = mpatches.Patch(color='lightgray', label='Sea ice fraction')
    w_Ekman_patch = mpatches.Patch(color='red', label=r'Ekman pumping (no $u_{geo}$)')
    w_Ekman_geo_patch = mpatches.Patch(color='black', label=r'Ekman pumping (with $u_{geo}$)')

    plt.legend(handles=[alpha_patch, w_Ekman_patch, w_Ekman_geo_patch], loc='lower center',
               bbox_to_anchor=(0, -0.2, 1, -0.2), ncol=3, mode='expand', borderaxespad=0, framealpha=0)

    png_filepath = os.path.join(figure_dir_path, 'WGR Ekman pumping time series.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')
    plt.close()


def make_u_geo_climo_fig():
    from os import path
    import netCDF4
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from constants import figure_dir_path, data_dir_path

    climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2011, year_end=2016)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
    #                                      year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM',
    #                                      year_start=2005, year_end=2015)

    lons, lats, climo_tau_x_field = get_field_from_netcdf(climo_filepath, 'tau_x')
    climo_tau_y_field = get_field_from_netcdf(climo_filepath, 'tau_y')[2]
    climo_alpha_field = get_field_from_netcdf(climo_filepath, 'alpha')[2]
    climo_u_geo_field = get_field_from_netcdf(climo_filepath, 'geo_u')[2]
    climo_v_geo_field = get_field_from_netcdf(climo_filepath, 'geo_v')[2]

    # climo_tau_x_field, lons_tau = cartopy.util.add_cyclic_point(climo_tau_x_field, coord=lons)
    # climo_tau_y_field, lons_tau = cartopy.util.add_cyclic_point(climo_tau_y_field, coord=lons)

    import astropy.convolution
    kernel = astropy.convolution.Box2DKernel(10)
    # kernel = astropy.convolution.Gaussian2DKernel(2)
    climo_u_geo_field = astropy.convolution.convolve(climo_u_geo_field, kernel, boundary='wrap')
    climo_v_geo_field = astropy.convolution.convolve(climo_v_geo_field, kernel, boundary='wrap')

    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    # Compute a circle in axes coordinates, which we can use as a boundary for the map. We can pan/zoom as much as we
    # like - the boundary will be permanently circular.
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(1, 2)
    matplotlib.rcParams.update({'font.size': 10})

    """ Plot u_geo """
    crs_sps = ccrs.SouthPolarStereo()
    crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

    ax1 = plt.subplot(121, projection=crs_sps)

    ax1.add_feature(land_50m)
    ax1.add_feature(ice_50m)
    ax1.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax1.set_boundary(circle, transform=ax1.transAxes)

    gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl1.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl1.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im1 = ax1.pcolormesh(lons, lats, climo_u_geo_field, transform=vector_crs, cmap=cmocean.cm.balance,
                         vmin=-0.10, vmax=0.10)
    # im1 = ax1.contourf(lons, lats, climo_u_geo_field, levels=np.arange(-0.10, 0.12, 0.02),
    #                    transform=vector_crs, cmap=cmocean.cm.balance, vmin=-0.10, vmax=0.10, extend='both')

    Q = ax1.quiver(lons[::10], lats[::10], climo_u_geo_field[::10, ::10], climo_v_geo_field[::10, ::10],
                   color='black', pivot='middle', transform=vector_crs, units='width', width=0.002, scale=4)

    plt.quiverkey(Q, 0.33, 0.88, 0.1, r'0.1 m/s', labelpos='E', coordinates='figure',
                  fontproperties={'size': 11}, transform=ax1.transAxes)

    # ax1.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
    #             levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    # ax1.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
    #             colors='green', linewidths=2, transform=vector_crs)

    ax1.add_patch(mpatches.Rectangle(xy=[-60, -68], width=90, height=13, transform=ccrs.PlateCarree(),
                                    edgecolor='gold', linewidth=2, linestyle='solid', facecolor='none'))

    ax1.text(0.49,  1.01,  '0°',   transform=ax1.transAxes)
    ax1.text(1.01,  0.49,  '90°E', transform=ax1.transAxes)
    ax1.text(0.47,  -0.03, '180°', transform=ax1.transAxes)
    ax1.text(-0.09, 0.49,  '90°W', transform=ax1.transAxes)
    ax1.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax1.transAxes)
    ax1.text(0.85,  0.125, '135°E', rotation=-45, transform=ax1.transAxes)
    ax1.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax1.transAxes)
    ax1.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax1.transAxes)

    ax1.text(0.50, 1.05, r'Zonal component $u_{geo}$', fontsize=14, va='bottom', ha='center', rotation='horizontal',
             rotation_mode='anchor', transform=ax1.transAxes)

    clb = fig.colorbar(im1, ax=ax1, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'm/s')

    # zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
    # ice_edge_patch = mpatches.Patch(color='black', label=r'15% ice edge')
    # plt.legend(handles=[zero_stress_line_patch, ice_edge_patch], loc='lower center',
    #            bbox_to_anchor=(0, -0.15, 1, -0.15), ncol=1, mode='expand', borderaxespad=0, framealpha=0)

    plt.suptitle(r'Southern Ocean geostrophic current velocity $\mathbf{u}_{geo}$', fontsize=16)

    """ Plot v_geo """
    crs_sps = ccrs.SouthPolarStereo()
    crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

    ax2 = plt.subplot(122, projection=crs_sps)

    ax2.add_feature(land_50m)
    ax2.add_feature(ice_50m)
    ax2.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    ax2.set_boundary(circle, transform=ax2.transAxes)

    gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl2.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl2.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    im2 = ax2.pcolormesh(lons, lats, climo_v_geo_field, transform=vector_crs, cmap=cmocean.cm.balance,
                         vmin=-0.10, vmax=0.10)

    Q = ax2.quiver(lons[::10], lats[::10], climo_u_geo_field[::10, ::10], climo_v_geo_field[::10, ::10],
                   pivot='middle', transform=vector_crs, units='width', width=0.002, scale=4)

    # plt.quiverkey(Q, 0.33, 0.88, 0.1, r'0.1 m/s', labelpos='E', coordinates='figure',
    #               fontproperties={'size': 11}, transform=ax1.transAxes)

    # ax2.contour(lons, lats, np.ma.array(climo_alpha_field, mask=np.isnan(climo_alpha_field)),
    #             levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    # ax2.contour(lons, lats, np.ma.array(climo_tau_x_field, mask=np.isnan(climo_alpha_field)), levels=[0],
    #             colors='green', linewidths=2, transform=vector_crs)

    ax2.add_patch(mpatches.Rectangle(xy=[-60, -68], width=90, height=13, transform=ccrs.PlateCarree(),
                                     edgecolor='gold', linewidth=2, linestyle='solid', facecolor='none'))

    ax2.text(0.49,  1.01,  '0°',   transform=ax2.transAxes)
    ax2.text(1.01,  0.49,  '90°E', transform=ax2.transAxes)
    ax2.text(0.47,  -0.03, '180°', transform=ax2.transAxes)
    ax2.text(-0.09, 0.49,  '90°W', transform=ax2.transAxes)
    ax2.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax2.transAxes)
    ax2.text(0.85,  0.125, '135°E', rotation=-45, transform=ax2.transAxes)
    ax2.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax2.transAxes)
    ax2.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax2.transAxes)

    ax2.text(0.50, 1.05, r'Meridional component $v_{geo}$', fontsize=14, va='bottom', ha='center',
             rotation='horizontal', rotation_mode='anchor', transform=ax2.transAxes)

    clb = fig.colorbar(im2, ax=ax2, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'm/s')

    png_filepath = os.path.join(figure_dir_path, 'u_geo_climo_figure.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False)


def make_melting_freezing_rate_term_plots():
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from constants import output_dir_path, figure_dir_path

    # climo_filepath = get_netCDF_filepath(field_type='climo', year_start=2011, year_end=2016)

    climo_filepath = os.path.join(output_dir_path, 'melting_freezing_rate_2011-01-01_2016-12-31.nc')

    # feb_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 2, 1),
    #                                          year_start=2005, year_end=2015)
    # sep_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 9, 1),
    #                                          year_start=2005, year_end=2015)

    lons, lats, div_field = get_field_from_netcdf(climo_filepath, 'div')

    Ekman_term_field = get_field_from_netcdf(climo_filepath, 'Ekman_term')[2]
    geo_term_field = get_field_from_netcdf(climo_filepath, 'geo_term')[2]
    diffusion_term_field = get_field_from_netcdf(climo_filepath, 'diffusion_term')[2]

    # climo_u_geo_field, lons_geo = cartopy.util.add_cyclic_point(climo_u_geo_field, coord=lons)
    # climo_v_geo_field, lons_geo = cartopy.util.add_cyclic_point(climo_v_geo_field, coord=lons)

    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 12))
    matplotlib.rcParams.update({'font.size': 10})

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax1 = plt.subplot(221, projection=ccrs.SouthPolarStereo())

    ax1.set_boundary(circle, transform=ax1.transAxes)
    ax1.add_feature(land_50m)
    ax1.add_feature(ice_50m)
    ax1.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    im1 = ax1.pcolormesh(lons, lats, div_field, transform=vector_crs, cmap=cmocean.cm.balance,
                         vmin=-1, vmax=1)

    ax1.text(0.49,  1.01,  '0°',   transform=ax1.transAxes)
    ax1.text(1.01,  0.49,  '90°E', transform=ax1.transAxes)
    ax1.text(0.47,  -0.03, '180°', transform=ax1.transAxes)
    ax1.text(-0.09, 0.49,  '90°W', transform=ax1.transAxes)
    ax1.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax1.transAxes)
    ax1.text(0.85,  0.125, '135°E', rotation=-45, transform=ax1.transAxes)
    ax1.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax1.transAxes)
    ax1.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax1.transAxes)

    ax1.text(0.505, 1.05, r'Sea ice advection $\nabla\cdot(\alpha h_{ice} \mathbf{u}_{ice})$', fontsize=14,
             va='bottom', ha='center', rotation='horizontal', rotation_mode='anchor', transform=ax1.transAxes)

    # plt.suptitle(r'Figure 3: Geostrophic current $\mathbf{u}_g$ and ice drift $\mathbf{u}_i$ observations, '
    #              'winter (JAS) mean', fontsize=16)

    ax2 = plt.subplot(222, projection=ccrs.SouthPolarStereo())

    ax2.set_boundary(circle, transform=ax2.transAxes)
    ax2.add_feature(land_50m)
    ax2.add_feature(ice_50m)
    ax2.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    im2 = ax2.pcolormesh(lons, lats, Ekman_term_field, transform=vector_crs, cmap=cmocean.cm.balance,
                         vmin=-1, vmax=1)

    ax2.text(0.49,  1.01,  '0°',   transform=ax2.transAxes)
    ax2.text(1.01,  0.49,  '90°E', transform=ax2.transAxes)
    ax2.text(0.47,  -0.03, '180°', transform=ax2.transAxes)
    ax2.text(-0.09, 0.49,  '90°W', transform=ax2.transAxes)
    ax2.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax2.transAxes)
    ax2.text(0.85,  0.125, '135°E', rotation=-45, transform=ax2.transAxes)
    ax2.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax2.transAxes)
    ax2.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax2.transAxes)

    ax1.text(0.505, 1.05, r'Ekman advection $\frac{\mathcal{U}_{Ek} \cdot \nabla S}{S}$', fontsize=14,
             va='bottom', ha='center', rotation='horizontal', rotation_mode='anchor', transform=ax2.transAxes)

    ax3 = plt.subplot(223, projection=ccrs.SouthPolarStereo())

    ax3.set_boundary(circle, transform=ax3.transAxes)
    ax3.add_feature(land_50m)
    ax3.add_feature(ice_50m)
    ax3.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    im3 = ax3.pcolormesh(lons, lats, geo_term_field, transform=vector_crs, cmap=cmocean.cm.balance,
                         vmin=-1, vmax=1)

    ax3.text(0.49,  1.01,  '0°',   transform=ax3.transAxes)
    ax3.text(1.01,  0.49,  '90°E', transform=ax3.transAxes)
    ax3.text(0.47,  -0.03, '180°', transform=ax3.transAxes)
    ax3.text(-0.09, 0.49,  '90°W', transform=ax3.transAxes)
    ax3.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax3.transAxes)
    ax3.text(0.85,  0.125, '135°E', rotation=-45, transform=ax3.transAxes)
    ax3.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax3.transAxes)
    ax3.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax3.transAxes)

    ax3.text(0.50, 1.05, r'Geostrophic advection $(D_e / S) \mathbf{u}_{geo} \cdot \nabla S$', fontsize=14, va='bottom', ha='center',
             rotation='horizontal', rotation_mode='anchor', transform=ax3.transAxes)

    ax4 = plt.subplot(224, projection=ccrs.SouthPolarStereo())

    ax4.set_boundary(circle, transform=ax4.transAxes)
    ax4.add_feature(land_50m)
    ax4.add_feature(ice_50m)
    ax4.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    im4 = ax4.pcolormesh(lons, lats, diffusion_term_field, transform=vector_crs, cmap=cmocean.cm.balance,
                         vmin=-1, vmax=1)

    ax4.text(0.49,  1.01,  '0°',   transform=ax4.transAxes)
    ax4.text(1.01,  0.49,  '90°E', transform=ax4.transAxes)
    ax4.text(0.47,  -0.03, '180°', transform=ax4.transAxes)
    ax4.text(-0.09, 0.49,  '90°W', transform=ax4.transAxes)
    ax4.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax4.transAxes)
    ax4.text(0.85,  0.125, '135°E', rotation=-45, transform=ax4.transAxes)
    ax4.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax4.transAxes)
    ax4.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax4.transAxes)

    ax4.text(0.50, 1.05, r'Diffusion $\kappa \nabla^2 S$', fontsize=14, va='bottom', ha='center',
             rotation='horizontal', rotation_mode='anchor', transform=ax4.transAxes)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    clb = fig.colorbar(im4, cax=cbar_ax, extend='both')
    clb.ax.set_title(r'm/year')

    png_filepath = os.path.join(figure_dir_path, 'melting_freezing_rate_terms.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False)


def plot_streamwise_velocity_integrals():
    import constants
    from constants import figure_dir_path, D_e
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from utils import get_northward_zero_zonal_stress_line, get_northward_ice_edge, get_coast_coordinates

    nogeo_output_dir_path = 'E:\\output\\'
    geo_output_dir_path = 'C:\\Users\\Ali\\Downloads\\output\\'

    constants.output_dir_path = nogeo_output_dir_path
    climo_nogeo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
                                               year_start=2005, year_end=2015)

    lons, lats, tau_x_nogeo_field = get_field_from_netcdf(climo_nogeo_filepath, 'tau_x')

    constants.output_dir_path = geo_output_dir_path
    climo_geo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
                                             year_start=2011, year_end=2016)

    u_ice_field = get_field_from_netcdf(climo_geo_filepath, 'ice_u')[2]
    v_ice_field = get_field_from_netcdf(climo_geo_filepath, 'ice_v')[2]
    u_geo_field = get_field_from_netcdf(climo_geo_filepath, 'geo_u')[2]
    v_geo_field = get_field_from_netcdf(climo_geo_filepath, 'geo_v')[2]

    contour_coordinates = np.empty((len(lats), len(lons)))
    contour_coordinates[:] = np.nan

    tau_x_lons, tau_x_lats = get_northward_zero_zonal_stress_line(climo_nogeo_filepath)
    alpha_lons, alpha_lats = get_northward_ice_edge(climo_nogeo_filepath)
    coast_lons, coast_lats = get_coast_coordinates(climo_nogeo_filepath)

    ice_mag_field = np.sqrt(u_ice_field*u_ice_field + v_ice_field*v_ice_field)
    geo_mag_field = np.sqrt(u_geo_field*u_geo_field + v_geo_field*v_geo_field)

    for i, lon in enumerate(lons):
        if alpha_lats[i] > tau_x_lats[i] > coast_lats[i]:
            lat_0 = coast_lats[i]
            lat_h = tau_x_lats[i]  # lat_h is short for lat_half ~ lat_1/2
            lat_1 = alpha_lats[i]

            for j, lat in enumerate(lats):
                if lat < lat_0 or lat > lat_1:
                    contour_coordinates[j][i] = np.nan
                elif lat_0 <= lat <= lat_h:
                    contour_coordinates[j][i] = (lat - lat_0) / (2 * (lat_h - lat_0))
                elif lat_h <= lat <= lat_1:
                    contour_coordinates[j][i] = 0.5 + ((lat - lat_h) / (2 * (lat_1 - lat_h)))

    c_bins = np.linspace(0, 1, 41)[:-1]
    delta_c = c_bins[1] - c_bins[0]
    c_bins = c_bins + (delta_c / 2)

    u_ice_cavg = np.zeros(c_bins.shape)
    v_ice_cavg = np.zeros(c_bins.shape)
    u_geo_cavg = np.zeros(c_bins.shape)
    v_geo_cavg = np.zeros(c_bins.shape)
    ice_mag_cavg = np.zeros(c_bins.shape)
    geo_mag_cavg = np.zeros(c_bins.shape)

    for i in range(len(c_bins)):
        c = c_bins[i]
        c_low = c - (delta_c / 2)
        c_high = c + (delta_c / 2)

        c_in_range = np.logical_and(contour_coordinates > c_low, contour_coordinates < c_high)

        u_ice_cavg[i] = np.nanmean(u_ice_field[c_in_range])
        v_ice_cavg[i] = np.nanmean(v_ice_field[c_in_range])
        u_geo_cavg[i] = np.nanmean(u_geo_field[c_in_range])
        v_geo_cavg[i] = np.nanmean(v_geo_field[c_in_range])
        ice_mag_cavg[i] = np.nanmean(ice_mag_field[c_in_range])
        geo_mag_cavg[i] = np.nanmean(geo_mag_field[c_in_range])

    fig = plt.figure(figsize=(20, 6))

    ax = fig.add_subplot(131)
    ax.plot(c_bins, u_ice_cavg, label=r'$u_{ice}$')
    ax.plot(c_bins, u_geo_cavg, label=r'$u_{geo}$')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], minor=False)
    ax.set_xlabel('streamwise coordinate', fontsize='large')
    ax.set_ylabel(r'Velocity (m/s)', fontsize='large')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(linestyle='--')
    ax.legend(fontsize='large')

    ax = fig.add_subplot(132)
    ax.plot(c_bins, v_ice_cavg, label=r'$v_{ice}$')
    ax.plot(c_bins, v_geo_cavg, label=r'$v_{geo }$')
    ax.set_xlabel('streamwise coordinate', fontsize='large')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], minor=False)
    ax.set_ylabel('Velocity (m/s)', fontsize='large')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(linestyle='--')
    ax.legend(fontsize='large')

    ax = fig.add_subplot(133)
    ax.plot(c_bins, ice_mag_cavg, label=r'$|\mathbf{u}_{ice}|$')
    ax.plot(c_bins, geo_mag_cavg, label=r'$|\mathbf{u}_{geo}|$')
    ax.set_xlabel('streamwise coordinate', fontsize='large')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], minor=False)
    ax.set_ylabel('Speed (m/s)', fontsize='large')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(linestyle='--')
    ax.legend(fontsize='large')

    plt.suptitle(r'Streamwise averages, winter (JAS) mean', fontsize=16)

    png_filepath = os.path.join(figure_dir_path, 'ice_geo_streamwise_average.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')


def plot_ice_ocean_pumping():
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from constants import figure_dir_path

    nc_filepath = "C:\\Users\\Ali\\Downloads\\output\\ice_ocean_govenor_2011-07-01_2016-09-30.nc"

    JAS_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
                                       year_start=2005, year_end=2015)
    feb_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 2, 1),
                                             year_start=2005, year_end=2015)
    sep_climo_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2005, 9, 1),
                                             year_start=2005, year_end=2015)

    lons, lats, tau_x_JAS_field = get_field_from_netcdf(JAS_filepath, 'tau_x')
    w_Ek_nogeo_field = get_field_from_netcdf(nc_filepath, 'w_Ek_nogeo')[2]
    w_ig_field = get_field_from_netcdf(nc_filepath, 'w_ig')[2]
    w_Ek_geo_field = get_field_from_netcdf(nc_filepath, 'w_Ek_geo')[2]
    tau_ig_x_field = get_field_from_netcdf(nc_filepath, 'tau_ig_x')[2]
    tau_ig_y_field = get_field_from_netcdf(nc_filepath, 'tau_ig_y')[2]

    feb_climo_alpha_field = get_field_from_netcdf(feb_climo_filepath, 'alpha')[2]
    sep_climo_alpha_field = get_field_from_netcdf(sep_climo_filepath, 'alpha')[2]

    import astropy.convolution
    kernel = astropy.convolution.Box2DKernel(5)
    w_Ek_nogeo_field = astropy.convolution.convolve(w_Ek_nogeo_field, kernel, boundary='wrap')
    w_ig_field = astropy.convolution.convolve(w_ig_field, kernel, boundary='wrap')
    tau_ig_x_field = astropy.convolution.convolve(tau_ig_x_field, kernel, boundary='wrap')
    tau_ig_y_field = astropy.convolution.convolve(tau_ig_y_field, kernel, boundary='wrap')

    # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
    # the smaller plots.
    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='dimgray',
                                                   linewidth=0)
    ice_50m = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', edgecolor='face',
                                                  facecolor='darkgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 9))
    matplotlib.rcParams.update({'font.size': 10})

    crs_sps = ccrs.SouthPolarStereo()
    crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363
    ax = plt.subplot(111, projection=crs_sps)

    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.add_feature(land_50m)
    ax.add_feature(ice_50m)
    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

    # im = ax.contourf(lons, lats, 365*24*3600 * w_Ek_geo_field, transform=vector_crs, cmap=cmocean.cm.balance,
    #                  vmin=-100, vmax=100, levels=np.linspace(-100, 100, 16), extend='both')
    im = ax.contourf(lons, lats, tau_ig_y_field, transform=vector_crs, cmap=cmocean.cm.balance,
                     vmin=-0.15, vmax=0.15, levels=np.linspace(-0.15, 0.15, 16), extend='both')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='black', alpha=.8, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl.ylocator = mticker.FixedLocator([-80, -70, -60, -50])

    ax.contour(lons, lats, np.ma.array(tau_x_JAS_field, mask=np.isnan(sep_climo_alpha_field)),
               levels=[0], colors='green', linewidths=2, transform=vector_crs)
    ax.contour(lons, lats, np.ma.array(feb_climo_alpha_field, mask=np.isnan(feb_climo_alpha_field)),
               levels=[0.15], colors='black', linewidths=2, transform=vector_crs)
    ax.contour(lons, lats, np.ma.array(sep_climo_alpha_field, mask=np.isnan(sep_climo_alpha_field)),
               levels=[0.15], colors='black', linewidths=2, transform=vector_crs)

    ax.text(0.49,  1.01,  '0°',   transform=ax.transAxes)
    ax.text(1.01,  0.49,  '90°E', transform=ax.transAxes)
    ax.text(0.47,  -0.03, '180°', transform=ax.transAxes)
    ax.text(-0.09, 0.49,  '90°W', transform=ax.transAxes)
    ax.text(0.855, 0.895, '45°E',  rotation=45,  transform=ax.transAxes)
    ax.text(0.85,  0.125, '135°E', rotation=-45, transform=ax.transAxes)
    ax.text(0.07,  0.90,  '45°W',  rotation=-45, transform=ax.transAxes)
    ax.text(0.06,  0.13,  '135°W', rotation=45,  transform=ax.transAxes)

    # clb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.1, ticks=np.linspace(-100, 100, 6))
    # clb.ax.set_title(r'm/year')
    clb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.1, ticks=np.linspace(-0.15, 0.15, 6))
    clb.ax.set_title(r'$N/m^2$')

    # plt.gcf().text(0.5, 0.95, 'Ekman pumping due to wind+ice, winter (JAS) mean', fontsize=14)
    # plt.gcf().text(0.5, 0.95, 'Ekman pumping due to geostrophic currents, winter (JAS) mean', fontsize=14)
    # plt.gcf().text(0.5, 0.95, 'Ekman pumping total, winter (JAS) mean', fontsize=14)
    plt.gcf().text(0.5, 0.95, 'Meridional ice-ocean stress due to geostrophic currents, winter (JAS) mean', fontsize=14)

    zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
    ice_edge_patch = mpatches.Patch(color='black', label=r'Feb (min) and Sep (max) 15% sea ice edge')
    plt.legend(handles=[zero_stress_line_patch, ice_edge_patch], loc='lower center', fontsize=13,
               bbox_to_anchor=(0, -0.1, 1, -0.1), ncol=1, mode='expand', borderaxespad=0, framealpha=0)

    png_filepath = os.path.join(figure_dir_path, 'tau_ig_y.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')


def antarctic_divergence_time_series():
    import pickle
    import constants
    from utils import get_northward_zero_zonal_stress_line, get_northward_ice_edge, get_coast_coordinates

    nogeo_output_dir_path = 'E:\\output\\'
    geo_output_dir_path = 'C:\\Users\\Ali\\Downloads\\output\\'
    govenor_output_dir_path = "E:\\output\\ice_ocean_govenor\\"

    constants.output_dir_path = nogeo_output_dir_path
    climo_nogeo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
                                               year_start=2005, year_end=2015)

    lons, lats, tau_x_nogeo_field = get_field_from_netcdf(climo_nogeo_filepath, 'tau_x')

    contour_coordinates = np.empty((len(lats), len(lons)))
    contour_coordinates[:] = np.nan

    tau_x_lons, tau_x_lats = get_northward_zero_zonal_stress_line(climo_nogeo_filepath)
    alpha_lons, alpha_lats = get_northward_ice_edge(climo_nogeo_filepath)
    coast_lons, coast_lats = get_coast_coordinates(climo_nogeo_filepath)

    for i, lon in enumerate(lons):
        if alpha_lats[i] > tau_x_lats[i] > coast_lats[i]:
            lat_0 = coast_lats[i]
            lat_h = tau_x_lats[i]  # lat_h is short for lat_half ~ lat_1/2
            lat_1 = alpha_lats[i]

            for j, lat in enumerate(lats):
                if lat < lat_0 or lat > lat_1:
                    contour_coordinates[j][i] = np.nan
                elif lat_0 <= lat <= lat_h:
                    contour_coordinates[j][i] = (lat - lat_0) / (2 * (lat_h - lat_0))
                elif lat_h <= lat <= lat_1:
                    contour_coordinates[j][i] = 0.5 + ((lat - lat_h) / (2 * (lat_1 - lat_h)))

    c_bins = np.linspace(0, 1, 41)[:-1]
    delta_c = c_bins[1] - c_bins[0]
    c_bins = c_bins + (delta_c / 2)

    year_start = 2011
    year_end = 2015
    dates = date_range(datetime.date(year_start, 1, 1), datetime.date(year_end, 12, 31))

    n_days = len(dates)

    pickle_filepath = os.path.join(figure_dir_path, 'Antarctic_Divergence_time_series.pickle')

    pickle_found = False
    try:
        with open(pickle_filepath, 'rb') as f:
            AD_time_series_dict = pickle.load(f)
            logger.info('Previous computation found. Loading {:s}...'.format(pickle_filepath))
            pickle_found = True
    except OSError:
        logger.info('Computing Antarctic Divergence time series...')

    if pickle_found:
        alpha_AD = AD_time_series_dict['alpha']
        u_wind_AD = AD_time_series_dict['u_wind']
        v_wind_AD = AD_time_series_dict['v_wind']
        wind_speed_AD = AD_time_series_dict['wind_speed']
        u_ice_AD = AD_time_series_dict['u_ice']
        v_ice_AD = AD_time_series_dict['v_ice']
        ice_speed_AD = AD_time_series_dict['ice_speed']
        u_geo_AD = AD_time_series_dict['u_geo']
        v_geo_AD = AD_time_series_dict['v_geo']
        geo_speed_AD = AD_time_series_dict['geo_speed']
        w_a_AD = AD_time_series_dict['w_a']
        w_A_AD = AD_time_series_dict['w_A']
        w_Ek_nogeo_AD = AD_time_series_dict['w_Ek_nogeo']
        w_Ek_geo_AD = AD_time_series_dict['w_Ek_geo']
        w_ig_AD = AD_time_series_dict['w_ig']
        w_i_AD = AD_time_series_dict['w_i']
        w_i0_AD = AD_time_series_dict['w_i0']

    else:
        alpha_AD = np.zeros(n_days)

        u_wind_AD = np.zeros(n_days)
        v_wind_AD = np.zeros(n_days)
        wind_speed_AD = np.zeros(n_days)

        u_ice_AD = np.zeros(n_days)
        v_ice_AD = np.zeros(n_days)
        ice_speed_AD = np.zeros(n_days)

        u_geo_AD = np.zeros(n_days)
        v_geo_AD = np.zeros(n_days)
        geo_speed_AD = np.zeros(n_days)

        w_a_AD = np.zeros(n_days)
        w_A_AD = np.zeros(n_days)
        w_Ek_nogeo_AD = np.zeros(n_days)
        w_Ek_geo_AD = np.zeros(n_days)
        w_ig_AD = np.zeros(n_days)
        w_i_AD = np.zeros(n_days)
        w_i0_AD = np.zeros(n_days)

        for d, date in enumerate(dates):
            constants.output_dir_path = nogeo_output_dir_path

            try:
                climo_filepath_nogeo = get_netCDF_filepath(field_type='daily', date=date)
                lons, lats, _ = get_field_from_netcdf(climo_filepath_nogeo, 'Ekman_w')
            except Exception as e:
                logger.error('{}'.format(e))
                logger.warning('{:s} not found. Proceeding without it...'.format(climo_filepath_nogeo))
                n_days = n_days - 1  # Must account for lost day if no data available for that day.
                continue

            constants.output_dir_path = geo_output_dir_path

            try:
                climo_filepath_geo = get_netCDF_filepath(field_type='daily', date=date)
                _, _, _ = get_field_from_netcdf(climo_filepath_geo, 'Ekman_w')
            except OSError as e:
                logger.error('{}'.format(e))
                logger.warning('{:s} not found. Proceeding without it...'.format(climo_filepath_geo))
                n_days = n_days - 1  # Must account for lost day if no data available for that day.
                continue

            try:
                govenor_filename = "ice_ocean_govenor_{:}.nc".format(date)
                govenor_filepath = os.path.join(govenor_output_dir_path, govenor_filename)
                _, _, _ = get_field_from_netcdf(govenor_filepath, 'alpha')
            except OSError as e:
                logger.error('{}'.format(e))
                logger.warning('{:s} not found. Proceeding without it...'.format(climo_filepath_geo))
                n_days = n_days - 1  # Must account for lost day if no data available for that day.
                continue

            logger.info('Averaging {:%b %d, %Y}...'.format(date))

            alpha_field = get_field_from_netcdf(climo_filepath_geo, 'alpha')[2]

            u_wind_field = get_field_from_netcdf(climo_filepath_geo, 'wind_u')[2]
            v_wind_field = get_field_from_netcdf(climo_filepath_geo, 'wind_v')[2]
            wind_speed_field = np.sqrt(u_wind_field*u_wind_field + v_wind_field*v_wind_field)

            u_ice_field = get_field_from_netcdf(climo_filepath_geo, 'ice_u')[2]
            v_ice_field = get_field_from_netcdf(climo_filepath_geo, 'ice_v')[2]
            ice_speed_field = np.sqrt(u_ice_field*u_ice_field + v_ice_field*v_ice_field)

            u_geo_field = get_field_from_netcdf(climo_filepath_geo, 'geo_u')[2]
            v_geo_field = get_field_from_netcdf(climo_filepath_geo, 'geo_v')[2]
            geo_speed_field = np.sqrt(u_geo_field*u_geo_field + v_geo_field*v_geo_field)

            w_a_field = get_field_from_netcdf(govenor_filepath, 'w_a')[2]
            w_A_field = get_field_from_netcdf(govenor_filepath, 'w_A')[2]
            w_Ek_geo_field = get_field_from_netcdf(govenor_filepath, 'w_Ek_geo')[2]
            w_Ek_nogeo_field = get_field_from_netcdf(govenor_filepath, 'w_Ek_nogeo')[2]
            w_i_field = get_field_from_netcdf(govenor_filepath, 'w_i')[2]
            w_i0_field = get_field_from_netcdf(govenor_filepath, 'w_i0')[2]
            w_ig_field = get_field_from_netcdf(govenor_filepath, 'w_ig')[2]

            import astropy.convolution
            kernel = astropy.convolution.Box2DKernel(4)

            c_south_of_AD = np.logical_and(contour_coordinates >= 0, contour_coordinates < 0.5)

            alpha_AD[d] = np.nanmean(alpha_field[c_south_of_AD])

            u_wind_AD[d] = np.nanmean(u_wind_field[c_south_of_AD])
            v_wind_AD[d] = np.nanmean(v_wind_field[c_south_of_AD])
            wind_speed_AD[d] = np.nanmean(wind_speed_field[c_south_of_AD])

            u_ice_AD[d] = np.nanmean(u_ice_field[c_south_of_AD])
            v_ice_AD[d] = np.nanmean(v_ice_field[c_south_of_AD])
            ice_speed_AD[d] = np.nanmean(ice_speed_field[c_south_of_AD])

            u_geo_AD[d] = np.nanmean(u_geo_field[c_south_of_AD])
            v_geo_AD[d] = np.nanmean(v_geo_field[c_south_of_AD])
            geo_speed_AD[d] = np.nanmean(geo_speed_field[c_south_of_AD])

            w_a_AD[d] = np.nanmean(w_a_field[c_south_of_AD])
            w_A_AD[d] = np.nanmean(w_A_field[c_south_of_AD])
            w_Ek_geo_AD[d] = np.nanmean(w_Ek_geo_field[c_south_of_AD])
            w_Ek_nogeo_AD[d] = np.nanmean(w_Ek_nogeo_field[c_south_of_AD])
            w_i_AD[d] = np.nanmean(w_i_field[c_south_of_AD])
            w_i0_AD[d] = np.nanmean(w_i0_field[c_south_of_AD])
            w_ig_AD[d] = np.nanmean(w_ig_field[c_south_of_AD])

    with open(pickle_filepath, 'wb') as f:
        AD_time_series_dict = {
            'alpha': alpha_AD,
            'u_wind': u_wind_AD,
            'v_wind': v_wind_AD,
            'wind_speed': wind_speed_AD,
            'u_ice': u_ice_AD,
            'v_ice': v_ice_AD,
            'ice_speed': ice_speed_AD,
            'u_geo': u_geo_AD,
            'v_geo': v_geo_AD,
            'geo_speed': geo_speed_AD,
            'w_a': w_a_AD,
            'w_A': w_A_AD,
            'w_Ek_nogeo': w_Ek_nogeo_AD,
            'w_Ek_geo': w_Ek_geo_AD,
            'w_ig': w_ig_AD,
            'w_i': w_i_AD,
            'w_i0': w_i0_AD
        }
        pickle.dump(AD_time_series_dict, f, pickle.HIGHEST_PROTOCOL)

    import astropy.convolution
    kernel = astropy.convolution.Box1DKernel(30)

    alpha_AD = astropy.convolution.convolve(alpha_AD, kernel, boundary='extend')

    u_wind_AD = astropy.convolution.convolve(u_wind_AD, kernel, boundary='extend')
    v_wind_AD = astropy.convolution.convolve(v_wind_AD, kernel, boundary='extend')
    wind_speed_AD = astropy.convolution.convolve(wind_speed_AD, kernel, boundary='extend')

    u_ice_AD = astropy.convolution.convolve(u_ice_AD, kernel, boundary='extend')
    v_ice_AD = astropy.convolution.convolve(v_ice_AD, kernel, boundary='extend')
    ice_speed_AD = astropy.convolution.convolve(ice_speed_AD, kernel, boundary='extend')

    u_geo_AD = astropy.convolution.convolve(u_geo_AD, kernel, boundary='extend')
    v_geo_AD = astropy.convolution.convolve(v_geo_AD, kernel, boundary='extend')
    geo_speed_AD = astropy.convolution.convolve(geo_speed_AD, kernel, boundary='extend')

    w_a_AD = astropy.convolution.convolve(w_a_AD, kernel, boundary='extend')
    w_A_AD = astropy.convolution.convolve(w_A_AD, kernel, boundary='extend')
    w_Ek_nogeo_AD = astropy.convolution.convolve(w_Ek_nogeo_AD, kernel, boundary='extend')
    w_Ek_geo_AD = astropy.convolution.convolve(w_Ek_geo_AD, kernel, boundary='extend')
    w_ig_AD = astropy.convolution.convolve(w_ig_AD, kernel, boundary='extend')
    w_i_AD = astropy.convolution.convolve(w_i_AD, kernel, boundary='extend')
    w_i0_AD = astropy.convolution.convolve(w_i0_AD, kernel, boundary='extend')

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig, host = plt.subplots(figsize=(16, 4.5))
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    # Offset the right spine of par2.  The ticks and label have already been placed on the right by twinx above.
    par2.spines['right'].set_position(('axes', 1.1))

    # Having been created by twinx, par2 has its frame off, so the line of its detached spine is invisible.  First,
    # activate the frame but make the patch and spines invisible.
    make_patch_spines_invisible(par2)

    # Second, show the right spine.
    par2.spines['right'].set_visible(True)

    host.fill_between(dates, 0, alpha_AD, color='lightgray')

    p1, = host.plot(dates, alpha_AD, 'lightgray')
    p2, = par1.plot(dates, u_wind_AD, 'red')
    p3, = par2.plot(dates, u_ice_AD, 'green')
    p4, = par2.plot(dates, u_geo_AD, 'dodgerblue')

    host.set_xlim(dates[0], dates[-1])
    host.set_ylim(0, 1)
    par1.set_ylim(-3, 3)
    par2.set_ylim(-0.10, 0.10)

    host.set_xlabel('Date')
    host.set_ylabel(r'Mean sea ice fraction $\alpha$')
    # par1.set_ylabel('Mean wind speed (m/s)', color='red')
    # par2.set_ylabel(r'Mean ice and geostrophic speed (m/s)')
    par1.set_ylabel('Mean zonal wind velocity (m/s)', color='red')
    par2.set_ylabel(r'Mean zonal ice and geostrophic velocity (m/s)')

    # host.yaxis.label.set_color(p1.get_color())
    # par1.yaxis.label.set_color(p2.get_color())
    # par2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    # host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    # par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    # lines = [p1, p2, p3]
    # host.legend(lines, [l.get_label() for l in lines])

    plt.title('30-day running mean of sea ice fraction, wind, ice, and geostrophic speeds south of the Antarctic Divergence')

    alpha_patch = mpatches.Patch(color='lightgray', label='Sea ice fraction')
    wind_speed_patch = mpatches.Patch(color='red', label='Wind speed')
    ice_speed_patch = mpatches.Patch(color='green', label='Ice speed')
    geostrophic_speed_patch = mpatches.Patch(color='dodgerblue', label='Geostrophic speed')

    plt.legend(handles=[alpha_patch, wind_speed_patch, ice_speed_patch, geostrophic_speed_patch], loc='lower center',
               bbox_to_anchor=(0, -0.2, 1, -0.2), ncol=4, mode='expand', borderaxespad=0, framealpha=0)

    png_filepath = os.path.join(figure_dir_path, 'Antarctic divergence velocity time series.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')
    plt.close()

    """ Second figure """
    fig, host = plt.subplots(figsize=(16, 4.5))
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    # Offset the right spine of par2.  The ticks and label have already been placed on the right by twinx above.
    par2.spines['right'].set_position(('axes', 1.1))

    # Having been created by twinx, par2 has its frame off, so the line of its detached spine is invisible.  First,
    # activate the frame but make the patch and spines invisible.
    make_patch_spines_invisible(par2)

    # Second, show the right spine.
    par2.spines['right'].set_visible(True)

    host.fill_between(dates, 0, alpha_AD, color='lightgray')

    p1, = host.plot(dates, alpha_AD, 'lightgray')
    p2, = par1.plot(dates, 3600*24*365 * w_Ek_nogeo_AD, 'red')
    p3, = par2.plot(dates, 3600*24*365 * w_Ek_geo_AD, 'black')

    par2.axhline(linewidth=1, color='black', linestyle='dashed')

    host.set_xlim(dates[0], dates[-1])
    host.set_ylim(0, 1)
    par1.set_ylim(-60, 100)
    par2.set_ylim(-60, 100)

    host.set_xlabel('Date')
    host.set_ylabel(r'Mean sea ice fraction $\alpha$')
    par1.set_ylabel(r'Mean Ekman pumping (no $u_{geo}$) (m/year)', color='red')
    par2.set_ylabel(r'Mean Ekman pumping (with $u_{geo}$) (m/year)')

    # host.yaxis.label.set_color(p1.get_color())
    # par1.yaxis.label.set_color(p2.get_color())
    # par2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    # host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    # lines = [p1, p2, p3]
    # host.legend(lines, [l.get_label() for l in lines])

    plt.title(r'30-day running mean of Ekman pumping (with and without $u_{geo}$) south of the Antarctic Divergence')

    alpha_patch = mpatches.Patch(color='lightgray', label='Sea ice fraction')
    w_Ekman_patch = mpatches.Patch(color='red', label=r'Ekman pumping (no $u_{geo}$)')
    w_Ekman_geo_patch = mpatches.Patch(color='black', label=r'Ekman pumping (with $u_{geo}$)')

    plt.legend(handles=[alpha_patch, w_Ekman_patch, w_Ekman_geo_patch], loc='lower center',
               bbox_to_anchor=(0, -0.2, 1, -0.2), ncol=3, mode='expand', borderaxespad=0, framealpha=0)

    png_filepath = os.path.join(figure_dir_path, 'Antarctic Divergence Ekman pumping time series.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')
    plt.close()


def ice_ocean_govenor_monthly_climo_barchart():
    import pickle
    import constants
    from utils import get_northward_zero_zonal_stress_line, get_northward_ice_edge, get_coast_coordinates

    climo_nogeo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
                                               year_start=2011, year_end=2015)

    lons, lats, tau_x_nogeo_field = get_field_from_netcdf(climo_nogeo_filepath, 'tau_x')

    contour_coordinates = np.empty((len(lats), len(lons)))
    contour_coordinates[:] = np.nan

    tau_x_lons, tau_x_lats = get_northward_zero_zonal_stress_line(climo_nogeo_filepath)
    alpha_lons, alpha_lats = get_northward_ice_edge(climo_nogeo_filepath)
    coast_lons, coast_lats = get_coast_coordinates(climo_nogeo_filepath)

    for i, lon in enumerate(lons):
        if alpha_lats[i] > tau_x_lats[i] > coast_lats[i]:
            lat_0 = coast_lats[i]
            lat_h = tau_x_lats[i]  # lat_h is short for lat_half ~ lat_1/2
            lat_1 = alpha_lats[i]

            for j, lat in enumerate(lats):
                if lat < lat_0 or lat > lat_1:
                    contour_coordinates[j][i] = np.nan
                elif lat_0 <= lat <= lat_h:
                    contour_coordinates[j][i] = (lat - lat_0) / (2 * (lat_h - lat_0))
                elif lat_h <= lat <= lat_1:
                    contour_coordinates[j][i] = 0.5 + ((lat - lat_h) / (2 * (lat_1 - lat_h)))

    c_bins = np.linspace(0, 1, 41)[:-1]
    delta_c = c_bins[1] - c_bins[0]
    c_bins = c_bins + (delta_c / 2)

    c_south_of_AD = np.logical_and(contour_coordinates >= 0, contour_coordinates < 0.5)

    year_start = 2011
    year_end = 2015
    dates = date_range(datetime.date(year_start, 1, 1), datetime.date(year_end, 12, 31))

    n_days = len(dates)
    month_days = np.zeros(12)

    pickle_filepath = os.path.join(figure_dir_path, 'Antarctic_Divergence_Ekman_pumping_barchart.pickle')

    pickle_found = False
    try:
        with open(pickle_filepath, 'rb') as f:
            monthly_barchart_dict = pickle.load(f)
            logger.info('Previous computation found. Loading {:s}...'.format(pickle_filepath))
            pickle_found = True
    except OSError:
        logger.info('Computing Antarctic Divergence time series...')

    if pickle_found:
        alpha_monthly = monthly_barchart_dict['alpha']
        w_a_monthly = monthly_barchart_dict['w_a']
        w_A_monthly = monthly_barchart_dict['w_A']
        w_Ek_nogeo_monthly = monthly_barchart_dict['w_Ek_nogeo']
        w_Ek_geo_monthly = monthly_barchart_dict['w_Ek_geo']
        w_ig_monthly = monthly_barchart_dict['w_ig']
        w_i_monthly = monthly_barchart_dict['w_i']
        w_i0_monthly = monthly_barchart_dict['w_i0']

    else:
        alpha_monthly = np.zeros(12)

        w_a_monthly = np.zeros(12)
        w_A_monthly = np.zeros(12)
        w_Ek_nogeo_monthly = np.zeros(12)
        w_Ek_geo_monthly = np.zeros(12)
        w_ig_monthly = np.zeros(12)
        w_i_monthly = np.zeros(12)
        w_i0_monthly = np.zeros(12)

        for d, date in enumerate(dates):

            try:
                daily_filepath = get_netCDF_filepath(field_type='daily', date=date)
                lons, lats, _ = get_field_from_netcdf(daily_filepath, 'Ekman_w')
            except Exception as e:
                logger.error('{}'.format(e))
                logger.warning('{:s} not found. Proceeding without it...'.format(daily_filepath))
                n_days = n_days - 1  # Must account for lost day if no data available for that day.
                continue

            logger.info('Averaging {:%b %d, %Y}...'.format(date))

            alpha_field = get_field_from_netcdf(daily_filepath, 'alpha')[2]

            w_a_field = get_field_from_netcdf(daily_filepath, 'w_a')[2]
            w_A_field = get_field_from_netcdf(daily_filepath, 'w_A')[2]
            w_Ek_geo_field = get_field_from_netcdf(daily_filepath, 'Ekman_w')[2]
            w_Ek_nogeo_field = get_field_from_netcdf(daily_filepath, 'w_Ekman_nogeo')[2]
            w_i_field = get_field_from_netcdf(daily_filepath, 'w_i')[2]
            w_i0_field = get_field_from_netcdf(daily_filepath, 'w_i0')[2]
            w_ig_field = get_field_from_netcdf(daily_filepath, 'w_ig')[2]

            import astropy.convolution
            kernel = astropy.convolution.Box2DKernel(4)
            # w_a_field = astropy.convolution.convolve(w_a_field, kernel, boundary='wrap')
            # w_A_field = astropy.convolution.convolve(w_A_field, kernel, boundary='wrap')
            # w_Ek_geo_field = astropy.convolution.convolve(w_Ek_geo_field, kernel, boundary='wrap')
            # w_Ek_nogeo_field = astropy.convolution.convolve(w_Ek_nogeo_field, kernel, boundary='wrap')
            # w_i_field = astropy.convolution.convolve(w_i_field, kernel, boundary='wrap')
            # w_i0_field = astropy.convolution.convolve(w_i0_field, kernel, boundary='wrap')
            # w_ig_field = astropy.convolution.convolve(w_ig_field, kernel, boundary='wrap')

            m = date.month - 1
            month_days[m] += 1

            alpha_monthly[m] += np.nanmean(alpha_field[c_south_of_AD])
            w_a_monthly[m] += np.nanmean(w_a_field[c_south_of_AD])
            w_A_monthly[m] += np.nanmean(w_A_field[c_south_of_AD])
            w_Ek_geo_monthly[m] += np.nanmean(w_Ek_geo_field[c_south_of_AD])
            w_Ek_nogeo_monthly[m] += np.nanmean(w_Ek_nogeo_field[c_south_of_AD])
            w_i_monthly[m] += np.nanmean(w_i_field[c_south_of_AD])
            w_i0_monthly[m] += np.nanmean(w_i0_field[c_south_of_AD])
            w_ig_monthly[m] += np.nanmean(w_ig_field[c_south_of_AD])

        for m in range(12):
            alpha_monthly[m] /= month_days[m]
            w_a_monthly[m] /= month_days[m]
            w_A_monthly[m] /= month_days[m]
            w_Ek_geo_monthly[m] /= month_days[m]
            w_Ek_nogeo_monthly[m] /= month_days[m]
            w_i_monthly[m] /= month_days[m]
            w_i0_monthly[m] /= month_days[m]
            w_ig_monthly[m] /= month_days[m]

    with open(pickle_filepath, 'wb') as f:
        monthly_barchart_dict = {
            'alpha': alpha_monthly,
            'w_a': w_a_monthly,
            'w_A': w_A_monthly,
            'w_Ek_nogeo': w_Ek_nogeo_monthly,
            'w_Ek_geo': w_Ek_geo_monthly,
            'w_ig': w_ig_monthly,
            'w_i': w_i_monthly,
            'w_i0': w_i0_monthly
        }
        pickle.dump(monthly_barchart_dict, f, pickle.HIGHEST_PROTOCOL)

    import calendar

    fig = plt.subplots(figsize=(16, 9))
    ax = plt.subplot(111)

    ax.bar(np.arange(1, 13)-0.30, 365*3600*24 * w_i0_monthly, width=0.15, color='green', align='center', label="ice")
    ax.bar(np.arange(1, 13)-0.15, 365*3600*24 * w_ig_monthly, width=0.15, color='blue', align='center', label="geo")
    ax.bar(np.arange(1, 13), 365*3600*24 * w_a_monthly, width=0.15, color='red', align='center', label="wind")
    ax.bar(np.arange(1, 13)+0.15, 365 * 3600 * 24 * w_i_monthly, width=0.15, color='orange', align='center', label="ice+geo")
    ax.bar(np.arange(1, 13)+0.15, 365 * 3600 * 24 * (w_i0_monthly + w_ig_monthly), width=0.15, color='orange', alpha=0.3, hatch='//', align='center', label="ice+geo (sum)")
    ax.bar(np.arange(1, 13)+0.30, 365 * 3600 * 24 * w_Ek_geo_monthly, width=0.15, color='black', align='center', label="total")
    ax.bar(np.arange(1, 13)+0.30, 365*3600*24 * (w_i0_monthly + w_ig_monthly + w_a_monthly), width=0.15, color='black', edgecolor='gray', alpha =0.3, hatch='//', align='center', label="total (sum)")
    ax.bar(np.arange(1, 13)+0.45, 365 * 3600 * 24 * w_Ek_nogeo_monthly, width=0.15, color='purple', align='center', label="total (no_geo)")

    ax.set_xlabel("Month", fontsize=18)
    ax.set_ylabel("Ekman pumping (m/year)", fontsize=18)
    plt.xticks(np.arange(1, 13), calendar.month_abbr[1:], fontsize=16)
    plt.yticks(fontsize=16)

    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), fontsize=18)

    ax2 = ax.twinx()
    ax2.plot(np.arange(1, 13), alpha_monthly, color='gray', label='sea ice fraction')
    ax2.set_ylim([-1, 1])
    ax2.tick_params('y', colors='gray')
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks(fontsize=16)
    ax2.set_ylabel('Mean Sea ice fraction', color='gray', fontsize=18)

    png_filepath = os.path.join(figure_dir_path, 'Ekman pumping south of the Antarctic Divergence bar chart.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # make_five_box_climo_fig('tau_x')
    # make_five_box_climo_fig('tau_y')
    # make_five_box_climo_fig('Ekman_u')
    # make_five_box_climo_fig('Ekman_v')

    # for var in ['wind_u', 'wind_v', 'ice_u', 'ice_v', 'alpha']
    #     make_five_box_climo_fig(var)

    # make_melt_rate_diagnostic_fig()
    # make_zonal_and_contour_averaged_plots()

    # plot_meridional_salinity_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=-135, split_depth=250)
    # plot_meridional_salinity_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=-30, split_depth=250)
    # plot_meridional_salinity_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=75, split_depth=250)

    # plot_meridional_temperature_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=-135, split_depth=500)
    # plot_meridional_temperature_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=-30, split_depth=500)
    # plot_meridional_temperature_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=75, split_depth=500)

    # look_at_neutral_density_contours(2005, 2012)
    # look_at_neutral_density_contours(2014, 2015)
    # look_at_neutral_density_contours(1992, 1993)
    #
    # plot_meridional_gamma_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=-135, split_depth=250)
    # plot_meridional_gamma_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=-30, split_depth=250)
    # plot_meridional_gamma_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=75, split_depth=250)

    # make_figure1()
    # make_tau_climo_fig()
    # make_ugeo_uice_figure()
    # make_tau_climo_fig()
    # make_uEk_climo_fig()
    # make_curl_climo_fig()
    # make_urel_figure()

    # make_salinity_figure()
    # make_streamwise_coordinate_map()
    # make_streamwise_averaged_plots()
    # make_zonal_average_plots()
    # make_melt_rate_plots()

    # compare_zzsl_with_gamma_contour()
    # compare_zzsl_with_pellichero_gamma()

    # look_for_ice_ocean_governor()
    # plot_Wedell_Gyre_Ekman_pumping()
    # plot_Weddell_Gyre_time_series()
    # make_u_geo_climo_fig()

    # make_melting_freezing_rate_term_plots()

    # plot_streamwise_velocity_integrals()
    # plot_ice_ocean_pumping()
    # antarctic_divergence_time_series()
    ice_ocean_govenor_monthly_climo_barchart()
