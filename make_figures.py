import os
import datetime
import calendar

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
from utils import date_range

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig('logging.ini')
logger = logging.getLogger(__name__)


def make_five_box_climo_fig(var):
    from utils import get_netCDF_filepath, get_field_from_netcdf
    from constants import figure_dir_path

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
    climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JAS',
                                         year_start=2005, year_end=2015)
    # climo_filepath = get_netCDF_filepath(field_type='seasonal_climo', season_str='JFM',
    #                                      year_start=2005, year_end=2015)

    lons, lats, climo_tau_x_field = get_field_from_netcdf(climo_filepath, 'tau_x')
    climo_tau_y_field = get_field_from_netcdf(climo_filepath, 'tau_y')[2]
    climo_alpha_field = get_field_from_netcdf(climo_filepath, 'alpha')[2]

    # climo_tau_x_field, lons_tau = cartopy.util.add_cyclic_point(climo_tau_x_field, coord=lons)
    # climo_tau_y_field, lons_tau = cartopy.util.add_cyclic_point(climo_tau_y_field, coord=lons)

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

    im1 = ax1.pcolormesh(lons, lats, climo_tau_x_field, transform=vector_crs, cmap=cmocean.cm.balance,
                         vmin=-0.20, vmax=0.20)

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

    clb = fig.colorbar(im1, ax=ax1, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'N/m$^2$')

    zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
    ice_edge_patch = mpatches.Patch(color='black', label=r'15% ice edge')
    gamma_patch = mpatches.Patch(color='red', label=r'$\gamma^n$ = 27.6 kg/m$^3$ contour')
    plt.legend(handles=[zero_stress_line_patch, gamma_patch, ice_edge_patch], loc='lower center',
               bbox_to_anchor=(0, -0.15, 1, -0.15), ncol=1, mode='expand', borderaxespad=0, framealpha=0)

    # plt.suptitle(r'Figure 2: Ocean surface stress $\mathbf{\tau}$ observations, winter (JAS) mean', fontsize=16)
    plt.suptitle(r'Figure 4: Ocean surface stress $\mathbf{\tau}$ observations, with geostrophic current, '
                 'winter (JAS) mean', fontsize=16)

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

    im2 = ax2.pcolormesh(lons, lats, climo_tau_y_field, transform=vector_crs, cmap=cmocean.cm.balance,
                         vmin=-0.20, vmax=0.20)

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

    clb = fig.colorbar(im2, ax=ax2, extend='both', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'N/m$^2$')

    png_filepath = os.path.join(figure_dir_path, 'tau_climo_figure.png')

    tau_dir = os.path.dirname(png_filepath)
    if not os.path.exists(tau_dir):
        logger.info('Creating directory: {:s}'.format(tau_dir))
        os.makedirs(tau_dir)

    logger.info('Saving diagnostic figure: {:s}'.format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='pdf', transparent=False)


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

    # make_salinity_figure()
    # make_streamwise_coordinate_map()
    # make_streamwise_averaged_plots()
    # make_zonal_average_plots()
    # make_melt_rate_plots()
    look_for_ice_ocean_governor()

    # compare_zzsl_with_gamma_contour()
    # compare_zzsl_with_pellichero_gamma()