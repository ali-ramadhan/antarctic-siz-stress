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

    png_filepath = os.path.join(figure_dir_path, 'melt_rate_diagnostic_figure.png')

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
    from constants import D_e

    tau_filepath = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(2000, 11, 1), year_start=2005, year_end=2015)
    lons, lats, ice_div_field = get_field_from_netcdf(tau_filepath, 'ice_flux_div')
    melt_rate_field = get_field_from_netcdf(tau_filepath, 'melt_rate')[2]
    psi_delta_field = get_field_from_netcdf(tau_filepath, 'psi_delta')[2]
    salinity_field = get_field_from_netcdf(tau_filepath, 'salinity')[2]
    temperature_field = get_field_from_netcdf(tau_filepath, 'temperature')[2]

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
    psi_delta_cavg = np.zeros(c_bins.shape)

    for i in range(len(c_bins)):
        c = c_bins[i]
        c_low = c - (delta_c / 2)
        c_high = c + (delta_c / 2)

        c_in_range = np.logical_and(contour_coordinate > c_low, contour_coordinate < c_high)

        ice_div_cavg[i] = np.nanmean(ice_div_field[c_in_range])
        melt_rate_cavg[i] = np.nanmean(melt_rate_field[c_in_range])
        psi_delta_cavg[i] = np.nanmean(psi_delta_field[c_in_range])

    fig = plt.figure(figsize=(20, 6))

    ax = fig.add_subplot(131)

    ax.plot(lats, np.nanmean(ice_div_field, axis=1) * 24 * 3600 * 365, label='div(α*h*u_ice) [m/year]')
    ax.plot(lats, np.nanmean(melt_rate_field, axis=1) * 24 * 3600 * 365, label='ψ(-δ) * 1/S * dS/dy ~ M-F [m/year]')
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
    # ax.legend(fontsize='large')

    # plt.show()

    plt.savefig('zonal_avg.png', dpi=300, format='png', transparent=False, bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(20, 6))

    ax = fig.add_subplot(131)
    ax.plot(c_bins, ice_div_cavg * 24 * 3600 * 365, label='div(α*h*u_ice) [m/year]')
    ax.plot(c_bins, melt_rate_cavg * 24 * 3600 * 365, label='(M-F) [m/year]')
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

    plt.savefig('contour_avg.png', dpi=300, format='png', transparent=False, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # make_five_box_climo_fig('tau_x')
    # make_five_box_climo_fig('tau_y')
    make_five_box_climo_fig('Ekman_u')
    make_five_box_climo_fig('Ekman_v')

    # for var in ['wind_u', 'wind_v', 'ice_u', 'ice_v', 'alpha']
    #     make_five_box_climo_fig(var)
    # make_melt_rate_diagnostic_fig()
    # make_zonal_and_contour_averaged_plots()
