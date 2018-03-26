import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy
import cartopy.crs as ccrs

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig('logging.ini')
logger = logging.getLogger(__name__)

from utils import get_netCDF_filepath, get_northward_zero_zonal_stress_line, get_northward_zero_zonal_wind_line
from utils import get_northward_ice_edge
from constants import output_dir_path, lon_min, lon_max, n_lon

np.set_printoptions(precision=4)


def analyze_zero_zonal_stress_line(custom_str=None):
    years = np.arange(1995, 2016, 1)

    lon_bins = np.linspace(lon_min, lon_max, n_lon)

    lat_northward = np.empty((len(lon_bins), len(years)))
    lat_northward[:] = -180

    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face',
                                                   facecolor='dimgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(111, projection=ccrs.SouthPolarStereo())
    ax.add_feature(land_50m)
    ax.set_extent([-110, -50, -75, -65], ccrs.PlateCarree())
    patches = []

    NUM_COLORS = len(years)
    PiYG = plt.get_cmap('PiYG')

    for i in range(len(years)):
        year = years[i]

        tau_filepath = get_netCDF_filepath(date=datetime.date(year, 1, 1), field_type='annual')
        # tau_filepath = get_netCDF_filepath(field_type='seasonal', date=datetime.date(year, 1, 1), season_str='JFM')

        lon_bins, lat_northward_i = get_northward_zero_zonal_stress_line(tau_filepath)
        lat_northward[:, i] = lat_northward_i

        # For some reason if I plot all lons and zoom in, the curves disappear... But if I just plot a smaller section,
        # it's fine...
        bellinghausen_lons = np.logical_and(lon_bins > -150, lon_bins < -30)

        logger.info('({:d}) lat_max={:f}'.format(year, np.nanmax(lat_northward[:, i])))

        # plt.plot(lon_bins, lat_northward_i, label=str(year), color=cm(1. * i/NUM_COLORS))
        # ax.plot(lon_bins, lat_northward_i, color=PiYG(1. * i/NUM_COLORS), linewidth=1, label=str(year),
        #         transform=vector_crs)
        ax.plot(lon_bins[bellinghausen_lons], lat_northward_i[bellinghausen_lons], color=PiYG(1. * i/NUM_COLORS),
                linewidth=1, label=str(year), transform=vector_crs)

        patches = patches + [mpatches.Patch(color=PiYG(1. * i/NUM_COLORS), label=str(year))]

    plt.title('zero zonal stress line (annual means)')
    ax.gridlines(crs=ccrs.PlateCarree())
    plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.03, 1), ncol=1, mode='expand', frameon=False,
               borderaxespad=0)
    plt.savefig('tau_x_zero_line' + custom_str + '.png', dpi=600, format='png', transparent=False, bbox_inches='tight')
    plt.show()
    plt.close()

    from scipy import stats

    slopes = np.zeros(len(lon_bins))
    intercepts = np.zeros(len(lon_bins))
    r_values = np.zeros(len(lon_bins))
    p_values = np.zeros(len(lon_bins))
    std_errs = np.zeros(len(lon_bins))

    # Do linear regression for the N/S position of the line at each longitude.
    for i in range(len(lon_bins)):
        lat_time_series = lat_northward[i, :]

        slope, intercept, r_value, p_value, std_err = stats.linregress(years, lat_time_series)
        slopes[i] = slope
        intercepts[i] = intercept
        r_values[i] = r_value**2
        p_values[i] = p_value
        std_errs[i] = std_err

    plt.plot(lon_bins, slopes, label='slopes')
    # plt.plot(lon_bins, intercepts, label='intercepts')
    plt.plot(lon_bins, r_values, label='r^2 values')
    plt.plot(lon_bins, p_values, label='p values')
    # plt.plot(lon_bins, std_errs, label='std_errs')

    plt.legend()
    plt.show()
    plt.close()

    # Repeat linear regression but now use a bin width of delta_lon
    # delta_lon = 72
    # bins = 360/delta_lon
    # for i in np.linspace(-180, 180, bins):
    #     lat_time_series = lat_northward[i, :]


def analyze_zero_zonal_wind_line(custom_str=None):
    years = np.arange(1995, 2016, 1)

    lon_bins = np.linspace(lon_min, lon_max, n_lon)

    lat_northward = np.empty((len(lon_bins), len(years)))
    lat_northward[:] = -180

    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face',
                                                   facecolor='dimgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(111, projection=ccrs.SouthPolarStereo())
    ax.add_feature(land_50m)

    # ax.set_extent([-180, 180, -75, -65], ccrs.PlateCarree())
    ax.set_extent([-110, -50, -75, -65], ccrs.PlateCarree())  # Bellinghausen Sea
    # ax.set_extent([0, 50, -75, -65], ccrs.PlateCarree())  # Ross Sea?

    patches = []

    NUM_COLORS = len(years)
    PiYG = plt.get_cmap('PiYG')

    for i in range(len(years)):
        year = years[i]

        tau_filepath = get_netCDF_filepath(date=datetime.date(year, 1, 1), field_type='annual')
        # tau_filepath = get_netCDF_filepath(field_type='seasonal', date=datetime.date(year, 1, 1), season_str='JFM')

        lon_bins, lat_northward_i = get_northward_zero_zonal_wind_line(tau_filepath)
        lat_northward[:, i] = lat_northward_i

        bellinghausen_lons = np.logical_and(lon_bins > -150, lon_bins < -30)
        # ross_lons = np.logical_and(lon_bins < -150, lon_bins > 150)

        logger.info('({:d}) lat_max={:f}'.format(year, np.nanmax(lat_northward[:, i])))

        # plt.plot(lon_bins, lat_northward_i, label=str(year), color=cm(1. * i/NUM_COLORS))
        # ax.plot(lon_bins, lat_northward_i, color=PiYG(1. * i/NUM_COLORS), linewidth=1, label=str(year),
        #         transform=vector_crs)
        ax.plot(lon_bins[bellinghausen_lons], lat_northward_i[bellinghausen_lons], color=PiYG(1. * i/NUM_COLORS),
                linewidth=1, label=str(year), transform=vector_crs)
        # ax.plot(lon_bins[ross_lons], lat_northward_i[ross_lons], color=PiYG(1. * i/NUM_COLORS),
        #         linewidth=1, label=str(year), transform=vector_crs)

        patches = patches + [mpatches.Patch(color=PiYG(1. * i/NUM_COLORS), label=str(year))]

    plt.title('Zero zonal wind line (annual means)')
    ax.gridlines(crs=ccrs.PlateCarree())
    plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.03, 1), ncol=1, mode='expand', frameon=False,
               borderaxespad=0)

    plt.savefig('wind_u_zero_line' + custom_str + '.png', dpi=600, format='png', transparent=False, bbox_inches='tight')
    plt.show()
    plt.close()

    from scipy import stats

    slopes = np.zeros(len(lon_bins))
    intercepts = np.zeros(len(lon_bins))
    r_values = np.zeros(len(lon_bins))
    p_values = np.zeros(len(lon_bins))
    std_errs = np.zeros(len(lon_bins))

    # Do linear regression for the N/S position of the line at each longitude.
    for i in range(len(lon_bins)):
        lat_time_series = lat_northward[i, :]

        slope, intercept, r_value, p_value, std_err = stats.linregress(years, lat_time_series)
        slopes[i] = slope
        intercepts[i] = intercept
        r_values[i] = r_value**2
        p_values[i] = p_value
        std_errs[i] = std_err

    plt.plot(lon_bins, slopes, label='slopes')
    # plt.plot(lon_bins, intercepts, label='intercepts')
    plt.plot(lon_bins, r_values, label='r^2 values')
    plt.plot(lon_bins, p_values, label='p values')
    # plt.plot(lon_bins, std_errs, label='std_errs')

    plt.legend()
    plt.show()
    plt.close()


def analyze_ice_edge_trend(custom_str=None):
    years = np.arange(1995, 2016, 1)

    lon_bins = np.linspace(lon_min, lon_max, n_lon)

    lat_northward = np.empty((len(lon_bins), len(years)))
    lat_northward[:] = -180

    land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face',
                                                   facecolor='dimgray', linewidth=0)
    vector_crs = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(111, projection=ccrs.SouthPolarStereo())
    ax.add_feature(land_50m)

    # ax.set_extent([-180, 180, -75, -65], ccrs.PlateCarree())
    ax.set_extent([-110, -50, -75, -60], ccrs.PlateCarree())  # Bellinghausen Sea
    # ax.set_extent([0, 50, -75, -65], ccrs.PlateCarree())  # Ross Sea?

    patches = []

    NUM_COLORS = len(years)
    PiYG = plt.get_cmap('PiYG')

    for i in range(len(years)):
        year = years[i]

        tau_filepath = get_netCDF_filepath(date=datetime.date(year, 1, 1), field_type='annual')
        # tau_filepath = get_netCDF_filepath(field_type='seasonal', date=datetime.date(year, 1, 1), season_str='JFM')

        lon_bins, lat_northward_i = get_northward_ice_edge(tau_filepath)
        lat_northward[:, i] = lat_northward_i

        bellinghausen_lons = np.logical_and(lon_bins > -150, lon_bins < -30)
        # ross_lons = np.logical_and(lon_bins < -150, lon_bins > 150)

        logger.info('({:d}) lat_max={:f}'.format(year, np.nanmax(lat_northward[:, i])))

        # plt.plot(lon_bins, lat_northward_i, label=str(year), color=cm(1. * i/NUM_COLORS))
        # ax.plot(lon_bins, lat_northward_i, color=PiYG(1. * i/NUM_COLORS), linewidth=1, label=str(year),
        #         transform=vector_crs)
        ax.plot(lon_bins[bellinghausen_lons], lat_northward_i[bellinghausen_lons], color=PiYG(1. * i/NUM_COLORS),
                linewidth=1, label=str(year), transform=vector_crs)
        # ax.plot(lon_bins[ross_lons], lat_northward_i[ross_lons], color=PiYG(1. * i/NUM_COLORS),
        #         linewidth=1, label=str(year), transform=vector_crs)

        patches = patches + [mpatches.Patch(color=PiYG(1. * i/NUM_COLORS), label=str(year))]

    plt.title('15% ice edge (annual means)')
    ax.gridlines(crs=ccrs.PlateCarree())
    plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.03, 1), ncol=1, mode='expand', frameon=False,
               borderaxespad=0)

    plt.savefig('ice_edge' + custom_str + '.png', dpi=600, format='png', transparent=False, bbox_inches='tight')
    plt.show()
    plt.close()

    from scipy import stats

    slopes = np.zeros(len(lon_bins))
    intercepts = np.zeros(len(lon_bins))
    r_values = np.zeros(len(lon_bins))
    p_values = np.zeros(len(lon_bins))
    std_errs = np.zeros(len(lon_bins))

    # Do linear regression for the N/S position of the line at each longitude.
    for i in range(len(lon_bins)):
        lat_time_series = lat_northward[i, :]

        slope, intercept, r_value, p_value, std_err = stats.linregress(years, lat_time_series)
        slopes[i] = slope
        intercepts[i] = intercept
        r_values[i] = r_value**2
        p_values[i] = p_value
        std_errs[i] = std_err

    plt.plot(lon_bins, slopes, label='slopes')
    # plt.plot(lon_bins, intercepts, label='intercepts')
    plt.plot(lon_bins, r_values, label='r^2 values')
    plt.plot(lon_bins, p_values, label='p values')
    # plt.plot(lon_bins, std_errs, label='std_errs')

    plt.legend()
    plt.show()
    plt.close()


# ice_lon_bins, ice_lat_northward_i = get_northward_ice_edge(tau_filepath)
# ice_lat_northward[:, i] = ice_lat_northward_i
# plt.plot(ice_lon_bins, ice_lat_northward_i, color=cm(1. * i/NUM_COLORS))

if __name__ == '__main__':
    analyze_zero_zonal_stress_line(custom_str='climo')
    # analyze_zero_zonal_wind_line(custom_str='climo')
    # analyze_ice_edge_trend(custom_str='climo')
