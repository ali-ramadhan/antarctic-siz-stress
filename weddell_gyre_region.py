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

import constants
from constants import figure_dir_path
from utils import date_range, distance, log_netCDF_dataset_metadata, get_netCDF_filepath, get_field_from_netcdf

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig('logging.ini')
logger = logging.getLogger(__name__)

nogeo_output_dir_path = 'E:\\output\\'
geo_output_dir_path = 'C:\\Users\\Ali\\Downloads\\output\\'


def retroactively_compute_ice_ocean_governor_variables(dates):
    try:
        tau_tmp_filepath = get_netCDF_filepath(field_type='daily', date=dates[0])
        tau_dataset = netCDF4.Dataset(tau_tmp_filepath)
        lats = np.array(tau_dataset.variables['lat'])
        lons = np.array(tau_dataset.variables['lon'])
    except Exception as e:
        logger.error('{}'.format(e))
        logger.error('{:s} not found. Could not load lat, lon arrays.'.format(tau_tmp_filepath))

    tau_io_nogeo_avg_field = np.zeros((len(lats), len(lons)))
    tau_io_geo_avg_field = np.zeros((len(lats), len(lons)))
    tau_ig_avg_field = np.zeros((len(lats), len(lons)))
    w_Ek_nogeo_avg_field = np.zeros((len(lats), len(lons)))
    w_Ek_geo_avg_field = np.zeros((len(lats), len(lons)))
    w_a_avg_field = np.zeros((len(lats), len(lons)))
    w_i_avg_field = np.zeros((len(lats), len(lons)))
    w_i0_avg_field = np.zeros((len(lats), len(lons)))
    w_ig_avg_field = np.zeros((len(lats), len(lons)))
    w_A_avg_field = np.zeros((len(lats), len(lons)))
    gamma_avg_field = np.zeros((len(lats), len(lons)))

    tau_io_nogeo_day_field = np.zeros((len(lats), len(lons)))
    tau_io_geo_day_field = np.zeros((len(lats), len(lons)))
    tau_ig_day_field = np.zeros((len(lats), len(lons)))
    w_Ek_nogeo_day_field = np.zeros((len(lats), len(lons)))
    w_Ek_geo_day_field = np.zeros((len(lats), len(lons)))
    w_a_day_field = np.zeros((len(lats), len(lons)))
    w_i_day_field = np.zeros((len(lats), len(lons)))
    w_i0_day_field = np.zeros((len(lats), len(lons)))
    w_ig_day_field = np.zeros((len(lats), len(lons)))
    w_A_day_field = np.zeros((len(lats), len(lons)))
    gamma_day_field = np.zeros((len(lats), len(lons)))

    for date in dates:
        # Load no_geo daily dataset first.
        constants.output_dir_path = nogeo_output_dir_path

        tau_nogeo_filepath = get_netCDF_filepath(field_type='daily', date=date)
        logger.info('Loading {:%b %d, %Y} nogeo ({:s})...'.format(date, tau_nogeo_filepath))

        try:
            current_tau_nogeo_dataset = netCDF4.Dataset(tau_nogeo_filepath)
            log_netCDF_dataset_metadata(current_tau_nogeo_dataset)
        except OSError as e:
            logger.error('{}'.format(e))
            logger.warning('{:s} not found. Proceeding without it...'.format(tau_nogeo_filepath))
            continue

        # Now load the corresponding geo daily dataset.
        constants.output_dir_path = geo_output_dir_path

        tau_geo_filepath = get_netCDF_filepath(field_type='daily', date=date)
        logger.info('Loading {:%b %d, %Y} geo ({:s})...'.format(date, tau_geo_filepath))

        try:
            current_tau_geo_dataset = netCDF4.Dataset(tau_geo_filepath)
            log_netCDF_dataset_metadata(current_tau_geo_dataset)
        except OSError as e:
            logger.error('{}'.format(e))
            logger.warning('{:s} not found. Proceeding without it...'.format(tau_geo_filepath))
            continue

        logger.info('Averaging {:%b %d, %Y}...'.format(date))

        u_ice_daily_field = np.array(current_tau_dataset.variables['ice_u'])
        v_ice_daily_field = np.array(current_tau_dataset.variables['ice_v'])

        h_ice_daily_field = np.zeros((len(lats), len(lons)))
        zonal_div_daily_field = np.zeros((len(lats), len(lons)))
        merid_div_daily_field = np.zeros((len(lats), len(lons)))
        div_daily_field = np.zeros((len(lats), len(lons)))

        hu_dadx_daily_field = np.zeros((len(lats), len(lons)))
        au_dhdx_daily_field = np.zeros((len(lats), len(lons)))
        ah_dudx_daily_field = np.zeros((len(lats), len(lons)))
        hv_dady_daily_field = np.zeros((len(lats), len(lons)))
        av_dhdy_daily_field = np.zeros((len(lats), len(lons)))
        ah_dvdy_daily_field = np.zeros((len(lats), len(lons)))

        div2_daily_field = np.zeros((len(lats), len(lons)))


if __name__ == '__main__':
    pass
