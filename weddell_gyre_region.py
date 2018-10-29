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
from constants import Omega, rho_0, output_dir_path, figure_dir_path
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

    alpha_avg_field = np.zeros((len(lats), len(lons)))
    tau_io_x_nogeo_avg_field = np.zeros((len(lats), len(lons)))
    tau_io_y_nogeo_avg_field = np.zeros((len(lats), len(lons)))
    tau_io_x_geo_avg_field = np.zeros((len(lats), len(lons)))
    tau_io_y_geo_avg_field = np.zeros((len(lats), len(lons)))
    tau_ig_x_avg_field = np.zeros((len(lats), len(lons)))
    tau_ig_y_avg_field = np.zeros((len(lats), len(lons)))
    tau_ao_x_avg_field = np.zeros((len(lats), len(lons)))
    tau_ao_y_avg_field = np.zeros((len(lats), len(lons)))
    w_Ek_nogeo_avg_field = np.zeros((len(lats), len(lons)))
    w_Ek_geo_avg_field = np.zeros((len(lats), len(lons)))
    w_a_avg_field = np.zeros((len(lats), len(lons)))
    w_i_avg_field = np.zeros((len(lats), len(lons)))
    w_i0_avg_field = np.zeros((len(lats), len(lons)))
    w_ig_avg_field = np.zeros((len(lats), len(lons)))
    w_A_avg_field = np.zeros((len(lats), len(lons)))
    gamma_avg_field = np.zeros((len(lats), len(lons)))

    alpha_day_field = np.zeros((len(lats), len(lons)))
    tau_io_x_nogeo_day_field = np.zeros((len(lats), len(lons)))
    tau_io_y_nogeo_day_field = np.zeros((len(lats), len(lons)))
    tau_io_x_geo_day_field = np.zeros((len(lats), len(lons)))
    tau_io_y_geo_day_field = np.zeros((len(lats), len(lons)))
    tau_ig_x_day_field = np.zeros((len(lats), len(lons)))
    tau_ig_y_day_field = np.zeros((len(lats), len(lons)))
    tau_ao_x_day_field = np.zeros((len(lats), len(lons)))
    tau_ao_y_day_field = np.zeros((len(lats), len(lons)))
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

        alpha_daily_field = np.array(current_tau_geo_dataset.variables['alpha'])
        tau_io_x_nogeo_daily_field = np.array(current_tau_nogeo_dataset.variables['tau_ice_x'])
        tau_io_y_nogeo_daily_field = np.array(current_tau_nogeo_dataset.variables['tau_ice_y'])
        tau_io_x_geo_daily_field = np.array(current_tau_geo_dataset.variables['tau_ice_x'])
        tau_io_y_geo_daily_field = np.array(current_tau_geo_dataset.variables['tau_ice_y'])
        tau_ao_x_daily_field = np.array(current_tau_geo_dataset.variables['tau_air_x'])
        tau_ao_y_daily_field = np.array(current_tau_geo_dataset.variables['tau_air_y'])
        w_Ek_nogeo_daily_field = np.array(current_tau_nogeo_dataset.variables['Ekman_w'])
        w_Ek_geo_daily_field = np.array(current_tau_geo_dataset.variables['Ekman_w'])

        import astropy.convolution
        kernel = astropy.convolution.Box2DKernel(5)
        alpha_daily_field = astropy.convolution.convolve(alpha_daily_field, kernel, boundary='wrap')
        tau_io_x_nogeo_daily_field = astropy.convolution.convolve(tau_io_x_nogeo_daily_field, kernel, boundary='wrap')
        tau_io_y_nogeo_daily_field = astropy.convolution.convolve(tau_io_y_nogeo_daily_field, kernel, boundary='wrap')
        tau_io_x_geo_daily_field = astropy.convolution.convolve(tau_io_x_geo_daily_field, kernel, boundary='wrap')
        tau_io_y_geo_daily_field = astropy.convolution.convolve(tau_io_y_geo_daily_field, kernel, boundary='wrap')
        tau_ao_x_daily_field = astropy.convolution.convolve(tau_ao_x_daily_field, kernel, boundary='wrap')
        tau_ao_y_daily_field = astropy.convolution.convolve(tau_ao_y_daily_field, kernel, boundary='wrap')
        w_Ek_nogeo_daily_field = astropy.convolution.convolve(w_Ek_nogeo_daily_field, kernel, boundary='wrap')
        w_Ek_geo_daily_field = astropy.convolution.convolve(w_Ek_geo_daily_field, kernel, boundary='wrap')

        tau_ig_x_daily_field = np.zeros((len(lats), len(lons)))
        tau_ig_y_daily_field = np.zeros((len(lats), len(lons)))
        w_a_daily_field = np.zeros((len(lats), len(lons)))
        w_i_daily_field = np.zeros((len(lats), len(lons)))
        w_i0_daily_field = np.zeros((len(lats), len(lons)))
        w_ig_daily_field = np.zeros((len(lats), len(lons)))
        w_A_daily_field = np.zeros((len(lats), len(lons)))
        gamma_daily_field = np.zeros((len(lats), len(lons)))

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                tau_ig_x_daily_field[i][j] = tau_io_x_geo_daily_field[i][j] - tau_io_x_nogeo_daily_field[i][j]
                tau_ig_y_daily_field[i][j] = tau_io_y_geo_daily_field[i][j] - tau_io_y_nogeo_daily_field[i][j]

        i_max = len(lats) - 1
        j_max = len(lons) - 1

        for i, lat in enumerate(lats[1:-1]):
            f = 2 * Omega * np.sin(np.deg2rad(lat))  # Coriolis parameter [s^-1]
            dx = distance(lats[i-1], lons[0], lats[i+1], lons[0])
            dy = distance(lats[i], lons[0], lats[i], lons[2])

            for j, lon in enumerate(lons):
                # Taking modulus of j-1 and j+1 to get the correct index in the special cases of
                #  * j=0 (180 W) and need to use the value from j=j_max (180 E)
                #  * j=j_max (180 E) and need to use the value from j=0 (180 W)
                jm1 = (j - 1) % j_max
                jp1 = (j + 1) % j_max

                alpha_ij = alpha_daily_field[i][j]
                alpha_i_jp1 = alpha_daily_field[i][jp1]
                alpha_i_jm1 = alpha_daily_field[i][jm1]
                alpha_ip1_j = alpha_daily_field[i+1][j]
                alpha_im1_j = alpha_daily_field[i-1][j]

                tau_ao_x_ij = tau_ao_x_daily_field[i][j]
                tau_ao_x_i_jp1 = tau_ao_x_daily_field[i][jp1]
                tau_ao_x_i_jm1 = tau_ao_x_daily_field[i][jm1]
                tau_ao_x_ip1_j = tau_ao_x_daily_field[i+1][j]
                tau_ao_x_im1_j = tau_ao_x_daily_field[i-1][j]

                tau_ao_y_ij = tau_ao_y_daily_field[i][j]
                tau_ao_y_i_jp1 = tau_ao_y_daily_field[i][jp1]
                tau_ao_y_i_jm1 = tau_ao_y_daily_field[i][jm1]
                tau_ao_y_ip1_j = tau_ao_y_daily_field[i+1][j]
                tau_ao_y_im1_j = tau_ao_y_daily_field[i-1][j]

                tau_io_x_nogeo_ij = tau_io_x_nogeo_daily_field[i][j]
                tau_io_x_nogeo_i_jp1 = tau_io_x_nogeo_daily_field[i][jp1]
                tau_io_x_nogeo_i_jm1 = tau_io_x_nogeo_daily_field[i][jm1]
                tau_io_x_nogeo_ip1_j = tau_io_x_nogeo_daily_field[i+1][j]
                tau_io_x_nogeo_im1_j = tau_io_x_nogeo_daily_field[i-1][j]

                tau_io_y_nogeo_ij = tau_io_y_nogeo_daily_field[i][j]
                tau_io_y_nogeo_i_jp1 = tau_io_y_nogeo_daily_field[i][jp1]
                tau_io_y_nogeo_i_jm1 = tau_io_y_nogeo_daily_field[i][jm1]
                tau_io_y_nogeo_ip1_j = tau_io_y_nogeo_daily_field[i+1][j]
                tau_io_y_nogeo_im1_j = tau_io_y_nogeo_daily_field[i-1][j]

                tau_io_x_geo_ij = tau_io_x_geo_daily_field[i][j]
                tau_io_x_geo_i_jp1 = tau_io_x_geo_daily_field[i][jp1]
                tau_io_x_geo_i_jm1 = tau_io_x_geo_daily_field[i][jm1]
                tau_io_x_geo_ip1_j = tau_io_x_geo_daily_field[i+1][j]
                tau_io_x_geo_im1_j = tau_io_x_geo_daily_field[i-1][j]

                tau_io_y_geo_ij = tau_io_y_geo_daily_field[i][j]
                tau_io_y_geo_i_jp1 = tau_io_y_geo_daily_field[i][jp1]
                tau_io_y_geo_i_jm1 = tau_io_y_geo_daily_field[i][jm1]
                tau_io_y_geo_ip1_j = tau_io_y_geo_daily_field[i+1][j]
                tau_io_y_geo_im1_j = tau_io_y_geo_daily_field[i-1][j]

                tau_ig_x_ij = tau_ig_x_daily_field[i][j]
                tau_ig_x_i_jp1 = tau_ig_x_daily_field[i][jp1]
                tau_ig_x_i_jm1 = tau_ig_x_daily_field[i][jm1]
                tau_ig_x_ip1_j = tau_ig_x_daily_field[i+1][j]
                tau_ig_x_im1_j = tau_ig_x_daily_field[i-1][j]

                tau_ig_y_ij = tau_ig_y_daily_field[i][j]
                tau_ig_y_i_jp1 = tau_ig_y_daily_field[i][jp1]
                tau_ig_y_i_jm1 = tau_ig_y_daily_field[i][jm1]
                tau_ig_y_ip1_j = tau_ig_y_daily_field[i+1][j]
                tau_ig_y_im1_j = tau_ig_y_daily_field[i-1][j]

                if not np.isnan(tau_ao_x_ip1_j) and not np.isnan(tau_ao_x_im1_j) \
                        and not np.isnan(tau_ao_y_i_jp1) and not np.isnan(tau_ao_y_i_jm1):
                    ddx_tau_ao_y = (tau_ao_y_i_jp1 - tau_ao_y_i_jm1) / dx
                    ddy_tau_ao_x = (tau_ao_x_ip1_j - tau_ao_x_im1_j) / dy
                    w_A_daily_field[i][j] = (ddx_tau_ao_y - ddy_tau_ao_x) / (rho_0 * f)

                if not np.isnan(alpha_ip1_j) and not np.isnan(alpha_im1_j) \
                        and not np.isnan(alpha_i_jp1) and not np.isnan(alpha_i_jm1):

                    if not np.isnan(tau_ao_x_ip1_j) and not np.isnan(tau_ao_x_im1_j) \
                            and not np.isnan(tau_ao_y_i_jp1) and not np.isnan(tau_ao_y_i_jm1):
                        ddx_1malpha_tau_ao_y = ((1-alpha_i_jp1)*tau_ao_y_i_jp1 - (1-alpha_i_jm1)*tau_ao_y_i_jm1) / dx
                        ddy_1malpha_tau_ao_x = ((1-alpha_ip1_j)*tau_ao_x_ip1_j - (1-alpha_im1_j)*tau_ao_x_im1_j) / dy
                        w_a_daily_field[i][j] = (ddx_1malpha_tau_ao_y - ddy_1malpha_tau_ao_x) / (rho_0 * f)

                    if not np.isnan(tau_io_x_geo_ip1_j) and not np.isnan(tau_io_x_geo_im1_j) \
                            and not np.isnan(tau_io_y_geo_i_jp1) and not np.isnan(tau_io_y_geo_i_jm1):
                        ddx_alpha_tau_io_y_geo = (alpha_i_jp1*tau_io_y_geo_i_jp1 - alpha_i_jm1*tau_io_y_geo_i_jm1) / dx
                        ddy_alpha_tau_io_x_geo = (alpha_ip1_j*tau_io_x_geo_ip1_j - alpha_im1_j*tau_io_x_geo_im1_j) / dy
                        w_i_daily_field[i][j] = (ddx_alpha_tau_io_y_geo - ddy_alpha_tau_io_x_geo) / (rho_0 * f)

                    if not np.isnan(tau_io_x_nogeo_ip1_j) and not np.isnan(tau_io_x_nogeo_im1_j) \
                            and not np.isnan(tau_io_y_nogeo_i_jp1) and not np.isnan(tau_io_y_nogeo_i_jm1):
                        ddx_alpha_tau_io_y_nogeo = (alpha_i_jp1*tau_io_y_nogeo_i_jp1 - alpha_i_jm1*tau_io_y_nogeo_i_jm1) / dx
                        ddy_alpha_tau_io_x_nogeo = (alpha_ip1_j*tau_io_x_nogeo_ip1_j - alpha_im1_j*tau_io_x_nogeo_im1_j) / dy
                        w_i0_daily_field[i][j] = (ddx_alpha_tau_io_y_nogeo - ddy_alpha_tau_io_x_nogeo) / (rho_0 * f)

                    if not np.isnan(tau_ig_x_ip1_j) and not np.isnan(tau_ig_x_im1_j) \
                            and not np.isnan(tau_ig_y_i_jp1) and not np.isnan(tau_ig_y_i_jm1):
                        ddx_alpha_tau_ig_y = (alpha_i_jp1*tau_ig_y_i_jp1 - alpha_i_jm1*tau_ig_y_i_jm1) / dx
                        ddy_alpha_tau_ig_x = (alpha_ip1_j*tau_ig_x_ip1_j - alpha_im1_j*tau_ig_x_im1_j) / dx
                        w_ig_daily_field[i][j] = (ddx_alpha_tau_ig_y - ddy_alpha_tau_ig_x) / (rho_0 * f)

                    gamma_daily_field[i][j] = np.abs(w_ig_daily_field[i][j]) \
                                              / (np.abs(w_a_daily_field[i][j]) + np.abs(w_i0_daily_field[i][j])
                                                 + np.abs(w_ig_daily_field[i][j]))

        nc_daily_dir = os.path.join(os.path.dirname(output_dir_path), 'ice_ocean_govenor')
        nc_daily_filepath = os.path.join(nc_daily_dir, 'ice_ocean_govenor_{:}.nc'.format(date))

        if not os.path.exists(nc_daily_dir):
            logger.info('Creating directory: {:s}'.format(nc_daily_dir))
            os.makedirs(nc_daily_dir)

        logger.info('Saving fields to netCDF file: {:s}'.format(nc_daily_filepath))

        tau_dataset = netCDF4.Dataset(nc_daily_filepath, 'w')

        tau_dataset.title = 'Ekman pumping in the Antarctic sea ice zone'
        tau_dataset.institution = 'Department of Earth, Atmospheric, and Planetary Science, ' \
                                  'Massachusetts Institute of Technology'

        tau_dataset.createDimension('time', None)
        tau_dataset.createDimension('lat', len(lats))
        tau_dataset.createDimension('lon', len(lons))

        # TODO: Actually store a date.
        time_var = tau_dataset.createVariable('time', np.float64, ('time',))
        time_var.units = 'hours since 0001-01-01 00:00:00'
        time_var.calendar = 'gregorian'

        lat_var = tau_dataset.createVariable('lat', np.float32, ('lat',))
        lat_var.units = 'degrees south'
        lat_var[:] = lats

        lon_var = tau_dataset.createVariable('lon', np.float32, ('lon',))
        lat_var.units = 'degrees west/east'
        lon_var[:] = lons

        var_fields = {
            'alpha': alpha_daily_field,
            'tau_io_x_nogeo': tau_io_x_nogeo_daily_field,
            'tau_io_y_nogeo': tau_io_y_nogeo_daily_field,
            'tau_io_x_geo': tau_io_x_geo_daily_field,
            'tau_io_y_geo': tau_io_y_geo_daily_field,
            'tau_ig_x': tau_ig_x_daily_field,
            'tau_ig_y': tau_ig_y_daily_field,
            'tau_ao_x': tau_ao_x_daily_field,
            'tau_ao_y': tau_ao_y_daily_field,
            'w_Ek_nogeo': w_Ek_nogeo_daily_field,
            'w_Ek_geo': w_Ek_geo_daily_field,
            'w_a': w_a_daily_field,
            'w_i': w_i_daily_field,
            'w_i0': w_i0_daily_field,
            'w_ig': w_ig_daily_field,
            'w_A': w_A_daily_field,
            'gamma': gamma_daily_field
        }

        for var_name in var_fields.keys():
            field_var = tau_dataset.createVariable(var_name, float, ('lat', 'lon'), zlib=True)
            field_var[:] = var_fields[var_name]

        tau_dataset.close()

        alpha_avg_field = alpha_avg_field + np.nan_to_num(alpha_daily_field)
        alpha_daily_field[~np.isnan(alpha_daily_field)] = 1
        alpha_daily_field[np.isnan(alpha_daily_field)] = 0
        alpha_day_field = alpha_day_field + alpha_daily_field

        tau_io_x_nogeo_avg_field = tau_io_x_nogeo_avg_field + np.nan_to_num(tau_io_x_nogeo_daily_field)
        tau_io_x_nogeo_daily_field[~np.isnan(tau_io_x_nogeo_daily_field)] = 1
        tau_io_x_nogeo_daily_field[np.isnan(tau_io_x_nogeo_daily_field)] = 0
        tau_io_x_nogeo_day_field = tau_io_x_nogeo_day_field + tau_io_x_nogeo_daily_field

        tau_io_y_nogeo_avg_field = tau_io_y_nogeo_avg_field + np.nan_to_num(tau_io_y_nogeo_daily_field)
        tau_io_y_nogeo_daily_field[~np.isnan(tau_io_y_nogeo_daily_field)] = 1
        tau_io_y_nogeo_daily_field[np.isnan(tau_io_y_nogeo_daily_field)] = 0
        tau_io_y_nogeo_day_field = tau_io_y_nogeo_day_field + tau_io_y_nogeo_daily_field

        tau_io_x_geo_avg_field = tau_io_x_geo_avg_field + np.nan_to_num(tau_io_x_geo_daily_field)
        tau_io_x_geo_daily_field[~np.isnan(tau_io_x_geo_daily_field)] = 1
        tau_io_x_geo_daily_field[np.isnan(tau_io_x_geo_daily_field)] = 0
        tau_io_x_geo_day_field = tau_io_x_geo_day_field + tau_io_x_geo_daily_field

        tau_io_y_geo_avg_field = tau_io_y_geo_avg_field + np.nan_to_num(tau_io_y_geo_daily_field)
        tau_io_y_geo_daily_field[~np.isnan(tau_io_y_geo_daily_field)] = 1
        tau_io_y_geo_daily_field[np.isnan(tau_io_y_geo_daily_field)] = 0
        tau_io_y_geo_day_field = tau_io_y_geo_day_field + tau_io_y_geo_daily_field

        tau_ig_x_avg_field = tau_ig_x_avg_field + np.nan_to_num(tau_ig_x_daily_field)
        tau_ig_x_daily_field[~np.isnan(tau_ig_x_daily_field)] = 1
        tau_ig_x_daily_field[np.isnan(tau_ig_x_daily_field)] = 0
        tau_ig_x_day_field = tau_ig_x_day_field + tau_ig_x_daily_field

        tau_ig_y_avg_field = tau_ig_y_avg_field + np.nan_to_num(tau_ig_y_daily_field)
        tau_ig_y_daily_field[~np.isnan(tau_ig_y_daily_field)] = 1
        tau_ig_y_daily_field[np.isnan(tau_ig_y_daily_field)] = 0
        tau_ig_y_day_field = tau_ig_y_day_field + tau_ig_y_daily_field

        tau_ao_x_avg_field = tau_ao_x_avg_field + np.nan_to_num(tau_ao_x_daily_field)
        tau_ao_x_daily_field[~np.isnan(tau_ao_x_daily_field)] = 1
        tau_ao_x_daily_field[np.isnan(tau_ao_x_daily_field)] = 0
        tau_ao_x_day_field = tau_ao_x_day_field + tau_ao_x_daily_field

        tau_ao_y_avg_field = tau_ao_y_avg_field + np.nan_to_num(tau_ao_y_daily_field)
        tau_ao_y_daily_field[~np.isnan(tau_ao_y_daily_field)] = 1
        tau_ao_y_daily_field[np.isnan(tau_ao_y_daily_field)] = 0
        tau_ao_y_day_field = tau_ao_y_day_field + tau_ao_y_daily_field

        w_Ek_nogeo_avg_field = w_Ek_nogeo_avg_field + np.nan_to_num(w_Ek_nogeo_daily_field)
        w_Ek_nogeo_daily_field[~np.isnan(w_Ek_nogeo_daily_field)] = 1
        w_Ek_nogeo_daily_field[np.isnan(w_Ek_nogeo_daily_field)] = 0
        w_Ek_nogeo_day_field = w_Ek_nogeo_day_field + w_Ek_nogeo_daily_field

        w_Ek_geo_avg_field = w_Ek_geo_avg_field + np.nan_to_num(w_Ek_geo_daily_field)
        w_Ek_geo_daily_field[~np.isnan(w_Ek_geo_daily_field)] = 1
        w_Ek_geo_daily_field[np.isnan(w_Ek_geo_daily_field)] = 0
        w_Ek_geo_day_field = w_Ek_geo_day_field + w_Ek_geo_daily_field

        w_a_avg_field = w_a_avg_field + np.nan_to_num(w_a_daily_field)
        w_a_daily_field[~np.isnan(w_a_daily_field)] = 1
        w_a_daily_field[np.isnan(w_a_daily_field)] = 0
        w_a_day_field = w_a_day_field + w_a_daily_field

        w_i_avg_field = w_i_avg_field + np.nan_to_num(w_i_daily_field)
        w_i_daily_field[~np.isnan(w_i_daily_field)] = 1
        w_i_daily_field[np.isnan(w_i_daily_field)] = 0
        w_i_day_field = w_i_day_field + w_i_daily_field

        w_i0_avg_field = w_i0_avg_field + np.nan_to_num(w_i0_daily_field)
        w_i0_daily_field[~np.isnan(w_i0_daily_field)] = 1
        w_i0_daily_field[np.isnan(w_i0_daily_field)] = 0
        w_i0_day_field = w_i0_day_field + w_i0_daily_field

        w_ig_avg_field = w_ig_avg_field + np.nan_to_num(w_ig_daily_field)
        w_ig_daily_field[~np.isnan(w_ig_daily_field)] = 1
        w_ig_daily_field[np.isnan(w_ig_daily_field)] = 0
        w_ig_day_field = w_ig_day_field + w_ig_daily_field

        w_A_avg_field = w_A_avg_field + np.nan_to_num(w_A_daily_field)
        w_A_daily_field[~np.isnan(w_A_daily_field)] = 1
        w_A_daily_field[np.isnan(w_A_daily_field)] = 0
        w_A_day_field = w_A_day_field + w_A_daily_field

        gamma_avg_field = gamma_avg_field + np.nan_to_num(gamma_daily_field)
        gamma_daily_field[~np.isnan(gamma_daily_field)] = 1
        gamma_daily_field[np.isnan(gamma_daily_field)] = 0
        gamma_day_field = gamma_day_field + gamma_daily_field

    alpha_avg_field = np.divide(alpha_avg_field, alpha_day_field)
    tau_io_x_nogeo_avg_field = np.divide(tau_io_x_nogeo_avg_field, tau_io_x_nogeo_day_field)
    tau_io_y_nogeo_avg_field = np.divide(tau_io_y_nogeo_avg_field, tau_io_y_nogeo_day_field)
    tau_io_x_geo_avg_field = np.divide(tau_io_x_geo_avg_field, tau_io_x_geo_day_field)
    tau_io_y_geo_avg_field = np.divide(tau_io_y_geo_avg_field, tau_io_y_geo_day_field)
    tau_ig_x_avg_field = np.divide(tau_ig_x_avg_field, tau_ig_x_day_field)
    tau_ig_y_avg_field = np.divide(tau_ig_y_avg_field, tau_ig_y_day_field)
    tau_ao_x_avg_field = np.divide(tau_ao_x_avg_field, tau_ao_x_day_field)
    tau_ao_y_avg_field = np.divide(tau_ao_y_avg_field, tau_ao_y_day_field)
    w_Ek_nogeo_avg_field = np.divide(w_Ek_nogeo_avg_field, w_Ek_nogeo_day_field)
    w_Ek_geo_avg_field = np.divide(w_Ek_geo_avg_field, w_Ek_geo_day_field)
    w_a_avg_field = np.divide(w_a_avg_field, w_a_day_field)
    w_i_avg_field = np.divide(w_i_avg_field, w_i_day_field)
    w_i0_avg_field = np.divide(w_i0_avg_field, w_i0_day_field)
    w_ig_avg_field = np.divide(w_ig_avg_field, w_ig_day_field)
    w_A_avg_field = np.divide(w_A_avg_field, w_A_day_field)
    gamma_avg_field = np.divide(gamma_avg_field, gamma_day_field)

    nc_dir = os.path.dirname(output_dir_path)
    nc_filepath = os.path.join(nc_dir, 'ice_ocean_govenor_{:}_{:}.nc'.format(dates[0], dates[-1]))

    if not os.path.exists(nc_dir):
        logger.info('Creating directory: {:s}'.format(nc_dir))
        os.makedirs(nc_dir)

    logger.info('Saving fields to netCDF file: {:s}'.format(nc_filepath))

    tau_dataset = netCDF4.Dataset(nc_filepath, 'w')

    tau_dataset.title = 'Ekman pumping in the Antarctic sea ice zone'
    tau_dataset.institution = 'Department of Earth, Atmospheric, and Planetary Science, ' \
                              'Massachusetts Institute of Technology'

    tau_dataset.createDimension('time', None)
    tau_dataset.createDimension('lat', len(lats))
    tau_dataset.createDimension('lon', len(lons))

    # TODO: Actually store a date.
    time_var = tau_dataset.createVariable('time', np.float64, ('time',))
    time_var.units = 'hours since 0001-01-01 00:00:00'
    time_var.calendar = 'gregorian'

    # d = datetime.datetime(dates[-1])
    # time_var[:] = netCDF4.date2num(d, units=time_var.units, calendar=time_var.calendar)

    lat_var = tau_dataset.createVariable('lat', np.float32, ('lat',))
    lat_var.units = 'degrees south'
    lat_var[:] = lats

    lon_var = tau_dataset.createVariable('lon', np.float32, ('lon',))
    lat_var.units = 'degrees west/east'
    lon_var[:] = lons

    var_fields = {
        'alpha': alpha_avg_field,
        'tau_io_x_nogeo': tau_io_x_nogeo_avg_field,
        'tau_io_y_nogeo': tau_io_y_nogeo_avg_field,
        'tau_io_x_geo': tau_io_x_geo_avg_field,
        'tau_io_y_geo': tau_io_y_geo_avg_field,
        'tau_ig_x': tau_ig_x_avg_field,
        'tau_ig_y': tau_ig_y_avg_field,
        'tau_ao_x': tau_ao_x_avg_field,
        'tau_ao_y': tau_ao_y_avg_field,
        'w_Ek_nogeo': w_Ek_nogeo_avg_field,
        'w_Ek_geo': w_Ek_geo_avg_field,
        'w_a': w_a_avg_field,
        'w_i': w_i_avg_field,
        'w_i0': w_i0_avg_field,
        'w_ig': w_ig_avg_field,
        'w_A': w_A_avg_field,
        'gamma': gamma_avg_field
    }

    for var_name in var_fields.keys():
        field_var = tau_dataset.createVariable(var_name, float, ('lat', 'lon'), zlib=True)
        field_var[:] = var_fields[var_name]

    tau_dataset.close()


if __name__ == '__main__':
    # dates = date_range(datetime.date(2011, 7, 1), datetime.date(2011, 9, 30))

    dates = []
    for year in range(2011, 2015 + 1):
        start_date = datetime.date(year, 7, 1)
        end_date = datetime.date(year, 9, 30)

        dates = dates + date_range(start_date, end_date)

    # retroactively_compute_ice_ocean_governor_variables(dates)
    start = datetime.date(2011, 1, 1)
    end = datetime.date(2015, 12, 31)
    retroactively_compute_ice_ocean_governor_variables(date_range(start, end))
