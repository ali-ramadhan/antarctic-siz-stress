import os
import datetime
import calendar

import netCDF4
import numpy as np

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig('logging.ini')
logger = logging.getLogger(__name__)

from SeaIceThicknessDataset import SeaIceThicknessDataset
from constants import output_dir_path, C_fw
from utils import date_range, distance, log_netCDF_dataset_metadata, get_netCDF_filepath

np.set_printoptions(precision=4)


def retroactively_compute_sea_ice_advection():
    start_date = datetime.date(2005, 1, 1)
    end_date = datetime.date(2015, 12, 31)
    dates = date_range(start_date, end_date)

    h_ice_dataset = SeaIceThicknessDataset(start_date)

    try:
        tau_filepath = get_netCDF_filepath(field_type='daily', date=start_date)
        tau_dataset = netCDF4.Dataset(tau_filepath)
        lats = np.array(tau_dataset.variables['lat'])
        lons = np.array(tau_dataset.variables['lon'])
    except OSError as e:
        logger.error('{}'.format(e))
        logger.error('{:s} not found. Could not load lat, lon arrays.'.format(tau_filepath))

    alpha_avg_field = np.zeros((len(lats), len(lons)))
    alpha_day_field = np.zeros((len(lats), len(lons)))
    u_ice_avg_field = np.zeros((len(lats), len(lons)))
    u_ice_day_field = np.zeros((len(lats), len(lons)))
    v_ice_avg_field = np.zeros((len(lats), len(lons)))
    v_ice_day_field = np.zeros((len(lats), len(lons)))
    h_ice_avg_field = np.zeros((len(lats), len(lons)))
    h_ice_day_field = np.zeros((len(lats), len(lons)))
    zonal_div_avg_field = np.zeros((len(lats), len(lons)))
    zonal_div_day_field = np.zeros((len(lats), len(lons)))
    merid_div_avg_field = np.zeros((len(lats), len(lons)))
    merid_div_day_field = np.zeros((len(lats), len(lons)))
    div_avg_field = np.zeros((len(lats), len(lons)))
    div_day_field = np.zeros((len(lats), len(lons)))

    for date in dates:
        tau_filepath = get_netCDF_filepath(field_type='daily', date=date)

        logger.info('Averaging {:%b %d, %Y} ({:s})...'.format(date, tau_filepath))

        try:
            current_tau_dataset = netCDF4.Dataset(tau_filepath)
            log_netCDF_dataset_metadata(current_tau_dataset)
        except OSError as e:
            logger.error('{}'.format(e))
            logger.warning('{:s} not found. Proceeding without it...'.format(tau_filepath))
            continue

        alpha_daily_field = np.array(current_tau_dataset.variables['alpha'])
        u_ice_daily_field = np.array(current_tau_dataset.variables['ice_u'])
        v_ice_daily_field = np.array(current_tau_dataset.variables['ice_v'])

        h_ice_daily_field = np.zeros((len(lats), len(lons)))
        zonal_div_daily_field = np.zeros((len(lats), len(lons)))
        merid_div_daily_field = np.zeros((len(lats), len(lons)))
        div_daily_field = np.zeros((len(lats), len(lons)))

        # Load h_ice field for the day (i.e. the correct seasonal field).
        for i in range(len(lats)):
            for j in range(len(lons)):
                h_ice_daily_field[i][j] = h_ice_dataset.sea_ice_thickness(i, j, date)

        i_max = len(lats) - 1
        j_max = len(lons) - 1

        for i in range(1, len(lats) - 1):
            # lat = lats[i]
            # progress_percent = 100 * i / (len(lats) - 2)
            # logger.info('({:}, ice_div) lat = {:.2f}/{:.2f} ({:.1f}%)'.format(date, lat, -40, progress_percent))

            dx = distance(lats[i-1], lons[0], lats[i+1], lons[0])
            dy = distance(lats[i], lons[0], lats[i], lons[2])

            for j in range(1, len(lons) - 1):
                # Taking modulus of j-1 and j+1 to get the correct index in the special cases of
                #  * j=0 (180 W) and need to use the value from j=j_max (180 E)
                #  * j=j_max (180 E) and need to use the value from j=0 (180 W)
                jm1 = (j - 1) % j_max
                jp1 = (j + 1) % j_max

                u_ice_i_jp1 = u_ice_daily_field[i][jp1]
                u_ice_i_jm1 = u_ice_daily_field[i][jm1]
                v_ice_ip1_j = v_ice_daily_field[i+1][j]
                v_ice_im1_j = v_ice_daily_field[i-1][j]

                alpha_i_jp1 = alpha_daily_field[i][jp1]
                alpha_i_jm1 = alpha_daily_field[i][jm1]
                alpha_ip1_j = alpha_daily_field[i+1][j]
                alpha_im1_j = alpha_daily_field[i-1][j]

                h_ice_i_jp1 = h_ice_daily_field[i][jp1]
                h_ice_i_jm1 = h_ice_daily_field[i][jm1]
                h_ice_ip1_j = h_ice_daily_field[i+1][j]
                h_ice_im1_j = h_ice_daily_field[i-1][j]

                if not np.isnan(u_ice_i_jm1) and not np.isnan(u_ice_i_jp1):
                    div_x = (alpha_i_jp1 * h_ice_i_jp1 * u_ice_i_jp1 - alpha_i_jm1 * h_ice_i_jm1 * u_ice_i_jm1) / dx
                    zonal_div_daily_field[i][j] = div_x
                else:
                    zonal_div_daily_field[i][j] = np.nan

                if not np.isnan(v_ice_im1_j) and not np.isnan(v_ice_ip1_j):
                    div_y = (alpha_ip1_j * h_ice_ip1_j * v_ice_ip1_j - alpha_im1_j * h_ice_im1_j * v_ice_im1_j) / dy
                    merid_div_daily_field[i][j] = div_y
                else:
                    merid_div_daily_field[i][j] = np.nan

                if not np.isnan(zonal_div_daily_field[i][j]) and not np.isnan(merid_div_daily_field[i][j]):
                    div_daily_field[i][j] = C_fw * (div_x + div_y)
                else:
                    div_daily_field[i][j] = np.nan

        alpha_avg_field = alpha_avg_field + np.nan_to_num(alpha_daily_field)
        alpha_daily_field[~np.isnan(alpha_daily_field)] = 1
        alpha_daily_field[np.isnan(alpha_daily_field)] = 0
        alpha_day_field = alpha_day_field + alpha_daily_field

        u_ice_avg_field = u_ice_avg_field + np.nan_to_num(u_ice_daily_field)
        u_ice_daily_field[~np.isnan(u_ice_daily_field)] = 1
        u_ice_daily_field[np.isnan(u_ice_daily_field)] = 0
        u_ice_day_field = u_ice_day_field + u_ice_daily_field

        v_ice_avg_field = v_ice_avg_field + np.nan_to_num(v_ice_daily_field)
        v_ice_daily_field[~np.isnan(v_ice_daily_field)] = 1
        v_ice_daily_field[np.isnan(v_ice_daily_field)] = 0
        v_ice_day_field = v_ice_day_field + v_ice_daily_field

        h_ice_avg_field = h_ice_avg_field + np.nan_to_num(h_ice_daily_field)
        h_ice_daily_field[~np.isnan(h_ice_daily_field)] = 1
        h_ice_daily_field[np.isnan(h_ice_daily_field)] = 0
        h_ice_day_field = h_ice_day_field + h_ice_daily_field

        zonal_div_avg_field = zonal_div_avg_field + np.nan_to_num(zonal_div_daily_field)
        zonal_div_daily_field[~np.isnan(zonal_div_daily_field)] = 1
        zonal_div_daily_field[np.isnan(zonal_div_daily_field)] = 0
        zonal_div_day_field = zonal_div_day_field + zonal_div_daily_field

        merid_div_avg_field = merid_div_avg_field + np.nan_to_num(merid_div_daily_field)
        merid_div_daily_field[~np.isnan(merid_div_daily_field)] = 1
        merid_div_daily_field[np.isnan(merid_div_daily_field)] = 0
        merid_div_day_field = merid_div_day_field + merid_div_daily_field

        div_avg_field = div_avg_field + np.nan_to_num(div_daily_field)
        div_daily_field[~np.isnan(div_daily_field)] = 1
        div_daily_field[np.isnan(div_daily_field)] = 0
        div_day_field = div_day_field + div_daily_field

    alpha_avg_field = np.divide(alpha_avg_field, alpha_day_field)
    u_ice_avg_field = np.divide(u_ice_avg_field, u_ice_day_field)
    v_ice_avg_field = np.divide(v_ice_avg_field, v_ice_day_field)
    h_ice_avg_field = np.divide(h_ice_avg_field, h_ice_day_field)
    zonal_div_avg_field = np.divide(zonal_div_avg_field, zonal_div_day_field)
    merid_div_avg_field = np.divide(merid_div_avg_field, merid_div_day_field)
    div_avg_field = np.divide(div_avg_field, div_day_field)

    nc_dir = os.path.dirname(output_dir_path)
    nc_filepath = os.path.join(nc_dir, 'ice_flux_div_{:}_{:}.nc'.format(start_date, end_date))

    if not os.path.exists(nc_dir):
        logger.info('Creating directory: {:s}'.format(nc_dir))
        os.makedirs(nc_dir)

    logger.info('Saving fields to netCDF file: {:s}'.format(nc_filepath))

    tau_dataset = netCDF4.Dataset(nc_filepath, 'w')

    tau_dataset.title = 'Ice flux divergence in the Antarctic sea ice zone'
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
        'alpha': alpha_avg_field,
        'u_ice': u_ice_avg_field,
        'v_ice': v_ice_avg_field,
        'h_ice': h_ice_avg_field,
        'zonal_div': zonal_div_avg_field,
        'merid_div': merid_div_avg_field,
        'div': div_avg_field
    }

    for var_name in var_fields.keys():
        field_var = tau_dataset.createVariable(var_name, float, ('lat', 'lon'), zlib=True)
        field_var[:] = var_fields[var_name]

    tau_dataset.close()


def retroactively_compute_melting_freezing_rate():
    from constants import D_e, kappa

    start_date = datetime.date(2011, 1, 1)
    end_date = datetime.date(2016, 12, 31)
    dates = date_range(start_date, end_date)

    h_ice_dataset = SeaIceThicknessDataset(start_date)

    try:
        tau_filepath = get_netCDF_filepath(field_type='daily', date=start_date)
        tau_dataset = netCDF4.Dataset(tau_filepath)
        lats = np.array(tau_dataset.variables['lat'])
        lons = np.array(tau_dataset.variables['lon'])
    except OSError as e:
        logger.error('{}'.format(e))
        logger.error('{:s} not found. Could not load lat, lon arrays.'.format(tau_filepath))

    alpha_avg_field = np.zeros((len(lats), len(lons)))
    alpha_day_field = np.zeros((len(lats), len(lons)))
    u_ice_avg_field = np.zeros((len(lats), len(lons)))
    u_ice_day_field = np.zeros((len(lats), len(lons)))
    v_ice_avg_field = np.zeros((len(lats), len(lons)))
    v_ice_day_field = np.zeros((len(lats), len(lons)))
    h_ice_avg_field = np.zeros((len(lats), len(lons)))
    h_ice_day_field = np.zeros((len(lats), len(lons)))
    zonal_div_avg_field = np.zeros((len(lats), len(lons)))
    zonal_div_day_field = np.zeros((len(lats), len(lons)))
    merid_div_avg_field = np.zeros((len(lats), len(lons)))
    merid_div_day_field = np.zeros((len(lats), len(lons)))
    div_avg_field = np.zeros((len(lats), len(lons)))
    div_day_field = np.zeros((len(lats), len(lons)))

    Ekman_term_avg_field = np.zeros((len(lats), len(lons)))
    Ekman_term_day_field = np.zeros((len(lats), len(lons)))
    geo_term_avg_field = np.zeros((len(lats), len(lons)))
    geo_term_day_field = np.zeros((len(lats), len(lons)))
    diffusion_term_avg_field = np.zeros((len(lats), len(lons)))
    diffusion_term_day_field = np.zeros((len(lats), len(lons)))

    for date in dates:
        tau_filepath = get_netCDF_filepath(field_type='daily', date=date)

        logger.info('Averaging {:%b %d, %Y} ({:s})...'.format(date, tau_filepath))

        try:
            current_tau_dataset = netCDF4.Dataset(tau_filepath)
            log_netCDF_dataset_metadata(current_tau_dataset)
        except OSError as e:
            logger.error('{}'.format(e))
            logger.warning('{:s} not found. Proceeding without it...'.format(tau_filepath))
            continue

        alpha_daily_field = np.array(current_tau_dataset.variables['alpha'])
        u_ice_daily_field = np.array(current_tau_dataset.variables['ice_u'])
        v_ice_daily_field = np.array(current_tau_dataset.variables['ice_v'])

        salinity_daily_field = np.array(current_tau_dataset.variables['salinity'])
        u_geo_daily_field = np.array(current_tau_dataset.variables['geo_u'])
        v_geo_daily_field = np.array(current_tau_dataset.variables['geo_v'])
        U_Ekman_daily_field = np.array(current_tau_dataset.variables['Ekman_U'])
        V_Ekman_daily_field = np.array(current_tau_dataset.variables['Ekman_V'])

        h_ice_daily_field = np.zeros((len(lats), len(lons)))
        zonal_div_daily_field = np.zeros((len(lats), len(lons)))
        merid_div_daily_field = np.zeros((len(lats), len(lons)))
        div_daily_field = np.zeros((len(lats), len(lons)))
        Ekman_term_daily_field = np.zeros((len(lats), len(lons)))
        geo_term_daily_field = np.zeros((len(lats), len(lons)))
        diffusion_term_daily_field = np.zeros((len(lats), len(lons)))

        # Load h_ice field for the day (i.e. the correct seasonal field).
        for i in range(len(lats)):
            for j in range(len(lons)):
                h_ice_daily_field[i][j] = h_ice_dataset.sea_ice_thickness(i, j, date)

        i_max = len(lats) - 1
        j_max = len(lons) - 1

        for i in range(1, len(lats) - 1):
            # lat = lats[i]
            # progress_percent = 100 * i / (len(lats) - 2)
            # logger.info('({:}, ice_div) lat = {:.2f}/{:.2f} ({:.1f}%)'.format(date, lat, -40, progress_percent))

            dx = distance(lats[i-1], lons[0], lats[i+1], lons[0])
            dy = distance(lats[i], lons[0], lats[i], lons[2])

            for j in range(len(lons)):
                # Taking modulus of j-1 and j+1 to get the correct index in the special cases of
                #  * j=0 (180 W) and need to use the value from j=j_max (180 E)
                #  * j=j_max (180 E) and need to use the value from j=0 (180 W)
                jm1 = (j - 1) % j_max
                jp1 = (j + 1) % j_max

                u_ice_i_jp1 = u_ice_daily_field[i][jp1]
                u_ice_i_jm1 = u_ice_daily_field[i][jm1]
                v_ice_ip1_j = v_ice_daily_field[i+1][j]
                v_ice_im1_j = v_ice_daily_field[i-1][j]

                alpha_i_jp1 = alpha_daily_field[i][jp1]
                alpha_i_jm1 = alpha_daily_field[i][jm1]
                alpha_ip1_j = alpha_daily_field[i+1][j]
                alpha_im1_j = alpha_daily_field[i-1][j]

                h_ice_i_jp1 = h_ice_daily_field[i][jp1]
                h_ice_i_jm1 = h_ice_daily_field[i][jm1]
                h_ice_ip1_j = h_ice_daily_field[i+1][j]
                h_ice_im1_j = h_ice_daily_field[i-1][j]

                S_ij = salinity_daily_field[i][j]
                S_i_jp1 = salinity_daily_field[i][jp1]
                S_i_jm1 = salinity_daily_field[i][jm1]
                S_ip1_j = salinity_daily_field[i+1][j]
                S_im1_j = salinity_daily_field[i-1][j]

                u_geo_ij = u_geo_daily_field[i][j]
                v_geo_ij = v_geo_daily_field[i][j]
                U_Ekman_ij = U_Ekman_daily_field[i][j]
                V_Ekman_ij = V_Ekman_daily_field[i][j]

                if not np.isnan(u_ice_i_jm1) and not np.isnan(u_ice_i_jp1):
                    div_x = (alpha_i_jp1 * h_ice_i_jp1 * u_ice_i_jp1 - alpha_i_jm1 * h_ice_i_jm1 * u_ice_i_jm1) / dx
                    zonal_div_daily_field[i][j] = div_x
                else:
                    zonal_div_daily_field[i][j] = np.nan

                if not np.isnan(v_ice_im1_j) and not np.isnan(v_ice_ip1_j):
                    div_y = (alpha_ip1_j * h_ice_ip1_j * v_ice_ip1_j - alpha_im1_j * h_ice_im1_j * v_ice_im1_j) / dy
                    merid_div_daily_field[i][j] = div_y
                else:
                    merid_div_daily_field[i][j] = np.nan

                if not np.isnan(zonal_div_daily_field[i][j]) and not np.isnan(merid_div_daily_field[i][j]):
                    div_daily_field[i][j] = div_x + div_y
                else:
                    div_daily_field[i][j] = np.nan

                if not np.isnan(S_ij) and not np.isnan(S_i_jm1) and not np.isnan(S_i_jp1) and not np.isnan(S_ip1_j) \
                        and not np.isnan(S_im1_j):
                    dSdx = (S_i_jp1 - S_i_jm1) / dx
                    dSdy = (S_ip1_j - S_im1_j) / dy
                    del2_S = ((S_i_jp1 - 2*S_ij + S_i_jm1) / dx**2) + ((S_ip1_j - 2*S_ij + S_im1_j) / dy**2)

                    Ekman_term_daily_field[i][j] = (U_Ekman_ij*dSdx + V_Ekman_ij*dSdy) / S_ij
                    geo_term_daily_field[i][j] = (D_e / S_ij) * (u_geo_ij*dSdx + v_geo_ij*dSdy)
                    diffusion_term_daily_field[i][j] = (kappa*D_e / S_ij) * del2_S
                else:
                    Ekman_term_daily_field[i][j] = np.nan
                    geo_term_daily_field[i][j] = np.nan
                    diffusion_term_daily_field[i][j] = np.nan

                if np.isnan(alpha_daily_field[i][j]) or alpha_daily_field[i][j] < 0.15:
                    Ekman_term_daily_field[i][j] = np.nan
                    geo_term_daily_field[i][j] = np.nan
                    diffusion_term_daily_field[i][j] = np.nan

        alpha_avg_field = alpha_avg_field + np.nan_to_num(alpha_daily_field)
        alpha_daily_field[~np.isnan(alpha_daily_field)] = 1
        alpha_daily_field[np.isnan(alpha_daily_field)] = 0
        alpha_day_field = alpha_day_field + alpha_daily_field

        u_ice_avg_field = u_ice_avg_field + np.nan_to_num(u_ice_daily_field)
        u_ice_daily_field[~np.isnan(u_ice_daily_field)] = 1
        u_ice_daily_field[np.isnan(u_ice_daily_field)] = 0
        u_ice_day_field = u_ice_day_field + u_ice_daily_field

        v_ice_avg_field = v_ice_avg_field + np.nan_to_num(v_ice_daily_field)
        v_ice_daily_field[~np.isnan(v_ice_daily_field)] = 1
        v_ice_daily_field[np.isnan(v_ice_daily_field)] = 0
        v_ice_day_field = v_ice_day_field + v_ice_daily_field

        h_ice_avg_field = h_ice_avg_field + np.nan_to_num(h_ice_daily_field)
        h_ice_daily_field[~np.isnan(h_ice_daily_field)] = 1
        h_ice_daily_field[np.isnan(h_ice_daily_field)] = 0
        h_ice_day_field = h_ice_day_field + h_ice_daily_field

        zonal_div_avg_field = zonal_div_avg_field + np.nan_to_num(zonal_div_daily_field)
        zonal_div_daily_field[~np.isnan(zonal_div_daily_field)] = 1
        zonal_div_daily_field[np.isnan(zonal_div_daily_field)] = 0
        zonal_div_day_field = zonal_div_day_field + zonal_div_daily_field

        merid_div_avg_field = merid_div_avg_field + np.nan_to_num(merid_div_daily_field)
        merid_div_daily_field[~np.isnan(merid_div_daily_field)] = 1
        merid_div_daily_field[np.isnan(merid_div_daily_field)] = 0
        merid_div_day_field = merid_div_day_field + merid_div_daily_field

        div_avg_field = div_avg_field + np.nan_to_num(div_daily_field)
        div_daily_field[~np.isnan(div_daily_field)] = 1
        div_daily_field[np.isnan(div_daily_field)] = 0
        div_day_field = div_day_field + div_daily_field

        Ekman_term_avg_field = Ekman_term_avg_field + np.nan_to_num(Ekman_term_daily_field)
        Ekman_term_daily_field[~np.isnan(Ekman_term_daily_field)] = 1
        Ekman_term_daily_field[np.isnan(Ekman_term_daily_field)] = 0
        Ekman_term_day_field = Ekman_term_day_field + Ekman_term_daily_field

        geo_term_avg_field = geo_term_avg_field + np.nan_to_num(geo_term_daily_field)
        geo_term_daily_field[~np.isnan(geo_term_daily_field)] = 1
        geo_term_daily_field[np.isnan(geo_term_daily_field)] = 0
        geo_term_day_field = geo_term_day_field + geo_term_daily_field

        diffusion_term_avg_field = diffusion_term_avg_field + np.nan_to_num(diffusion_term_daily_field)
        diffusion_term_daily_field[~np.isnan(diffusion_term_daily_field)] = 1
        diffusion_term_daily_field[np.isnan(diffusion_term_daily_field)] = 0
        diffusion_term_day_field = diffusion_term_day_field + diffusion_term_daily_field

    alpha_avg_field = np.divide(alpha_avg_field, alpha_day_field)
    u_ice_avg_field = np.divide(u_ice_avg_field, u_ice_day_field)
    v_ice_avg_field = np.divide(v_ice_avg_field, v_ice_day_field)
    h_ice_avg_field = np.divide(h_ice_avg_field, h_ice_day_field)
    zonal_div_avg_field = np.divide(zonal_div_avg_field, zonal_div_day_field)
    merid_div_avg_field = np.divide(merid_div_avg_field, merid_div_day_field)
    div_avg_field = 3600*24*365 * np.divide(div_avg_field, div_day_field)
    Ekman_term_avg_field = 3600*24*365 * np.divide(Ekman_term_avg_field, Ekman_term_day_field)
    geo_term_avg_field = 3600*24*365 * np.divide(geo_term_avg_field, geo_term_day_field)
    diffusion_term_avg_field = 3600*24*365 * np.divide(diffusion_term_avg_field, diffusion_term_day_field)

    nc_dir = os.path.dirname(output_dir_path)
    nc_filepath = os.path.join(nc_dir, 'melting_freezing_rate_{:}_{:}.nc'.format(start_date, end_date))

    if not os.path.exists(nc_dir):
        logger.info('Creating directory: {:s}'.format(nc_dir))
        os.makedirs(nc_dir)

    logger.info('Saving fields to netCDF file: {:s}'.format(nc_filepath))

    tau_dataset = netCDF4.Dataset(nc_filepath, 'w')

    tau_dataset.title = 'Melting and freezing rates in the Antarctic sea ice zone'
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
        'alpha': alpha_avg_field,
        'u_ice': u_ice_avg_field,
        'v_ice': v_ice_avg_field,
        'h_ice': h_ice_avg_field,
        'zonal_div': zonal_div_avg_field,
        'merid_div': merid_div_avg_field,
        'div': div_avg_field,
        'Ekman_term': Ekman_term_avg_field,
        'geo_term': geo_term_avg_field,
        'diffusion_term': diffusion_term_avg_field
    }

    for var_name in var_fields.keys():
        field_var = tau_dataset.createVariable(var_name, float, ('lat', 'lon'), zlib=True)
        field_var[:] = var_fields[var_name]

    tau_dataset.close()


if __name__ == '__main__':
    # retroactively_compute_sea_ice_advection()
    retroactively_compute_melting_freezing_rate()
