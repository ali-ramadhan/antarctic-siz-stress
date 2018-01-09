import os
import numpy as np
import netCDF4

from GeostrophicVelocityDataReader import GeostrophicVelocityDataReader
from OceanSurfaceWindVectorDataReader import OceanSurfaceWindVectorDataReader
from SeaIceConcentrationDataReader import SeaIceConcentrationDataReader
from SeaIceMotionDataReader import SeaIceMotionDataReader

from utils import distance
from constants import output_dir_path
from constants import lat_min, lat_max, lat_step, n_lat, lon_min, lon_max, lon_step, n_lon
from constants import rho_air, rho_seawater, C_air, C_seawater
from constants import Omega, rho_0, D_e

import logging
logger = logging.getLogger(__name__)


class SurfaceStressDataWriter(object):
    """
    Such an object should mainly compute daily (averaged) wind stress and wind stress curl fields and write them out
    to netCDF files. Computing monthly means makes sense here. But plotting should go elsewhere.
    """

    from constants import output_dir_path
    surface_stress_dir = os.path.join(output_dir_path, 'surface_stress')

    R_45deg = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]])

    def __init__(self, date):
        self.lats = np.linspace(lat_min, lat_max, n_lat)
        self.lons = np.linspace(lon_min, lon_max, n_lon)
        self.lons = self.lons[:-1]  # Remove the +180 longitude as it coincides with the -180 longitude.

        # Initializing all the fields we want to write to the netCDF file.
        self.tau_air_x_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_air_y_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_ice_x_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_ice_y_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_SIZ_x_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_SIZ_y_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_x_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_y_field = np.zeros((len(self.lats), len(self.lons)))

        self.u_Ekman_field = np.zeros((len(self.lats), len(self.lons)))
        self.v_Ekman_field = np.zeros((len(self.lats), len(self.lons)))
        self.u_Ekman_SIZ_field = np.zeros((len(self.lats), len(self.lons)))
        self.v_Ekman_SIZ_field = np.zeros((len(self.lats), len(self.lons)))

        self.u_geo_field = np.zeros((len(self.lats), len(self.lons)))
        self.v_geo_field = np.zeros((len(self.lats), len(self.lons)))
        self.u_wind_field = np.zeros((len(self.lats), len(self.lons)))
        self.v_wind_field = np.zeros((len(self.lats), len(self.lons)))
        self.alpha_field = np.zeros((len(self.lats), len(self.lons)))
        self.u_ice_field = np.zeros((len(self.lats), len(self.lons)))
        self.v_ice_field = np.zeros((len(self.lats), len(self.lons)))

        self.wind_stress_curl_field = np.zeros((len(self.lats), len(self.lons)))
        self.w_Ekman_field = np.zeros((len(self.lats), len(self.lons)))

        if date is not None:
            self.date = date

            self.u_geo_data = GeostrophicVelocityDataReader(self.date)
            self.sea_ice_conc_data = SeaIceConcentrationDataReader(self.date)
            self.sea_ice_motion_data = SeaIceMotionDataReader(self.date)
            self.u_wind_data = OceanSurfaceWindVectorDataReader(self.date)

    def surface_stress(self, f, u_geo_vec, u_wind_vec, alpha, u_ice_vec):
        # Use the Modified Richardson iteration to calculate tau and u_Ekman. Here we set the variables to arbitrary
        # initial guesses.
        iter_count = 0
        tau_vec_residual = np.array([1, 1])
        tau_relative_error = 1
        tau_air_vec = np.array([0, 0])
        tau_ice_vec = np.array([0, 0])
        tau_vec = np.array([0, 0])
        u_Ekman_vec = np.array([0.001, 0.001])
        omega = 0.01  # Richardson relaxation parameter

        while np.linalg.norm(tau_vec_residual) > 1e-5:
            iter_count = iter_count + 1
            if iter_count > 50:
                logger.warning('iter_acount exceeded 50 during calculation of tau and u_Ekman.')
                logger.warning('tau = {}, u_Ekman = {}, tau_residual = {}, tau_rel_error = {:.4f}'
                               .format(tau_vec, u_Ekman_vec, tau_vec_residual, tau_relative_error))
                break

            if np.linalg.norm(tau_vec) > 10:
                logger.warning('Large tau = {}, u_geo_mean = {}, u_wind = {}, alpha = {:.4f}, u_ice = {}'
                               .format(tau_vec, u_geo_vec, u_wind_vec, alpha, u_ice_vec))
                break

            tau_air_vec = rho_air * C_air * np.linalg.norm(u_wind_vec) * u_wind_vec

            u_Ekman_vec = (np.sqrt(2) / (f * rho_0 * D_e)) * np.matmul(self.R_45deg, tau_vec)
            u_rel_vec = u_ice_vec - (u_geo_vec - u_Ekman_vec)
            tau_ice_vec = rho_0 * C_seawater * np.linalg.norm(u_rel_vec) * u_rel_vec
            tau_vec = alpha * tau_ice_vec + (1 - alpha) * tau_air_vec

            tau_vec_residual = tau_vec - (alpha * tau_ice_vec + (1 - alpha) * tau_air_vec)
            tau_relative_error = np.linalg.norm(tau_vec_residual) / np.linalg.norm(tau_vec)

            tau_vec = tau_vec + omega * tau_vec_residual

            if np.isnan(tau_vec[0]) or np.isnan(tau_vec[1]):
                logger.warning('NaN tau = {}, u_geo_mean = {}, u_wind = {}, alpha = {:.4f}, u_ice = {}'
                               .format(tau_vec, u_geo_vec, u_wind_vec, alpha, u_ice_vec))

        return tau_vec, tau_air_vec, tau_ice_vec

    def compute_daily_surface_stress_field(self):
        logger.info('Calculating surface stress field (tau_x, tau_y) for:')
        logger.info('lat_min = {}, lat_max = {}, lat_step = {}, n_lat = {}'.format(lat_min, lat_max, lat_step, n_lat))
        logger.info('lon_min = {}, lon_max = {}, lon_step = {}, n_lon = {}'.format(lon_min, lon_max, lon_step, n_lon))

        for i in range(len(self.lats)):
            lat = self.lats[i]
            f = 2 * Omega * np.sin(np.deg2rad(lat))  # Coriolis parameter [s^-1]

            progress_percent = 100 * i / (len(self.lats) - 1)
            logger.info('lat = {:.2f}/{:.2f} ({:.1f}%)'.format(lat, lat_max, progress_percent))

            for j in range(len(self.lons)):
                lon = self.lons[j]

                u_geo_vec = self.u_geo_data.absolute_geostrophic_velocity(lat, lon, self.date, 'interp')
                u_wind_vec = self.u_wind_data.ocean_surface_wind_vector(lat, lon, self.date, 'interp')
                alpha = self.sea_ice_conc_data.sea_ice_concentration(lat, lon, self.date, 'interp')
                u_ice_vec = self.sea_ice_motion_data.seaice_motion_vector(lat, lon, self.date, 'interp')

                # # TODO: Bit of a hack lol to get u_wind values for the 358-360 longitude sector.
                # if np.isnan(u_wind_vec[0]) or np.isnan(u_wind_vec[1]):
                #     u_wind_vec = self.u_wind_data.ocean_surface_wind_vector(lat, lon, self.date, 'product')

                self.u_geo_field[i][j] = u_geo_vec[0]
                self.v_geo_field[i][j] = u_geo_vec[1]
                self.u_wind_field[i][j] = u_wind_vec[0]
                self.v_wind_field[i][j] = u_wind_vec[1]
                self.alpha_field[i][j] = alpha
                self.u_ice_field[i][j] = u_ice_vec[0]
                self.v_ice_field[i][j] = u_ice_vec[1]

                # If there's no sea ice at a point and we have data at that point (i.e. the point is still in the ocean)
                # then tau is just tau_air and easy to calculate.
                if ((alpha == 0 or np.isnan(alpha)) and np.isnan(u_ice_vec[0])) \
                        and not np.isnan(u_geo_vec[0]) and not np.isnan(u_wind_vec[0]):

                    tau_air_vec = rho_air * C_air * np.linalg.norm(u_wind_vec) * u_wind_vec

                    self.tau_air_x_field[i][j] = tau_air_vec[0]
                    self.tau_air_y_field[i][j] = tau_air_vec[1]
                    self.tau_ice_x_field[i][j] = 0
                    self.tau_ice_y_field[i][j] = 0

                    self.tau_x_field[i][j] = tau_air_vec[0]
                    self.tau_y_field[i][j] = tau_air_vec[1]
                    self.tau_SIZ_x_field[i][j] = np.nan
                    self.tau_SIZ_y_field[i][j] = np.nan

                    # Not sure why I have to recalculate u_Ekman_vec otherwise it's just zero.
                    u_Ekman_vec = (np.sqrt(2) / (f * rho_0 * D_e)) * np.matmul(self.R_45deg, tau_air_vec)
                    self.u_Ekman_field[i][j] = u_Ekman_vec[0]
                    self.v_Ekman_field[i][j] = u_Ekman_vec[1]
                    self.u_Ekman_SIZ_field[i][j] = np.nan
                    self.v_Ekman_SIZ_field[i][j] = np.nan
                    continue

                # If we have data missing, then we're probably on land or somewhere where we cannot calculate tau.
                if np.isnan(alpha) or np.isnan(u_geo_vec[0]) or np.isnan(u_wind_vec[0]) or np.isnan(u_ice_vec[0]):
                    self.tau_air_x_field[i][j] = np.nan
                    self.tau_air_y_field[i][j] = np.nan
                    self.tau_ice_x_field[i][j] = np.nan
                    self.tau_ice_y_field[i][j] = np.nan
                    self.tau_x_field[i][j] = np.nan
                    self.tau_y_field[i][j] = np.nan
                    self.tau_SIZ_x_field[i][j] = np.nan
                    self.tau_SIZ_y_field[i][j] = np.nan
                    self.u_Ekman_field[i][j] = np.nan
                    self.v_Ekman_field[i][j] = np.nan
                    continue

                tau_vec, tau_air_vec, tau_ice_vec = self.surface_stress(f, u_geo_vec, u_wind_vec, alpha, u_ice_vec)

                self.tau_air_x_field[i][j] = tau_air_vec[0]
                self.tau_air_y_field[i][j] = tau_air_vec[1]
                self.tau_ice_x_field[i][j] = tau_ice_vec[0]
                self.tau_ice_y_field[i][j] = tau_ice_vec[1]
                self.tau_x_field[i][j] = tau_vec[0]
                self.tau_y_field[i][j] = tau_vec[1]
                self.tau_SIZ_x_field[i][j] = tau_vec[0]
                self.tau_SIZ_y_field[i][j] = tau_vec[1]

                # Not sure why I have to recalculate u_Ekman_vec otherwise it's just zero.
                u_Ekman_vec = (np.sqrt(2) / (f * rho_0 * D_e)) * np.matmul(self.R_45deg, tau_vec)
                self.u_Ekman_field[i][j] = u_Ekman_vec[0]
                self.v_Ekman_field[i][j] = u_Ekman_vec[1]
                self.u_Ekman_SIZ_field[i][j] = u_Ekman_vec[0]
                self.v_Ekman_SIZ_field[i][j] = u_Ekman_vec[1]

    def compute_daily_ekman_pumping_field(self):
        from constants import Omega, rho_0
        logger.info('Calculating wind stress curl and Ekman pumping fields...')

        for i in range(1, len(self.lats) - 1):
            lat = self.lats[i]
            f = 2 * Omega * np.sin(np.deg2rad(lat))  # Coriolis parameter [s^-1]

            for j in range(1, len(self.lons) - 1):
                if not np.isnan(self.tau_x_field[i][j - 1]) and not np.isnan(self.tau_x_field[i][j + 1]) \
                        and not np.isnan(self.tau_y_field[i - 1][j]) and not np.isnan(self.tau_y_field[i + 1][j]):
                    dx = distance(self.lats[i-1], self.lons[j], self.lats[i+1], self.lons[j])
                    dy = distance(self.lats[i], self.lons[j-1], self.lats[i], self.lons[j+1])

                    dtauxdy = (self.tau_x_field[i][j+1] - self.tau_x_field[i][j-1]) / (2*dy)
                    dtauydx = (self.tau_y_field[i+1][j] - self.tau_y_field[i-1][j]) / (2*dx)

                    self.wind_stress_curl_field[i][j] = dtauydx - dtauxdy
                    self.w_Ekman_field[i][j] = (dtauydx - dtauxdy) / (rho_0 * f)

                else:
                    self.wind_stress_curl_field[i][j] = np.nan
                    self.w_Ekman_field[i][j] = np.nan

    def compute_monthly_mean_fields(self, date_in_month, method):
        import datetime
        from calendar import monthrange
        from constants import output_dir_path
        from utils import log_netCDF_dataset_metadata

        self.date = date_in_month
        n_days = monthrange(date_in_month.year, date_in_month.month)[1]

        # Initializing all the fields we want to calculate a monthly average for.
        tau_air_x_field_avg = np.zeros((len(self.lats), len(self.lons)))
        tau_air_y_field_avg = np.zeros((len(self.lats), len(self.lons)))
        tau_ice_x_field_avg = np.zeros((len(self.lats), len(self.lons)))
        tau_ice_y_field_avg = np.zeros((len(self.lats), len(self.lons)))
        tau_SIZ_x_field_avg = np.zeros((len(self.lats), len(self.lons)))
        tau_SIZ_y_field_avg = np.zeros((len(self.lats), len(self.lons)))
        tau_x_field_avg = np.zeros((len(self.lats), len(self.lons)))
        tau_y_field_avg = np.zeros((len(self.lats), len(self.lons)))

        u_Ekman_field_avg = np.zeros((len(self.lats), len(self.lons)))
        v_Ekman_field_avg = np.zeros((len(self.lats), len(self.lons)))
        u_Ekman_SIZ_field_avg = np.zeros((len(self.lats), len(self.lons)))
        v_Ekman_SIZ_field_avg = np.zeros((len(self.lats), len(self.lons)))

        u_geo_field_avg = np.zeros((len(self.lats), len(self.lons)))
        v_geo_field_avg = np.zeros((len(self.lats), len(self.lons)))
        u_wind_field_avg = np.zeros((len(self.lats), len(self.lons)))
        v_wind_field_avg = np.zeros((len(self.lats), len(self.lons)))
        alpha_field_avg = np.zeros((len(self.lats), len(self.lons)))
        u_ice_field_avg = np.zeros((len(self.lats), len(self.lons)))
        v_ice_field_avg = np.zeros((len(self.lats), len(self.lons)))

        wind_stress_curl_field_avg = np.zeros((len(self.lats), len(self.lons)))
        w_Ekman_field_avg = np.zeros((len(self.lats), len(self.lons)))

        # Number of days with available data for each grid point.
        tau_air_x_field_days = np.zeros((len(self.lats), len(self.lons)))
        tau_air_y_field_days = np.zeros((len(self.lats), len(self.lons)))
        tau_ice_x_field_days = np.zeros((len(self.lats), len(self.lons)))
        tau_ice_y_field_days = np.zeros((len(self.lats), len(self.lons)))
        tau_SIZ_x_field_days = np.zeros((len(self.lats), len(self.lons)))
        tau_SIZ_y_field_days = np.zeros((len(self.lats), len(self.lons)))
        tau_x_field_days = np.zeros((len(self.lats), len(self.lons)))
        tau_y_field_days = np.zeros((len(self.lats), len(self.lons)))

        u_Ekman_field_days = np.zeros((len(self.lats), len(self.lons)))
        v_Ekman_field_days = np.zeros((len(self.lats), len(self.lons)))
        u_Ekman_SIZ_field_days = np.zeros((len(self.lats), len(self.lons)))
        v_Ekman_SIZ_field_days = np.zeros((len(self.lats), len(self.lons)))

        u_geo_field_days = np.zeros((len(self.lats), len(self.lons)))
        v_geo_field_days = np.zeros((len(self.lats), len(self.lons)))
        u_wind_field_days = np.zeros((len(self.lats), len(self.lons)))
        v_wind_field_days = np.zeros((len(self.lats), len(self.lons)))
        alpha_field_days = np.zeros((len(self.lats), len(self.lons)))
        u_ice_field_days = np.zeros((len(self.lats), len(self.lons)))
        v_ice_field_days = np.zeros((len(self.lats), len(self.lons)))

        wind_stress_curl_field_days = np.zeros((len(self.lats), len(self.lons)))
        w_Ekman_field_days = np.zeros((len(self.lats), len(self.lons)))

        for day in range(1, n_days+1):
            date = datetime.date(date_in_month.year, date_in_month.month, day)
            logger.info('Averaging {:%b %d, %Y}...'.format(date))

            tau_nc_filename = 'surface_stress_' + '2015' + '07' + str(day).zfill(2) + '.nc'
            tau_filepath = os.path.join(output_dir_path, 'surface_stress', '2015', tau_nc_filename)

            current_tau_dataset = netCDF4.Dataset(tau_filepath)
            log_netCDF_dataset_metadata(current_tau_dataset)

            self.lats = np.array(current_tau_dataset.variables['lat'])
            self.lons = np.array(current_tau_dataset.variables['lon'])

            tau_air_x_field = np.array(current_tau_dataset.variables['tau_air_x'])
            tau_air_y_field = np.array(current_tau_dataset.variables['tau_air_y'])
            tau_ice_x_field = np.array(current_tau_dataset.variables['tau_ice_x'])
            tau_ice_y_field = np.array(current_tau_dataset.variables['tau_ice_y'])
            tau_SIZ_x_field = np.array(current_tau_dataset.variables['tau_SIZ_x'])
            tau_SIZ_y_field = np.array(current_tau_dataset.variables['tau_SIZ_y'])
            tau_x_field = np.array(current_tau_dataset.variables['tau_x'])
            tau_y_field = np.array(current_tau_dataset.variables['tau_y'])

            u_Ekman_field = np.array(current_tau_dataset.variables['Ekman_u'])
            v_Ekman_field = np.array(current_tau_dataset.variables['Ekman_v'])
            u_Ekman_SIZ_field = np.array(current_tau_dataset.variables['Ekman_SIZ_u'])
            v_Ekman_SIZ_field = np.array(current_tau_dataset.variables['Ekman_SIZ_v'])

            u_geo_field = np.array(current_tau_dataset.variables['geo_u'])
            v_geo_field = np.array(current_tau_dataset.variables['geo_v'])
            u_wind_field = np.array(current_tau_dataset.variables['wind_u'])
            v_wind_field = np.array(current_tau_dataset.variables['wind_v'])
            alpha_field = np.array(current_tau_dataset.variables['alpha'])
            u_ice_field = np.array(current_tau_dataset.variables['ice_u'])
            v_ice_field = np.array(current_tau_dataset.variables['ice_v'])

            wind_stress_curl_field = np.array(current_tau_dataset.variables['wind_stress_curl'])
            w_Ekman_field = np.array(current_tau_dataset.variables['Ekman_w'])

            if method == 'full_data_only':
                tau_air_x_field_avg = tau_air_x_field_avg + tau_air_x_field/n_days
                tau_air_y_field_avg = tau_air_y_field_avg + tau_air_y_field/n_days
                tau_ice_x_field_avg = tau_ice_x_field_avg + tau_ice_x_field/n_days
                tau_ice_y_field_avg = tau_ice_y_field_avg + tau_ice_y_field/n_days
                tau_SIZ_x_field_avg = tau_SIZ_x_field_avg + tau_SIZ_x_field/n_days
                tau_SIZ_y_field_avg = tau_SIZ_y_field_avg + tau_SIZ_y_field/n_days
                tau_x_field_avg = tau_x_field_avg + tau_x_field/n_days
                tau_y_field_avg = tau_y_field_avg + tau_y_field/n_days

                u_Ekman_field_avg = u_Ekman_field_avg + u_Ekman_field/n_days
                v_Ekman_field_avg = v_Ekman_field_avg + v_Ekman_field/n_days
                u_Ekman_SIZ_field_avg = u_Ekman_SIZ_field_avg + u_Ekman_SIZ_field/n_days
                v_Ekman_SIZ_field_avg = v_Ekman_SIZ_field_avg + v_Ekman_SIZ_field/n_days

                u_geo_field_avg = u_geo_field_avg + u_geo_field/n_days
                v_geo_field_avg = v_geo_field_avg + v_geo_field/n_days
                u_wind_field_avg = u_wind_field_avg + u_wind_field/n_days
                v_wind_field_avg = v_wind_field_avg + v_wind_field/n_days
                alpha_field_avg = alpha_field_avg + alpha_field/n_days
                u_ice_field_avg = u_ice_field_avg + u_ice_field/n_days
                v_ice_field_avg = v_ice_field_avg + v_ice_field/n_days

                wind_stress_curl_field_avg = wind_stress_curl_field_avg + wind_stress_curl_field/n_days
                w_Ekman_field_avg = w_Ekman_field_avg + w_Ekman_field/n_days

            elif method == 'partial_data_ok':
                tau_air_x_field_avg = tau_air_x_field_avg + np.nan_to_num(tau_air_x_field)
                tau_air_x_field[~np.isnan(tau_air_x_field)] = 1
                tau_air_x_field[np.isnan(tau_air_x_field)] = 0
                tau_air_x_field_days = tau_air_x_field_days + tau_air_x_field

                tau_air_y_field_avg = tau_air_y_field_avg + np.nan_to_num(tau_air_y_field)
                tau_air_y_field[~np.isnan(tau_air_y_field)] = 1
                tau_air_y_field[np.isnan(tau_air_y_field)] = 0
                tau_air_y_field_days = tau_air_y_field_days + tau_air_y_field

                tau_ice_x_field_avg = tau_ice_x_field_avg + np.nan_to_num(tau_ice_x_field)
                tau_ice_x_field[~np.isnan(tau_ice_x_field)] = 1
                tau_ice_x_field[np.isnan(tau_ice_x_field)] = 0
                tau_ice_x_field_days = tau_ice_x_field_days + tau_ice_x_field

                tau_ice_y_field_avg = tau_ice_y_field_avg + np.nan_to_num(tau_ice_y_field)
                tau_ice_y_field[~np.isnan(tau_ice_y_field)] = 1
                tau_ice_y_field[np.isnan(tau_ice_y_field)] = 0
                tau_ice_y_field_days = tau_ice_y_field_days + tau_ice_y_field

                tau_SIZ_x_field_avg = tau_SIZ_x_field_avg + np.nan_to_num(tau_SIZ_x_field)
                tau_SIZ_x_field[~np.isnan(tau_SIZ_x_field)] = 1
                tau_SIZ_x_field[np.isnan(tau_SIZ_x_field)] = 0
                tau_SIZ_x_field_days = tau_SIZ_x_field_days + tau_SIZ_x_field

                tau_SIZ_y_field_avg = tau_SIZ_y_field_avg + np.nan_to_num(tau_SIZ_y_field)
                tau_SIZ_y_field[~np.isnan(tau_SIZ_y_field)] = 1
                tau_SIZ_y_field[np.isnan(tau_SIZ_y_field)] = 0
                tau_SIZ_y_field_days = tau_SIZ_y_field_days + tau_SIZ_y_field

                tau_x_field_avg = tau_x_field_avg + np.nan_to_num(tau_x_field)
                tau_x_field[~np.isnan(tau_x_field)] = 1
                tau_x_field[np.isnan(tau_x_field)] = 0
                tau_x_field_days = tau_x_field_days + tau_x_field

                tau_y_field_avg = tau_y_field_avg + np.nan_to_num(tau_y_field)
                tau_y_field[~np.isnan(tau_y_field)] = 1
                tau_y_field[np.isnan(tau_y_field)] = 0
                tau_y_field_days = tau_y_field_days + tau_y_field

                u_Ekman_field_avg = u_Ekman_field_avg + np.nan_to_num(u_Ekman_field)
                u_Ekman_field[~np.isnan(u_Ekman_field)] = 1
                u_Ekman_field[np.isnan(u_Ekman_field)] = 0
                u_Ekman_field_days = u_Ekman_field_days + u_Ekman_field

                v_Ekman_field_avg = v_Ekman_field_avg + np.nan_to_num(v_Ekman_field)
                v_Ekman_field[~np.isnan(v_Ekman_field)] = 1
                v_Ekman_field[np.isnan(v_Ekman_field)] = 0
                v_Ekman_field_days = v_Ekman_field_days + v_Ekman_field

                u_Ekman_SIZ_field_avg = u_Ekman_SIZ_field_avg + np.nan_to_num(u_Ekman_SIZ_field)
                u_Ekman_SIZ_field[~np.isnan(u_Ekman_SIZ_field)] = 1
                u_Ekman_SIZ_field[np.isnan(u_Ekman_SIZ_field)] = 0
                u_Ekman_SIZ_field_days = u_Ekman_SIZ_field_days + u_Ekman_SIZ_field

                v_Ekman_SIZ_field_avg = v_Ekman_SIZ_field_avg + np.nan_to_num(v_Ekman_SIZ_field)
                v_Ekman_SIZ_field[~np.isnan(v_Ekman_SIZ_field)] = 1
                v_Ekman_SIZ_field[np.isnan(v_Ekman_SIZ_field)] = 0
                v_Ekman_SIZ_field_days = v_Ekman_SIZ_field_days + v_Ekman_SIZ_field

                u_geo_field_avg = u_geo_field_avg + np.nan_to_num(u_geo_field)
                u_geo_field[~np.isnan(u_geo_field)] = 1
                u_geo_field[np.isnan(u_geo_field)] = 0
                u_geo_field_days = u_geo_field_days + u_geo_field

                v_geo_field_avg = v_geo_field_avg + np.nan_to_num(v_geo_field)
                v_geo_field[~np.isnan(v_geo_field)] = 1
                v_geo_field[np.isnan(v_geo_field)] = 0
                v_geo_field_days = v_geo_field_days + v_geo_field

                u_wind_field_avg = u_wind_field_avg + np.nan_to_num(u_wind_field)
                u_wind_field[~np.isnan(u_wind_field)] = 1
                u_wind_field[np.isnan(u_wind_field)] = 0
                u_wind_field_days = u_wind_field_days + u_wind_field

                v_wind_field_avg = v_wind_field_avg + np.nan_to_num(v_wind_field)
                v_wind_field[~np.isnan(v_wind_field)] = 1
                v_wind_field[np.isnan(v_wind_field)] = 0
                v_wind_field_days = v_wind_field_days + v_wind_field

                alpha_field_avg = alpha_field_avg + np.nan_to_num(alpha_field)
                alpha_field[~np.isnan(alpha_field)] = 1
                alpha_field[np.isnan(alpha_field)] = 0
                alpha_field_days = alpha_field_days + alpha_field

                u_ice_field_avg = u_ice_field_avg + np.nan_to_num(u_ice_field)
                u_ice_field[~np.isnan(u_ice_field)] = 1
                u_ice_field[np.isnan(u_ice_field)] = 0
                u_ice_field_days = u_ice_field_days + u_ice_field

                v_ice_field_avg = v_ice_field_avg + np.nan_to_num(v_ice_field)
                v_ice_field[~np.isnan(v_ice_field)] = 1
                v_ice_field[np.isnan(v_ice_field)] = 0
                v_ice_field_days = v_ice_field_days + v_ice_field

                wind_stress_curl_field_avg = wind_stress_curl_field_avg + np.nan_to_num(wind_stress_curl_field)
                wind_stress_curl_field[~np.isnan(wind_stress_curl_field)] = 1
                wind_stress_curl_field[np.isnan(wind_stress_curl_field)] = 0
                wind_stress_curl_field_days = wind_stress_curl_field_days + wind_stress_curl_field

                w_Ekman_field_avg = w_Ekman_field_avg + np.nan_to_num(w_Ekman_field)
                w_Ekman_field[~np.isnan(w_Ekman_field)] = 1
                w_Ekman_field[np.isnan(w_Ekman_field)] = 0
                w_Ekman_field_days = w_Ekman_field_days + w_Ekman_field

        if method == 'full_data_only':
            self.tau_air_x_field = tau_air_x_field_avg
            self.tau_air_y_field = tau_air_y_field_avg
            self.tau_ice_x_field = tau_ice_x_field_avg
            self.tau_ice_y_field = tau_ice_y_field_avg
            self.tau_SIZ_x_field = tau_SIZ_x_field_avg
            self.tau_SIZ_y_field = tau_SIZ_y_field_avg
            self.tau_x_field = tau_x_field_avg
            self.tau_y_field = tau_y_field_avg

            self.u_Ekman_field = u_Ekman_field_avg
            self.v_Ekman_field = v_Ekman_field_avg
            self.u_Ekman_SIZ_field = u_Ekman_SIZ_field_avg
            self.v_Ekman_SIZ_field = v_Ekman_SIZ_field_avg

            self.u_geo_field = u_geo_field_avg
            self.v_geo_field = v_geo_field_avg
            self.u_wind_field = u_wind_field_avg
            self.v_wind_field = v_wind_field_avg
            self.alpha_field = alpha_field_avg
            self.u_ice_field = u_ice_field_avg
            self.v_ice_field = v_ice_field_avg

            self.wind_stress_curl_field = wind_stress_curl_field_avg
            self.w_Ekman_field = w_Ekman_field_avg

        elif method == 'partial_data_ok':
            self.tau_air_x_field = np.divide(tau_air_x_field_avg, tau_air_x_field_days)
            self.tau_air_y_field = np.divide(tau_air_y_field_avg, tau_air_y_field_days)
            self.tau_ice_x_field = np.divide(tau_ice_x_field_avg, tau_ice_x_field_days)
            self.tau_ice_y_field = np.divide(tau_ice_y_field_avg, tau_ice_y_field_days)
            self.tau_SIZ_x_field = np.divide(tau_SIZ_x_field_avg, tau_SIZ_x_field_days)
            self.tau_SIZ_y_field = np.divide(tau_SIZ_y_field_avg, tau_SIZ_y_field_days)
            self.tau_x_field = np.divide(tau_x_field_avg, tau_x_field_days)
            self.tau_y_field = np.divide(tau_y_field_avg, tau_y_field_days)

            self.u_Ekman_field = np.divide(u_Ekman_field_avg, u_Ekman_field_days)
            self.v_Ekman_field = np.divide(v_Ekman_field_avg, v_Ekman_field_days)
            self.u_Ekman_SIZ_field = np.divide(u_Ekman_SIZ_field_avg, u_Ekman_SIZ_field_days)
            self.v_Ekman_SIZ_field = np.divide(v_Ekman_SIZ_field_avg, v_Ekman_SIZ_field_days)

            self.u_geo_field = np.divide(u_geo_field_avg, u_geo_field_days)
            self.v_geo_field = np.divide(v_geo_field_avg, v_geo_field_days)
            self.u_wind_field = np.divide(u_wind_field_avg, u_wind_field_days)
            self.v_wind_field = np.divide(v_wind_field_avg, v_wind_field_days)
            self.alpha_field = np.divide(alpha_field_avg, alpha_field_days)
            self.u_ice_field = np.divide(u_ice_field_avg, u_ice_field_days)
            self.v_ice_field = np.divide(v_ice_field_avg, v_ice_field_days)

            self.wind_stress_curl_field = np.divide(wind_stress_curl_field_avg, wind_stress_curl_field_days)
            self.w_Ekman_field = np.divide(w_Ekman_field_avg, w_Ekman_field_days)

        self.plot_diagnostic_fields(type='monthly_avg')

    def plot_diagnostic_fields(self, type):
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        import cartopy
        import cartopy.crs as ccrs

        from constants import titles, gs_coords, scale_factor, colorbar_label, cmaps, cmap_ranges

        logger.info('Creating diagnostic figure...')

        fields = {
            'u_geo': self.u_geo_field,
            'v_geo': self.v_geo_field,
            'u_wind': self.u_wind_field,
            'v_wind': self.v_wind_field,
            'u_ice': self.u_ice_field,
            'v_ice': self.v_ice_field,
            'alpha': self.alpha_field,
            'tau_air_x': self.tau_air_x_field,
            'tau_air_y': self.tau_air_y_field,
            'tau_ice_x': self.tau_ice_x_field,
            'tau_ice_y': self.tau_ice_y_field,
            'tau_x': self.tau_x_field,
            'tau_y': self.tau_y_field,
            'u_Ekman': self.u_Ekman_field,
            'v_Ekman': self.v_Ekman_field,
            'curl_tau': self.wind_stress_curl_field,
            'tau_SIZ_x': self.tau_SIZ_x_field,
            'tau_SIZ_y': self.tau_SIZ_y_field,
            'w_Ekman': self.w_Ekman_field
        }

        # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
        # the smaller plots.
        land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face',
                                                       facecolor='dimgray', linewidth=0)
        vector_crs = ccrs.PlateCarree()

        # Figure size with an aspect ratio of 16:9 so it fits perfectly on a 1080p or 4K screen.
        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(5, 9)
        matplotlib.rcParams.update({'font.size': 6})

        # Plot all the scalar fields
        for var in fields.keys():
            ax = plt.subplot(gs[gs_coords[var]], projection=ccrs.SouthPolarStereo())
            ax.add_feature(land_50m)
            ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
            ax.set_title(var)

            im = ax.pcolormesh(self.lons, self.lats, scale_factor[var] * fields[var], transform=vector_crs,
                               cmap=cmaps[var], vmin=cmap_ranges[var][0], vmax=cmap_ranges[var][1])
            clb = fig.colorbar(im, ax=ax)
            clb.ax.set_title(colorbar_label[var])

            # Add selected vector fields.
            if var == 'u_ice':
                ax.quiver(self.lons[::10], self.lats[::10], self.u_ice_field[::10, ::10], self.v_ice_field[::10, ::10],
                          transform=vector_crs, units='width', width=0.002, scale=4)
            elif var == 'tau_x':
                ax.quiver(self.lons[::10], self.lats[::10], self.tau_x_field[::10, ::10], self.tau_y_field[::10, ::10],
                          transform=vector_crs, units='width', width=0.002, scale=8)

        # Add date label to bottom left.
        if type == 'daily':
            date_str = str(self.date.year) + '/' + str(self.date.month).zfill(2) + '/' + str(self.date.day).zfill(2)
            plt.gcf().text(0.1, 0.1, date_str, fontsize=10)
        elif type == 'monthly_avg':
            date_str = '{:%b %Y} average'.format(self.date)
            plt.gcf().text(0.1, 0.1, date_str, fontsize=10)

        logger.info('Saving diagnostic figures to disk...')

        if type == 'daily':
            tau_filename = 'surface_stress_' + str(self.date.year) + str(self.date.month).zfill(2) \
                           + str(self.date.day).zfill(2)
        elif type == 'monthly_avg':
            tau_filename = 'surfce_stress_' + '{:%b%Y}_avg'.format(self.date)

        # Saving diagnostic figure to disk. Only in .png as .pdf takes forever to write and is MASSIVE.
        tau_png_filepath = os.path.join(self.surface_stress_dir, str(self.date.year), tau_filename + '.png')
        # tau_pdf_filepath = os.path.join(self.surface_stress_dir, str(self.date.year), tau_filename + '.pdf')

        tau_dir = os.path.dirname(tau_png_filepath)
        if not os.path.exists(tau_dir):
            logger.info('Creating directory: {:s}'.format(tau_dir))
            os.makedirs(tau_dir)

        plt.savefig(tau_png_filepath, dpi=600, format='png', transparent=False, bbox_inches='tight')
        logger.info('Saved diagnostic figure: {:s}'.format(tau_png_filepath))

        # plt.savefig(tau_pdf_filepath, dpi=300, format='pdf', transparent=True)
        # logger.info('Saved diagnostic figure: {:s}'.format(tau_pdf_filepath))

    def write_fields_to_netcdf(self):
        tau_nc_filename = 'surface_stress_' + str(self.date.year) + str(self.date.month).zfill(2) \
                       + str(self.date.day).zfill(2) + '.nc'
        tau_filepath = os.path.join(self.surface_stress_dir, str(self.date.year), tau_nc_filename)
        tau_dir = os.path.dirname(tau_filepath)
        if not os.path.exists(tau_dir):
            logger.info('Creating directory: {:s}'.format(tau_dir))
            os.makedirs(tau_dir)

        tau_dataset = netCDF4.Dataset(tau_filepath, 'w')

        tau_dataset.title = 'Antarctic sea ice zone surface stress'
        tau_dataset.institution = 'Department of Earth, Atmospheric, and Planetary Science, ' \
                                  'Massachusetts Institute of Technology'
        # tau_dataset.history = 'Created ' + datetime.time.ctime(datetime.time.time()) + '.'

        tau_dataset.createDimension('time', None)
        tau_dataset.createDimension('lat', len(self.lats))
        tau_dataset.createDimension('lon', len(self.lons))

        # TODO: Actually store a date.
        time_var = tau_dataset.createVariable('time', np.float64, ('time',))
        time_var.units = 'hours since 0001-01-01 00:00:00'
        time_var.calendar = 'gregorian'

        lat_var = tau_dataset.createVariable('lat', np.float32, ('lat',))
        lat_var.units = 'degrees south'
        lat_var[:] = self.lats

        lon_var = tau_dataset.createVariable('lon', np.float32, ('lon',))
        lat_var.units = 'degrees west/east'
        lon_var[:] = self.lons

        tau_air_x_var = tau_dataset.createVariable('tau_air_x', float, ('lat', 'lon'), zlib=True)
        tau_air_x_var.units = 'N/m^2'
        tau_air_x_var.positive = 'up'
        tau_air_x_var.long_name = 'Zonal air surface stress'
        tau_air_x_var[:] = self.tau_air_x_field

        tau_air_y_var = tau_dataset.createVariable('tau_air_y', float, ('lat', 'lon'), zlib=True)
        tau_air_y_var.units = 'N/m^2'
        tau_air_y_var.positive = 'up'
        tau_air_y_var.long_name = 'Meridional air surface stress'
        tau_air_y_var[:] = self.tau_air_y_field

        tau_ice_x_var = tau_dataset.createVariable('tau_ice_x', float, ('lat', 'lon'), zlib=True)
        tau_ice_x_var.units = 'N/m^2'
        tau_ice_x_var.positive = 'up'
        tau_ice_x_var.long_name = 'Zonal ice surface stress'
        tau_ice_x_var[:] = self.tau_ice_x_field

        tau_ice_y_var = tau_dataset.createVariable('tau_ice_y', float, ('lat', 'lon'), zlib=True)
        tau_ice_y_var.units = 'N/m^2'
        tau_ice_y_var.positive = 'up'
        tau_ice_y_var.long_name = 'Meridional ice surface stress'
        tau_ice_y_var[:] = self.tau_ice_y_field

        tau_x_var = tau_dataset.createVariable('tau_x', float, ('lat', 'lon'), zlib=True)
        tau_x_var.units = 'N/m^2'
        tau_x_var.positive = 'up'
        tau_x_var.long_name = 'Zonal surface stress'
        tau_x_var[:] = self.tau_x_field

        tau_y_var = tau_dataset.createVariable('tau_y', float, ('lat', 'lon'), zlib=True)
        tau_y_var.units = 'N/m^2'
        tau_y_var.positive = 'up'
        tau_y_var.long_name = 'Meridional surface stress'
        tau_y_var[:] = self.tau_y_field

        tau_SIZ_x_var = tau_dataset.createVariable('tau_SIZ_x', float, ('lat', 'lon'), zlib=True)
        tau_SIZ_x_var.units = 'N/m^2'
        tau_SIZ_x_var.positive = 'up'
        tau_SIZ_x_var.long_name = 'Zonal surface stress in the SIZ'
        tau_SIZ_x_var[:] = self.tau_SIZ_x_field

        tau_SIZ_y_var = tau_dataset.createVariable('tau_SIZ_y', float, ('lat', 'lon'), zlib=True)
        tau_SIZ_y_var.units = 'N/m^2'
        tau_SIZ_y_var.positive = 'up'
        tau_SIZ_y_var.long_name = 'Meridional surface stress in the SIZ'
        tau_SIZ_y_var[:] = self.tau_SIZ_y_field

        curl_tau_var = tau_dataset.createVariable('wind_stress_curl', float, ('lat', 'lon'), zlib=True)
        curl_tau_var.units = 'N/m^3'
        curl_tau_var.positive = 'up'
        curl_tau_var.long_name = 'Wind stress curl'
        curl_tau_var[:] = self.wind_stress_curl_field

        w_Ekman_var = tau_dataset.createVariable('Ekman_w', float, ('lat', 'lon'), zlib=True)
        w_Ekman_var.units = 'm/year'
        w_Ekman_var.positive = 'up'
        w_Ekman_var.long_name = 'Ekman pumping'
        w_Ekman_var[:] = self.w_Ekman_field

        u_Ekman_var = tau_dataset.createVariable('Ekman_u', float, ('lat', 'lon'), zlib=True)
        u_Ekman_var.units = 'm/s'
        u_Ekman_var.positive = 'up'
        u_Ekman_var.long_name = 'Zonal Ekman transport velocity'
        u_Ekman_var[:] = self.u_Ekman_field

        v_Ekman_var = tau_dataset.createVariable('Ekman_v', float, ('lat', 'lon'), zlib=True)
        v_Ekman_var.units = 'm/s'
        v_Ekman_var.positive = 'up'
        v_Ekman_var.long_name = 'Meridional Ekman transport velocity'
        v_Ekman_var[:] = self.v_Ekman_field

        u_Ekman_SIZ_var = tau_dataset.createVariable('Ekman_SIZ_u', float, ('lat', 'lon'), zlib=True)
        u_Ekman_SIZ_var.units = 'm/s'
        u_Ekman_SIZ_var.positive = 'up'
        u_Ekman_SIZ_var.long_name = 'Zonal Ekman transport velocity in the SIZ'
        u_Ekman_SIZ_var[:] = self.u_Ekman_SIZ_field

        v_Ekman_SIZ_var = tau_dataset.createVariable('Ekman_SIZ_v', float, ('lat', 'lon'), zlib=True)
        v_Ekman_SIZ_var.units = 'm/s'
        v_Ekman_SIZ_var.positive = 'up'
        v_Ekman_SIZ_var.long_name = 'Meridional Ekman transport velocity in the SIZ'
        v_Ekman_SIZ_var[:] = self.v_Ekman_SIZ_field

        u_geo_var = tau_dataset.createVariable('geo_u', float, ('lat', 'lon'), zlib=True)
        u_geo_var.units = 'm/s'
        u_geo_var.positive = 'up'
        u_geo_var.long_name = 'Mean zonal geostrophic velocity'
        u_geo_var[:] = self.u_geo_field

        v_geo_var = tau_dataset.createVariable('geo_v', float, ('lat', 'lon'), zlib=True)
        v_geo_var.units = 'm/s'
        v_geo_var.positive = 'up'
        v_geo_var.long_name = 'Mean meridional geostrophic velocity'
        v_geo_var[:] = self.v_geo_field

        u_wind_var = tau_dataset.createVariable('wind_u', float, ('lat', 'lon'), zlib=True)
        u_wind_var.units = 'm/s'
        u_wind_var.positive = 'up'
        u_wind_var.long_name = 'Zonal wind velocity'
        u_wind_var[:] = self.u_wind_field

        v_wind_var = tau_dataset.createVariable('wind_v', float, ('lat', 'lon'), zlib=True)
        v_wind_var.units = 'm/s'
        v_wind_var.positive = 'up'
        v_wind_var.long_name = 'Meridional wind velocity'
        v_wind_var[:] = self.v_wind_field

        alpha_var = tau_dataset.createVariable('alpha', float, ('lat', 'lon'), zlib=True)
        alpha_var.units = 'fractional'
        alpha_var.long_name = 'Sea ice concentration'
        alpha_var[:] = self.alpha_field

        u_ice_var = tau_dataset.createVariable('ice_u', float, ('lat', 'lon'), zlib=True)
        u_ice_var.units = 'm/s'
        u_ice_var.positive = 'up'
        u_ice_var.long_name = 'Zonal sea ice motion'
        u_ice_var[:] = self.u_ice_field

        v_ice_var = tau_dataset.createVariable('ice_v', float, ('lat', 'lon'), zlib=True)
        v_ice_var.units = 'm/s'
        v_ice_var.positive = 'up'
        v_ice_var.long_name = 'Meridional sea ice motion'
        v_ice_var[:] = self.v_ice_field

        tau_dataset.close()
