import os
import numpy as np

import netCDF4
import matplotlib.colors as colors

from GeostrophicVelocityDataset import GeostrophicVelocityDataset
from SurfaceWindDataset import SurfaceWindDataset
from SeaIceConcentrationDataset import SeaIceConcentrationDataset
from SeaIceMotionDataset import SeaIceMotionDataset

from utils import distance
from constants import lat_min, lat_max, lat_step, n_lat, lon_min, lon_max, lon_step, n_lon
from constants import rho_air, rho_seawater, C_air, C_seawater
from constants import Omega, rho_0, D_e

import logging
logger = logging.getLogger(__name__)


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Set NaN values to zero so they appear as white (i.e. not at all if using the 'seismic' colormap).
        value[np.isnan(value)] = 0

        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class SurfaceStressDataWriter(object):
    """
    Such an object should mainly compute daily (averaged) wind stress and wind stress curl fields and write them out
    to netCDF files. Computing monthly means makes sense here. But plotting should go elsewhere.
    """

    from constants import output_dir_path
    surface_stress_dir = os.path.join(output_dir_path, 'surface_stress')

    R_45deg = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]])
    R_m45deg = np.array([[np.cos(-np.pi/4), -np.sin(-np.pi/4)], [np.sin(-np.pi/4), np.cos(-np.pi/4)]])

    def __init__(self, date):
        self.lats = np.linspace(lat_min, lat_max, n_lat)
        self.lons = np.linspace(lon_min, lon_max, n_lon)

        # Remove the +180 longitude as it coincides with the -180 longitude.
        # Actually no, it should not be removed. It's important when plotting the fields if we want the last sector to
        # be plotted as well.
        # self.lons = self.lons[:-1]

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

        self.dtauxdy_field = np.zeros((len(self.lats), len(self.lons)))
        self.dtauydx_field = np.zeros((len(self.lats), len(self.lons)))

        # The fields below are really only used for mean fields, not daily fields.
        self.freshwater_flux_field =np.zeros((len(self.lats), len(self.lons)))
        self.ice_div_field = np.zeros((len(self.lats), len(self.lons)))

        self.salinity_field = np.zeros((len(self.lats), len(self.lons)))
        self.temperature_field = np.zeros((len(self.lats), len(self.lons)))
        self.neutral_density_field = np.zeros((len(self.lats), len(self.lons)))

        self.var_fields = {
            'tau_air_x': self.tau_air_x_field,
            'tau_air_y': self.tau_air_y_field,
            'tau_ice_x': self.tau_ice_x_field,
            'tau_ice_y': self.tau_ice_y_field,
            'tau_x': self.tau_x_field,
            'tau_y': self.tau_y_field,
            'tau_SIZ_x': self.tau_SIZ_x_field,
            'tau_SIZ_y': self.tau_SIZ_y_field,
            'wind_stress_curl': self.wind_stress_curl_field,
            'Ekman_w': self.w_Ekman_field,
            'Ekman_u': self.u_Ekman_field,
            'Ekman_v': self.v_Ekman_field,
            'Ekman_SIZ_u': self.u_Ekman_SIZ_field,
            'Ekman_SIZ_v': self.v_Ekman_SIZ_field,
            'geo_u': self.u_geo_field,
            'geo_v': self.v_geo_field,
            'wind_u': self.u_wind_field,
            'wind_v': self.v_wind_field,
            'alpha': self.alpha_field,
            'ice_u': self.u_ice_field,
            'ice_v': self.v_ice_field,
            'dtauydx': self.dtauydx_field,
            'dtauxdy': self.dtauxdy_field
        }

        self.figure_fields = {
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
            'dtauydx': self.dtauydx_field,
            'dtauxdy': self.dtauxdy_field,
            'curl_tau': self.wind_stress_curl_field,
            # 'tau_SIZ_x': self.tau_SIZ_x_field,
            # 'tau_SIZ_y': self.tau_SIZ_y_field,
            'w_Ekman': self.w_Ekman_field,
            'freshwater_flux': self.freshwater_flux_field,
            'ice_div': self.ice_div_field,
            'temperature': self.temperature_field,
            'salinity': self.salinity_field,
            'neutral_density': self.neutral_density_field
        }

        if date is not None:
            self.date = date

            self.u_geo_data = GeostrophicVelocityDataset(self.date)
            self.sea_ice_conc_data = SeaIceConcentrationDataset(self.date)
            self.sea_ice_motion_data = SeaIceMotionDataset(self.date)
            self.u_wind_data = SurfaceWindDataset(self.date)

    def surface_stress(self, f, u_geo_vec, u_wind_vec, alpha, u_ice_vec, u_Ekman_vec_type='vertical_avg'):
        """
        Use the modified Richardson iteration to calculate tau and u_Ekman.
        :param f:
        :param u_geo_vec:
        :param u_wind_vec:
        :param alpha:
        :param u_ice_vec:
        :param u_Ekman_vec_type:
        :return:
        """

        # Here we set the variables to arbitrary initial guesses before iteratively calculating tau and u_Ekman.
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

            if u_Ekman_vec_type == 'surface':
                u_Ekman_vec = (np.sqrt(2) / (f * rho_0 * D_e)) * np.matmul(self.R_m45deg, tau_vec)
            elif u_Ekman_vec_type == 'vertical_avg':
                tau_x_scalar = tau_vec[0]
                tau_y_scalar = tau_vec[1]
                u_Ekman_scalar = tau_y_scalar / (f * rho_0 * D_e)
                v_Ekman_scalar = -tau_x_scalar / (f * rho_0 * D_e)
                u_Ekman_vec = np.array([u_Ekman_scalar, v_Ekman_scalar])

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

    def compute_daily_surface_stress_field(self, u_Ekman_vec_type='vertical_avg'):
        logger.info('Calculating surface stress field (tau_x, tau_y) for:')
        logger.info('lat_min = {}, lat_max = {}, lat_step = {}, n_lat = {}'.format(lat_min, lat_max, lat_step, n_lat))
        logger.info('lon_min = {}, lon_max = {}, lon_step = {}, n_lon = {}'.format(lon_min, lon_max, lon_step, n_lon))

        for i in range(len(self.lats)):
            lat = self.lats[i]
            f = 2 * Omega * np.sin(np.deg2rad(lat))  # Coriolis parameter [s^-1]

            progress_percent = 100 * i / (len(self.lats) - 1)
            logger.info('({}) lat = {:.2f}/{:.2f} ({:.1f}%)'.format(self.date, lat, lat_max, progress_percent))

            for j in range(len(self.lons)):
                lon = self.lons[j]

                u_geo_vec = self.u_geo_data.absolute_geostrophic_velocity(lat, lon, 'interp')
                u_wind_vec = self.u_wind_data.ocean_surface_wind_vector(lat, lon, 'interp')
                alpha = self.sea_ice_conc_data.sea_ice_concentration(lat, lon, 'interp')
                u_ice_vec = self.sea_ice_motion_data.seaice_motion_vector(lat, lon, 'interp')

                self.u_geo_field[i][j] = u_geo_vec[0]
                self.v_geo_field[i][j] = u_geo_vec[1]
                self.u_wind_field[i][j] = u_wind_vec[0]
                self.v_wind_field[i][j] = u_wind_vec[1]
                self.alpha_field[i][j] = alpha
                self.u_ice_field[i][j] = u_ice_vec[0]
                self.v_ice_field[i][j] = u_ice_vec[1]

                # If there's no sea ice at a point and we have data at that point (i.e. the point is still in the ocean)
                # then tau is just tau_air and easy to calculate. Note that this encompasses regions of alpha < 0.15 as
                # well since SeaIceConcentrationDataset returns 0 for alpha < 0.15.
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

                    # Not sure why I have to recalculate u_Ekman_vec, otherwise I just the zero vector.
                    if u_Ekman_vec_type == 'surface':
                        u_Ekman_vec = (np.sqrt(2) / (f * rho_0 * D_e)) * np.matmul(self.R_m45deg, tau_air_vec)
                    elif u_Ekman_vec_type == 'vertical_avg':
                        tau_x_scalar = tau_air_vec[0]
                        tau_y_scalar = tau_air_vec[1]
                        u_Ekman_scalar = tau_y_scalar / (f * rho_0 * D_e)
                        v_Ekman_scalar = -tau_x_scalar / (f * rho_0 * D_e)
                        u_Ekman_vec = np.array([u_Ekman_scalar, v_Ekman_scalar])

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

                # Not sure why I have to recalculate u_Ekman_vec, otherwise I just the zero vector.
                if u_Ekman_vec_type == 'surface':
                    u_Ekman_vec = (np.sqrt(2) / (f * rho_0 * D_e)) * np.matmul(self.R_m45deg, tau_vec)
                elif u_Ekman_vec_type == 'vertical_avg':
                    tau_x_scalar = tau_vec[0]
                    tau_y_scalar = tau_vec[1]
                    u_Ekman_scalar = tau_y_scalar / (f * rho_0 * D_e)
                    v_Ekman_scalar = -tau_x_scalar / (f * rho_0 * D_e)
                    u_Ekman_vec = np.array([u_Ekman_scalar, v_Ekman_scalar])

                self.u_Ekman_field[i][j] = u_Ekman_vec[0]
                self.v_Ekman_field[i][j] = u_Ekman_vec[1]
                self.u_Ekman_SIZ_field[i][j] = u_Ekman_vec[0]
                self.v_Ekman_SIZ_field[i][j] = u_Ekman_vec[1]

    def compute_daily_ekman_pumping_field(self):
        from constants import Omega, rho_0
        logger.info('Calculating wind stress curl and Ekman pumping fields...')

        j_max = len(self.lons) - 1
        logger.info('j_max = {}'.format(j_max))

        for i in range(1, len(self.lats) - 1):
            lat = self.lats[i]
            f = 2 * Omega * np.sin(np.deg2rad(lat))  # Coriolis parameter [s^-1]

            progress_percent = 100 * i / (len(self.lats) - 2)
            logger.info('({} w_Ekman) lat = {:.2f}/{:.2f} ({:.1f}%)'.format(self.date, lat, lat_max, progress_percent))

            dx = distance(self.lats[i-1], self.lons[0], self.lats[i+1], self.lons[0])
            dy = distance(self.lats[i], self.lons[0], self.lats[i], self.lons[2])

            for j in range(len(self.lons)):
                # TODO: Can we lump all three test cases together by using the mod function to calculate indices?
                # Special case for when we're at j=0 (180 W) and need to use the value from j=j_max (180 E)
                if j == 0:
                    if not np.isnan(self.tau_x_field[i][j_max]) and not np.isnan(self.tau_x_field[i][1]) \
                            and not np.isnan(self.tau_y_field[i-1][j]) and not np.isnan(self.tau_y_field[i+1][j]):
                        # dtauxdy = (self.tau_x_field[i][1] - self.tau_x_field[i][j_max]) / dy
                        # dtauydx = (self.tau_y_field[i+1][j] - self.tau_y_field[i-1][j]) / dx

                        dtauydx = (self.tau_y_field[i][j+1] - self.tau_y_field[i][j_max]) / dx
                        dtauxdy = (self.tau_x_field[i+1][j] - self.tau_x_field[i-1][j]) / dy

                        self.dtauydx_field[i][j] = dtauydx
                        self.dtauxdy_field[i][j] = dtauxdy
                        self.wind_stress_curl_field[i][j] = dtauydx - dtauxdy
                        self.w_Ekman_field[i][j] = (dtauydx - dtauxdy) / (rho_0 * f)
                    else:
                        self.dtauydx_field[i][j] = np.nan
                        self.dtauxdy_field[i][j] = np.nan
                        self.wind_stress_curl_field[i][j] = np.nan
                        self.w_Ekman_field[i][j] = np.nan

                    continue

                # Special case for when we're at j=j_max (180 E) and need to use the value from j=0 (180 W)
                if j == j_max:
                    if not np.isnan(self.tau_x_field[i][j-1]) and not np.isnan(self.tau_x_field[i][0]) \
                            and not np.isnan(self.tau_y_field[i-1][j]) and not np.isnan(self.tau_y_field[i+1][j]):
                        # dtauydx = (self.tau_y_field[i+1][j] - self.tau_y_field[i-1][j]) / dx
                        # dtauxdy = (self.tau_x_field[i][0] - self.tau_x_field[i][j-1]) / dy

                        dtauydx = (self.tau_y_field[i][0] - self.tau_y_field[i][j-1]) / dx
                        dtauxdy = (self.tau_x_field[i+1][j] - self.tau_x_field[i-1][j]) / dy

                        self.dtauydx_field[i][j] = dtauydx
                        self.dtauxdy_field[i][j] = dtauxdy
                        self.wind_stress_curl_field[i][j] = dtauydx - dtauxdy
                        self.w_Ekman_field[i][j] = (dtauydx - dtauxdy) / (rho_0 * f)
                    else:
                        self.dtauydx_field[i][j] = np.nan
                        self.dtauxdy_field[i][j] = np.nan
                        self.wind_stress_curl_field[i][j] = np.nan
                        self.w_Ekman_field[i][j] = np.nan

                    continue

                if not np.isnan(self.tau_x_field[i][j-1]) and not np.isnan(self.tau_x_field[i][j+1]) \
                        and not np.isnan(self.tau_y_field[i-1][j]) and not np.isnan(self.tau_y_field[i+1][j]):
                    # Second-order centered difference scheme where we divide by the distance between the i+1 and i-1
                    # cells, which is just dx as defined in the above line. Textbook formulas will usually have a 2*dx
                    # in the denominator because dx is the width of just one cell.
                    # TODO: Why does it look like accessing the wrong axis gives the right derivative!?
                    # dtauydx = (self.tau_y_field[i+1][j] - self.tau_y_field[i-1][j]) / dx
                    # dtauxdy = (self.tau_x_field[i][j+1] - self.tau_x_field[i][j-1]) / dy
                    dtauydx = (self.tau_y_field[i][j+1] - self.tau_y_field[i][j-1]) / dx
                    dtauxdy = (self.tau_x_field[i+1][j] - self.tau_x_field[i-1][j]) / dy

                    self.dtauydx_field[i][j] = dtauydx
                    self.dtauxdy_field[i][j] = dtauxdy
                    self.wind_stress_curl_field[i][j] = dtauydx - dtauxdy
                    self.w_Ekman_field[i][j] = (dtauydx - dtauxdy) / (rho_0 * f)
                else:
                    self.dtauydx_field[i][j] = np.nan
                    self.dtauxdy_field[i][j] = np.nan
                    self.wind_stress_curl_field[i][j] = np.nan
                    self.w_Ekman_field[i][j] = np.nan

    def compute_salt_transport_fields(self):
        pass

    def compute_ice_divergence_field(self):
        pass

    def compute_mean_fields(self, dates, avg_method):
        from constants import output_dir_path, netcdf_var_names
        from utils import log_netCDF_dataset_metadata

        n_days = len(dates)

        # Dictionary storing all the averaged fields.
        field_avg = {}

        # Dictionary storing the number of days of available data at each gridpoint for each field (these are used when
        # avg_method = 'partial_data_ok').
        field_days = {}

        # Initializing all the fields we want to calculate an average for.
        for var_name in netcdf_var_names:
            field_avg[var_name] = np.zeros((len(self.lats), len(self.lons)))
            field_days[var_name] = np.zeros((len(self.lats), len(self.lons)))

        for date in dates:
            tau_nc_filename = 'surface_stress_' + str(date.year) + str(date.month).zfill(2) \
                              + str(date.day).zfill(2) + '.nc'
            tau_filepath = os.path.join(output_dir_path, 'surface_stress', str(date.year), tau_nc_filename)

            logger.info('Averaging {:%b %d, %Y} ({:s})...'.format(date, tau_filepath))

            try:
                current_tau_dataset = netCDF4.Dataset(tau_filepath)
                log_netCDF_dataset_metadata(current_tau_dataset)
            except OSError as e:
                logger.error('{}'.format(e))
                logger.warning('{:s} not found. Proceeding without it...'.format(tau_filepath))
                n_days = n_days - 1  # Must account for lost day if using avg_method='full_data_only'.
                continue

            self.lats = np.array(current_tau_dataset.variables['lat'])
            self.lons = np.array(current_tau_dataset.variables['lon'])

            daily_fields = {}
            for var_name in netcdf_var_names:
                daily_fields[var_name] = np.array(current_tau_dataset.variables[var_name])

            if avg_method == 'full_data_only':
                for var_name in netcdf_var_names:
                    field_avg[var_name] = field_avg[var_name] + daily_fields[var_name]/n_days

            elif avg_method == 'partial_data_ok':
                field_avg[var_name] = field_avg[var_name] + np.nan_to_num(daily_fields[var_name])
                daily_fields[var_name][~np.isnan(daily_fields[var_name])] = 1
                daily_fields[np.isnan(daily_fields[var_name])] = 0
                field_days[var_name] = field_days[var_name] + daily_fields[var_name]

        if avg_method == 'full_data_only':
            for var_name in netcdf_var_names:
                self.var_fields[var_name] = field_avg[var_name]

        elif avg_method == 'partial_data_ok':
            for var_name in netcdf_var_names:
                self.var_fields[var_name] = np.divide(field_avg[var_name], field_days[var_name])

    def plot_diagnostic_fields(self, plot_type, custom_label=None, avg_period=None):
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.colors as colors
        from matplotlib.gridspec import GridSpec

        import cartopy
        import cartopy.crs as ccrs
        import cmocean.cm

        from constants import titles, gs_coords, scale_factor, colorbar_label, cmaps, cmap_ranges

        logger.info('Converting and recalculating tau_air and tau_ice fields...')

        # Convert tau_air fields into (1-alpha)*tau_air, and tau_ice fields into alpha*tau_ice.
        self.tau_air_x_field = np.multiply(1 - self.alpha_field, self.tau_air_x_field)
        self.tau_air_y_field = np.multiply(1 - self.alpha_field, self.tau_air_y_field)
        self.tau_ice_x_field = np.multiply(self.alpha_field, self.tau_ice_x_field)
        self.tau_ice_y_field = np.multiply(self.alpha_field, self.tau_ice_y_field)

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
            'dtauydx': self.dtauydx_field,
            'dtauxdy': self.dtauxdy_field,
            'curl_tau': self.wind_stress_curl_field,
            # 'tau_SIZ_x': self.tau_SIZ_x_field,
            # 'tau_SIZ_y': self.tau_SIZ_y_field,
            'w_Ekman': self.w_Ekman_field
        }

        # Add land to the plot with a 1:50,000,000 scale. Line width is set to 0 so that the edges aren't poofed up in
        # the smaller plots.
        land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face',
                                                       facecolor='dimgray', linewidth=0)
        vector_crs = ccrs.PlateCarree()

        # Figure size with an aspect ratio of 16:9 so it fits perfectly on a 1080p/4K screen. Well now it's 20:9.
        fig = plt.figure(figsize=(20, 9))
        gs = GridSpec(5, 11)
        matplotlib.rcParams.update({'font.size': 6})

        # Plot all the scalar fields
        for var in fields.keys():
            ax = plt.subplot(gs[gs_coords[var]], projection=ccrs.SouthPolarStereo())
            ax.add_feature(land_50m)
            ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
            ax.set_title(titles[var])

            # if var == 'v_Ekman':
            #     im = ax.pcolormesh(self.lons, self.lats, scale_factor[var] * fields[var], transform=vector_crs,
            #                        cmap=cmaps[var], vmin=cmap_ranges[var][0], vmax=cmap_ranges[var][1],
            #                        # norm=MidpointNormalize(midpoint=0))
            #                        norm=colors.SymLogNorm(linthresh=0.2, linscale=0.2,
            #                                               vmin=cmap_ranges[var][0], vmax=cmap_ranges[var][1]))
            # else:
            #     im = ax.pcolormesh(self.lons, self.lats, scale_factor[var] * fields[var], transform=vector_crs,
            #                        cmap=cmaps[var], vmin=cmap_ranges[var][0], vmax=cmap_ranges[var][1])

            im = ax.pcolormesh(self.lons, self.lats, scale_factor[var] * fields[var], transform=vector_crs,
                               cmap=cmaps[var], vmin=cmap_ranges[var][0], vmax=cmap_ranges[var][1])

            clb = fig.colorbar(im, ax=ax, extend='both')
            clb.ax.set_title(colorbar_label[var])

            # Add vector fields to the u_ice and tau_x fields.
            if var == 'u_ice':
                ax.quiver(self.lons[::10], self.lats[::10], self.u_ice_field[::10, ::10], self.v_ice_field[::10, ::10],
                          transform=vector_crs, units='width', width=0.002, scale=2)
            elif var == 'tau_x':
                ax.quiver(self.lons[::10], self.lats[::10], self.tau_x_field[::10, ::10], self.tau_y_field[::10, ::10],
                          transform=vector_crs, units='width', width=0.002, scale=4)

            # Plot zero stress line, zero wind line, and ice edge on tau_x and w_Ekman plots (plus legends).
            if var == 'tau_x' or var == 'tau_y' or var == 'w_Ekman':
                ax.contour(self.lons, self.lats, np.ma.array(self.tau_x_field, mask=np.isnan(self.alpha_field)),
                           levels=[0], colors='green', linewidths=1, transform=vector_crs)
                ax.contour(self.lons, self.lats, np.ma.array(self.u_wind_field, mask=np.isnan(self.alpha_field)),
                           levels=[0], colors='gold', linewidths=1, transform=vector_crs)
                ax.contour(self.lons, self.lats, np.ma.array(self.alpha_field, mask=np.isnan(self.alpha_field)),
                           levels=[0.15], colors='black', linewidths=1, transform=vector_crs)

                zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
                zero_wind_line_patch = mpatches.Patch(color='gold', label='zero zonal wind line')
                ice_edge_patch = mpatches.Patch(color='black', label='15% ice edge')
                plt.legend(handles=[zero_stress_line_patch, zero_wind_line_patch, ice_edge_patch], loc='lower center',
                           bbox_to_anchor=(0, -0.05, 1, -0.05), ncol=3, mode='expand', borderaxespad=0)

            # Plot zero stress line and ice edge on d/dx (tau_y) and d/dy (tau_x) plots.
            if var == 'u_Ekman' or var == 'v_Ekman' or var == 'dtauydx' or var == 'dtauxdy':
                ax.contour(self.lons, self.lats, np.ma.array(self.tau_x_field, mask=np.isnan(self.alpha_field)),
                           levels=[0], colors='green', linewidths=0.5, transform=vector_crs)
                ax.contour(self.lons, self.lats, np.ma.array(self.alpha_field, mask=np.isnan(self.alpha_field)),
                           levels=[0.15], colors='black', linewidths=0.5, transform=vector_crs)

        from SalinityDataset import SalinityDataset
        from TemperatureDataset import TemperatureDataset
        from NeutralDensityDataset import NeutralDensityDataset

        levels = [0, 1, 2, 3, 4, 5, 6, 7]
        salinity_dataset = SalinityDataset(time_span='A5B2', avg_period=avg_period, grid_size='04', field_type='an')
        temperature_dataset = TemperatureDataset(time_span='A5B2', avg_period=avg_period, grid_size='04',
                                                 field_type='an')
        neutral_density_dataset = NeutralDensityDataset(time_span='A5B2', avg_period=avg_period, grid_size='04',
                                                        field_type='an', depth_levels=levels)

        salinity_field = np.zeros((len(self.lats), len(self.lons)))
        temperature_field = np.zeros((len(self.lats), len(self.lons)))
        neutral_density_field = np.zeros((len(self.lats), len(self.lons)))

        logger.info('Calculating average T, S, gamma_n...')
        for i in range(len(self.lats)):
            lat = self.lats[i]
            for j in range(len(self.lons)):
                lon = self.lons[j]
                salinity_field[i][j] = salinity_dataset.salinity(lat, lon, levels)
                temperature_field[i][j] = temperature_dataset.temperature(lat, lon, levels)
                neutral_density_field[i][j] = neutral_density_dataset.gamma_n_depth_averaged(lat, lon, levels)

        # Calculate and plot salt transport.
        from SalinityDataset import SalinityDataset
        salinity_dataset = SalinityDataset(time_span='A5B2', avg_period=avg_period, grid_size='04', field_type='an')

        zonal_salt_transport_field = np.zeros((len(self.lats), len(self.lons)))
        merid_salt_transport_field = np.zeros((len(self.lats), len(self.lons)))

        for i in range(1, len(self.lats) - 1):
            lat = self.lats[i]

            progress_percent = 100 * i / (len(self.lats) - 2)
            logger.info('(salt_transport) lat = {:.2f}/{:.2f} ({:.1f}%)'.format(lat, lat_max, progress_percent))

            dx = distance(self.lats[i-1], self.lons[0], self.lats[i+1], self.lons[0])
            dy = distance(self.lats[i], self.lons[0], self.lats[i], self.lons[2])

            for j in range(1, len(self.lons) - 1):
                lon = self.lons[j]

                u_Ekman_scalar = self.u_Ekman_field[i][j]
                v_Ekman_scalar = self.v_Ekman_field[i][j]

                salinity_ip1_j = salinity_dataset.salinity(self.lats[i+1], lon, 0)
                salinity_im1_j = salinity_dataset.salinity(self.lats[i-1], lon, 0)
                salinity_i_jp1 = salinity_dataset.salinity(lat, self.lons[j+1], 0)
                salinity_i_jm1 = salinity_dataset.salinity(lat, self.lons[j-1], 0)

                if not np.isnan(u_Ekman_scalar) and not np.isnan(salinity_i_jm1) \
                        and not np.isnan(salinity_i_jp1):
                    dSdx = (salinity_i_jp1 - salinity_i_jm1) / dx
                    # dSdx = (salinity_ip1_j - salinity_im1_j) / dx
                    zonal_salt_transport_field[i][j] = u_Ekman_scalar * dSdx
                else:
                    zonal_salt_transport_field[i][j] = np.nan

                if not np.isnan(v_Ekman_scalar) and not np.isnan(salinity_im1_j) \
                        and not np.isnan(salinity_ip1_j):
                    dSdy = (salinity_ip1_j - salinity_im1_j) / dy
                    # dSdy = (salinity_i_jp1 - salinity_i_jm1) / dy
                    merid_salt_transport_field[i][j] = v_Ekman_scalar * dSdy
                else:
                    merid_salt_transport_field[i][j] = np.nan

        # Zonal freshwater flux plot.
        ax = plt.subplot(gs[0, 9], projection=ccrs.SouthPolarStereo())
        ax.add_feature(land_50m)
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        ax.set_title('u_Ekman*(dS/dx)')

        max_zst = np.abs(np.nanmax(zonal_salt_transport_field)) / 10
        im = ax.pcolormesh(self.lons, self.lats, zonal_salt_transport_field, transform=vector_crs,
                           cmap=cmocean.cm.balance, vmin=-max_zst, vmax=max_zst)

        clb = fig.colorbar(im, ax=ax, extend='both')
        clb.ax.set_title('m*g/(s*kg)')

        ax.contour(self.lons, self.lats, np.ma.array(self.tau_x_field, mask=np.isnan(self.alpha_field)),
                   levels=[0], colors='green', linewidths=0.5, transform=vector_crs)
        ax.contour(self.lons, self.lats, np.ma.array(self.alpha_field, mask=np.isnan(self.alpha_field)),
                   levels=[0.15], colors='black', linewidths=0.5, transform=vector_crs)

        # Meridional freshwater flux plot.
        ax = plt.subplot(gs[1, 9], projection=ccrs.SouthPolarStereo())
        ax.add_feature(land_50m)
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        ax.set_title('v_Ekman*(dS/dy)')

        max_mst = np.abs(np.nanmax(merid_salt_transport_field)) / 10
        im = ax.pcolormesh(self.lons, self.lats, merid_salt_transport_field, transform=vector_crs,
                           cmap=cmocean.cm.balance, vmin=-max_mst, vmax=max_mst)

        clb = fig.colorbar(im, ax=ax, extend='both')
        clb.ax.set_title('m*g/(s*kg)')

        ax.contour(self.lons, self.lats, np.ma.array(self.tau_x_field, mask=np.isnan(self.alpha_field)),
                   levels=[0], colors='green', linewidths=0.5, transform=vector_crs)
        ax.contour(self.lons, self.lats, np.ma.array(self.alpha_field, mask=np.isnan(self.alpha_field)),
                   levels=[0.15], colors='black', linewidths=0.5, transform=vector_crs)

        # Temperature plot.
        ax = plt.subplot(gs[0, 10], projection=ccrs.SouthPolarStereo())
        ax.add_feature(land_50m)
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        ax.set_title('surface temperature')

        im = ax.pcolormesh(self.lons, self.lats, temperature_field, transform=vector_crs,
                           cmap=cmocean.cm.thermal, vmin=-2.5, vmax=2.5)

        clb = fig.colorbar(im, ax=ax, extend='both')
        clb.ax.set_title('deg C')

        ax.contour(self.lons, self.lats, np.ma.array(self.tau_x_field, mask=np.isnan(self.alpha_field)),
                   levels=[0], colors='green', linewidths=0.5, transform=vector_crs)
        ax.contour(self.lons, self.lats, np.ma.array(self.alpha_field, mask=np.isnan(self.alpha_field)),
                   levels=[0.15], colors='black', linewidths=0.5, transform=vector_crs)

        # Salinity plot.
        ax = plt.subplot(gs[1, 10], projection=ccrs.SouthPolarStereo())
        ax.add_feature(land_50m)
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        ax.set_title('surface salinity')

        im = ax.pcolormesh(self.lons, self.lats, salinity_field, transform=vector_crs,
                           cmap=cmocean.cm.haline, vmin=33.75, vmax=35)

        clb = fig.colorbar(im, ax=ax, extend='both')
        clb.ax.set_title('g/kg')

        ax.contour(self.lons, self.lats, np.ma.array(self.tau_x_field, mask=np.isnan(self.alpha_field)),
                   levels=[0], colors='green', linewidths=0.5, transform=vector_crs)
        ax.contour(self.lons, self.lats, np.ma.array(self.alpha_field, mask=np.isnan(self.alpha_field)),
                   levels=[0.15], colors='black', linewidths=0.5, transform=vector_crs)

        # Neutral density plot.
        ax = plt.subplot(gs[slice(2, 4), slice(9, 11)], projection=ccrs.SouthPolarStereo())
        ax.add_feature(land_50m)
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        ax.set_title('neutral density')

        im = ax.pcolormesh(self.lons, self.lats, neutral_density_field, transform=vector_crs,
                           cmap=cmocean.cm.dense, vmin=26, vmax=28)

        clb = fig.colorbar(im, ax=ax, extend='both')
        clb.ax.set_title('kg/m$^3$')

        ax.contour(self.lons, self.lats, np.ma.array(self.tau_x_field, mask=np.isnan(self.alpha_field)),
                   levels=[0], colors='green', linewidths=1, transform=vector_crs)
        ax.contour(self.lons, self.lats, np.ma.array(self.alpha_field, mask=np.isnan(self.alpha_field)),
                   levels=[0.15], colors='black', linewidths=1, transform=vector_crs)
        ax.contour(self.lons, self.lats, np.ma.array(neutral_density_field, mask=np.isnan(self.alpha_field)),
                   levels=[27.4], colors='lemonchiffon', linewidths=1, transform=vector_crs)
        ax.contour(self.lons, self.lats, np.ma.array(neutral_density_field, mask=np.isnan(self.alpha_field)),
                   levels=[27.5], colors='burlywood', linewidths=1, transform=vector_crs)
        ax.contour(self.lons, self.lats, np.ma.array(neutral_density_field, mask=np.isnan(self.alpha_field)),
                   levels=[27.6], colors='orange', linewidths=1, transform=vector_crs)
        ax.contour(self.lons, self.lats, np.ma.array(neutral_density_field, mask=np.isnan(self.alpha_field)),
                   levels=[27.7], colors='darkorange', linewidths=1, transform=vector_crs)
        ax.contour(self.lons, self.lats, np.ma.array(neutral_density_field, mask=np.isnan(self.alpha_field)),
                   levels=[27.8], colors='crimson', linewidths=1, transform=vector_crs)

        zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
        ice_edge_patch = mpatches.Patch(color='black', label='15% ice edge')
        density_274_patch = mpatches.Patch(color='lemonchiffon', label='$\gamma$=27.4 kg\m$^3$')
        density_275_patch = mpatches.Patch(color='burlywood', label='$\gamma$=27.5 kg\m$^3$')
        density_276_patch = mpatches.Patch(color='orange', label='$\gamma$=27.6 kg\m$^3$')
        density_277_patch = mpatches.Patch(color='darkorange', label='$\gamma$=27.7 kg\m$^3$')
        density_278_patch = mpatches.Patch(color='crimson', label='$\gamma$=27.8 kg\m$^3$')
        plt.legend(handles=[zero_stress_line_patch, ice_edge_patch, density_274_patch, density_275_patch,
                            density_276_patch, density_277_patch, density_278_patch],
                   loc='lower center', bbox_to_anchor=(0, -0.3, 1, -0.3), ncol=2, mode='expand', borderaxespad=0)

        # Add date label to bottom left.
        if plot_type == 'daily':
            date_str = str(self.date.year) + '/' + str(self.date.month).zfill(2) + '/' + str(self.date.day).zfill(2)
            plt.gcf().text(0.1, 0.1, date_str, fontsize=10)
        elif plot_type == 'monthly':
            date_str = '{:%b %Y} average'.format(self.date)
            plt.gcf().text(0.1, 0.1, date_str, fontsize=10)
        elif plot_type == 'annual':
            date_str = '{:%Y} (annual mean)'.format(self.date)
            plt.gcf().text(0.1, 0.1, date_str, fontsize=10)
        elif plot_type == 'custom':
            plt.gcf().text(0.1, 0.1, custom_label, fontsize=10)

        logger.info('Saving diagnostic figures to disk...')

        if plot_type == 'daily':
            tau_filename = 'surface_stress_' + str(self.date.year) + str(self.date.month).zfill(2) \
                           + str(self.date.day).zfill(2)
        elif plot_type == 'monthly':
            tau_filename = 'surface_stress_' + '{:%b%Y}_avg'.format(self.date)
        elif plot_type == 'annual':
            tau_filename = 'surface_stress_' + '{:%Y}_avg'.format(self.date)
        elif plot_type == 'custom':
            tau_filename = 'surface_stress_' + custom_label

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

        """ Calculate ice divergence \grad \dot (h*u_ice) ~ f - r """
        ice_div_x_field = np.zeros((len(self.lats), len(self.lons)))
        ice_div_y_field = np.zeros((len(self.lats), len(self.lons)))
        ice_div_field = np.zeros((len(self.lats), len(self.lons)))

        h = 1  # [m]
        for i in range(1, len(self.lats) - 1):
            lat = self.lats[i]

            progress_percent = 100 * i / (len(self.lats) - 2)
            logger.info('(ice_div) lat = {:.2f}/{:.2f} ({:.1f}%)'.format(lat, lat_max, progress_percent))

            dx = distance(self.lats[i-1], self.lons[0], self.lats[i+1], self.lons[0])
            dy = distance(self.lats[i], self.lons[0], self.lats[i], self.lons[2])
            for j in range(1, len(self.lons) - 1):
                lon = self.lons[j]

                u_ice_i_jp1 = self.u_ice_field[i][j+1]
                u_ice_i_jm1 = self.u_ice_field[i][j-1]
                v_ice_ip1_j = self.v_ice_field[i+1][j]
                v_ice_im1_j = self.v_ice_field[i-1][j]

                if not np.isnan(u_ice_i_jm1) and not np.isnan(u_ice_i_jp1):
                    dudx = (u_ice_i_jp1 - u_ice_i_jm1) / dx
                    # dSdx = (salinity_ip1_j - salinity_im1_j) / dx
                    ice_div_x_field[i][j] = h * dudx
                else:
                    ice_div_x_field[i][j] = np.nan

                if not np.isnan(v_ice_im1_j) and not np.isnan(v_ice_ip1_j):
                    dvdy = (v_ice_ip1_j - v_ice_im1_j) / dy
                    # dSdy = (salinity_i_jp1 - salinity_i_jm1) / dy
                    ice_div_y_field[i][j] = h * dvdy
                else:
                    ice_div_y_field[i][j] = np.nan

                if not np.isnan(ice_div_x_field[i][j]) and not np.isnan(ice_div_y_field[i][j]):
                    ice_div_field[i][j] = ice_div_x_field[i][j] + ice_div_y_field[i][j]
                else:
                    ice_div_field[i][j] = np.nan

        fig = plt.figure(figsize=(4, 4))
        land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face',
                                                       facecolor='dimgray', linewidth=0)
        vector_crs = ccrs.PlateCarree()

        ax = plt.axes(projection=ccrs.SouthPolarStereo())
        ax.add_feature(land_50m)
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        ax.set_title('d/dx (h*u_ice) + d/dy (h*v_ice)')

        max_val = 5e-7
        im = ax.pcolormesh(self.lons, self.lats, ice_div_field, transform=vector_crs,
                           cmap=cmocean.cm.balance, vmin=-max_val, vmax=max_val)

        clb = fig.colorbar(im, ax=ax, extend='both')
        clb.ax.set_title('m/s')

        ax.contour(self.lons, self.lats, np.ma.array(self.tau_x_field, mask=np.isnan(self.alpha_field)),
                   levels=[0], colors='green', linewidths=0.5, transform=vector_crs)
        ax.contour(self.lons, self.lats, np.ma.array(self.alpha_field, mask=np.isnan(self.alpha_field)),
                   levels=[0.15], colors='black', linewidths=0.5, transform=vector_crs)

        ice_div_png_filepath = os.path.join(self.surface_stress_dir, str(self.date.year), 'surface_stress_' + custom_label + '_ice_div.png')

        plt.savefig(ice_div_png_filepath, dpi=600, format='png', transparent=False, bbox_inches='tight')
        logger.info('Saved diagnostic figure: {:s}'.format(ice_div_png_filepath))

    def write_fields_to_netcdf(self):
        from constants import netcdf_var_names, var_units, var_positive, var_long_names

        tau_nc_filename = 'surface_stress_' + str(self.date.year) + str(self.date.month).zfill(2) \
                       + str(self.date.day).zfill(2) + '.nc'
        tau_filepath = os.path.join(self.surface_stress_dir, str(self.date.year), tau_nc_filename)
        tau_dir = os.path.dirname(tau_filepath)
        if not os.path.exists(tau_dir):
            logger.info('Creating directory: {:s}'.format(tau_dir))
            os.makedirs(tau_dir)

        logger.info('Saving fields to netCDF file: {:s}'.format(tau_filepath))

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

        for var_name in netcdf_var_names:
            field_var = tau_dataset.createVariable(var_name, float, ('lat', 'lon'), zlib=True)
            field_var.units = var_units[var_name]
            field_var.positive = var_positive[var_name]
            field_var.long_name = var_long_names[var_name]
            field_var[:] = self.var_fields[var_name]

        tau_dataset.close()
