import os
import numpy as np

import netCDF4
import matplotlib.colors as colors

from MeanDynamicTopographyDataReader import MeanDynamicTopographyDataReader
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

        # tmp
        self.zonal_ice_div = np.zeros((len(self.lats), len(self.lons)))
        self.merid_ice_div = np.zeros((len(self.lats), len(self.lons)))
        self.melt_rate = np.zeros((len(self.lats), len(self.lons)))
        self.psi_delta = np.zeros((len(self.lats), len(self.lons)))

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
            self.u_geo_data = MeanDynamicTopographyDataReader()
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

    def compute_daily_surface_stress_field(self, u_geo_source='zero', u_Ekman_vec_type='vertical_avg'):
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

                u_wind_vec = self.u_wind_data.ocean_surface_wind_vector(lat, lon, 'interp')
                alpha = self.sea_ice_conc_data.sea_ice_concentration(lat, lon, 'interp')
                u_ice_vec = self.sea_ice_motion_data.seaice_motion_vector(lat, lon, 'interp')

                if u_geo_source == 'zero':
                    u_geo_vec = np.array([0, 0])
                elif u_geo_source == 'climo':
                    u_geo_vec = self.u_geo_data.u_geo_mean(lat, lon, 'interp')

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

                # If we have data for everything, and we're in the SIZ then use the Richardson method to compute tau.
                tau_vec, tau_air_vec, tau_ice_vec = self.surface_stress(f, u_geo_vec, u_wind_vec, alpha, u_ice_vec)

                self.tau_air_x_field[i][j] = tau_air_vec[0]
                self.tau_air_y_field[i][j] = tau_air_vec[1]
                self.tau_ice_x_field[i][j] = tau_ice_vec[0]
                self.tau_ice_y_field[i][j] = tau_ice_vec[1]
                self.tau_x_field[i][j] = tau_vec[0]
                self.tau_y_field[i][j] = tau_vec[1]
                self.tau_SIZ_x_field[i][j] = tau_vec[0]
                self.tau_SIZ_y_field[i][j] = tau_vec[1]

                # Recalculate u_Ekman_vec. Not sure why I have to do this, but otherwise I just get the zero vector...
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
        """ Compute daily Ekman pumping field w_Ekman = curl(tau / rho * f). """

        from constants import Omega, rho_0
        logger.info('Calculating wind stress curl and Ekman pumping fields...')

        i_max = len(self.lats) - 1
        j_max = len(self.lons) - 1

        # Setting NaN field values for i=0 (lat=lat_min) and i=i_max (lat=lat_max) where w_Ekman cannot be calculated
        # as there is no northern or southern value to use to calculate d/dy (tau_x) or d/dx (tau_y).
        self.ddx_tau_y_field[0, :] = np.nan
        self.ddx_tau_y_field[i_max, :] = np.nan
        self.ddy_tau_x_field[0, :] = np.nan
        self.ddy_tau_x_field[i_max, :] = np.nan
        self.stress_curl_field[0, :] = np.nan
        self.stress_curl_field[i_max, :] = np.nan
        self.w_Ekman_field[0, :] = np.nan
        self.w_Ekman_field[i_max, :] = np.nan

        for i in range(1, len(self.lats) - 1):
            lat = self.lats[i]
            f = 2 * Omega * np.sin(np.deg2rad(lat))  # Coriolis parameter [s^-1]

            progress_percent = 100 * i / (len(self.lats) - 2)
            logger.info('({} w_Ekman) lat = {:.2f}/{:.2f} ({:.1f}%)'.format(self.date, lat, lat_max, progress_percent))

            dx = distance(self.lats[i-1], self.lons[0], self.lats[i+1], self.lons[0])
            dy = distance(self.lats[i], self.lons[0], self.lats[i], self.lons[2])

            for j in range(len(self.lons)):

                # Taking modulus of j-1 and j+1 to get the correct index in the special cases of
                #  * j=0 (180 W) and need to use the value from j=j_max (180 E)
                #  * j=j_max (180 E) and need to use the value from j=0 (180 W)
                jm1 = (j-1) % j_max
                jp1 = (j+1) % j_max

                if not np.isnan(self.tau_x_field[i][jp1]) and not np.isnan(self.tau_x_field[i][jp1]) \
                        and not np.isnan(self.tau_y_field[i-1][j]) and not np.isnan(self.tau_y_field[i+1][j]):

                    # Second-order centered difference scheme where we divide by the distance between the i+1 and i-1
                    # cells, which is just dx as defined in the above line. Textbook formulas will usually have a 2*dx
                    # in the denominator because dx is the width of just one cell.

                    # TODO: Why does it look like accessing the wrong axis gives the right derivative!?
                    # dtauydx = (self.tau_y_field[i+1][j] - self.tau_y_field[i-1][j]) / dx
                    # dtauxdy = (self.tau_x_field[i][j+1] - self.tau_x_field[i][j-1]) / dy
                    dtauydx = (self.tau_y_field[i][jp1] - self.tau_y_field[i][jm1]) / dx
                    dtauxdy = (self.tau_x_field[i+1][j] - self.tau_x_field[i-1][j]) / dy

                    self.ddx_tau_y_field[i][j] = dtauydx
                    self.ddy_tau_x_field[i][j] = dtauxdy
                    self.stress_curl_field[i][j] = dtauydx - dtauxdy
                    self.w_Ekman_field[i][j] = (dtauydx - dtauxdy) / (rho_0 * f)
                else:
                    self.ddx_tau_y_field[i][j] = np.nan
                    self.ddy_tau_x_field[i][j] = np.nan
                    self.stress_curl_field[i][j] = np.nan
                    self.w_Ekman_field[i][j] = np.nan

        from SalinityDataset import SalinityDataset
        salinity_dataset = SalinityDataset(time_span='A5B2', avg_period=avg_period, grid_size='04', field_type='an')

    def compute_daily_freshwater_ekman_advection_field(self):
        """ Calculate freshwater flux div(u_Ek*S).  """

        i_max = len(self.lats) - 1
        j_max = len(self.lons) - 1

        # TODO: Do all this in a loop in __init__?
        self.dSdx_field[0, :] = np.nan
        self.dSdx_field[i_max, :] = np.nan
        self.dSdy_field[0, :] = np.nan
        self.dSdy_field[i_max, :] = np.nan
        self.ddx_uEk_S_field[0, :] = np.nan
        self.ddx_uEk_S_field[i_max, :] = np.nan
        self.ddy_vEk_S_field[0, :] = np.nan
        self.ddy_vEk_S_field[i_max, :] = np.nan
        self.freshwater_ekman_advection_field[0, :] = np.nan
        self.freshwater_ekman_advection_field[i_max, :] = np.nan

        for i in range(1, len(self.lats) - 1):
            lat = self.lats[i]
            f = 2 * Omega * np.sin(np.deg2rad(lat))  # Coriolis parameter [s^-1]

            progress_percent = 100 * i / (len(self.lats) - 2)
            logger.info('({} freshwater_flux) lat = {:.2f}/{:.2f} ({:.1f}%)'.format(self.date, lat, lat_max,
                                                                                    progress_percent))

            dx = distance(self.lats[i-1], self.lons[0], self.lats[i+1], self.lons[0])
            dy = distance(self.lats[i], self.lons[0], self.lats[i], self.lons[2])

            for j in range(1, len(self.lons) - 1):
                lon = self.lons[j]

                # Taking modulus of j-1 and j+1 to get the correct index in the special cases of
                #  * j=0 (180 W) and need to use the value from j=j_max (180 E)
                #  * j=j_max (180 E) and need to use the value from j=0 (180 W)
                jm1 = (j-1) % j_max
                jp1 = (j+1) % j_max

                u_Ekman_scalar = self.u_Ekman_field[i][j]
                v_Ekman_scalar = self.v_Ekman_field[i][j]

                salinity_ij = self.salinity_field[i][j]
                salinity_ip1_j = self.salinity_field[i+1][j]
                salinity_im1_j = self.salinity_field[i-1][j]
                salinity_i_jp1 = self.salinity_field[i][jp1]
                salinity_i_jm1 = self.salinity_field[i][jm1]

                if not np.isnan(u_Ekman_scalar) and not np.isnan(salinity_i_jm1) \
                        and not np.isnan(salinity_i_jp1):
                    # dSdx = (salinity_ip1_j - salinity_im1_j) / dx
                    dSdx = (salinity_i_jp1 - salinity_i_jm1) / dx

                    self.dSdx_field[i][j] = dSdx
                    self.ddx_uEk_S_field[i][j] = u_Ekman_scalar * dSdx
                else:
                    self.dSdx_field = np.nan
                    self.ddx_uEk_S_field[i][j] = np.nan

                if not np.isnan(v_Ekman_scalar) and not np.isnan(salinity_im1_j) \
                        and not np.isnan(salinity_ip1_j):
                    # dSdy = (salinity_i_jp1 - salinity_i_jm1) / dy
                    dSdy = (salinity_ip1_j - salinity_im1_j) / dy

                    self.dSdy_field[i][j] = dSdy
                    self.ddy_vEk_S_field[i][j] = v_Ekman_scalar * dSdy
                else:
                    self.dSdy_field[i][j] = np.nan
                    self.ddy_vEk_S_field[i][j] = np.nan

                if not np.isnan(self.ddx_uEk_S_field[i][j]) and not np.isnan(self.ddy_vEk_S_field[i][j]):
                    self.freshwater_ekman_advection_field[i][j] \
                        = self.ddx_uEk_S_field[i][j] + self.ddy_vEk_S_field[i][j]

                    # self.psi_delta[i][j] = -self.tau_x_field[i][j] / (1025 * f)

                    # zonal = -(self.tau_x_field[i][j] / (1025 * f)) * (1 / salinity_ij) * dSdy
                    # merid = (self.tau_y_field[i][j] / (1025 * f)) * (1 / salinity_ij) * dSdx

                    # self.melt_rate[i][j] = zonal + merid

                    # self.psi_delta[i][j] = self.psi_delta[i][j] * (2*np.pi*6371e3*np.cos(np.deg2rad(lat))) / 1e6
                else:
                    self.freshwater_flux_field[i][j] = np.nan
                    # self.psi_delta[i][j] = np.nan
                    # self.melt_rate[i][j] = np.nan

    def compute_ice_divergence_field(self):
        """ Calculate ice divergence \grad \cdot (h*u_ice) ~ f - r """
        ice_div_x_field = np.zeros((len(self.lats), len(self.lons)))
        ice_div_y_field = np.zeros((len(self.lats), len(self.lons)))

        h = 1  # [m]

        for i in range(1, len(self.lats) - 1):
            lat = self.lats[i]

            progress_percent = 100 * i / (len(self.lats) - 2)
            logger.info('(ice_div) lat = {:.2f}/{:.2f} ({:.1f}%)'.format(lat, lat_max, progress_percent))

            dx = distance(self.lats[i-1], self.lons[0], self.lats[i+1], self.lons[0])
            dy = distance(self.lats[i], self.lons[0], self.lats[i], self.lons[2])

            # TODO: Consider the j=0 (180 W) and j=j_max (180 E) edge cases.
            for j in range(1, len(self.lons) - 1):
                lon = self.lons[j]

                u_ice_i_jp1 = self.u_ice_field[i][j+1]
                u_ice_i_jm1 = self.u_ice_field[i][j-1]
                v_ice_ip1_j = self.v_ice_field[i+1][j]
                v_ice_im1_j = self.v_ice_field[i-1][j]

                alpha_i_jp1 = self.alpha_field[i][j+1]
                alpha_i_jm1 = self.alpha_field[i][j-1]
                alpha_ip1_j = self.alpha_field[i+1][j]
                alpha_im1_j = self.alpha_field[i-1][j]

                if not np.isnan(u_ice_i_jm1) and not np.isnan(u_ice_i_jp1):
                    dudx = (alpha_i_jp1 * u_ice_i_jp1 - alpha_i_jm1 * u_ice_i_jm1) / dx
                    # dSdx = (salinity_ip1_j - salinity_im1_j) / dx
                    ice_div_x_field[i][j] = h * dudx
                else:
                    ice_div_x_field[i][j] = np.nan

                if not np.isnan(v_ice_im1_j) and not np.isnan(v_ice_ip1_j):
                    dvdy = (alpha_ip1_j * v_ice_ip1_j - alpha_im1_j * v_ice_im1_j) / dy
                    # dSdy = (salinity_i_jp1 - salinity_i_jm1) / dy
                    ice_div_y_field[i][j] = h * dvdy
                else:
                    ice_div_y_field[i][j] = np.nan

                if not np.isnan(ice_div_x_field[i][j]) and not np.isnan(ice_div_y_field[i][j]):
                    self.ice_div_field[i][j] = ice_div_x_field[i][j] + ice_div_y_field[i][j]
                    self.zonal_ice_div[i][j] = ice_div_x_field[i][j]
                    self.merid_ice_div[i][j] = ice_div_y_field[i][j]
                else:
                    self.ice_div_field[i][j] = np.nan
                    self.zonal_ice_div[i][j] = np.nan
                    self.merid_ice_div[i][j] = np.nan

    def process_thermodynamic_fields(self, avg_period):
        logger.info('Calculating average T, S, gamma_n...')
        from SalinityDataset import SalinityDataset
        from TemperatureDataset import TemperatureDataset
        from NeutralDensityDataset import NeutralDensityDataset

        levels = [0, 1, 2, 3, 4, 5, 6, 7]
        salinity_dataset = SalinityDataset(time_span='A5B2', avg_period=avg_period, grid_size='04', field_type='an')
        temperature_dataset = TemperatureDataset(time_span='A5B2', avg_period=avg_period, grid_size='04',
                                                 field_type='an')
        neutral_density_dataset = NeutralDensityDataset(time_span='A5B2', avg_period=avg_period, grid_size='04',
                                                        field_type='an', depth_levels=levels)

        for i in range(len(self.lats)):
            lat = self.lats[i]

            progress_percent = 100 * i / (len(self.lats))
            logger.info('(T, S, gamma) lat = {:.2f}/{:.2f} ({:.1f}%)'.format(lat, lat_max, progress_percent))
            for j in range(len(self.lons)):
                lon = self.lons[j]
                self.salinity_field[i][j] = salinity_dataset.salinity(lat, lon, levels)
                self.temperature_field[i][j] = temperature_dataset.temperature(lat, lon, levels)
                self.neutral_density_field[i][j] = neutral_density_dataset.gamma_n_depth_averaged(lat, lon, levels)

    def compute_mean_fields(self, dates, avg_method, sum_monthly_climo=False, tau_filepath=None):
        import netCDF4
        from constants import output_dir_path
        from utils import log_netCDF_dataset_metadata, get_netCDF_filepath

        if tau_filepath is not None:
            try:
                tau_dataset = netCDF4.Dataset(tau_filepath)
                log_netCDF_dataset_metadata(tau_dataset)
                dataset_found = True
            except OSError as e:
                logger.error('{}'.format(e))
                logger.error('Dataset not found, will compute mean fields: {:s}'.format(tau_filepath))
                dataset_found = False

            if dataset_found:
                logger.info('Dataset found! Loading fields from: {:s}'.format(tau_filepath))
                self.lats = np.array(tau_dataset.variables['lat'])
                self.lons = np.array(tau_dataset.variables['lon'])

                for var in self.var_fields.keys():
                    loaded_field = np.array(tau_dataset.variables[var])
                    self.var_fields[var][:] = loaded_field[:]

                return

        n_days = len(dates)

        # Dictionary storing all the averaged fields.
        field_avg = {}

        # Dictionary storing the number of days of available data at each gridpoint for each field (these are used when
        # avg_method = 'partial_data_ok').
        field_days = {}

        # Initializing all the fields we want to calculate an average for.
        for var_name in self.var_fields.keys():
            field_avg[var_name] = np.zeros((len(self.lats), len(self.lons)))
            field_days[var_name] = np.zeros((len(self.lats), len(self.lons)))

        if sum_monthly_climo:
            for month in range(1,13):
                month_str = str(month).zfill(2)

                tau_nc_filename = 'surface_stress_' + month_str + '_2005-2012_monthly_climo.nc'
                tau_filepath = os.path.join(output_dir_path, 'surface_stress', 'monthly_climo', tau_nc_filename)

                logger.info('Averaging month {:s} ({:s})...'.format(month_str, tau_filepath))

                current_tau_dataset = netCDF4.Dataset(tau_filepath)
                log_netCDF_dataset_metadata(current_tau_dataset)

                self.lats = np.array(current_tau_dataset.variables['lat'])
                self.lons = np.array(current_tau_dataset.variables['lon'])

                daily_fields = {}
                for var_name in self.var_fields.keys():
                    daily_fields[var_name] = np.array(current_tau_dataset.variables[var_name])

                if avg_method == 'full_data_only':
                    for var_name in self.var_fields.keys():
                        field_avg[var_name] = field_avg[var_name] + daily_fields[var_name]/n_days

                elif avg_method == 'partial_data_ok':
                    for var_name in self.var_fields.keys():
                        field_avg[var_name] = field_avg[var_name] + np.nan_to_num(daily_fields[var_name])
                        daily_fields[var_name][~np.isnan(daily_fields[var_name])] = 1
                        daily_fields[var_name][np.isnan(daily_fields[var_name])] = 0
                        field_days[var_name] = field_days[var_name] + daily_fields[var_name]
        else:
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
                for var_name in self.var_fields.keys():
                    daily_fields[var_name] = np.array(current_tau_dataset.variables[var_name])

                if avg_method == 'full_data_only':
                    for var_name in self.var_fields.keys():
                        field_avg[var_name] = field_avg[var_name] + daily_fields[var_name]/n_days

                elif avg_method == 'partial_data_ok':
                    for var_name in self.var_fields.keys():
                        field_avg[var_name] = field_avg[var_name] + np.nan_to_num(daily_fields[var_name])
                        daily_fields[var_name][~np.isnan(daily_fields[var_name])] = 1
                        daily_fields[var_name][np.isnan(daily_fields[var_name])] = 0
                        field_days[var_name] = field_days[var_name] + daily_fields[var_name]

        # Remember that the [:] syntax is used is so that we perform deep copies. Otherwise, e.g.
        # self.var_fields['ice_u'] will point to a different array than the original self.u_ice_field, and will NOT be
        # the same object as self.figure_fields['u_ice']!.
        if avg_method == 'full_data_only':
            for var_name in self.var_fields.keys():
                self.var_fields[var_name][:] = field_avg[var_name][:]

        elif avg_method == 'partial_data_ok':
            for var_name in self.var_fields.keys():
                field_avg[var_name] = np.divide(field_avg[var_name], field_days[var_name])
                self.var_fields[var_name][:] = field_avg[var_name][:]

    def plot_diagnostic_fields(self, plot_type, custom_label=None, avg_period=None):
        logger.info('Injecting u_geo field...')
        from MeanDynamicTopographyDataReader import MeanDynamicTopographyDataReader
        geo = MeanDynamicTopographyDataReader()
        for i in range(len(self.lats)):
            lat = self.lats[i]
            for j in range(len(self.lons)):
                lon = self.lons[j]
                u_geo_mean = geo.u_geo_mean(lat, lon, 'interp')
                # self.u_geo_field[i][j] = u_geo_mean[0]
                # self.v_geo_field[i][j] = u_geo_mean[1]
                self.u_geo_field[i][j] = 0
                self.v_geo_field[i][j] = 0

        # logger.info('Smoothing u_geo field using Gaussian filter...')
        # from scipy.ndimage import gaussian_filter
        # u_geo_smoothed = gaussian_filter(self.u_geo_field, sigma=1)
        # v_geo_smoothed = gaussian_filter(self.v_geo_field, sigma=1)
        #
        # self.u_geo_field[:] = u_geo_smoothed[:]
        # self.v_geo_field[:] = v_geo_smoothed[:]

        # logger.info('Recalculating stresses...')
        # self.compute_daily_surface_stress_field()
        # logger.info('Recalculating Ekman pumping field...')
        # self.compute_daily_ekman_pumping_field()

        logger.info('Smoothing salinity field using Gaussian filter...')
        from scipy.ndimage import gaussian_filter
        salinity_smoothed = gaussian_filter(self.salinity_field, sigma=1)

        self.salinity_field[:] = salinity_smoothed[:]

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

        self.compute_freshwater_flux_field(avg_period)
        self.compute_ice_divergence_field()
        self.process_thermodynamic_fields(avg_period)

        logger.info('Creating diagnostic figure...')

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
        for var in self.figure_fields.keys():
            ax = plt.subplot(gs[gs_coords[var]], projection=ccrs.SouthPolarStereo())
            ax.add_feature(land_50m)
            ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
            ax.set_title(titles[var])

            # Use symmetric logarithmic scale for v_Ekman.
            # if var == 'v_Ekman':
            #     im = ax.pcolormesh(self.lons, self.lats, scale_factor[var] * fields[var], transform=vector_crs,
            #                        cmap=cmaps[var], vmin=cmap_ranges[var][0], vmax=cmap_ranges[var][1],
            #                        # norm=MidpointNormalize(midpoint=0))
            #                        norm=colors.SymLogNorm(linthresh=0.2, linscale=0.2,
            #                                               vmin=cmap_ranges[var][0], vmax=cmap_ranges[var][1]))
            # else:
            #     im = ax.pcolormesh(self.lons, self.lats, scale_factor[var] * fields[var], transform=vector_crs,
            #                        cmap=cmaps[var], vmin=cmap_ranges[var][0], vmax=cmap_ranges[var][1])

            im = ax.pcolormesh(self.lons, self.lats, scale_factor[var] * self.figure_fields[var], transform=vector_crs,
                               cmap=cmaps[var], vmin=cmap_ranges[var][0], vmax=cmap_ranges[var][1])

            clb = fig.colorbar(im, ax=ax, extend='both')
            # clb = fig.colorbar(im, ax=ax, extend='both', fraction=0.046, pad=0.04)
            clb.ax.set_title(colorbar_label[var])

            # Add vector fields to the u_ice and tau_x fields.
            if var == 'u_ice':
                ax.quiver(self.lons[::10], self.lats[::10], self.u_ice_field[::10, ::10], self.v_ice_field[::10, ::10],
                          transform=vector_crs, units='width', width=0.002, scale=2)
            elif var == 'tau_x':
                ax.quiver(self.lons[::10], self.lats[::10], self.tau_x_field[::10, ::10], self.tau_y_field[::10, ::10],
                          transform=vector_crs, units='width', width=0.002, scale=4)

            # Plot zero stress line, zero wind line, and ice edge on tau_x and w_Ekman plots (plus legends).
            if var in ['tau_x', 'tau_y', 'w_Ekman', 'neutral_density']:
                cs = ax.contour(self.lons, self.lats, np.ma.array(self.tau_x_field, mask=np.isnan(self.alpha_field)),
                                levels=[0], colors='green', linewidths=1, transform=vector_crs)
                # ax.contour(self.lons, self.lats, np.ma.array(self.u_wind_field, mask=np.isnan(self.alpha_field)),
                #            levels=[0], colors='gold', linewidths=1, transform=vector_crs)
                ax.contour(self.lons, self.lats, np.ma.array(self.alpha_field, mask=np.isnan(self.alpha_field)),
                           levels=[0.15], colors='black', linewidths=1, transform=vector_crs)

                zero_stress_line_patch = mpatches.Patch(color='green', label='zero zonal stress line')
                # zero_wind_line_patch = mpatches.Patch(color='gold', label='zero zonal wind line')
                ice_edge_patch = mpatches.Patch(color='black', label='15% ice edge')

                if var == 'neutral_density':
                    plt.legend(handles=[zero_stress_line_patch, ice_edge_patch],
                               loc='lower center', bbox_to_anchor=(0, -0.05, 1, -0.05), ncol=2, mode='expand',
                               borderaxespad=0)
                else:
                    plt.legend(handles=[zero_stress_line_patch, ice_edge_patch],
                               loc='lower center', bbox_to_anchor=(0, -0.05, 1, -0.05), ncol=2, mode='expand',
                               borderaxespad=0)

            # Plot zero stress line and ice edge on d/dx (tau_y) and d/dy (tau_x) plots.
            if var in ['u_Ekman', 'v_Ekman', 'dtauydx', 'dtauxdy', 'freshwater_flux', 'ice_div', 'temperature',
                       'salinity']:
                ax.contour(self.lons, self.lats, np.ma.array(self.tau_x_field, mask=np.isnan(self.alpha_field)),
                           levels=[0], colors='green', linewidths=0.5, transform=vector_crs)
                ax.contour(self.lons, self.lats, np.ma.array(self.alpha_field, mask=np.isnan(self.alpha_field)),
                           levels=[0.15], colors='black', linewidths=0.5, transform=vector_crs)

        # Extra plot of \Psi_{-\delta}*1/S*dS/dy.
        ax = plt.subplot(gs[1, 8], projection=ccrs.SouthPolarStereo())
        ax.add_feature(land_50m)
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        ax.set_title('M-F = ψ(-δ)/S*dS/dy')
        im = ax.pcolormesh(self.lons, self.lats, self.melt_rate*24*3600*365,
                           transform=vector_crs, cmap='seismic', vmin=-1, vmax=1)
        clb = fig.colorbar(im, ax=ax, extend='both')
        clb.ax.set_title('')
        ax.contour(self.lons, self.lats, np.ma.array(self.tau_x_field, mask=np.isnan(self.alpha_field)),
                   levels=[0], colors='green', linewidths=0.5, transform=vector_crs)
        ax.contour(self.lons, self.lats, np.ma.array(self.alpha_field, mask=np.isnan(self.alpha_field)),
                   levels=[0.15], colors='black', linewidths=0.5, transform=vector_crs)

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

        logger.info('Saving diagnostic figure: {:s}'.format(tau_png_filepath))
        plt.savefig(tau_png_filepath, dpi=600, format='png', transparent=False, bbox_inches='tight')

        # plt.savefig(tau_pdf_filepath, dpi=300, format='pdf', transparent=True)
        # logger.info('Saved diagnostic figure: {:s}'.format(tau_pdf_filepath))

        plt.close()

    def write_fields_to_netcdf(self, field_type='daily', season_str=None, year_start=None, year_end=None):
        from constants import var_units, var_positive, var_long_names
        from utils import get_netCDF_filepath

        tau_filepath = get_netCDF_filepath(field_type=field_type, date=self.date, season_str=season_str,
                                           year_start=year_start, year_end=year_end)

        tau_dir = os.path.dirname(tau_filepath)
        if not os.path.exists(tau_dir):
            logger.info('Creating directory: {:s}'.format(tau_dir))
            os.makedirs(tau_dir)

        logger.info('Saving fields (field_type={:s}) to netCDF file: {:s}'.format(field_type, tau_filepath))

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

        for var_name in self.var_fields.keys():
            field_var = tau_dataset.createVariable(var_name, float, ('lat', 'lon'), zlib=True)
            field_var.units = var_units[var_name]
            field_var.positive = var_positive[var_name]
            field_var.long_name = var_long_names[var_name]
            field_var[:] = self.var_fields[var_name]

        tau_dataset.close()

        # tmp: plot zonally averaged div(alpha*u_ice*h_ice) and \Psi_{-\delta}*1/S*dS/dy.
        import matplotlib.pyplot as plt

        from utils import get_northward_zero_zonal_stress_line, get_northward_ice_edge, get_coast_coordinates
        from constants import D_e

        contour_coordinate = np.empty((len(self.lats), len(self.lons)))
        contour_coordinate[:] = np.nan

        tau_x_lons, tau_x_lats = get_northward_zero_zonal_stress_line(tau_filepath)
        alpha_lons, alpha_lats = get_northward_ice_edge(tau_filepath)
        coast_lons, coast_lats = get_coast_coordinates(tau_filepath)

        for i in range(len(self.lons)):
            lon = self.lons[i]

            if alpha_lats[i] > tau_x_lats[i] > coast_lats[i]:
                lat_0 = coast_lats[i]
                lat_h = tau_x_lats[i]  # lat_h ~ lat_half ~ lat_1/2
                lat_1 = alpha_lats[i]

                for j in range(len(self.lats)):
                    lat = self.lats[j]

                    if lat < lat_0 or lat > lat_1:
                        contour_coordinate[j][i] = np.nan
                    elif lat_0 <= lat <= lat_h:
                        contour_coordinate[j][i] = (lat - lat_0) / (2 * (lat_h - lat_0))
                    elif lat_h <= lat <= lat_1:
                        contour_coordinate[j][i] = 0.5 + ((lat - lat_h) / (2 * (lat_1 - lat_h)))

        import matplotlib.pyplot as plt
        import cartopy
        import cartopy.crs as ccrs

        # fig = plt.figure()
        #
        # ax = plt.axes(projection=ccrs.SouthPolarStereo())
        # land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face',
        #                                                facecolor='dimgray', linewidth=0)
        # ax.add_feature(land_50m)
        # ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        # vector_crs = ccrs.PlateCarree()
        #
        # im = ax.pcolormesh(self.lons, self.lats, contour_coordinate, transform=vector_crs, cmap='PuOr')
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
        c_bins = c_bins + (delta_c/2)

        ice_div_cavg = np.zeros(c_bins.shape)
        melt_rate_cavg = np.zeros(c_bins.shape)
        psi_delta_cavg = np.zeros(c_bins.shape)

        for i in range(len(c_bins)):
            c = c_bins[i]
            c_low = c - (delta_c/2)
            c_high = c + (delta_c/2)

            c_in_range = np.logical_and(contour_coordinate > c_low, contour_coordinate < c_high)

            ice_div_cavg[i] = np.nanmean(self.ice_div_field[c_in_range])
            melt_rate_cavg[i] = np.nanmean(self.melt_rate[c_in_range])
            psi_delta_cavg[i] = np.nanmean(self.psi_delta[c_in_range])

        fig = plt.figure(figsize=(16, 9))

        ax = fig.add_subplot(131)
        # ax.title('div(α*h*u_ice) ~ M - F [m/year]')
        # ax.plot(self.lats, np.nanmean(self.ice_div_field, axis=1)*24*3600*365, label='div(α*h*u_ice) [m/year]')
        # ax.plot(self.lats, np.nanmean(self.zonal_ice_div, axis=1)*24*3600*365, label='d/dx(α*h*u_ice) [m/year]')
        # ax.plot(self.lats, np.nanmean(self.merid_ice_div, axis=1)*24*3600*365, label='d/dy(α*h*u_ice) [m/year]')

        # ax.plot(self.lats, np.nanmean(self.melt_rate, axis=1) * 24 * 3600 * 365,
        #         label='ψ(-δ) * 1/S * dS/dy ~ M-F [m/year]')
        # ax.plot(self.lats, np.nanmean(np.multiply(self.melt_rate, self.alpha_field), axis=1) * 24 * 3600 * 365 * D_e,
        #         label='α*(M-F) [m/year]')

        # ax.plot(self.lats, np.nanmean(self.salinity_field, axis=1), label='salinity')

        # ax.set_xlim(-80, -50)
        # ax.set_ylim(-5, 15)
        # ax.set_xlabel('latitude', fontsize='xx-large')
        # ax.set_ylabel('m/year', fontsize='xx-large')

        ax.plot(c_bins, ice_div_cavg * 24 * 3600 * 365, label='div(α*h*u_ice) [m/year]')
        ax.plot(c_bins, melt_rate_cavg * 24 * 3600 * 365, label='(M-F) [m/year]')

        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], minor=False)
        ax.set_xlabel('\"green line coordinates\" (0=coast, 0.5=zero stress line, 1=ice edge)', fontsize='large')
        ax.set_ylabel('m/year', fontsize='xx-large')

        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(linestyle='--')
        ax.legend(fontsize='xx-large')

        ax1 = fig.add_subplot(132)
        # ax.title('M - F = ψ(-δ) * 1/S * dS/dy [m/year]')
        ax1.plot(self.lats, np.nanmean(self.salinity_field, axis=1), label='salinity [g/kg]', color='b')
        ax1.set_ylabel('salinity [g/kg]', color='b', fontsize='xx-large')
        ax1.tick_params('y', colors='b')
        ax1.tick_params(axis='both', which='major', labelsize=10)

        ax2 = ax1.twinx()
        ax2.plot(self.lats, np.nanmean(self.temperature_field, axis=1), label='temperature [°C]', color='r')
        ax2.set_ylabel('temperature [°C]', color='r', fontsize='xx-large')
        ax2.tick_params('y', colors='r')
        ax2.tick_params(axis='both', which='major', labelsize=10)

        ax1.set_xlim(-80, -50)
        ax1.set_xlabel('latitude', fontsize='xx-large')

        ax1.grid(linestyle='--')
        # ax.legend()

        ax = fig.add_subplot(133)
        # ax.title('Streamfunction ψ(-δ) [Sv]')
        # ax.plot(self.lats, np.nanmean(self.psi_delta, axis=1), label='Ekman transport streamfunction ψ(-δ) [Sv]')
        # ax.set_xlim(-80, -50)
        # ax.set_xlabel('latitude', fontsize='xx-large')
        # ax.set_ylabel('Sverdrups', fontsize='xx-large')

        ax.plot(c_bins, psi_delta_cavg, label='Ekman transport streamfunction ψ(-δ) [Sv]')
        ax.set_xlabel('\"green line coordinates\" (0=coast, 0.5=zero stress line, 1=ice edge)', fontsize='large')

        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], minor=False)
        ax.set_ylabel('Sverdrups', fontsize='xx-large')

        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(linestyle='--')
        ax.legend(fontsize='xx-large')

        # plt.show()

        tau_png_filepath = os.path.join(tau_dir, str(self.date.month).zfill(2) + '_zonal_avg.png')
        plt.savefig(tau_png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')
        plt.close()
