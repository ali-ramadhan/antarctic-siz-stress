import os
import time
import datetime
import numpy as np

import netCDF4
import matplotlib
import matplotlib.colors as colors

# matplotlib.use('AGG')

from MeanDynamicTopographyDataReader import MeanDynamicTopographyDataReader
from SurfaceWindDataset import SurfaceWindDataset
from SeaIceConcentrationDataset import SeaIceConcentrationDataset
from SeaIceMotionDataset import SeaIceMotionDataset
from GeostrophicCurrentDataset import GeostrophicCurrentDataset

from SalinityDataset import SalinityDataset
from TemperatureDataset import TemperatureDataset
from NeutralDensityDataset import NeutralDensityDataset

from utils import distance, get_netCDF_filepath, get_WOA_parameters
from constants import output_dir_path, figure_dir_path
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

    surface_stress_dir = os.path.join(output_dir_path, 'surface_stress')

    # 2D rotation matrices for +45 degree and -45 degree rotation.
    R_45deg = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]])
    R_m45deg = np.array([[np.cos(-np.pi/4), -np.sin(-np.pi/4)], [np.sin(-np.pi/4), np.cos(-np.pi/4)]])

    def __init__(self, field_type, date=None, season_str=None, year_start=None, year_end=None):
        self.field_type = field_type
        self.date = date
        self.season_str = season_str
        self.year_start = year_start
        self.year_end = year_end

        self.lats = np.linspace(lat_min, lat_max, n_lat)
        self.lons = np.linspace(lon_min, lon_max, n_lon)

        # Remove the +180 longitude as it coincides with the -180 longitude.
        # Actually no, it should not be removed. It's important when plotting the fields if we want the last sector to
        # be plotted as well.
        # self.lons = self.lons[:-1]

        # Filepath to the netCDF file that will be used to store all the fields. Other functions that save figures to
        # disk will use the netCDF filename to assign filenames their figures in other directories.
        self.nc_filepath = get_netCDF_filepath(field_type=field_type, date=date, season_str=season_str,
                                               year_start=year_start, year_end=year_end)

        # Figure out which data fields to load from the World Ocean Atlas (WOA) based on the type of surface stress
        # field we're going to calculate.
        WOA_parameters = get_WOA_parameters(field_type=field_type, date=date, season_str=season_str,
                                            year_start=year_start, year_end=year_end)

        self.WOA_time_span = WOA_parameters['time_span']
        self.WOA_avg_period = WOA_parameters['avg_period']
        self.WOA_grid_size = WOA_parameters['grid_size']
        self.WOA_field_type = WOA_parameters['field_type']

        logger.info('Assigned WOA parameters: time_span={:s}, avg_period={:s}, grid_size={:s}, field_type={:s}'
                    .format(self.WOA_time_span, self.WOA_avg_period, self.WOA_grid_size, self.WOA_field_type))

        # Initializing all the fields we want to write to the netCDF file.
        # Data (from gridded products) fields.
        self.u_geo_field = np.zeros((len(self.lats), len(self.lons)))
        self.v_geo_field = np.zeros((len(self.lats), len(self.lons)))
        self.u_wind_field = np.zeros((len(self.lats), len(self.lons)))
        self.v_wind_field = np.zeros((len(self.lats), len(self.lons)))
        self.alpha_field = np.zeros((len(self.lats), len(self.lons)))
        self.u_ice_field = np.zeros((len(self.lats), len(self.lons)))
        self.v_ice_field = np.zeros((len(self.lats), len(self.lons)))

        # Surface stress fields.
        self.tau_air_x_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_air_y_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_ice_x_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_ice_y_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_SIZ_x_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_SIZ_y_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_x_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_y_field = np.zeros((len(self.lats), len(self.lons)))

        # Surface stress fields (neglecting geostrophic currents).
        self.tau_nogeo_air_x_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_nogeo_air_y_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_nogeo_ice_x_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_nogeo_ice_y_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_nogeo_SIZ_x_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_nogeo_SIZ_y_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_nogeo_x_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_nogeo_y_field = np.zeros((len(self.lats), len(self.lons)))

        # Difference between tau (with u_geo) and tau (without u_geo).
        self.tau_ig_x_field = np.zeros((len(self.lats), len(self.lons)))
        self.tau_ig_y_field = np.zeros((len(self.lats), len(self.lons)))

        # Ekman surface velocity (u_Ekman) and Ekman volume transport (U_Ekman) fields.
        self.u_Ekman_field = np.zeros((len(self.lats), len(self.lons)))
        self.v_Ekman_field = np.zeros((len(self.lats), len(self.lons)))
        self.u_Ekman_SIZ_field = np.zeros((len(self.lats), len(self.lons)))
        self.v_Ekman_SIZ_field = np.zeros((len(self.lats), len(self.lons)))
        self.U_Ekman_field = np.zeros((len(self.lats), len(self.lons)))
        self.V_Ekman_field = np.zeros((len(self.lats), len(self.lons)))
        self.U_Ekman_SIZ_field = np.zeros((len(self.lats), len(self.lons)))
        self.V_Ekman_SIZ_field = np.zeros((len(self.lats), len(self.lons)))

        # Ekman pumping fields.
        self.ddy_tau_x_field = np.zeros((len(self.lats), len(self.lons)))
        self.ddx_tau_y_field = np.zeros((len(self.lats), len(self.lons)))
        self.stress_curl_field = np.zeros((len(self.lats), len(self.lons)))
        self.w_Ekman_field = np.zeros((len(self.lats), len(self.lons)))

        # Ekman pumping fields (neglecting geostrophic currents).
        self.ddy_tau_nogeo_x_field = np.zeros((len(self.lats), len(self.lons)))
        self.ddx_tau_nogeo_y_field = np.zeros((len(self.lats), len(self.lons)))
        self.stress_curl_nogeo_field = np.zeros((len(self.lats), len(self.lons)))
        self.w_Ekman_nogeo_field = np.zeros((len(self.lats), len(self.lons)))

        # Ekman pumping components.
        self.w_a_field = np.zeros((len(self.lats), len(self.lons)))
        self.w_i_field = np.zeros((len(self.lats), len(self.lons)))
        self.w_i0_field = np.zeros((len(self.lats), len(self.lons)))
        self.w_ig_field = np.zeros((len(self.lats), len(self.lons)))
        self.w_A_field = np.zeros((len(self.lats), len(self.lons)))
        self.gamma_metric_field = np.zeros((len(self.lats), len(self.lons)))

        # Thermodynamic fields.
        self.salinity_field = np.zeros((len(self.lats), len(self.lons)))
        self.temperature_field = np.zeros((len(self.lats), len(self.lons)))
        self.neutral_density_field = np.zeros((len(self.lats), len(self.lons)))

        # Freshwater Ekman advection fields.
        self.dSdx_field = np.zeros((len(self.lats), len(self.lons)))
        self.dSdy_field = np.zeros((len(self.lats), len(self.lons)))
        self.ddx_uEk_S_field = np.zeros((len(self.lats), len(self.lons)))
        self.ddy_vEk_S_field = np.zeros((len(self.lats), len(self.lons)))
        self.freshwater_ekman_advection_field = np.zeros((len(self.lats), len(self.lons)))

        # Ice-flux divergence fields.
        self.zonal_ice_flux_div_field = np.zeros((len(self.lats), len(self.lons)))
        self.merid_ice_flux_div_field = np.zeros((len(self.lats), len(self.lons)))
        self.ice_flux_div_field = np.zeros((len(self.lats), len(self.lons)))

        # Meridional streamfunction and melting/freezing rate.
        self.psi_delta_field = np.zeros((len(self.lats), len(self.lons)))
        self.zonal_melt_rate_field = np.zeros((len(self.lats), len(self.lons)))
        self.merid_melt_rate_field = np.zeros((len(self.lats), len(self.lons)))
        self.melt_rate_field = np.zeros((len(self.lats), len(self.lons)))

        self.h_ice_field = np.zeros((len(self.lats), len(self.lons)))

        # Dictionary of all fields that will be saved into netCDF.
        self.var_fields = {
            'geo_u': self.u_geo_field,
            'geo_v': self.v_geo_field,
            'wind_u': self.u_wind_field,
            'wind_v': self.v_wind_field,
            'alpha': self.alpha_field,
            'ice_u': self.u_ice_field,
            'ice_v': self.v_ice_field,
            'tau_air_x': self.tau_air_x_field,
            'tau_air_y': self.tau_air_y_field,
            'tau_ice_x': self.tau_ice_x_field,
            'tau_ice_y': self.tau_ice_y_field,
            'tau_SIZ_x': self.tau_SIZ_x_field,
            'tau_SIZ_y': self.tau_SIZ_y_field,
            'tau_x': self.tau_x_field,
            'tau_y': self.tau_y_field,
            'tau_nogeo_air_x': self.tau_nogeo_air_x_field,
            'tau_nogeo_air_y': self.tau_nogeo_air_y_field,
            'tau_nogeo_ice_x': self.tau_nogeo_ice_x_field,
            'tau_nogeo_ice_y': self.tau_nogeo_ice_y_field,
            'tau_nogeo_SIZ_x': self.tau_nogeo_SIZ_x_field,
            'tau_nogeo_SIZ_y': self.tau_nogeo_SIZ_y_field,
            'tau_nogeo_x': self.tau_nogeo_x_field,
            'tau_nogeo_y': self.tau_nogeo_y_field,
            'tau_ig_x': self.tau_ig_x_field,
            'tau_ig_y': self.tau_ig_y_field,
            'Ekman_u': self.u_Ekman_field,
            'Ekman_v': self.v_Ekman_field,
            'Ekman_SIZ_u': self.u_Ekman_SIZ_field,
            'Ekman_SIZ_v': self.v_Ekman_SIZ_field,
            'Ekman_U': self.U_Ekman_field,
            'Ekman_V': self.V_Ekman_field,
            'Ekman_SIZ_U': self.U_Ekman_SIZ_field,
            'Ekman_SIZ_V': self.V_Ekman_SIZ_field,
            'dtau_x_dy': self.ddy_tau_x_field,
            'dtau_y_dx': self.ddx_tau_y_field,
            'curl_stress': self.stress_curl_field,
            'Ekman_w': self.w_Ekman_field,
            'ddy_tau_nogeo_x': self.ddy_tau_nogeo_x_field,
            'ddx_tau_nogeo_y': self.ddx_tau_nogeo_y_field,
            'stress_curl_nogeo': self.stress_curl_nogeo_field,
            'w_Ekman_nogeo': self.w_Ekman_nogeo_field,
            'w_a': self.w_a_field,
            'w_i': self.w_i_field,
            'w_i0': self.w_i0_field,
            'w_ig': self.w_ig_field,
            'w_A': self.w_A_field,
            'gamma_metric': self.gamma_metric_field,
            'salinity': self.salinity_field,
            'temperature': self.temperature_field,
            'neutral_density': self.neutral_density_field,
            'dSdx': self.dSdx_field,
            'dSdy': self.dSdy_field,
            'uEk_S_ddx': self.ddx_uEk_S_field,
            'uEk_S_ddy': self.ddy_vEk_S_field,
            'fwf_uEk_S': self.freshwater_ekman_advection_field,
            'ice_flux_div_x': self.zonal_ice_flux_div_field,
            'ice_flux_div_y': self.merid_ice_flux_div_field,
            'ice_flux_div': self.ice_flux_div_field,
            'psi_delta': self.psi_delta_field,
            'melt_rate_x': self.zonal_melt_rate_field,
            'melt_rate_y': self.merid_melt_rate_field,
            'melt_rate': self.melt_rate_field,
            'h_ice': self.h_ice_field
        }

        # Dictionary of all fields to be plotted.
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
            'U_Ekman': self.U_Ekman_field,
            'V_Ekman': self.V_Ekman_field,
            'dtauydx': self.ddx_tau_y_field,
            'dtauxdy': self.ddy_tau_x_field,
            'curl_tau': self.stress_curl_field,
            'w_Ekman': self.w_Ekman_field,
            'freshwater_flux': self.freshwater_ekman_advection_field,
            'ice_div': self.ice_flux_div_field,
            'temperature': self.temperature_field,
            'salinity': self.salinity_field,
            'neutral_density': self.neutral_density_field
        }

        # Setting NaN field values for i=0 (lat=lat_min) and i=i_max (lat=lat_max) where derivative fields, (e.g.
        # w_Ekman, ice_flux_div) cannot be calculated as there is no northern or southern value to use to calculate
        # meridional (y) derivatives.
        derivative_fields = [self.ddx_tau_y_field, self.ddy_tau_x_field, self.stress_curl_field, self.w_Ekman_field,
                             self.ddx_tau_nogeo_y_field, self.ddy_tau_nogeo_x_field, self.stress_curl_nogeo_field,
                             self.w_Ekman_nogeo_field,
                             self.dSdx_field, self.dSdy_field, self.ddx_uEk_S_field, self.ddy_vEk_S_field,
                             self.freshwater_ekman_advection_field,
                             self.zonal_ice_flux_div_field, self.merid_ice_flux_div_field, self.ice_flux_div_field,
                             self.psi_delta_field,
                             self.zonal_melt_rate_field, self.merid_melt_rate_field, self.melt_rate_field]

        i_max = len(self.lats) - 1
        for field in derivative_fields:
            field[0, :] = np.nan
            field[i_max, :] = np.nan

        if date is not None and field_type == 'daily':
            # self.u_geo_data = MeanDynamicTopographyDataReader()
            self.u_geo_data = GeostrophicCurrentDataset(self.date)
            self.sea_ice_conc_data = SeaIceConcentrationDataset(self.date)
            self.sea_ice_motion_data = SeaIceMotionDataset(self.date)
            self.u_wind_data = SurfaceWindDataset(self.date)

    def surface_stress(self, f, u_geo_vec, u_wind_vec, alpha, u_ice_vec):
        """
        Use the modified Richardson iteration to calculate tau and u_Ekman.
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
                logger.warning('iter_count exceeded 50 during calculation of tau and u_Ekman.')
                logger.warning('tau = {}, u_Ekman = {}, tau_residual = {}, tau_rel_error = {:.4f}'
                               .format(tau_vec, u_Ekman_vec, tau_vec_residual, tau_relative_error))
                break

            if np.linalg.norm(tau_vec) > 10:
                logger.warning('Large tau = {}, u_geo_mean = {}, u_wind = {}, alpha = {:.4f}, u_ice = {}'
                               .format(tau_vec, u_geo_vec, u_wind_vec, alpha, u_ice_vec))
                break

            tau_air_vec = rho_air * C_air * np.linalg.norm(u_wind_vec) * u_wind_vec

            u_Ekman_vec = (np.sqrt(2) / (f * rho_0 * D_e)) * np.matmul(self.R_m45deg, tau_vec)

            # if u_Ekman_vec_type == 'surface':
            #     u_Ekman_vec = (np.sqrt(2) / (f * rho_0 * D_e)) * np.matmul(self.R_m45deg, tau_vec)
            # elif u_Ekman_vec_type == 'vertical_avg':
            #     tau_x_scalar = tau_vec[0]
            #     tau_y_scalar = tau_vec[1]
            #     u_Ekman_scalar = tau_y_scalar / (f * rho_0 * D_e)
            #     v_Ekman_scalar = -tau_x_scalar / (f * rho_0 * D_e)
            #     u_Ekman_vec = np.array([u_Ekman_scalar, v_Ekman_scalar])

            u_rel_vec = u_ice_vec - (u_geo_vec + u_Ekman_vec)
            tau_ice_vec = rho_0 * C_seawater * np.linalg.norm(u_rel_vec) * u_rel_vec
            tau_vec = alpha * tau_ice_vec + (1 - alpha) * tau_air_vec

            tau_vec_residual = tau_vec - (alpha * tau_ice_vec + (1 - alpha) * tau_air_vec)
            tau_relative_error = np.linalg.norm(tau_vec_residual) / np.linalg.norm(tau_vec)

            tau_vec = tau_vec + omega * tau_vec_residual

            if np.isnan(tau_vec[0]) or np.isnan(tau_vec[1]):
                logger.warning('NaN tau = {}, u_geo_mean = {}, u_wind = {}, alpha = {:.4f}, u_ice = {}'
                               .format(tau_vec, u_geo_vec, u_wind_vec, alpha, u_ice_vec))

        return tau_vec, tau_air_vec, tau_ice_vec

    def compute_daily_surface_stress_field(self, u_geo_source):
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
                elif u_geo_source == 'CS2':
                    u_geo_vec = self.u_geo_data.geostrophic_current_velocity(lat, lon)
                # elif u_geo_source == 'climo':
                #     u_geo_vec = self.u_geo_data.u_geo_mean(lat, lon, 'interp')

                self.alpha_field[i][j] = alpha
                self.u_geo_field[i][j] = u_geo_vec[0]
                self.v_geo_field[i][j] = u_geo_vec[1]
                self.u_wind_field[i][j] = u_wind_vec[0]
                self.v_wind_field[i][j] = u_wind_vec[1]
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

                    u_Ekman_vec = (np.sqrt(2) / (f * rho_0 * D_e)) * np.matmul(self.R_m45deg, tau_air_vec)

                    tau_x_scalar = tau_air_vec[0]
                    tau_y_scalar = tau_air_vec[1]
                    U_Ekman_scalar = tau_y_scalar / (f * rho_0)
                    V_Ekman_scalar = -tau_x_scalar / (f * rho_0)

                    # if u_Ekman_vec_type == 'surface':
                    #     u_Ekman_vec = (np.sqrt(2) / (f * rho_0 * D_e)) * np.matmul(self.R_m45deg, tau_air_vec)
                    # elif u_Ekman_vec_type == 'vertical_avg':
                    #     tau_x_scalar = tau_air_vec[0]
                    #     tau_y_scalar = tau_air_vec[1]
                    #     u_Ekman_scalar = tau_y_scalar / (f * rho_0 * D_e)
                    #     v_Ekman_scalar = -tau_x_scalar / (f * rho_0 * D_e)
                    #     u_Ekman_vec = np.array([u_Ekman_scalar, v_Ekman_scalar])

                    self.u_Ekman_field[i][j] = u_Ekman_vec[0]
                    self.v_Ekman_field[i][j] = u_Ekman_vec[1]
                    self.u_Ekman_SIZ_field[i][j] = np.nan
                    self.v_Ekman_SIZ_field[i][j] = np.nan

                    self.U_Ekman_field[i][j] = U_Ekman_scalar
                    self.V_Ekman_field[i][j] = V_Ekman_scalar
                    self.U_Ekman_SIZ_field[i][j] = np.nan
                    self.V_Ekman_SIZ_field[i][j] = np.nan

                    # In the absence of ice, the geostrophic current doesn't matter for the stress so it's
                    # the same as the surface stress including geostrophic currents.
                    self.tau_nogeo_air_x_field[i][j] = tau_air_vec[0]
                    self.tau_nogeo_air_y_field[i][j] = tau_air_vec[1]
                    self.tau_nogeo_ice_x_field[i][j] = 0
                    self.tau_nogeo_ice_y_field[i][j] = 0
                    self.tau_nogeo_SIZ_x_field[i][j] = np.nan
                    self.tau_nogeo_SIZ_y_field[i][j] = np.nan
                    self.tau_nogeo_x_field[i][j] = tau_air_vec[0]
                    self.tau_nogeo_y_field[i][j] = tau_air_vec[1]

                    self.tau_ig_x_field[i][j] = np.nan
                    self.tau_ig_y_field[i][j] = np.nan
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
                    self.u_Ekman_SIZ_field[i][j] = np.nan
                    self.v_Ekman_SIZ_field[i][j] = np.nan

                    self.U_Ekman_field[i][j] = np.nan
                    self.V_Ekman_field[i][j] = np.nan
                    self.U_Ekman_SIZ_field[i][j] = np.nan
                    self.V_Ekman_SIZ_field[i][j] = np.nan

                    self.tau_nogeo_air_x_field[i][j] = np.nan
                    self.tau_nogeo_air_y_field[i][j] = np.nan
                    self.tau_nogeo_ice_x_field[i][j] = np.nan
                    self.tau_nogeo_ice_y_field[i][j] = np.nan
                    self.tau_nogeo_SIZ_x_field[i][j] = np.nan
                    self.tau_nogeo_SIZ_y_field[i][j] = np.nan
                    self.tau_nogeo_x_field[i][j] = np.nan
                    self.tau_nogeo_y_field[i][j] = np.nan

                    self.tau_ig_x_field[i][j] = np.nan
                    self.tau_ig_y_field[i][j] = np.nan
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
                # This is the Ekman velocity vector at the ocean surface where it is 45 degrees to the left of the
                # stress (in the Southern Hemisphere).
                u_Ekman_vec = (np.sqrt(2) / (f * rho_0 * D_e)) * np.matmul(self.R_m45deg, tau_vec)

                # if u_Ekman_vec_type == 'surface':
                #     u_Ekman_vec = (np.sqrt(2) / (f * rho_0 * D_e)) * np.matmul(self.R_m45deg, tau_vec)
                # elif u_Ekman_vec_type == 'vertical_avg':
                #     tau_x_scalar = tau_vec[0]
                #     tau_y_scalar = tau_vec[1]
                #     u_Ekman_scalar = tau_y_scalar / (f * rho_0 * D_e)
                #     v_Ekman_scalar = -tau_x_scalar / (f * rho_0 * D_e)
                #     u_Ekman_vec = np.array([u_Ekman_scalar, v_Ekman_scalar])

                self.u_Ekman_field[i][j] = u_Ekman_vec[0]
                self.v_Ekman_field[i][j] = u_Ekman_vec[1]
                self.u_Ekman_SIZ_field[i][j] = u_Ekman_vec[0]
                self.v_Ekman_SIZ_field[i][j] = u_Ekman_vec[1]

                # Calculate Ekman volume transport
                tau_x_scalar = tau_air_vec[0]
                tau_y_scalar = tau_air_vec[1]
                U_Ekman_scalar = tau_y_scalar / (f * rho_0)
                V_Ekman_scalar = -tau_x_scalar / (f * rho_0)

                self.U_Ekman_field[i][j] = U_Ekman_scalar
                self.V_Ekman_field[i][j] = V_Ekman_scalar
                self.U_Ekman_SIZ_field[i][j] = U_Ekman_scalar
                self.V_Ekman_SIZ_field[i][j] = V_Ekman_scalar

                # Calculate the ice-ocean and air-ocean surface stresses neglecting geostrophic currents. Here we
                # are going to calculate it by assuming the Ekman velocity is the same as the case with geostrophic
                # currents and use u_rel = u_ice - u_Ekman. Since we know u_Ekman there is no need to perform an
                # iteration and we can just straight away compute tau.

                tau_nogeo_air_vec = rho_air * C_air * np.linalg.norm(u_wind_vec) * u_wind_vec
                u_nogeo_rel_vec = u_ice_vec - u_Ekman_vec
                tau_nogeo_ice_vec = rho_0 * C_seawater * np.linalg.norm(u_nogeo_rel_vec) * u_nogeo_rel_vec
                tau_nogeo_vec = alpha * tau_nogeo_ice_vec + (1 - alpha) * tau_nogeo_air_vec

                self.tau_nogeo_air_x_field[i][j] = tau_nogeo_air_vec[0]
                self.tau_nogeo_air_y_field[i][j] = tau_nogeo_air_vec[1]
                self.tau_nogeo_ice_x_field[i][j] = tau_nogeo_ice_vec[0]
                self.tau_nogeo_ice_y_field[i][j] = tau_nogeo_ice_vec[1]
                self.tau_nogeo_SIZ_x_field[i][j] = tau_nogeo_vec[0]
                self.tau_nogeo_SIZ_y_field[i][j] = tau_nogeo_vec[1]
                self.tau_nogeo_x_field[i][j] = tau_nogeo_vec[0]
                self.tau_nogeo_y_field[i][j] = tau_nogeo_vec[1]

                self.tau_ig_x_field[i][j] = self.tau_ice_x_field[i][j] - self.tau_nogeo_ice_x_field[i][j]
                self.tau_ig_y_field[i][j] = self.tau_ice_y_field[i][j] - self.tau_nogeo_ice_y_field[i][j]

    def compute_daily_ekman_pumping_field(self):
        """ Compute daily Ekman pumping field w_Ekman = curl(tau / rho * f). """

        from constants import Omega, rho_0
        logger.info('Calculating wind stress curl and Ekman pumping fields...')

        i_max = len(self.lats) - 1
        j_max = len(self.lons) - 1

        for i, lat in enumerate(self.lats[1:-1]):
            f = 2 * Omega * np.sin(np.deg2rad(lat))  # Coriolis parameter [s^-1]

            progress_percent = 100 * i / (len(self.lats) - 2)
            logger.info('({} w_Ekman) lat = {:.2f}/{:.2f} ({:.1f}%)'.format(self.date, lat, lat_max, progress_percent))

            dx = distance(self.lats[i-1], self.lons[0], self.lats[i+1], self.lons[0])
            dy = distance(self.lats[i], self.lons[0], self.lats[i], self.lons[2])

            for j, lon in enumerate(self.lons):
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
                    # dtauydx = (self.tau_y_field[i+1][j] - self.tau_y_field[i-1][j]) / dx
                    # dtauxdy = (self.tau_x_field[i][j+1] - self.tau_x_field[i][j-1]) / dy

                    # Calculate Ekman pumping rates (including geostrophic currents).
                    # TODO: Why does it look like accessing the wrong axis gives the right derivative!?
                    dtauydx = (self.tau_y_field[i][jp1] - self.tau_y_field[i][jm1]) / dx
                    dtauxdy = (self.tau_x_field[i+1][j] - self.tau_x_field[i-1][j]) / dy

                    self.ddx_tau_y_field[i][j] = dtauydx
                    self.ddy_tau_x_field[i][j] = dtauxdy
                    self.stress_curl_field[i][j] = dtauydx - dtauxdy
                    self.w_Ekman_field[i][j] = (dtauydx - dtauxdy) / (rho_0 * f)

                    dtauydx_nogeo = (self.tau_nogeo_y_field[i][jp1] - self.tau_nogeo_y_field[i][jm1]) / dx
                    dtauxdy_nogeo = (self.tau_nogeo_x_field[i+1][j] - self.tau_nogeo_x_field[i-1][j]) / dy

                    self.ddx_tau_nogeo_y_field[i][j] = dtauydx_nogeo
                    self.ddy_tau_nogeo_x_field[i][j] = dtauxdy_nogeo
                    self.stress_curl_nogeo_field[i][j] = dtauydx_nogeo - dtauxdy_nogeo
                    self.w_Ekman_nogeo_field[i][j] = (dtauydx_nogeo - dtauxdy_nogeo) / (rho_0 * f)

                    alpha_i_jp1 = self.alpha_field[i][jp1]
                    alpha_i_jm1 = self.alpha_field[i][jm1]
                    alpha_ip1_j = self.alpha_field[i+1][j]
                    alpha_im1_j = self.alpha_field[i-1][j]

                    tau_ao_x_ip1_j = self.tau_air_x_field[i+1][j]
                    tau_ao_x_im1_j = self.tau_air_x_field[i-1][j]
                    tau_ao_y_i_jp1 = self.tau_air_y_field[i][jp1]
                    tau_ao_y_i_jm1 = self.tau_air_y_field[i][jm1]

                    tau_io_x_geo_ip1_j = self.tau_ice_x_field[i+1][j]
                    tau_io_x_geo_im1_j = self.tau_ice_x_field[i-1][j]
                    tau_io_y_geo_i_jp1 = self.tau_ice_y_field[i][jp1]
                    tau_io_y_geo_i_jm1 = self.tau_ice_y_field[i][jm1]

                    tau_io_x_nogeo_ip1_j = self.tau_nogeo_ice_x_field[i+1][j]
                    tau_io_x_nogeo_im1_j = self.tau_nogeo_ice_x_field[i-1][j]
                    tau_io_y_nogeo_i_jp1 = self.tau_nogeo_ice_y_field[i][jp1]
                    tau_io_y_nogeo_i_jm1 = self.tau_nogeo_ice_y_field[i][jm1]

                    tau_ig_x_ip1_j = self.tau_ig_x_field[i+1][j]
                    tau_ig_x_im1_j = self.tau_ig_x_field[i-1][j]
                    tau_ig_y_i_jp1 = self.tau_ig_y_field[i][jp1]
                    tau_ig_y_i_jm1 = self.tau_ig_y_field[i][jm1]

                    ddx_tau_ao_y = (tau_ao_y_i_jp1 - tau_ao_y_i_jm1) / dx
                    ddy_tau_ao_x = (tau_ao_x_ip1_j - tau_ao_x_im1_j) / dy
                    self.w_A_field[i][j] = (ddx_tau_ao_y - ddy_tau_ao_x) / (rho_0 * f)

                    ddx_1malpha_tau_ao_y = ((1-alpha_i_jp1) * tau_ao_y_i_jp1 - (1-alpha_i_jm1) * tau_ao_y_i_jm1) / dx
                    ddy_1malpha_tau_ao_x = ((1-alpha_ip1_j) * tau_ao_x_ip1_j - (1-alpha_im1_j) * tau_ao_x_im1_j) / dy
                    self.w_a_field[i][j] = (ddx_1malpha_tau_ao_y - ddy_1malpha_tau_ao_x) / (rho_0 * f)

                    ddx_alpha_tau_io_y_geo = (alpha_i_jp1 * tau_io_y_geo_i_jp1 - alpha_i_jm1 * tau_io_y_geo_i_jm1) / dx
                    ddy_alpha_tau_io_x_geo = (alpha_ip1_j * tau_io_x_geo_ip1_j - alpha_im1_j * tau_io_x_geo_im1_j) / dy
                    self.w_i_field[i][j] = (ddx_alpha_tau_io_y_geo - ddy_alpha_tau_io_x_geo) / (rho_0 * f)

                    ddx_alpha_tau_io_y_nogeo = (alpha_i_jp1 * tau_io_y_nogeo_i_jp1 - alpha_i_jm1 * tau_io_y_nogeo_i_jm1) / dx
                    ddy_alpha_tau_io_x_nogeo = (alpha_ip1_j * tau_io_x_nogeo_ip1_j - alpha_im1_j * tau_io_x_nogeo_im1_j) / dy
                    self.w_i0_field[i][j] = (ddx_alpha_tau_io_y_nogeo - ddy_alpha_tau_io_x_nogeo) / (rho_0 * f)

                    ddx_alpha_tau_ig_y = (alpha_i_jp1 * tau_ig_y_i_jp1 - alpha_i_jm1 * tau_ig_y_i_jm1) / dx
                    ddy_alpha_tau_ig_x = (alpha_ip1_j * tau_ig_x_ip1_j - alpha_im1_j * tau_ig_x_im1_j) / dx
                    self.w_ig_field[i][j] = (ddx_alpha_tau_ig_y - ddy_alpha_tau_ig_x) / (rho_0 * f)

                    self.gamma_metric_field[i][j] = np.abs(self.w_ig_field[i][j]) \
                        / (np.abs(self.w_a_field[i][j]) + np.abs(self.w_i0_field[i][j]) + np.abs(self.w_ig_field[i][j]))
                else:
                    self.ddx_tau_y_field[i][j] = np.nan
                    self.ddy_tau_x_field[i][j] = np.nan
                    self.stress_curl_field[i][j] = np.nan
                    self.w_Ekman_field[i][j] = np.nan

                    self.w_A_field[i][j] = np.nan
                    self.w_a_field[i][j] = np.nan
                    self.w_i_field[i][j] = np.nan
                    self.w_i0_field[i][j] = np.nan
                    self.w_ig_field[i][j] = np.nan
                    self.gamma_metric_field[i][j] = np.nan

    def process_thermodynamic_fields(self, levels=None, process_neutral_density=False):
        logger.info('Calculating average T, S, gamma_n...')

        salinity_dataset = SalinityDataset(time_span=self.WOA_time_span, avg_period=self.WOA_avg_period,
                                           grid_size=self.WOA_grid_size, field_type=self.WOA_field_type)
        temperature_dataset = TemperatureDataset(time_span=self.WOA_time_span, avg_period=self.WOA_avg_period,
                                                 grid_size=self.WOA_grid_size, field_type=self.WOA_field_type)

        if process_neutral_density:
            gamma_dataset = NeutralDensityDataset(time_span=self.WOA_time_span, avg_period=self.WOA_avg_period,
                                                  grid_size=self.WOA_grid_size, field_type=self.WOA_field_type)
        else:
            self.neutral_density_field[:] = np.nan

        for i in range(len(self.lats)):
            lat = self.lats[i]

            progress_percent = 100 * i / (len(self.lats) - 1)

            if process_neutral_density:
                logger.info('({} T, S, gamma^n) lat = {:.2f}/{:.2f} ({:.1f}%)'
                            .format(self.date, lat, lat_max, progress_percent))
            else:
                logger.info('({} T, S) lat = {:.2f}/{:.2f} ({:.1f}%)'
                            .format(self.date, lat, lat_max, progress_percent))

            for j in range(len(self.lons)):
                lon = self.lons[j]
                self.salinity_field[i][j] = salinity_dataset.salinity(lat, lon, levels)
                self.temperature_field[i][j] = temperature_dataset.temperature(lat, lon, levels)

                if process_neutral_density:
                    self.neutral_density_field[i][j] = gamma_dataset.gamma_n_depth_averaged(lat, lon, levels)

    def load_sea_ice_thickness_field(self):
        from SeaIceThicknessDataset import SeaIceThicknessDataset
        h_ice_dataset = SeaIceThicknessDataset(self.date)

        for i in range(len(self.lats)):
            lat = self.lats[i]

            progress_percent = 100 * i / (len(self.lats) - 1)
            logger.info('({} h_ice) lat = {:.2f}/{:.2f} ({:.1f}%)'.format(self.date, lat, lat_max, progress_percent))
            for j in range(len(self.lons)):
                self.h_ice_field[i][j] = h_ice_dataset.sea_ice_thickness(i, j, self.date)

    def compute_daily_freshwater_ekman_advection_field(self):
        """ Calculate freshwater flux div(u_Ek*S).  """

        j_max = len(self.lons) - 1

        for i in range(1, len(self.lats) - 1):
            lat = self.lats[i]
            f = 2 * Omega * np.sin(np.deg2rad(lat))  # Coriolis parameter [s^-1]

            progress_percent = 100 * i / (len(self.lats) - 2)
            logger.info('({} freshwater_flux) lat = {:.2f}/{:.2f} ({:.1f}%)'
                        .format(self.date, lat, lat_max, progress_percent))

            dx = distance(self.lats[i-1], self.lons[0], self.lats[i+1], self.lons[0])
            dy = distance(self.lats[i], self.lons[0], self.lats[i], self.lons[2])

            for j in range(len(self.lons)):
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
                    self.dSdx_field[i][j] = np.nan
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
                else:
                    self.freshwater_ekman_advection_field[i][j] = np.nan
                    # self.psi_delta_field[i][j] = np.nan
                    # self.melt_rate_field[i][j] = np.nan

    def compute_daily_ice_flux_divergence_field(self):
        """ Calculate ice divergence \grad \cdot (h*u_ice) ~ f - r """

        i_max = len(self.lats) - 1
        j_max = len(self.lons) - 1

        for i in range(1, len(self.lats) - 1):
            lat = self.lats[i]

            progress_percent = 100 * i / (len(self.lats) - 2)
            logger.info('(ice_div) lat = {:.2f}/{:.2f} ({:.1f}%)'.format(lat, lat_max, progress_percent))

            dx = distance(self.lats[i-1], self.lons[0], self.lats[i+1], self.lons[0])
            dy = distance(self.lats[i], self.lons[0], self.lats[i], self.lons[2])

            for j in range(1, len(self.lons) - 1):
                lon = self.lons[j]

                # Taking modulus of j-1 and j+1 to get the correct index in the special cases of
                #  * j=0 (180 W) and need to use the value from j=j_max (180 E)
                #  * j=j_max (180 E) and need to use the value from j=0 (180 W)
                jm1 = (j - 1) % j_max
                jp1 = (j + 1) % j_max

                u_ice_i_jp1 = self.u_ice_field[i][jp1]
                u_ice_i_jm1 = self.u_ice_field[i][jm1]
                v_ice_ip1_j = self.v_ice_field[i+1][j]
                v_ice_im1_j = self.v_ice_field[i-1][j]

                alpha_i_jp1 = self.alpha_field[i][jp1]
                alpha_i_jm1 = self.alpha_field[i][jm1]
                alpha_ip1_j = self.alpha_field[i+1][j]
                alpha_im1_j = self.alpha_field[i-1][j]

                h_ice_i_jp1 = self.h_ice_field[i][jp1]
                h_ice_i_jm1 = self.h_ice_field[i][jm1]
                h_ice_ip1_j = self.h_ice_field[i+1][j]
                h_ice_im1_j = self.h_ice_field[i-1][j]

                if not np.isnan(u_ice_i_jm1) and not np.isnan(u_ice_i_jp1):
                    div_x = (alpha_i_jp1 * h_ice_i_jp1 * u_ice_i_jp1 - alpha_i_jm1 * h_ice_i_jm1 * u_ice_i_jm1) / dx
                    # dSdx = (salinity_ip1_j - salinity_im1_j) / dx

                    self.zonal_ice_flux_div_field[i][j] = div_x
                else:
                    self.zonal_ice_flux_div_field[i][j] = np.nan

                if not np.isnan(v_ice_im1_j) and not np.isnan(v_ice_ip1_j):
                    div_y = (alpha_ip1_j * h_ice_ip1_j * v_ice_ip1_j - alpha_im1_j * h_ice_im1_j * v_ice_im1_j) / dy
                    # dSdy = (salinity_i_jp1 - salinity_i_jm1) / dy

                    self.merid_ice_flux_div_field[i][j] = div_y
                else:
                    self.merid_ice_flux_div_field[i][j] = np.nan

                if not np.isnan(self.zonal_ice_flux_div_field[i][j]) \
                        and not np.isnan(self.merid_ice_flux_div_field[i][j]):
                    self.ice_flux_div_field[i][j] = div_x + div_y
                else:
                    self.ice_flux_div_field[i][j] = np.nan

    def compute_meridional_streamfunction_and_melt_rate(self):
        from constants import R

        for i in range(1, len(self.lats) - 1):
            lat = self.lats[i]
            f = 2 * Omega * np.sin(np.deg2rad(lat))  # Coriolis parameter [s^-1]

            progress_percent = 100 * i / (len(self.lats) - 2)
            logger.info('({} Psi_delta, M-F) lat = {:.2f}/{:.2f} ({:.1f}%)'
                        .format(self.date, lat, lat_max, progress_percent))

            for j in range(len(self.lons)):
                tau_x = self.tau_x_field[i][j]
                tau_y = self.tau_y_field[i][j]
                S = self.salinity_field[i][j]
                dSdx = self.dSdx_field[i][j]
                dSdy = self.dSdy_field[i][j]

                Psi_delta = np.nan

                if not np.isnan(tau_x):
                    Psi_delta = -self.tau_x_field[i][j] / (rho_0 * f)
                    self.psi_delta_field[i][j] = Psi_delta

                if not np.isnan(tau_y) and not np.isnan(Psi_delta) and not np.isnan(S) and not np.isnan(dSdy):
                    self.zonal_melt_rate_field[i][j] = Psi_delta * (1 / S) * dSdy
                    self.merid_melt_rate_field[i][j] = (tau_y / (rho_0 * f)) * (1 / S) * dSdx
                    self.melt_rate_field[i][j] = self.zonal_melt_rate_field[i][j] + self.merid_melt_rate_field[i][j]

                if not np.isnan(Psi_delta):
                    # Convert [m^2/s] -> [Sv] and account for the latitudinal dependence of the circumpolar distance.
                    L = 2 * np.pi * R * np.cos(np.deg2rad(lat))  # Circumpolar distance [m].
                    self.psi_delta_field[i][j] = self.psi_delta_field[i][j] * L / 1e6

    def compute_daily_auxillary_fields(self):
        self.compute_daily_ekman_pumping_field()
        self.process_thermodynamic_fields(levels=[0, 1, 2, 3, 4, 5], process_neutral_density=False)
        self.load_sea_ice_thickness_field()
        self.compute_daily_freshwater_ekman_advection_field()
        self.compute_daily_ice_flux_divergence_field()
        self.compute_meridional_streamfunction_and_melt_rate()

    def compute_mean_fields(self, dates, avg_method):
        import netCDF4
        from utils import log_netCDF_dataset_metadata

        try:
            tau_dataset = netCDF4.Dataset(self.nc_filepath)
            log_netCDF_dataset_metadata(tau_dataset)
            dataset_found = True
        except OSError as e:
            logger.info('Dataset not found, will compute mean fields: {:s}'.format(self.nc_filepath))
            dataset_found = False

        if dataset_found:
            logger.info('Dataset found! Loading fields from: {:s}'.format(self.nc_filepath))
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

        for date in dates:
            tau_filepath = get_netCDF_filepath(field_type='daily', date=date)

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
        # the same object as self.figure_fields['u_ice']!. The figure plots will come out all empty.
        if avg_method == 'full_data_only':
            for var_name in self.var_fields.keys():
                self.var_fields[var_name][:] = field_avg[var_name][:]

        elif avg_method == 'partial_data_ok':
            for var_name in self.var_fields.keys():
                field_avg[var_name] = np.divide(field_avg[var_name], field_days[var_name])
                self.var_fields[var_name][:] = field_avg[var_name][:]

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
        self.tau_air_x_field[:] = np.multiply(1 - self.alpha_field, self.tau_air_x_field)[:]
        self.tau_air_y_field[:] = np.multiply(1 - self.alpha_field, self.tau_air_y_field)[:]
        self.tau_ice_x_field[:] = np.multiply(self.alpha_field, self.tau_ice_x_field)[:]
        self.tau_ice_y_field[:] = np.multiply(self.alpha_field, self.tau_ice_y_field)[:]

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

        # # Extra plot of \Psi_{-\delta}*1/S*dS/dy.
        # ax = plt.subplot(gs[1, 8], projection=ccrs.SouthPolarStereo())
        # ax.add_feature(land_50m)
        # ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        # ax.set_title('M-F = ψ(-δ)/S*dS/dy')
        # im = ax.pcolormesh(self.lons, self.lats, self.melt_rate_field * 24 * 3600 * 365,
        #                    transform=vector_crs, cmap='seismic', vmin=-1, vmax=1)
        # clb = fig.colorbar(im, ax=ax, extend='both')
        # clb.ax.set_title('')
        # ax.contour(self.lons, self.lats, np.ma.array(self.tau_x_field, mask=np.isnan(self.alpha_field)),
        #            levels=[0], colors='green', linewidths=0.5, transform=vector_crs)
        # ax.contour(self.lons, self.lats, np.ma.array(self.alpha_field, mask=np.isnan(self.alpha_field)),
        #            levels=[0.15], colors='black', linewidths=0.5, transform=vector_crs)

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

        # Saving diagnostic figure to disk.
        tau_png_filepath = os.path.join(figure_dir_path, tau_filename + '.png')

        tau_dir = os.path.dirname(tau_png_filepath)
        if not os.path.exists(tau_dir):
            logger.info('Creating directory: {:s}'.format(tau_dir))
            os.makedirs(tau_dir)

        logger.info('Saving diagnostic figure: {:s}...'.format(tau_png_filepath))
        plt.savefig(tau_png_filepath, dpi=600, format='png', transparent=False, bbox_inches='tight')

        # Only going to save in .png as .pdf takes forever to write and is MASSIVE.
        # tau_pdf_filepath = os.path.join(figure_dir_path, tau_filename + '.pdf')
        # logger.info('Saving diagnostic figure: {:s}...'.format(tau_pdf_filepath))
        # plt.savefig(tau_pdf_filepath, dpi=300, format='pdf', transparent=True)

        plt.close()

    def write_fields_to_netcdf(self):
        from constants import var_units, var_positive, var_long_names

        nc_dir = os.path.dirname(self.nc_filepath)
        if not os.path.exists(nc_dir):
            logger.info('Creating directory: {:s}'.format(nc_dir))
            os.makedirs(nc_dir)

        logger.info('Saving fields (field_type={:s}) to netCDF file: {:s}'.format(self.field_type, self.nc_filepath))

        tau_dataset = netCDF4.Dataset(self.nc_filepath, 'w')

        tau_dataset.title = 'Antarctic sea ice zone surface stress and related fields'
        tau_dataset.institution = 'Department of Earth, Atmospheric, and Planetary Science, ' \
                                  'Massachusetts Institute of Technology'
        tau_dataset.history = 'Created ' + time.ctime() + '.'

        tau_dataset.createDimension('time', None)
        tau_dataset.createDimension('lat', len(self.lats))
        tau_dataset.createDimension('lon', len(self.lons))

        time_var = tau_dataset.createVariable('time', np.float64, ('time',))
        time_var.units = 'hours since 0001-01-01 00:00:00'
        time_var.calendar = 'gregorian'

        d = datetime.datetime(self.date.year, self.date.month, self.date.day)
        time_var[:] = netCDF4.date2num(d, units=time_var.units, calendar=time_var.calendar)

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
