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
    # Such an object should mainly compute daily (averaged) wind stress and wind stress curl fields and write them out
    # to netCDF files. Computing monthly means makes sense here. But plotting should go elsewhere.

    R_45deg = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]])

    def __init__(self, date):
        self.date = date

        self.u_geo_data = GeostrophicVelocityDataReader(self.date)
        self.sea_ice_conc_data = SeaIceConcentrationDataReader(self.date)
        self.sea_ice_motion_data = SeaIceMotionDataReader(self.date)
        self.u_wind_data = OceanSurfaceWindVectorDataReader(self.date)

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

    def compute_daily_wind_stress_curl_field(self):
        for i in range(1, len(self.lats) - 1):
            for j in range(1, len(self.lons) - 1):
                if not np.isnan(self.tau_x_field[i][j - 1]) and not np.isnan(self.tau_x_field[i][j + 1]) \
                        and not np.isnan(self.tau_y_field[i - 1][j]) and not np.isnan(self.tau_y_field[i + 1][j]):
                    dx = distance(self.lats[i-1], self.lons[j], self.lats[i+1], self.lons[j])
                    dy = distance(self.lats[i], self.lons[j-1], self.lats[i], self.lons[j+1])

                    dtauxdy = (self.tau_x_field[i][j+1] - self.tau_x_field[i][j-1]) / (2*dy)
                    dtauydx = (self.tau_y_field[i+1][j] - self.tau_x_field[i-1][j]) / (2*dx)

                    self.wind_stress_curl_field[i][j] = dtauydx - dtauxdy
                else:
                    self.wind_stress_curl_field[i][j] = np.nan

    def compute_monthly_mean_field(self):
        pass

    def plot_diagnostic_fields(self):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        import cartopy
        import cartopy.crs as ccrs

        gs_coords = {
            'tau_x': (0, 0),
            'tau_y': (0, 1),
            'curl_tau': (1, 0),
            'alpha': (1, 1)
        }

        land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face',
                                                       facecolor='dimgray')

        fig = plt.figure()
        gs = GridSpec(2, 2)

        ax1.add_feature(land_50m)
        # ax1.coastlines(resolution='50m')
        ax1.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

        vector_crs = ccrs.PlateCarree()
        im1 = ax1.pcolormesh(self.lons, self.lats, self.tau_x_field, transform=vector_crs, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        fig.colorbar(im1, ax=ax1)

        nlats = len(self.lats)
        nlons = len(self.lons)
        lats = np.repeat(self.lats, nlons)
        lons = np.tile(self.lons, nlats)
        tau_x_field = np.reshape(self.tau_x_field, (nlats*nlons,))
        tau_y_field = np.reshape(self.tau_y_field, (nlats*nlons,))
        ax1.quiver(lons[::10], lats[::10], tau_x_field[::10], tau_y_field[::10], transform=vector_crs, units='width', width=0.001, scale=10)

        ax2 = plt.subplot(gs[1, 0], projection=ccrs.SouthPolarStereo())
        ax2.add_feature(land_50m, edgecolor='black')
        ax2.coastlines(resolution='50m')
        ax2.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

        vector_crs = ccrs.PlateCarree()
        im2 = ax2.pcolormesh(self.lons, self.lats, self.tau_y_field, transform=vector_crs, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        fig.colorbar(im2, ax=ax2)

        ax3 = plt.subplot(gs[0, 1], projection=ccrs.SouthPolarStereo())
        ax3.add_feature(land_50m, edgecolor='black')
        ax3.coastlines(resolution='50m')
        ax3.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

        vector_crs = ccrs.PlateCarree()
        im3 = ax3.pcolormesh(self.lons, self.lats, self.wind_stress_curl_field, transform=vector_crs, cmap='RdBu_r')
        fig.colorbar(im3, ax=ax3)

        ax4 = plt.subplot(gs[1, 1], projection=ccrs.SouthPolarStereo())
        ax4.add_feature(land_50m, edgecolor='black')
        ax4.coastlines(resolution='50m')
        ax4.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

        vector_crs = ccrs.PlateCarree()
        im4 = ax4.pcolormesh(self.lons, self.lats, self.alpha_field, transform=vector_crs, cmap='plasma')
        fig.colorbar(im4, ax=ax4)

        plt.show()

    def write_fields_to_netcdf(self):
        tau_filepath = os.path.join(output_dir_path, str(self.date.year), 'tau.nc')
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
