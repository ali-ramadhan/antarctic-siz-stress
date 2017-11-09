# TODO: Use the typing module.
# TODO: Add test case units?
# TODO: Switch from printf style logging to Python3 style formatting.
# TODO: Use propoer docstrings for functions.
# TODO: Estimate tau_error.
# TODO: Calculate wind stress curl field. Gotta interpolate your tau field first.

# Conventions
# Latitude = -90 to +90
# Longitude = -180 to 180

import datetime
import numpy as np

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config
logging.config.fileConfig('logging.ini')
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    from MeanDynamicTopographyDataReader import MeanDynamicTopographyDataReader
    from OceanSurfaceWindVectorDataReader import OceanSurfaceWindVectorDataReader
    from SeaIceConcentrationDataReader import SeaIceConcentrationDataReader
    from SeaIceMotionDataReader import SeaIceMotionDataReader

    from constants import lat_min, lat_max, lat_step, lon_min, lon_max, lon_step
    from constants import rho_air, rho_seawater, C_air, C_seawater
    from constants import f_0, rho_0, D_e

    test_date = datetime.date(2015, 7, 1)

    mdt = MeanDynamicTopographyDataReader()
    seaice_conc = SeaIceConcentrationDataReader(test_date)
    wind_vectors = OceanSurfaceWindVectorDataReader(test_date)
    seaice_motion = SeaIceMotionDataReader(test_date)

    R_45deg = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]])

    n_lat = int((lat_max - lat_min) / lat_step)
    n_lon = int((lon_max - lon_min) / lon_step)

    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)

    logger.info('lat_min = {}, lat_max = {}, lat_step = {}, n_lat = {}'.format(lat_min, lat_max, lat_step, n_lat))
    logger.info('lon_min = {}, lon_max = {}, lon_step = {}, n_lon = {}'.format(lon_min, lon_max, lon_step, n_lon))

    tau_x = np.zeros((len(lats), len(lons)))
    tau_y = np.zeros((len(lats), len(lons)))

    for i in range(len(lats)):
        lat = lats[i]
        logger.info('lat = {}'.format(lat))

        for j in range(len(lons)):
            lon = lons[j]

            u_geo_mean = mdt.u_geo_mean(lat, lon)
            u_wind = wind_vectors.ocean_surface_wind_vector(lat, lon, test_date)
            alpha = seaice_conc.sea_ice_concentration(lat, lon, test_date)
            u_ice = seaice_motion.seaice_motion_vector(lat, lon, test_date)

            if np.isnan(alpha) or np.isnan(u_geo_mean[0]):
                tau_x[i][j] = np.nan
                tau_y[i][j] = np.nan
                continue

            np.set_printoptions(precision=4)
            iter_count = 0
            tau_residual = np.array([1, 1])
            tau_relative_error = 1
            tau = np.array([0, 0])
            u_Ekman = np.array([0.001, 0.001])
            omega = 0.01

            while np.linalg.norm(tau_residual) > 1e-5:
                iter_count = iter_count + 1
                if iter_count > 50:
                    logger.warning('iter_acount exceeded 50 during calculation of tau and u_Ekman.')
                    logger.warning('tau = {}, u_Ekman = {}, tau_residual = {}, tau_rel_error = {:.4f}'
                                   .format(tau, u_Ekman, tau_residual, tau_relative_error))
                    break

                if np.linalg.norm(tau) > 10:
                    logger.warning('Large tau = {}, u_geo_mean = {}, u_wind = {}, alpha = {:.4f}, u_ice = {}'
                                   .format(tau, u_geo_mean, u_wind, alpha, u_ice))
                    iter_count = 0
                    tau_error = 1
                    tau_relative_error = 1
                    tau = np.array([0, 0])

                tau_air = rho_air * C_air * np.linalg.norm(u_wind) * u_wind

                u_Ekman = (np.sqrt(2) / (f_0 * rho_0 * D_e)) * np.matmul(R_45deg, tau)
                u_rel = u_ice - (u_geo_mean - u_Ekman)
                tau_ice = rho_0 * C_seawater * np.linalg.norm(u_rel) * u_rel
                tau = alpha * tau_ice + (1 - alpha) * tau_air

                tau_residual = tau - (alpha * tau_ice + (1 - alpha) * tau_air)

                tau_relative_error = np.linalg.norm(tau_residual)/np.linalg.norm(tau)

                tau = tau + omega*tau_residual

            tau_x[i][j] = tau[0]
            tau_y[i][j] = tau[1]

            # # Complex version using the modified Richardson iterative method
            # u_geo_mean = u_geo_mean[0] + 1j*u_geo_mean[1]
            # u_wind = u_wind[0] + 1j*u_wind[1]
            # u_ice = u_ice[0] + 1j*u_ice[1]
            #
            # tau_air = rho_air * C_air * np.linalg.norm(u_wind) * u_wind
            #
            # iter_count = 0
            # tau_error = 1
            # tau_relative_error = 1
            # tau = 0 + 1j*0
            # u_Ekman = 0 + 1j*0
            # omega = 0.01  # Richardson relaxation parameter
            #
            # while tau_relative_error > 1e-5:
            #     iter_count = iter_count + 1
            #     if iter_count > 50:
            #         logger.warning('iter_acount exceeded 50 during calculation of tau and u_Ekman.')
            #         logger.warning('tau = {:.4f}, u_Ekman = {:.4f}, tau_rel_error = {:.4f}'
            #                        .format(tau, u_Ekman, tau_relative_error))
            #         break
            #
            #     if np.linalg.norm(tau) > 10:
            #         logger.warning('Large tau:')
            #         logger.warning('tau = {:.4f}, u_geo_mean = {:.4f}, u_wind = {:.4f}, alpha = {:.4f}, u_ice = {:.4f}'
            #                        .format(tau, u_geo_mean, u_wind, alpha, u_ice))
            #
            #     u_Ekman = (np.sqrt(2) / (f_0 * rho_0 * D_e)) * np.exp(-1j * np.pi / 4) * tau
            #     u_rel = u_ice - (u_geo_mean - u_Ekman)
            #     tau_ice = rho_0 * C_seawater * np.linalg.norm(u_rel) * u_rel
            #
            #     tau_error = tau - (alpha * tau_ice + (1 - alpha) * tau_air)
            #     tau_residual = tau - tau_error
            #     tau = tau + omega*tau_residual
            #
            #     tau_relative_error = np.linalg.norm(tau_error) / np.linalg.norm(tau)
            #     # omega = 0.75*omega
            #
            # if np.isnan(np.real(tau)) or np.isnan(np.imag(tau)):
            #     logger.warning('tau = {:.4f}, u_geo_mean = {:.4f}, u_wind = {:.4f}, alpha = {:.4f}, u_ice = {:.4f}'
            #                   .format(tau, u_geo_mean, u_wind, alpha, u_ice))
            #
            # tau_x[i][j] = np.real(tau)
            # tau_y[i][j] = np.imag(tau)

    import matplotlib.pyplot as plt
    plt.contourf(lons, lats, tau_x, 25)
    plt.colorbar()
    plt.show()

    import netCDF4 as nc
    tau_dataset = nc.Dataset('tau.nc', 'w')

    tau_dataset.title = 'Antarctic sea ice zone surface stress'
    tau_dataset.institution = 'Department of Earth, Atmospheric, and Planetary Science, ' \
                              'Massachusetts Institute of Technology'
    # tau_dataset.history = 'Created ' + datetime.time.ctime(datetime.time.time()) + '.'

    tau_dataset.createDimension('time', None)
    tau_dataset.createDimension('lat', len(lats))
    tau_dataset.createDimension('lon', len(lons))

    time_var = tau_dataset.createVariable('time', np.float64, ('time',))
    time_var.units = 'hours since 0001-01-01 00:00:00'
    time_var.calendar = 'gregorian'

    lat_var = tau_dataset.createVariable('lat', np.float32, ('lat',))
    lat_var.units = 'degrees south'
    lat_var[:] = lats

    lon_var = tau_dataset.createVariable('lon', np.float32, ('lon',))
    lat_var.units = 'degrees west/east'
    lon_var[:] = lons

    tau_var = tau_dataset.createVariable('tau_x', float, ('lat', 'lon'), zlib=True)
    tau_var.units = 'N/m^2'
    tau_var.long_name = 'Zonal surface stress'
    tau_var.positive = 'up'
    tau_var[:] = tau_x

    tau_var = tau_dataset.createVariable('tau_y', float, ('lat', 'lon'), zlib=True)
    tau_var.units = 'N/m^2'
    tau_var.long_name = 'Meridional surface stress'
    tau_var.positive = 'up'
    tau_var[:] = tau_y

    tau_dataset.close()
