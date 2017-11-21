# TODO: Use the typing module.
# TODO: Add test case units?
# TODO: Use propoer docstrings for functions.
# TODO: Estimate tau_error? Can you? NCEP Reanalysis doesn't really provide a "measurement error".
# TODO: Calculate wind stress curl field. Gotta interpolate your tau field first.
# TODO: Output more statistics during the analysis?
# TODO: Plot everything but draw the ice line where alpha drops below 0.15.
# TODO: Plot the zero stress line. We expect a unique position for it right?
# TODO: Stop pickling and start just saving as netCDF? Pickle might not be best for long-term storage. NetCDF is.
# TODO: Is it possible to make a general purpose interpolate_dataset function for all my datasets?

# Conventions
# Latitude = -90 to +90
# Longitude = -180 to 180

import datetime
import numpy as np

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config
logging.config.fileConfig('logging.ini')
logger = logging.getLogger(__name__)

np.set_printoptions(precision=4)

if __name__ == '__main__':
    from MeanDynamicTopographyDataReader import MeanDynamicTopographyDataReader
    from OceanSurfaceWindVectorDataReader import OceanSurfaceWindVectorDataReader
    from SeaIceConcentrationDataReader import SeaIceConcentrationDataReader
    from SeaIceMotionDataReader import SeaIceMotionDataReader

    from constants import lat_min, lat_max, lat_step, n_lat, lon_min, lon_max, lon_step, n_lon
    from constants import rho_air, rho_seawater, C_air, C_seawater
    from constants import Omega, rho_0, D_e

    test_date = datetime.date(2015, 7, 1)

    mdt = MeanDynamicTopographyDataReader()
    seaice_conc = SeaIceConcentrationDataReader(test_date)
    seaice_motion = SeaIceMotionDataReader(test_date)
    wind_vectors = OceanSurfaceWindVectorDataReader(test_date)
    exit()

    R_45deg = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]])

    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)

    logger.info('Calculating surface stress field (tau_x, tau_y) for:')
    logger.info('lat_min = {}, lat_max = {}, lat_step = {}, n_lat = {}'.format(lat_min, lat_max, lat_step, n_lat))
    logger.info('lon_min = {}, lon_max = {}, lon_step = {}, n_lon = {}'.format(lon_min, lon_max, lon_step, n_lon))

    tau_x = np.zeros((len(lats), len(lons)))
    tau_y = np.zeros((len(lats), len(lons)))

    for i in range(len(lats)):
        lat = lats[i]
        f = 2*Omega * np.sin(np.deg2rad(lat))

        progress_percent = 100 * (lat - lat_min)/(lat_max - lat_min)
        logger.info('lat = {:.2f}/{:.2f} ({:.1f}%)'.format(lat, lat_max, progress_percent))

        for j in range(len(lons)):
            lon = lons[j]

            u_geo_mean = mdt.u_geo_mean_interp(lat, lon)
            u_wind = wind_vectors.ocean_surface_wind_vector(lat, lon, test_date)
            alpha = seaice_conc.sea_ice_concentration(lat, lon, test_date)
            u_ice = seaice_motion.seaice_motion_vector(lat, lon, test_date)

            if np.isnan(alpha) or np.isnan(u_geo_mean[0]):
                tau_x[i][j] = np.nan
                tau_y[i][j] = np.nan
                continue

            iter_count = 0
            tau_residual = np.array([1, 1])
            tau_relative_error = 1
            tau = np.array([0, 0])
            u_Ekman = np.array([0.001, 0.001])
            omega = 0.01  # Richardson relaxation parameter

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
                    break

                tau_air = rho_air * C_air * np.linalg.norm(u_wind) * u_wind

                u_Ekman = (np.sqrt(2) / (f * rho_0 * D_e)) * np.matmul(R_45deg, tau)
                u_rel = u_ice - (u_geo_mean - u_Ekman)
                tau_ice = rho_0 * C_seawater * np.linalg.norm(u_rel) * u_rel
                tau = alpha * tau_ice + (1 - alpha) * tau_air

                tau_residual = tau - (alpha * tau_ice + (1 - alpha) * tau_air)
                tau_relative_error = np.linalg.norm(tau_residual)/np.linalg.norm(tau)

                tau = tau + omega*tau_residual

                if np.isnan(tau[0]) or np.isnan(tau[1]):
                    logger.warning('NaN tau = {}, u_geo_mean = {}, u_wind = {}, alpha = {:.4f}, u_ice = {}'
                                   .format(tau, u_geo_mean, u_wind, alpha, u_ice))

            tau_x[i][j] = tau[0]
            tau_y[i][j] = tau[1]

    import matplotlib.pyplot as plt
    plt.pcolormesh(lons, lats, tau_x)
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
