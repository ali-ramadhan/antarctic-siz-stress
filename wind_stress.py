# TODO: Use the typing module.
# TODO: Add test case units?
# TODO: Use propoer docstrings for functions.
# TODO: Estimate tau_error? Can you? NCEP Reanalysis doesn't really provide a "measurement error".
# TODO: Calculate wind stress curl field. Gotta interpolate your tau field first.
# TODO: Output more statistics during the analysis?
# TODO: Plot everything but draw the ice line where alpha drops below 0.15.
# TODO: Plot the zero stress line. We expect a unique position for it right?

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
    from GeostrophicVelocityDataReader import GeostrophicVelocityDataReader
    from OceanSurfaceWindVectorDataReader import OceanSurfaceWindVectorDataReader
    from SeaIceConcentrationDataReader import SeaIceConcentrationDataReader
    from SeaIceMotionDataReader import SeaIceMotionDataReader

    from constants import lat_min, lat_max, lat_step, n_lat, lon_min, lon_max, lon_step, n_lon
    from constants import rho_air, rho_seawater, C_air, C_seawater
    from constants import Omega, rho_0, D_e

    test_date = datetime.date(2015, 7, 1)

    u_geo = GeostrophicVelocityDataReader(test_date)
    seaice_conc = SeaIceConcentrationDataReader(test_date)
    seaice_motion = SeaIceMotionDataReader(test_date)
    wind_vectors = OceanSurfaceWindVectorDataReader(test_date)

    R_45deg = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]])

    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)
    lons = lons[:-1]  # Remove the +180 longitude as it coincides with the -180 longitude.

    logger.info('Calculating surface stress field (tau_x, tau_y) for:')
    logger.info('lat_min = {}, lat_max = {}, lat_step = {}, n_lat = {}'.format(lat_min, lat_max, lat_step, n_lat))
    logger.info('lon_min = {}, lon_max = {}, lon_step = {}, n_lon = {}'.format(lon_min, lon_max, lon_step, n_lon))

    # All the fields to save to the netCDF file.
    tau_air_x_field = np.zeros((len(lats), len(lons)))
    tau_air_y_field = np.zeros((len(lats), len(lons)))
    tau_ice_x_field = np.zeros((len(lats), len(lons)))
    tau_ice_y_field = np.zeros((len(lats), len(lons)))
    tau_SIZ_x_field = np.zeros((len(lats), len(lons)))
    tau_SIZ_y_field = np.zeros((len(lats), len(lons)))
    tau_x_field = np.zeros((len(lats), len(lons)))
    tau_y_field = np.zeros((len(lats), len(lons)))

    u_Ekman_field = np.zeros((len(lats), len(lons)))
    v_Ekman_field = np.zeros((len(lats), len(lons)))
    u_Ekman_SIZ_field = np.zeros((len(lats), len(lons)))
    v_Ekman_SIZ_field = np.zeros((len(lats), len(lons)))

    u_geo_mean_field = np.zeros((len(lats), len(lons)))
    v_geo_mean_field = np.zeros((len(lats), len(lons)))
    u_wind_field = np.zeros((len(lats), len(lons)))
    v_wind_field = np.zeros((len(lats), len(lons)))
    alpha_field = np.zeros((len(lats), len(lons)))
    u_ice_field = np.zeros((len(lats), len(lons)))
    v_ice_field = np.zeros((len(lats), len(lons)))

    for i in range(len(lats)):
        lat = lats[i]
        f = 2*Omega * np.sin(np.deg2rad(lat))  # Coriolis parameter [s^-1]

        progress_percent = 100 * i/(len(lats)-1)
        logger.info('lat = {:.2f}/{:.2f} ({:.1f}%)'.format(lat, lat_max, progress_percent))

        for j in range(len(lons)):
            lon = lons[j]

            u_geo_vec = u_geo.absolute_geostrophic_velocity(lat, lon, test_date, 'interp')
            u_wind_vec = wind_vectors.ocean_surface_wind_vector(lat, lon, test_date, 'interp')
            alpha = seaice_conc.sea_ice_concentration(lat, lon, test_date, 'interp')
            u_ice_vec = seaice_motion.seaice_motion_vector(lat, lon, test_date, 'interp')

            u_geo_mean_field[i][j] = u_geo_vec[0]
            v_geo_mean_field[i][j] = u_geo_vec[1]
            u_wind_field[i][j] = u_wind_vec[0]
            v_wind_field[i][j] = u_wind_vec[1]
            alpha_field[i][j] = alpha
            u_ice_field[i][j] = u_ice_vec[0]
            v_ice_field[i][j] = u_ice_vec[1]

            # If there's no sea ice at a point and we have data at that point (i.e. the point is still in the ocean)
            # then tau is just tau_air and easy to calculate.
            if ((alpha == 0 or np.isnan(alpha)) and np.isnan(u_ice_vec[0])) \
                    and not np.isnan(u_geo_vec[0]) and not np.isnan(u_wind_vec[0]):

                tau_air_vec = rho_air * C_air * np.linalg.norm(u_wind_vec) * u_wind_vec

                tau_air_x_field[i][j] = tau_air_vec[0]
                tau_air_y_field[i][j] = tau_air_vec[1]
                tau_ice_x_field[i][j] = 0
                tau_ice_y_field[i][j] = 0

                tau_x_field[i][j] = tau_air_vec[0]
                tau_y_field[i][j] = tau_air_vec[1]
                tau_SIZ_x_field[i][j] = np.nan
                tau_SIZ_y_field[i][j] = np.nan

                # Not sure why I have to recalculate u_Ekman_vec otherwise it's just zero.
                u_Ekman_vec = (np.sqrt(2) / (f * rho_0 * D_e)) * np.matmul(R_45deg, tau_vec)
                u_Ekman_field[i][j] = u_Ekman_vec[0]
                v_Ekman_field[i][j] = u_Ekman_vec[1]
                u_Ekman_SIZ_field[i][j] = np.nan
                v_Ekman_SIZ_field[i][j] = np.nan
                continue

            # If we have data missing, then we're probably on land or somewhere where we cannot calculate tau.
            if np.isnan(alpha) or np.isnan(u_geo_vec[0]) or np.isnan(u_wind_vec[0]) or np.isnan(u_ice_vec[0]):
                tau_air_x_field[i][j] = np.nan
                tau_air_y_field[i][j] = np.nan
                tau_ice_x_field[i][j] = np.nan
                tau_ice_y_field[i][j] = np.nan
                tau_x_field[i][j] = np.nan
                tau_y_field[i][j] = np.nan
                tau_SIZ_x_field[i][j] = np.nan
                tau_SIZ_y_field[i][j] = np.nan
                u_Ekman_field[i][j] = np.nan
                v_Ekman_field[i][j] = np.nan
                continue

            # Use the Modified Richardson iteration to calculate tau and u_Ekman. Here we set the variables to arbitrary
            # initial guesses.
            iter_count = 0
            tau_vec_residual = np.array([1, 1])
            tau_relative_error = 1
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

                u_Ekman_vec = (np.sqrt(2) / (f * rho_0 * D_e)) * np.matmul(R_45deg, tau_vec)
                u_rel_vec = u_ice_vec - (u_geo_vec - u_Ekman_vec)
                tau_ice_vec = rho_0 * C_seawater * np.linalg.norm(u_rel_vec) * u_rel_vec
                tau_vec = alpha * tau_ice_vec + (1 - alpha) * tau_air_vec

                tau_vec_residual = tau_vec - (alpha * tau_ice_vec + (1 - alpha) * tau_air_vec)
                tau_relative_error = np.linalg.norm(tau_vec_residual)/np.linalg.norm(tau_vec)

                tau_vec = tau_vec + omega*tau_vec_residual

                if np.isnan(tau_vec[0]) or np.isnan(tau_vec[1]):
                    logger.warning('NaN tau = {}, u_geo_mean = {}, u_wind = {}, alpha = {:.4f}, u_ice = {}'
                                   .format(tau_vec, u_geo_vec, u_wind_vec, alpha, u_ice_vec))

            tau_air_x_field[i][j] = tau_air_vec[0]
            tau_air_y_field[i][j] = tau_air_vec[1]
            tau_ice_x_field[i][j] = tau_ice_vec[0]
            tau_ice_y_field[i][j] = tau_ice_vec[1]
            tau_x_field[i][j] = tau_vec[0]
            tau_y_field[i][j] = tau_vec[1]
            tau_SIZ_x_field[i][j] = tau_vec[0]
            tau_SIZ_y_field[i][j] = tau_vec[1]

            # Not sure why I have to recalculate u_Ekman_vec otherwise it's just zero.
            u_Ekman_vec = (np.sqrt(2) / (f * rho_0 * D_e)) * np.matmul(R_45deg, tau_vec)
            u_Ekman_field[i][j] = u_Ekman_vec[0]
            v_Ekman_field[i][j] = u_Ekman_vec[1]
            u_Ekman_SIZ_field[i][j] = u_Ekman_vec[0]
            v_Ekman_SIZ_field[i][j] = u_Ekman_vec[1]

    # Calculate wind stress curl field
    # from utils import interpolate_scalar_field
    #
    # repeat0tile1 = True
    # convert_lon_range = False
    # mask_value_cond = lambda x : np.isnan(x)
    # tau_x_interp, latgrid_interp, longrid_interp = interpolate_scalar_field(
    #     tau_x_field, lats, lons, None, mask_value_cond, 'latlon', 'linear', repeat0tile1, convert_lon_range)
    # tau_y_interp, latgrid_interp, longrid_interp = interpolate_scalar_field(
    #     tau_y_field, lats, lons, None, mask_value_cond, 'latlon', 'linear', repeat0tile1, convert_lon_range)

    # import matplotlib.pyplot as plt
    # import cartopy
    # import cartopy.crs as ccrs
    # ax = plt.axes(projection=ccrs.SouthPolarStereo())
    #
    # # ax.add_feature(cartopy.feature.OCEAN, zorder=0)
    # land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
    #                                         edgecolor='face',
    #                                         facecolor=cartopy.feature.COLORS['land'])
    # ax.add_feature(land_50m, edgecolor='black')
    # # ax.stock_img()
    # ax.coastlines(resolution='50m')
    #
    # ax.set_global()
    # ax.gridlines()
    #
    # vector_crs = ccrs.PlateCarree()
    # ax.pcolormesh(lons, lats, tau_x_field, transform=vector_crs)
    #
    # nlats = len(lats)
    # nlons = len(lons)
    # lats = np.repeat(lats, nlons)
    # lons = np.tile(lons, nlats)
    # tau_x_field = np.reshape(tau_x_field, (nlats*nlons,))
    # tau_y_field = np.reshape(tau_y_field, (nlats*nlons,))
    # ax.quiver(lons[::10], lats[::10], tau_x_field[::10], tau_y_field[::10], transform=vector_crs, units='width', width=0.001, scale=10)
    # ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    # ax.gridlines()
    #
    # # import matplotlib.path as mpath
    # # # Compute a circle in axes coordinates, which we can use as a boundary
    # # # for the map. We can pan/zoom as much as we like - the boundary will be
    # # # permanently circular.
    # # theta = np.linspace(0, 2*np.pi, 1000)
    # # center, radius = [0.5, 0.5], 0.5
    # # verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    # # circle = mpath.Path(verts * radius + center)
    # # ax.set_boundary(circle, transform=ax.transAxes)
    #
    # plt.show()
    # exit()

    from utils import distance
    wind_stress_curl_field = np.zeros((len(lats), len(lons)))
    for i in range(1, len(lats)-1):
        for j in range(1, len(lons)-1):
            if not np.isnan(tau_x_field[i][j-1]) and not np.isnan(tau_x_field[i][j+1]) \
                    and not np.isnan(tau_y_field[i-1][j]) and not np.isnan(tau_y_field[i+1][j]):
                dtauxdy = (tau_x_field[i][j+1] - tau_x_field[i][j-1]) / (2*distance(lats[i], lons[j-1], lats[i], lons[j+1]))
                dtauydx = (tau_y_field[i+1][j] - tau_x_field[i-1][j]) / (2*distance(lats[i-1], lons[j], lats[i+1], lons[j]))
                wind_stress_curl_field[i][j] = dtauydx - dtauxdy
            else:
                wind_stress_curl_field[i][j] = np.nan

    # import matplotlib.pyplot as plt
    # plt.pcolormesh(lons, lats, tau_x_field)
    # plt.colorbar()
    # plt.show()

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

    tau_air_x_var = tau_dataset.createVariable('tau_air_x', float, ('lat', 'lon'), zlib=True)
    tau_air_x_var.units = 'N/m^2'
    tau_air_x_var.positive = 'up'
    tau_air_x_var.long_name = 'Zonal air surface stress'
    tau_air_x_var[:] = tau_air_x_field

    tau_air_y_var = tau_dataset.createVariable('tau_air_y', float, ('lat', 'lon'), zlib=True)
    tau_air_y_var.units = 'N/m^2'
    tau_air_y_var.positive = 'up'
    tau_air_y_var.long_name = 'Meridional air surface stress'
    tau_air_y_var[:] = tau_air_y_field

    tau_ice_x_var = tau_dataset.createVariable('tau_ice_x', float, ('lat', 'lon'), zlib=True)
    tau_ice_x_var.units = 'N/m^2'
    tau_ice_x_var.positive = 'up'
    tau_ice_x_var.long_name = 'Zonal ice surface stress'
    tau_ice_x_var[:] = tau_ice_x_field

    tau_ice_y_var = tau_dataset.createVariable('tau_ice_y', float, ('lat', 'lon'), zlib=True)
    tau_ice_y_var.units = 'N/m^2'
    tau_ice_y_var.positive = 'up'
    tau_ice_y_var.long_name = 'Meridional ice surface stress'
    tau_ice_y_var[:] = tau_ice_y_field

    tau_x_var = tau_dataset.createVariable('tau_x', float, ('lat', 'lon'), zlib=True)
    tau_x_var.units = 'N/m^2'
    tau_x_var.positive = 'up'
    tau_x_var.long_name = 'Zonal surface stress'
    tau_x_var[:] = tau_x_field

    tau_y_var = tau_dataset.createVariable('tau_y', float, ('lat', 'lon'), zlib=True)
    tau_y_var.units = 'N/m^2'
    tau_y_var.positive = 'up'
    tau_y_var.long_name = 'Meridional surface stress'
    tau_y_var[:] = tau_y_field

    tau_SIZ_x_var = tau_dataset.createVariable('tau_SIZ_x', float, ('lat', 'lon'), zlib=True)
    tau_SIZ_x_var.units = 'N/m^2'
    tau_SIZ_x_var.positive = 'up'
    tau_SIZ_x_var.long_name = 'Zonal surface stress in the SIZ'
    tau_SIZ_x_var[:] = tau_SIZ_x_field

    tau_SIZ_y_var = tau_dataset.createVariable('tau_SIZ_y', float, ('lat', 'lon'), zlib=True)
    tau_SIZ_y_var.units = 'N/m^2'
    tau_SIZ_y_var.positive = 'up'
    tau_SIZ_y_var.long_name = 'Meridional surface stress in the SIZ'
    tau_SIZ_y_var[:] = tau_SIZ_y_field

    curl_tau_var = tau_dataset.createVariable('wind_stress_curl', float, ('lat', 'lon'), zlib=True)
    curl_tau_var.units = 'N/m^3'
    curl_tau_var.positive = 'up'
    curl_tau_var.long_name = 'Wind stress curl'
    curl_tau_var[:] = wind_stress_curl_field

    u_Ekman_var = tau_dataset.createVariable('Ekman_u', float, ('lat', 'lon'), zlib=True)
    u_Ekman_var.units = 'm/s'
    u_Ekman_var.positive = 'up'
    u_Ekman_var.long_name = 'Zonal Ekman transport velocity'
    u_Ekman_var[:] = u_Ekman_field

    v_Ekman_var = tau_dataset.createVariable('Ekman_v', float, ('lat', 'lon'), zlib=True)
    v_Ekman_var.units = 'm/s'
    v_Ekman_var.positive = 'up'
    v_Ekman_var.long_name = 'Meridional Ekman transport velocity'
    v_Ekman_var[:] = v_Ekman_field

    u_Ekman_SIZ_var = tau_dataset.createVariable('Ekman_SIZ_u', float, ('lat', 'lon'), zlib=True)
    u_Ekman_SIZ_var.units = 'm/s'
    u_Ekman_SIZ_var.positive = 'up'
    u_Ekman_SIZ_var.long_name = 'Zonal Ekman transport velocity in the SIZ'
    u_Ekman_SIZ_var[:] = u_Ekman_SIZ_field

    v_Ekman_SIZ_var = tau_dataset.createVariable('Ekman_SIZ_v', float, ('lat', 'lon'), zlib=True)
    v_Ekman_SIZ_var.units = 'm/s'
    v_Ekman_SIZ_var.positive = 'up'
    v_Ekman_SIZ_var.long_name = 'Meridional Ekman transport velocity in the SIZ'
    v_Ekman_SIZ_var[:] = v_Ekman_SIZ_field

    u_geo_mean_var = tau_dataset.createVariable('geo_mean_u', float, ('lat', 'lon'), zlib=True)
    u_geo_mean_var.units = 'm/s'
    u_geo_mean_var.positive = 'up'
    u_geo_mean_var.long_name = 'Mean zonal geostrophic velocity'
    u_geo_mean_var[:] = u_geo_mean_field

    v_geo_mean_var = tau_dataset.createVariable('geo_mean_v', float, ('lat', 'lon'), zlib=True)
    v_geo_mean_var.units = 'm/s'
    v_geo_mean_var.positive = 'up'
    v_geo_mean_var.long_name = 'Mean meridional geostrophic velocity'
    v_geo_mean_var[:] = v_geo_mean_field

    u_wind_var = tau_dataset.createVariable('wind_u', float, ('lat', 'lon'), zlib=True)
    u_wind_var.units = 'm/s'
    u_wind_var.positive = 'up'
    u_wind_var.long_name = 'Zonal wind velocity'
    u_wind_var[:] = u_wind_field

    v_wind_var = tau_dataset.createVariable('wind_v', float, ('lat', 'lon'), zlib=True)
    v_wind_var.units = 'm/s'
    v_wind_var.positive = 'up'
    v_wind_var.long_name = 'Meridional wind velocity'
    v_wind_var[:] = v_wind_field

    alpha_var = tau_dataset.createVariable('alpha', float, ('lat', 'lon'), zlib=True)
    alpha_var.units = 'fractional'
    alpha_var.long_name = 'Sea ice concentration'
    alpha_var[:] = alpha_field

    u_ice_var = tau_dataset.createVariable('ice_u', float, ('lat', 'lon'), zlib=True)
    u_ice_var.units = 'm/s'
    u_ice_var.positive = 'up'
    u_ice_var.long_name = 'Zonal sea ice motion'
    u_ice_var[:] = u_ice_field

    v_ice_var = tau_dataset.createVariable('ice_v', float, ('lat', 'lon'), zlib=True)
    v_ice_var.units = 'm/s'
    v_ice_var.positive = 'up'
    v_ice_var.long_name = 'Meridional sea ice motion'
    v_ice_var[:] = v_ice_field

    tau_dataset.close()
