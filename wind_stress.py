# TODO: Use the typing module.
# TODO: Add test case units?
# TODO: Split into multiple files.
# TODO: Switch from printf style logging to Python3 style formatting.
# TODO: Use propoer docstrings for functions.
# TODO: Choose a lat/lon convention for the frontend and convert as required for each product.

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

    lat = -65
    lon = -175

    n_lat = int((lat_max - lat_min) / lat_step)
    n_lon = int((lon_max - lon_min) / lon_step)

    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)

    tau = np.zeros(len(lats), len(lons))

    u_geo_mean = mdt.u_geo_mean(lat, lon)
    u_wind = wind_vectors.ocean_surface_wind_vector(lat, lon, test_date)
    alpha = seaice_conc.sea_ice_concentration(lat, lon, test_date)
    u_ice = seaice_motion.seaice_motion_vector(lat, lon, test_date)

    print('u_geo_mean = {}'.format(u_geo_mean))
    print('wind = {}'.format(u_wind))
    print('SIC = {}'.format(alpha))
    print('SIM = {}'.format(u_ice))

    R_45deg = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]])

    tau_error = 1
    tau = np.array([0, 0])
    u_Ekman = np.array([0.001, 0.001])
    while tau_error > 1e-10:
        tau_air = rho_air * C_air * np.linalg.norm(u_wind) * u_wind

        u_rel = u_ice - (u_geo_mean - u_Ekman)
        tau_ice = rho_0 * C_seawater * np.linalg.norm(u_rel) * u_rel

        tau_error = np.linalg.norm(tau - (alpha * tau_ice + (1 - alpha) * tau_air))
        tau = alpha * tau_ice + (1 - alpha) * tau_air
        u_Ekman = (np.sqrt(2) / (f_0 * rho_0 * D_e)) * np.matmul(R_45deg, tau)
