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

    jan1_2015 = datetime.date(2015, 7, 1)

    mdt = MeanDynamicTopographyDataReader()
    sea_ice = SeaIceConcentrationDataReader(jan1_2015)
    wind_vectors = OceanSurfaceWindVectorDataReader(jan1_2015)
    seaice_motion = SeaIceMotionDataReader(jan1_2015)

    lat = -65
    lon = -175

    print('u_geo_mean = {}'.format(mdt.u_geo_mean(lat, lon)))
    print('wind = {}'.format(wind_vectors.ocean_surface_wind_vector(lat, lon, jan1_2015)))
    print('SIM = {}'.format(seaice_motion.seaice_motion_vector(lat, lon, jan1_2015)))
