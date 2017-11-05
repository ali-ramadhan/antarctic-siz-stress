# TODO: Use the typing module.
# TODO: Add test case units?
# TODO: Split into multiple files.
# TODO: Switch from printf style logging to Python3 style formatting.
# TODO: Use propoer docstrings for functions.
# TODO: Choose a lat/lon convention for the frontend and convert as required for each product.

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

    from utils import distance

    dist = distance(24, 25, 26, 27)
    logger.info("That distance is %f m or %f km.", dist, dist/1000)

    MDT = MeanDynamicTopographyDataReader()
    print(MDT.get_MDT(-60, 135+180))
    print(MDT.u_geo_mean(-60, 135+180))

    sea_ice = SeaIceConcentrationDataReader()
    print(sea_ice.sea_ice_concentration(-60.0, 133.0, datetime.date(2015, 7, 31)))
    print(sea_ice.sea_ice_concentration(-71.4, 24.5, datetime.date(2015, 7, 31)))
    print(sea_ice.sea_ice_concentration(-70, 180, datetime.date(2015, 7, 31)))

    wind_vectors = OceanSurfaceWindVectorDataReader()
    print(wind_vectors.ocean_surface_wind_vector(-60, 20, datetime.date(2015, 10, 10)))

    seaice_drift = SeaIceMotionDataReader()
    print(seaice_drift.seaice_drift_vector(-60, 20, datetime.date(2015, 1, 1)))
