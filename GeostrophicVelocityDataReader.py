from os import path
import numpy as np
import netCDF4

import logging
logger = logging.getLogger(__name__)


class GeostrophicVelocityDataReader(object):
    from constants import data_dir_path

    def __init__(self, date=None):
        pass

    def date_to_u_geo_dataset_filepath(self, date):
        pass

    def load_u_geo_dataset(self):
        pass

    def interpolate_u_geo_field(self):
        pass

    def absolute_geostrophic_velocity(self, lat, lon, date, data_source):
        pass