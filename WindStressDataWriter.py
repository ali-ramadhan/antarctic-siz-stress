from os import path
import numpy as np
import netCDF4

import logging
logger = logging.getLogger(__name__)

class WindStressDataWriter(object):
    # Such an object should mainly compute daily (averaged) wind stress and wind stress curl fields and write them out
    # to netCDF files. Computing monthly means makes sense here. But plotting should go elsewhere.
    from constants import data_dir_path
    from constants import data_dir_path

    def __init__(self, date):
        pass

    def surface_stress(self):
        pass

    def plot_field(self):
        pass

    def write_fields_to_netcdf(self):
        pass

    def compute_daily_fields(self):
        pass

    def compute_monthly_mean_field(self):
        pass
