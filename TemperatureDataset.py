from os import path
import numpy as np
import netCDF4

import logging
logger = logging.getLogger(__name__)


class TemperatureDataset(object):
    from constants import data_dir_path

    salinity_data_dir = path.join(data_dir_path, 'WOA13', 'DATAv2', 'temperature', 'netcdf')

    def __init__(self, time_span, avg_period, grid_size, field_type):
        """
        :param time_span: Choose from '5564', '6574', '7584', '8594', '95A4', 'A5B2', 'decav', and 'all'.
        :param avg_period: Choose from annual ('00'), monthly ('01'-'12'), and seasonal ('13' for JFM, '14' for AMJ,
                           '15' for JAS, and '16' for OND).
        :param grid_size: Choose from '04', '01', and '5d'.
        :param field_type: Choose from 'an', 'mn', 'dd', 'ma', 'sd', 'se', 'oa', and 'gp'.
        """
        self.temperature_dataset = None
        self.time_span = time_span
        self.avg_period = avg_period
        self.grid_size = grid_size
        self.field_type = field_type

        self.grid_size_dir = None
        if grid_size == '04':
            self.grid_size_dir = '0.25'
        elif grid_size == '01':
            self.grid_size_dir = '1.00'
        elif grid_size == '5d':
            self.grid_size_dir = '5deg'

        self.lats = None
        self.lons = None
        self.temperature_data = None

        logger.info('TemperatureDataset object initializing for time span {} and averaging period {}...'
                    .format(self.time_span, self.avg_period))

        self.load_temperature_dataset()

    def temperature_dataset_filepath(self):
        filename = 'woa13_' + self.time_span + '_t' + self.avg_period + '_' + self.grid_size + 'v2.nc'

        return path.join(self.salinity_data_dir, self.time_span, self.grid_size_dir, filename)

    def load_temperature_dataset(self):
        from utils import log_netCDF_dataset_metadata

        dataset_filepath = self.temperature_dataset_filepath()
        logger.info('Loading temperature dataset: {}'.format(dataset_filepath))
        self.temperature_dataset = netCDF4.Dataset(dataset_filepath)
        logger.info('Successfully loaded temperature dataset: {}'.format(dataset_filepath))
        log_netCDF_dataset_metadata(self.temperature_dataset)

        self.lats = np.array(self.temperature_dataset.variables['lat'])
        self.lons = np.array(self.temperature_dataset.variables['lon'])

        field_var = 't_' + self.field_type

        self.temperature_data = np.array(self.temperature_dataset.variables[field_var])

    def temperature(self, lat, lon, depth_levels):
        """
        :param lat:
        :param lon:
        :param depth_levels: Single depth level (int) or multiple depth levels (List[int]).
        :return: Temperature in degrees Celcius. If given an integer depth level, the temperature for that depth level
                 will be returned. If given a list of integer depth levels, the average temperature over the given depth
                 levels will be returned.
        """
        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert -180 <= lon <= 180, "Longitude value {} out of bounds!".format(lon)

        idx_lat = np.abs(self.lats - lat).argmin()
        idx_lon = np.abs(self.lons - lon).argmin()

        if isinstance(depth_levels, int):
            temperature_scalar = self.temperature_data[0][depth_levels][idx_lat][idx_lon]

            if temperature_scalar > 1e3:
                return np.nan
            else:
                return temperature_scalar

        elif isinstance(depth_levels, list):
            temperature_avg = 0
            for level in depth_levels:
                temperature_scalar = self.temperature_data[0][level][idx_lat][idx_lon]
                if temperature_scalar > 1e3:
                    return np.nan
                else:
                    temperature_avg = temperature_avg + (temperature_scalar/len(depth_levels))

            return temperature_avg
