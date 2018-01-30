from os import path
import numpy as np
import netCDF4

import logging
logger = logging.getLogger(__name__)


class SalinityDataset(object):
    from constants import data_dir_path

    salinity_data_dir = path.join(data_dir_path, 'WOA13', 'DATAv2', 'salinity', 'netcdf')

    def __init__(self, time_span, avg_period, grid_size, field_type):
        """
        :param time_span: Choose from '5564', '6574', '7584', '8594', '95A4', 'A5B2', 'decav', and 'all'.
        :param avg_period: Choose from annual ('00'), monthly ('01'-'12'), and seasonal ('13' for JFM, '14' for AMJ,
                           '15' for JAS, and '16' for OND).
        :param grid_size: Choose from '04', '01', and '5d'.
        :param field_type: Choose from 'an', 'mn', 'dd', 'ma', 'sd', 'se', 'oa', and 'gp'.
        """
        self.salinity_dataset = None
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
        self.depths = None
        self.salinity_data = None

        logger.info('SalinityDataset object initializing for time span {} and averaging period {}...'
                    .format(self.time_span, self.avg_period))

        self.load_salinity_dataset()

    def salinity_dataset_filepath(self):
        filename = 'woa13_' + self.time_span + '_s' + self.avg_period + '_' + self.grid_size + 'v2.nc'

        return path.join(self.salinity_data_dir, self.time_span, self.grid_size_dir, filename)

    def load_salinity_dataset(self):
        from utils import log_netCDF_dataset_metadata

        dataset_filepath = self.salinity_dataset_filepath()
        logger.info('Loading salinity_data dataset: {}'.format(dataset_filepath))
        self.salinity_dataset = netCDF4.Dataset(dataset_filepath)
        logger.info('Successfully loaded salinity_data dataset: {}'.format(dataset_filepath))
        log_netCDF_dataset_metadata(self.salinity_dataset)

        self.lats = np.array(self.salinity_dataset.variables['lat'])
        self.lons = np.array(self.salinity_dataset.variables['lon'])
        self.depths = np.array(self.salinity_dataset.variables['depth'])

        field_var = 's_' + self.field_type

        self.salinity_data = np.array(self.salinity_dataset.variables[field_var])

    def meridional_salinity_profile(self, lon, lat_min, lat_max):
        _, n_levels, n_lats, n_lons = self.salinity_data.shape
        n_depths = len(self.depths)

        idx_lon = np.abs(self.lons - lon).argmin()
        idx_lat_min = np.abs(self.lats - lat_min).argmin()
        idx_lat_max = np.abs(self.lats - lat_max).argmin() + 1

        n_lats = idx_lat_max - idx_lat_min
        lats = self.lats[idx_lat_min:idx_lat_max]

        salinity_profile = np.zeros((n_lats, n_levels))
        for i in range(n_depths):
            salinity_profile[:, i] = self.salinity_data[0, i, idx_lat_min:idx_lat_max, idx_lon]

        salinity_profile[salinity_profile > 1e3] = np.nan

        return lats, self.depths, salinity_profile.transpose()

    def salinity(self, lat, lon, depth_levels):
        """
        :param lat:
        :param lon:
        :param depth_levels: Single depth level (int) or multiple depth levels (List[int]).
        :return: Salinity in SI units (g/kg). If given an integer depth level, the salinity for that depth level will be
                 returned. If given a list of integer depth levels, the average salinity over the given depth levels
                 will be returned.
        """
        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert -180 <= lon <= 180, "Longitude value {} out of bounds!".format(lon)

        idx_lat = np.abs(self.lats - lat).argmin()
        idx_lon = np.abs(self.lons - lon).argmin()

        if isinstance(depth_levels, int):
            salinity_scalar = self.salinity_data[0][depth_levels][idx_lat][idx_lon]

            if salinity_scalar > 1e3:
                return np.nan
            else:
                return salinity_scalar

        elif isinstance(depth_levels, list):
            salinity_avg = 0
            for level in depth_levels:
                salinity_scalar = self.salinity_data[0][level][idx_lat][idx_lon]
                if salinity_scalar > 1e3:
                    return np.nan
                else:
                    salinity_avg = salinity_avg + (salinity_scalar/len(depth_levels))

            return salinity_avg
