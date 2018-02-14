import os
from os import path
import numpy as np
import netCDF4

from SalinityDataset import SalinityDataset
from TemperatureDataset import TemperatureDataset

from utils import log_netCDF_dataset_metadata
from constants import lat_min, lat_max, lat_step, n_lat, lon_min, lon_max, lon_step, n_lon

import logging
logger = logging.getLogger(__name__)


class NeutralDensityDataset(object):
    from constants import output_dir_path

    neutral_density_output_dir = path.join(output_dir_path, 'neutral_density')

    def __init__(self, time_span, avg_period, grid_size, field_type, depth_levels):
        """
        :param time_span: Choose from '5564', '6574', '7584', '8594', '95A4', 'A5B2', 'decav', and 'all'.
        :param avg_period: Choose from annual ('00'), monthly ('01'-'12'), and seasonal ('13' for JFM, '14' for AMJ,
                           '15' for JAS, and '16' for OND).
        :param grid_size: Choose from '04', '01', and '5d'.
        :param field_type: Choose from 'an', 'mn', 'dd', 'ma', 'sd', 'se', 'oa', and 'gp'.
        :param depth_level:
        """
        self.neutral_density_dataset = None

        self.time_span = time_span
        self.avg_period = avg_period
        self.grid_size = grid_size
        self.field_type = field_type
        self.depth_levels = depth_levels

        self.grid_size_dir = None
        if grid_size == '04':
            self.grid_size_dir = '0.25'
        elif grid_size == '01':
            self.grid_size_dir = '1.00'
        elif grid_size == '5d':
            self.grid_size_dir = '5deg'

        self.lats = None
        self.lons = None

        logger.info('NeutralDensityDataset object initializing for time span {} and averaging period {}...'
                    .format(self.time_span, self.avg_period))

        self.lats = np.linspace(lat_min, lat_max, n_lat)
        self.lons = np.linspace(lon_min, lon_max, n_lon)

        self.neutral_density_field = np.zeros((len(self.depth_levels), len(self.lats), len(self.lons)))
        self.salinity_field = np.zeros((len(self.depth_levels), len(self.lats), len(self.lons)))
        self.temperature_field = np.zeros((len(self.depth_levels), len(self.lats), len(self.lons)))

        self.salinity_dataset = SalinityDataset(time_span, avg_period, grid_size, field_type)
        self.temperature_dataset = TemperatureDataset(time_span, avg_period, grid_size, field_type)

        # If dataset already exists and is stored, load it up.
        for i in range(len(self.depth_levels)):
            neutral_density_dataset_filepath = self.neutral_density_dataset_filepath(self.depth_levels[i])

            try:
                self.neutral_density_dataset = netCDF4.Dataset(neutral_density_dataset_filepath)
                log_netCDF_dataset_metadata(self.neutral_density_dataset)
                self.salinity_field[i] = np.array(self.neutral_density_dataset.variables['salinity'])
                self.temperature_field[i] = np.array(self.neutral_density_dataset.variables['temperature'])
                self.neutral_density_field[i] = np.array(self.neutral_density_dataset.variables['neutral_density'])
            except Exception as e:
                logger.error('{}'.format(e))
                logger.warning('{:s} not found. Neutral density field will now be computed...'
                               .format(neutral_density_dataset_filepath))
        return

        self.neutral_density_dataset = None
        self.salinity_field = np.zeros((len(self.lats), len(self.lons)))
        self.temperature_field = np.zeros((len(self.lats), len(self.lons)))
        self.neutral_density_field = np.zeros((len(self.lats), len(self.lons)))

        self.calculate_neutral_density_field()
        self.save_neutral_density_dataset()

    def neutral_density_dataset_filepath(self, depth_level):
        filename = 'neutral_density_woa13_' + self.time_span + '_' + self.avg_period + '_' + self.grid_size + \
                   'depth_' + str(depth_level) + '_v2.nc'

        return path.join(self.neutral_density_output_dir, filename)

    def calculate_neutral_density_field(self):
        logger.info('Starting MATLAB engine...')
        import matlab.engine
        eng = matlab.engine.start_matlab()

        for i in range(len(self.lats)):

            progress_percent = 100 * i / (len(self.lats) - 1)
            logger.info('(gamma_{:s}_{:s}_d{:d}) lat = {:.2f}/{:.2f} ({:.1f}%)'.format(self.time_span, self.avg_period,
                                                                                       self.depth_level, self.lats[i],
                                                                                       lat_max, progress_percent))

            for j in range(len(self.lons)):
                salinity = self.salinity_dataset.salinity(self.lats[i], self.lons[j], self.depth_level)
                temperature = self.temperature_dataset.temperature(self.lats[i], self.lons[j], self.depth_level)

                if np.isnan(salinity) or np.isnan(temperature):
                    self.salinity_field[i][j] = np.nan
                    self.temperature_field[i][j] = np.nan
                    self.neutral_density_field[i][j] = np.nan
                else:
                    self.salinity_field[i][j] = salinity
                    self.temperature_field[i][j] = temperature
                    self.neutral_density_field[i][j] = eng.eos80_legacy_gamma_n(float(salinity), float(temperature),
                                                                                  0.0, float(self.lons[j]),
                                                                                  float(self.lats[i]))

    def save_neutral_density_dataset(self):
        gamma_n_nc_filepath = self.neutral_density_dataset_filepath()

        gamma_n_dir = path.dirname(gamma_n_nc_filepath)
        if not path.exists(gamma_n_dir):
            logger.info('Creating directory: {:s}'.format(gamma_n_dir))
            os.makedirs(gamma_n_dir)

        logger.info('Saving {:s}...'.format(gamma_n_nc_filepath))

        gamma_n_dataset = netCDF4.Dataset(gamma_n_nc_filepath, 'w')

        gamma_n_dataset.title = 'Antarctic sea ice zone neutral density'
        gamma_n_dataset.institution = 'Department of Earth, Atmospheric, and Planetary Science, ' \
                                  'Massachusetts Institute of Technology'

        gamma_n_dataset.createDimension('time', None)
        gamma_n_dataset.createDimension('lat', len(self.lats))
        gamma_n_dataset.createDimension('lon', len(self.lons))

        # TODO: Actually store a date.
        time_var = gamma_n_dataset.createVariable('time', np.float64, ('time',))
        time_var.units = 'hours since 0001-01-01 00:00:00'
        time_var.calendar = 'gregorian'

        lat_var = gamma_n_dataset.createVariable('lat', np.float32, ('lat',))
        lat_var.units = 'degrees north'
        lat_var[:] = self.lats

        lon_var = gamma_n_dataset.createVariable('lon', np.float32, ('lon',))
        lat_var.units = 'degrees east'
        lon_var[:] = self.lons

        salinity_var = gamma_n_dataset.createVariable('salinity', float, ('lat', 'lon'), zlib=True)
        salinity_var.units = 'g/kg'
        salinity_var.long_name = 'Salinity'
        salinity_var[:] = self.salinity_field

        temperature_var = gamma_n_dataset.createVariable('temperature', float, ('lat', 'lon'), zlib=True)
        temperature_var.units = 'deg C'
        temperature_var.long_name = 'Temperature'
        temperature_var[:] = self.temperature_field

        gamma_n_var = gamma_n_dataset.createVariable('neutral_density', float, ('lat', 'lon'), zlib=True)
        gamma_n_var.units = 'kg/m^3'
        gamma_n_var.long_name = 'Neutral density (gamma_n)'
        gamma_n_var[:] = self.neutral_density_field

        gamma_n_dataset.close()

    def meridional_gamma_profile(self, lon, lat_min, lat_max):
        n_levels, n_lats, n_lons = self.neutral_density_field.shape
        n_depths = len(self.depth_levels)

        idx_lon = np.abs(self.lons - lon).argmin()
        idx_lat_min = np.abs(self.lats - lat_min).argmin()
        idx_lat_max = np.abs(self.lats - lat_max).argmin() + 1

        n_lats = idx_lat_max - idx_lat_min
        lats = self.lats[idx_lat_min:idx_lat_max]

        gamma_profile = np.zeros((n_lats, n_levels))
        for i in range(n_depths):
            gamma_profile[:, i] = self.neutral_density_field[i, idx_lat_min:idx_lat_max, idx_lon]

        # gamma_profile[gamma_profile > 1e3] = np.nan

        return lats, self.depth_levels, gamma_profile.transpose()

    def gamma_n(self, lat, lon, depth_level):
        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert -180 <= lon <= 180, "Longitude value {} out of bounds!".format(lon)

        idx_lat = np.abs(self.lats - lat).argmin()
        idx_lon = np.abs(self.lons - lon).argmin()
        neutral_density_scalar = self.neutral_density_field[depth_level][idx_lat][idx_lon]

        return neutral_density_scalar

    def gamma_n_depth_averaged(self, lat, lon, depth_levels):
        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert -180 <= lon <= 180, "Longitude value {} out of bounds!".format(lon)

        idx_lat = np.abs(self.lats - lat).argmin()
        idx_lon = np.abs(self.lons - lon).argmin()

        gamma_n_bar = 0
        for lvl in depth_levels:
            gamma_n_bar = gamma_n_bar + (self.neutral_density_field[lvl][idx_lat][idx_lon] / len(depth_levels))

        return gamma_n_bar
