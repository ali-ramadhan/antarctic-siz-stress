from os import path
from enum import Enum, auto
import numpy as np
import netCDF4

import logging
logger = logging.getLogger(__name__)


class OSWVProduct(Enum):
    NCEP = auto()  # NCEP/NCAR Reanalysis 1 project wind data
    CCMP = auto()  # Cross-Calibrated Multi-Platform Ocean Surface Wind Vector L3.0 First-Look Analyses


class OceanSurfaceWindVectorDataReader(object):
    from constants import data_dir_path

    oswv_ccmp_data_dir_path = path.join(data_dir_path, 'CCMP_MEASURES_ATLAS_L4_OW_L3_0_WIND_VECTORS_FLK')
    oswv_ncep_data_dir_path = path.join(data_dir_path, 'ncep.reanalysis.dailyavgs', 'surface_gauss')

    def date_to_OSWV_dataset_filepath(self, date):
        if self.current_product is OSWVProduct.NCEP:
            uwind_filepath = path.join(self.oswv_ncep_data_dir_path, 'uwnd.10m.gauss.' + str(date.year) + '.nc')
            vwind_filepath = path.join(self.oswv_ncep_data_dir_path, 'vwnd.10m.gauss.' + str(date.year) + '.nc')
            return uwind_filepath, vwind_filepath
        elif self.current_product is OSWVProduct.CCMP:
            filename = 'analysis_' + str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)\
                       + '_v11l30flk.nc'
            return path.join(self.oswv_ccmp_data_dir_path, str(date.year), str(date.month).zfill(2), filename)
        else:
            logger.error('Invalid Enum value for current_product: {}'.format(self.current_product))
            raise ValueError('Invalid Enum value for current_product: {}'.format(self.current_product))

    def load_OSWV_dataset(self, date):
        from utils import log_netCDF_dataset_metadata

        if self.current_product is OSWVProduct.NCEP:
            uwind_dataset_filepath, vwind_dataset_filepath = self.date_to_OSWV_dataset_filepath(date)
            logger.info('Loading ocean surface wind vector NCEP datasets...', )

            logger.info('Loading NCEP uwind dataset: {}'.format(uwind_dataset_filepath))
            uwind_dataset = netCDF4.Dataset(uwind_dataset_filepath)
            logger.info('Successfully loaded NCEP uwind dataset: %s', uwind_dataset_filepath)
            log_netCDF_dataset_metadata(uwind_dataset)

            logger.info('Loading NCEP vwind dataset: {}'.format(vwind_dataset_filepath))
            vwind_dataset = netCDF4.Dataset(vwind_dataset_filepath)
            logger.info('Successfully loaded NCEP vwind dataset: %s', vwind_dataset_filepath)
            log_netCDF_dataset_metadata(vwind_dataset)

            return uwind_dataset, vwind_dataset
        elif self.current_product is OSWVProduct.CCMP:
            dataset_filepath = self.date_to_OSWV_dataset_filepath(date)
            logger.info('Loading ocean surface wind vector CCMP dataset: %s', dataset_filepath)
            dataset = netCDF4.Dataset(dataset_filepath)
            dataset.set_auto_mask(False)  # TODO: Why is most of the CCMP wind data masked?
            logger.info('Successfully ocean surface wind vector CCMP dataset: %s', dataset_filepath)
            log_netCDF_dataset_metadata(dataset)
            return dataset
        else:
            logger.error('Invalid value for current_product: {}'.format(self.current_product))
            raise ValueError('Invalid value for current_product: {}'.format(self.current_product))

    def __init__(self, date=None, product=OSWVProduct.NCEP):
        logger.warning('Watch out!')
        print('TRUTH:')
        print(logger.propagate)
        if date is None:
            logger.info('OceanSurfaceWindVectorDataReader object initialized but no dataset was loaded.')
            self.current_product = product
            self.current_date = None
            self.current_OSWV_dataset = None
        else:
            logger.info('OceanSurfaceWindVectorDataReader object initializing...')
            self.current_product = product
            self.current_date = date
            if self.current_product is OSWVProduct.NCEP:
                self.current_uwind_dataset, self.current_vwind_dataset = self.load_OSWV_dataset(date)
            elif self.current_product is OSWVProduct.CCMP:
                self.current_OSWV_dataset = self.load_OSWV_dataset(date)
            else:
                logger.error('Invalid value for current_product: {}'.format(self.current_product))
                raise ValueError('Invalid value for current_product: {}'.format(self.current_product))

    def ocean_surface_wind_vector(self, lat, lon, date):
        if self.current_OSWV_dataset is None:
            logger.info('ocean_surface_wind_vector called with no current dataset loaded.')
            logger.info('Loading ocean surface wind vector dataset for date requested: {}'.format(date))
            self.current_date = date

            if self.current_product is OSWVProduct.NCEP:
                self.current_uwind_dataset, self.current_vwind_dataset = self.load_OSWV_dataset(date)
            elif self.current_product is OSWVProduct.CCMP:
                self.current_OSWV_dataset = self.load_OSWV_dataset(date)
            else:
                logger.error('Invalid value for current_product: {}'.format(self.current_product))
                raise ValueError('Invalid value for current_product: {}'.format(self.current_product))

        if date != self.current_date:
            logger.info('OSWV at different date requested: {} -> {}.'.format(self.current_date, date))
            logger.info('Changing OSWV dataset...')
            self.current_date = date

            if self.current_product is OSWVProduct.NCEP:
                self.current_uwind_dataset, self.current_vwind_dataset = self.load_OSWV_dataset(date)
            elif self.current_product is OSWVProduct.CCMP:
                self.current_OSWV_dataset = self.load_OSWV_dataset(date)
            else:
                logger.error('Invalid value for current_product: {}'.format(self.current_product))
                raise ValueError('Invalid value for current_product: {}'.format(self.current_product))

        if self.current_product is OSWVProduct.NCEP:
            assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
            assert 0 <= lon <= 360, "Longitude value {} out of bounds!".format(lon)

            day_of_year = date.timetuple().tm_yday
            idx_lat = np.abs(np.array(self.current_uwind_dataset.variables['lat']) - lat).argmin()
            idx_lon = np.abs(np.array(self.current_uwind_dataset.variables['lon']) - lon).argmin()

            logger.debug("lat = %f, lon = %f", lat, lon)
            logger.debug("idx_lat = %d, idx_lon = %d", idx_lat, idx_lon)
            logger.debug("lat[idx_lat] = %f, lon[idx_lon] = %f", self.current_uwind_dataset.variables['lat'][idx_lat],
                         self.current_uwind_dataset.variables['lon'][idx_lon])
            logger.debug('time = {}'.format(self.current_uwind_dataset.variables['time'][day_of_year]))

            u_wind = self.current_uwind_dataset.variables['uwnd'][day_of_year][idx_lat][idx_lon]
            v_wind = self.current_vwind_dataset.variables['vwnd'][day_of_year][idx_lat][idx_lon]

            return np.array([u_wind, v_wind])
        elif self.current_product is OSWVProduct.CCMP:
            assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
            assert -180 <= lon <= 180, "Longitude value {} out of bounds!".format(lon)

            idx_lat = np.abs(np.array(self.current_OSWV_dataset.variables['lat']) - lat).argmin()
            idx_lon = np.abs(np.array(self.current_OSWV_dataset.variables['lon']) - lon).argmin()

            logger.debug("lat = %f, lon = %f", lat, lon)
            logger.debug("idx_lat = %d, idx_lon = %d", idx_lat, idx_lon)
            logger.debug("lat[idx_lat] = %f, lon[idx_lon] = %f", self.current_OSWV_dataset.variables['lat'][idx_lat],
                         self.current_OSWV_dataset.variables['lon'][idx_lon])

            u_wind = self.current_OSWV_dataset.variables['uwnd'][0][idx_lat][idx_lon]
            v_wind = self.current_OSWV_dataset.variables['vwnd'][0][idx_lat][idx_lon]

            return np.array([u_wind, v_wind])
        else:
            logger.error('Invalid value for current_product: {}'.format(self.current_product))
            raise ValueError('Invalid value for current_product: {}'.format(self.current_product))
