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
    from constants import data_dir_path, output_dir_path

    oswv_ccmp_data_dir_path = path.join(data_dir_path, 'CCMP_MEASURES_ATLAS_L4_OW_L3_0_WIND_VECTORS_FLK')
    oswv_ncep_data_dir_path = path.join(data_dir_path, 'ncep.reanalysis.dailyavgs', 'surface_gauss')

    oswv_ncep_interp_dir = path.join(output_dir_path, 'ncep.reanalysis.dailyavgs', 'surface_gauss')

    def __init__(self, date=None, product=OSWVProduct.NCEP):
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

                self.lats = np.array(self.current_uwind_dataset.variables['lat'])
                self.lons = np.array(self.current_uwind_dataset.variables['lon'])

                self.day_of_year = date.timetuple().tm_yday
                self.uwind = np.array(self.current_uwind_dataset.variables['uwnd'][self.day_of_year])
                self.vwind = np.array(self.current_vwind_dataset.variables['vwnd'][self.day_of_year])

                self.u_wind_interp = None
                self.v_wind_interp = None
                self.latgrid_interp = None
                self.longrid_interp = None
                self.interpolate_wind_field()

            elif self.current_product is OSWVProduct.CCMP:
                self.current_OSWV_dataset = self.load_OSWV_dataset(date)

            else:
                logger.error('Invalid value for current_product: {}'.format(self.current_product))
                raise ValueError('Invalid value for current_product: {}'.format(self.current_product))

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
            logger.info('Successfully loaded NCEP uwind dataset: {}'.format(uwind_dataset_filepath))
            log_netCDF_dataset_metadata(uwind_dataset)

            logger.info('Loading NCEP vwind dataset: {}'.format(vwind_dataset_filepath))
            vwind_dataset = netCDF4.Dataset(vwind_dataset_filepath)
            logger.info('Successfully loaded NCEP vwind dataset: {}'.format(vwind_dataset_filepath))
            log_netCDF_dataset_metadata(vwind_dataset)

            return uwind_dataset, vwind_dataset
        elif self.current_product is OSWVProduct.CCMP:
            dataset_filepath = self.date_to_OSWV_dataset_filepath(date)
            logger.info('Loading ocean surface wind vector CCMP dataset: {}'.format(dataset_filepath))
            dataset = netCDF4.Dataset(dataset_filepath)
            dataset.set_auto_mask(False)  # TODO: Why is most of the CCMP wind data masked?
            logger.info('Successfully ocean surface wind vector CCMP dataset: {}'.format(dataset_filepath))
            log_netCDF_dataset_metadata(dataset)
            return dataset
        else:
            logger.error('Invalid value for current_product: {}'.format(self.current_product))
            raise ValueError('Invalid value for current_product: {}'.format(self.current_product))

    def interpolate_wind_field(self):
        from utils import interpolate_scalar_field
        from constants import u_wind_interp_method
        from constants import lat_min, lat_max, n_lat, lon_min, lon_max, n_lon

        if self.current_product is OSWVProduct.NCEP:
            interp_filename_suffix = 'lat' + str(lat_min) + '-' + str(lat_max) + '_n' + str(n_lat) + '_' \
                                     + 'lon' + str(lon_min) + '-' + str(lon_max) + '_n' + str(n_lon) + '.pickle'

            day_of_year = str(self.current_date.timetuple().tm_yday)
            u_wind_interp_filename = 'uwnd.10m.gauss.' + str(self.current_date.year) + 'day' + day_of_year + \
                                     '_interp_uwind_' + interp_filename_suffix
            v_wind_interp_filename = 'vwnd.10m.gauss.' + str(self.current_date.year) + 'day' + day_of_year + \
                                     '_interp_vwind_' + interp_filename_suffix

            u_wind_interp_filepath = path.join(self.oswv_ncep_interp_dir, str(self.current_date.year), u_wind_interp_filename)
            v_wind_interp_filepath = path.join(self.oswv_ncep_interp_dir, str(self.current_date.year), v_wind_interp_filename)

            # TODO: Properly check for masked/filled values.
            mask_value_cond = lambda x: np.full(x.shape, False, dtype=bool)

            logger.info('lats.shape={}, lons.shape={}, u_wind.shape={}'.format(self.lats.shape, self.lons.shape, self.uwind.shape))

            repeat0tile1 = True
            convert_lon_range = True
            u_wind_interp, latgrid_interp, longrid_interp = interpolate_scalar_field(
                self.uwind, self.lats, self.lons, u_wind_interp_filepath, mask_value_cond, 'latlon',
                u_wind_interp_method, repeat0tile1, convert_lon_range)
            v_wind_interp, latgrid_interp, longrid_interp = interpolate_scalar_field(
                self.vwind, self.lats, self.lons, v_wind_interp_filepath, mask_value_cond, 'latlon',
                u_wind_interp_method, repeat0tile1, convert_lon_range)

            self.u_wind_interp = np.array(u_wind_interp)
            self.v_wind_interp = np.array(v_wind_interp)
            self.latgrid_interp = np.array(latgrid_interp)
            self.longrid_interp = np.array(longrid_interp)

        elif self.current_product is OSWVProduct.CCMP:
            pass
        else:
            logger.error('Invalid value for current_product: {}'.format(self.current_product))
            raise ValueError('Invalid value for current_product: {}'.format(self.current_product))

    def ocean_surface_wind_vector(self, lat, lon, date, data_source):
        if self.current_product is OSWVProduct.NCEP:
            if self.current_uwind_dataset is None or self.current_vwind_dataset is None:
                logger.info('ocean_surface_wind_vector called with no current dataset loaded.')
                logger.info('Loading ocean surface wind vector dataset for date requested: {}'.format(date))
                self.current_date = date
                self.current_uwind_dataset, self.current_vwind_dataset = self.load_OSWV_dataset(date)

                self.lats = np.array(self.current_uwind_dataset.variables['lat'])
                self.lons = np.array(self.current_uwind_dataset.variables['lon'])

                self.day_of_year = date.timetuple().tm_yday
                self.uwind = np.array(self.current_uwind_dataset.variables['uwnd'][self.day_of_year])
                self.vwind = np.array(self.current_vwind_dataset.variables['vwnd'][self.day_of_year])

            if date != self.current_date:
                logger.info('OSWV at different date requested: {} -> {}.'.format(self.current_date, date))
                logger.info('Changing NCEP OSWV dataset...')
                self.current_date = date
                self.current_uwind_dataset, self.current_vwind_dataset = self.load_OSWV_dataset(date)

                self.lats = np.array(self.current_uwind_dataset.variables['lat'])
                self.lons = np.array(self.current_uwind_dataset.variables['lon'])

                self.day_of_year = date.timetuple().tm_yday
                self.uwind = np.array(self.current_uwind_dataset.variables['uwnd'][self.day_of_year])
                self.vwind = np.array(self.current_vwind_dataset.variables['vwnd'][self.day_of_year])

            lon = lon + 180  # Change from our convention lon = [-180, 180] to [0, 360]

            assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
            assert 0 <= lon <= 360, "Longitude value {} out of bounds!".format(lon)

            if data_source == 'product':
                day_of_year = date.timetuple().tm_yday
                idx_lat = np.abs(self.lats - lat).argmin()
                idx_lon = np.abs(self.lons - lon).argmin()
                u_wind = self.uwind[idx_lat][idx_lon]
                v_wind = self.vwind[idx_lat][idx_lon]

                logger.debug("lat = {}, lon = {}".format(lat, lon))
                logger.debug("idx_lat = {}, idx_lon = {}".format(idx_lat, idx_lon))
                logger.debug("lat[idx_lat] = {}, lon[idx_lon] = {}".format(self.lats[idx_lat], self.lons[idx_lon]))
                logger.debug('time = {}'.format(self.current_uwind_dataset.variables['time'][day_of_year]))

            elif data_source == 'interp':
                idx_lat = np.abs(self.latgrid_interp - lat).argmin()
                idx_lon = np.abs(self.longrid_interp - lon).argmin()
                u_wind = self.u_wind_interp[idx_lat][idx_lon]
                v_wind = self.v_wind_interp[idx_lat][idx_lon]

            return np.array([u_wind, v_wind])

        elif self.current_product is OSWVProduct.CCMP:
            if self.current_OSWV_dataset is None:
                logger.info('ocean_surface_wind_vector called with no current dataset loaded.')
                logger.info('Loading ocean surface wind vector dataset for date requested: {}'.format(date))
                self.current_date = date
                self.current_OSWV_dataset = self.load_OSWV_dataset(date)

            if date != self.current_date:
                logger.info('OSWV at different date requested: {} -> {}.'.format(self.current_date, date))
                logger.info('Changing CCMP OSWV dataset...')
                self.current_date = date
                self.current_OSWV_dataset = self.load_OSWV_dataset(date)

            assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
            assert -180 <= lon <= 180, "Longitude value {} out of bounds!".format(lon)

            idx_lat = np.abs(np.array(self.current_OSWV_dataset.variables['lat']) - lat).argmin()
            idx_lon = np.abs(np.array(self.current_OSWV_dataset.variables['lon']) - lon).argmin()

            logger.debug("lat = {}, lon = {}".format(lat, lon))
            logger.debug("idx_lat = {}, idx_lon = {}".format(idx_lat, idx_lon))
            logger.debug("lat[idx_lat] = {}, lon[idx_lon] = {}"
                         .format(self.current_OSWV_dataset.variables['lat'][idx_lat],
                                 self.current_OSWV_dataset.variables['lon'][idx_lon]))

            u_wind = self.current_OSWV_dataset.variables['uwnd'][0][idx_lat][idx_lon]
            v_wind = self.current_OSWV_dataset.variables['vwnd'][0][idx_lat][idx_lon]

            return np.array([u_wind, v_wind])

        else:
            logger.error('Invalid value for current_product: {}'.format(self.current_product))
            raise ValueError('Invalid value for current_product: {}'.format(self.current_product))
