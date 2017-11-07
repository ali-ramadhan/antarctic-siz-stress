from os import path
import numpy as np
import netCDF4

import logging
logger = logging.getLogger(__name__)


class SeaIceConcentrationDataReader(object):
    from constants import data_dir_path

    sic_data_dir_path = path.join(data_dir_path, 'NOAA_NSIDC_G02202_V3_SEA_ICE_CONCENTRATION', 'south', 'daily')

    def __init__(self, date=None):
        if date is None:
            logger.info('SeaIceConcentrationDataReader object initialized but no dataset was loaded.')
            self.current_SIC_dataset = None
            self.current_date = None

            self.lats = None
            self.lons = None
            self.xgrid = None
            self.ygrid = None
            self.alpha = None

        else:
            logger.info('SeaIceConcentrationDataReader object initializing...')
            self.current_SIC_dataset = self.load_SIC_dataset(date)
            self.current_date = date

            self.lats = np.array(self.current_SIC_dataset.variables['latitude'])
            self.lons = np.array(self.current_SIC_dataset.variables['longitude'])
            self.xgrid = np.array(self.current_SIC_dataset.variables['xgrid'])
            self.ygrid = np.array(self.current_SIC_dataset.variables['ygrid'])
            self.alpha = np.array(self.current_SIC_dataset.variables['goddard_nt_seaice_conc'][0])

    def date_to_SIC_dataset_filepath(self, date):
        filename = 'seaice_conc_daily_sh_f17_' + str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)\
                   + '_v03r00.nc'
        return path.join(self.sic_data_dir_path, str(date.year), filename)

    def load_SIC_dataset(self, date):
        from utils import log_netCDF_dataset_metadata

        dataset_filepath = self.date_to_SIC_dataset_filepath(date)
        logger.info('Loading sea ice concentration dataset: %s', dataset_filepath)
        dataset = netCDF4.Dataset(dataset_filepath)
        logger.info('Successfully loaded sea ice concentration dataset: %s', dataset_filepath)
        log_netCDF_dataset_metadata(dataset)
        return dataset

    def sea_ice_concentration(self, lat, lon, date):
        from utils import latlon_to_polar_stereographic_xy

        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert -180 <= lon <= 180, "Longitude value {} out of bounds!".format(lon)

        if self.current_SIC_dataset is None:
            logger.info('sea_ice_concentration called with no current dataset loaded.')
            logger.info('Loading sea ice concentration dataset for date requested: {}'.format(date))
            self.current_SIC_dataset = self.load_SIC_dataset(date)
            self.current_date = date

            self.lats = np.array(self.current_SIC_dataset.variables['latitude'])
            self.lons = np.array(self.current_SIC_dataset.variables['longitude'])
            self.xgrid = np.array(self.current_SIC_dataset.variables['xgrid'])
            self.ygrid = np.array(self.current_SIC_dataset.variables['ygrid'])
            self.alpha = np.array(self.current_SIC_dataset.variables['goddard_nt_seaice_conc'][0])

        if date != self.current_date:
            logger.info('SIC at different date requested: {} -> {}.'.format(self.current_date, date))
            logger.info('Changing SIC dataset...')
            self.current_SIC_dataset = self.load_SIC_dataset(date)
            self.current_date = date

            self.lats = np.array(self.current_SIC_dataset.variables['latitude'])
            self.lons = np.array(self.current_SIC_dataset.variables['longitude'])
            self.xgrid = np.array(self.current_SIC_dataset.variables['xgrid'])
            self.ygrid = np.array(self.current_SIC_dataset.variables['ygrid'])
            self.alpha = np.array(self.current_SIC_dataset.variables['goddard_nt_seaice_conc'][0])

        x, y = latlon_to_polar_stereographic_xy(lat, lon)
        idx_x = np.abs(self.xgrid - x).argmin()
        idx_y = np.abs(self.ygrid - y).argmin()
        lat_xy = self.lats[idx_y][idx_x]
        lon_xy = self.lons[idx_y][idx_x]

        if (np.abs(lat - lat_xy) > 0.5 or np.abs(lon - lon_xy) > 0.5) \
                and np.abs(lat) - 180 > 0.5 and np.abs(lat_xy) - 180 > 0.5:
            logger.warning('Lat or lon obtained from SIC dataset differ by more than 0.5 deg from input lat/lon!')
            logger.warning("lat = %f, lon = %f (input)", lat, lon)
            logger.warning("x = %f, y = %f (polar stereographic)", x, y)
            logger.warning("idx_x = %d, idx_y = %d", idx_x, idx_y)
            logger.warning("lat_xy = %f, lon_xy = %f (from SIC dataset)", lat_xy, lon_xy)

        # TODO: check for masked values, etc.
        return self.alpha[idx_y][idx_x]
