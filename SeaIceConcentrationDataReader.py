from os import path
import numpy as np
import netCDF4

import logging
logger = logging.getLogger(__name__)


class SeaIceConcentrationDataReader(object):
    from constants import data_dir_path

    sic_data_dir_path = path.join(data_dir_path, 'NOAA_NSIDC_G02202_V3_SEA_ICE_CONCENTRATION', 'south', 'daily')

    def __init__(self, date=None):
        self.current_SIC_dataset = None
        self.current_date = None
        self.lats = None
        self.lons = None
        self.xgrid = None
        self.ygrid = None
        self.alpha = None

        self.alpha_interp = None
        self.xgrid_interp = None
        self.ygrid_interp = None

        if date is None:
            logger.info('SeaIceConcentrationDataReader object initialized but no dataset was loaded.')
        else:
            logger.info('SeaIceConcentrationDataReader object initializing...')
            self.current_date = date
            self.current_SIC_dataset = self.load_SIC_dataset(date)

            self.interpolate_sea_ice_concentration_dataset()

    def date_to_SIC_dataset_filepath(self, date):
        filename = 'seaice_conc_daily_sh_f17_' + str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)\
                   + '_v03r00.nc'
        return path.join(self.sic_data_dir_path, str(date.year), filename)

    def load_SIC_dataset(self, date):
        from utils import log_netCDF_dataset_metadata

        dataset_filepath = self.date_to_SIC_dataset_filepath(date)
        logger.info('Loading sea ice concentration dataset: {}'.format(dataset_filepath))
        dataset = netCDF4.Dataset(dataset_filepath)
        logger.info('Successfully loaded sea ice concentration dataset: {}'.format(dataset_filepath))
        log_netCDF_dataset_metadata(dataset)

        self.lats = np.array(dataset.variables['latitude'])
        self.lons = np.array(dataset.variables['longitude'])
        self.xgrid = np.array(dataset.variables['xgrid'])
        self.ygrid = np.array(dataset.variables['ygrid'])
        self.alpha = np.array(dataset.variables['goddard_nt_seaice_conc'][0])

        return dataset

    def interpolate_sea_ice_concentration_dataset(self):
        from utils import interpolate_dataset
        from constants import data_dir_path
        from constants import lat_min, lat_max, n_lat, lon_min, lon_max, n_lon

        interp_filename_prefix = 'seaice_conc_daily_sh_f17_' + str(self.current_date.year) \
                                 + str(self.current_date.month).zfill(2) + str(self.current_date.day).zfill(2) \
                                 + '_v03r00'

        interp_filename_suffix = 'lat' + str(lat_min) + '-' + str(lat_max) + '_n' + str(n_lat) + '_' \
            + 'lon' + str(lon_min) + '-' + str(lon_max) + '_n' + str(n_lon) + '.pickle'

        alpha_interp_filename = interp_filename_prefix + '_interp_alpha_' + interp_filename_suffix
        # x_interp_filename = interp_filename_prefix + '_interp_x_' + interp_filename_suffix
        # y_interp_filename = interp_filename_prefix + '_interp_y_' + interp_filename_suffix
        alpha_interp_filepath = path.join(data_dir_path, 'mdt_cnes_cls2013_global', alpha_interp_filename)
        # x_interp_filepath = path.join(data_dir_path, 'mdt_cnes_cls2013_global', x_interp_filename)
        # y_interp_filepath = path.join(data_dir_path, 'mdt_cnes_cls2013_global', y_interp_filename)

        # TODO: Properly check for masked/filled values.
        mask_value_cond = lambda x: x > 1

        alpha_interp, xgrid_interp, ygrid_interp = \
            interpolate_dataset(self.alpha, self.xgrid, self.ygrid, alpha_interp_filepath, mask_value_cond, False, True)

        self.alpha_interp = alpha_interp
        self.xgrid_interp = xgrid_interp
        self.ygrid_interp = ygrid_interp

        # import matplotlib.pyplot as plt
        # plt.pcolormesh(ygrid_interp, xgrid_interp, alpha_interp)
        # plt.colorbar()
        # plt.show()

    def sea_ice_concentration(self, lat, lon, date):
        from utils import latlon_to_polar_stereographic_xy

        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert -180 <= lon <= 180, "Longitude value {} out of bounds!".format(lon)

        if self.current_SIC_dataset is None:
            logger.info('sea_ice_concentration called with no current dataset loaded.')
            logger.info('Loading sea ice concentration dataset for date requested: {}'.format(date))
            self.current_SIC_dataset = self.load_SIC_dataset(date)
            self.current_date = date

        if date != self.current_date:
            logger.info('SIC at different date requested: {} -> {}.'.format(self.current_date, date))
            logger.info('Changing SIC dataset...')
            self.current_SIC_dataset = self.load_SIC_dataset(date)
            self.current_date = date

        x, y = latlon_to_polar_stereographic_xy(lat, lon)
        idx_x = np.abs(self.xgrid - x).argmin()
        idx_y = np.abs(self.ygrid - y).argmin()
        lat_xy = self.lats[idx_y][idx_x]
        lon_xy = self.lons[idx_y][idx_x]

        if (np.abs(lat - lat_xy) > 0.5 or np.abs(lon - lon_xy) > 0.5) \
                and np.abs(lat) - 180 > 0.5 and np.abs(lat_xy) - 180 > 0.5:
            logger.warning('Lat or lon obtained from SIC dataset differ by more than 0.5 deg from input lat/lon!')
            logger.warning("lat = {}, lon = {} (input)", lat, lon)
            logger.warning("x = {}, y = {} (polar stereographic)", x, y)
            logger.warning("idx_x = {}, idx_y = {}", idx_x, idx_y)
            logger.warning("lat_xy = {}, lon_xy = {} (from SIC dataset)", lat_xy, lon_xy)

        alpha = self.alpha[idx_y][idx_x]

        # TODO: Properly check for masked values.
        if alpha > 1:
            return np.nan

        return alpha
