from os import path
import numpy as np
import netCDF4

import logging
logger = logging.getLogger(__name__)


class SeaIceConcentrationDataReader(object):
    from constants import data_dir_path, output_dir_path

    sic_data_dir_path = path.join(data_dir_path, 'NOAA_NSIDC_G02202_V3_SEA_ICE_CONCENTRATION', 'south', 'daily')
    sic_interp_dir = path.join(output_dir_path, 'NOAA_NSIDC_G02202_V3_SEA_ICE_CONCENTRATION', 'south', 'daily')

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
            self.load_SIC_dataset()

    def date_to_SIC_dataset_filepath(self, date):
        filename = 'seaice_conc_daily_sh_f17_' + str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2) \
                   + '_v03r00.nc'
        return path.join(self.sic_data_dir_path, str(date.year), filename)

    def load_SIC_dataset(self):
        from utils import log_netCDF_dataset_metadata

        dataset_filepath = self.date_to_SIC_dataset_filepath(self.current_date)
        logger.info('Loading sea ice concentration dataset: {}'.format(dataset_filepath))
        self.current_SIC_dataset = netCDF4.Dataset(dataset_filepath)
        logger.info('Successfully loaded sea ice concentration dataset: {}'.format(dataset_filepath))
        log_netCDF_dataset_metadata(self.current_SIC_dataset)

        self.lats = np.array(self.current_SIC_dataset.variables['latitude'])
        self.lons = np.array(self.current_SIC_dataset.variables['longitude'])
        self.xgrid = np.array(self.current_SIC_dataset.variables['xgrid'])
        self.ygrid = np.array(self.current_SIC_dataset.variables['ygrid'])
        self.alpha = np.array(self.current_SIC_dataset.variables['goddard_nt_seaice_conc'][0])

        self.interpolate_sea_ice_concentration_field()

    def interpolate_sea_ice_concentration_field(self):
        from utils import interpolate_scalar_field
        from constants import n_x, n_y, alpha_interp_method

        interp_filename_prefix = 'seaice_conc_daily_sh_f17_' + str(self.current_date.year) \
                                 + str(self.current_date.month).zfill(2) + str(self.current_date.day).zfill(2) \
                                 + '_v03r00'

        interp_filename_suffix = str(n_x) + 'x_' + str(n_y) + 'y.pickle'

        alpha_interp_filename = interp_filename_prefix + '_interp_alpha_' + interp_filename_suffix
        alpha_interp_filepath = path.join(self.sic_interp_dir, str(self.current_date.year), alpha_interp_filename)

        # TODO: Properly check for masked/filled values.
        mask_value_cond = lambda x: x > 1

        repeat0tile1 = False
        convert_lon_range = False
        alpha_interp, xgrid_interp, ygrid_interp = interpolate_scalar_field(
            self.alpha, self.xgrid, self.ygrid, alpha_interp_filepath, mask_value_cond, 'polar_stereographic_xy',
            alpha_interp_method, repeat0tile1, convert_lon_range)

        self.alpha_interp = alpha_interp
        self.xgrid_interp = xgrid_interp
        self.ygrid_interp = ygrid_interp

    def sea_ice_concentration(self, lat, lon, date, data_source):
        from utils import latlon_to_polar_stereographic_xy

        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert -180 <= lon <= 180, "Longitude value {} out of bounds!".format(lon)

        if self.current_SIC_dataset is None:
            logger.info('sea_ice_concentration called with no current dataset loaded.')
            logger.info('Loading sea ice concentration dataset for date requested: {}'.format(date))
            self.current_date = date
            self.load_SIC_dataset()

        if date != self.current_date:
            logger.info('SIC at different date requested: {} -> {}.'.format(self.current_date, date))
            logger.info('Changing SIC dataset...')
            self.current_date = date
            self.load_SIC_dataset()

        x, y = latlon_to_polar_stereographic_xy(lat, lon)

        if data_source == 'product':
            idx_x = np.abs(self.xgrid - x).argmin()
            idx_y = np.abs(self.ygrid - y).argmin()
            lat_xy = self.lats[idx_y][idx_x]
            lon_xy = self.lons[idx_y][idx_x]
            alpha = self.alpha[idx_y][idx_x]

            if (np.abs(lat - lat_xy) > 0.5 or np.abs(lon - lon_xy) > 0.5) \
                    and np.abs(lat) - 180 > 0.5 and np.abs(lat_xy) - 180 > 0.5:
                logger.warning('Lat or lon obtained from SIC dataset differ by more than 0.5 deg from input lat/lon!')
                logger.warning("lat = {}, lon = {} (input)", lat, lon)
                logger.warning("x = {}, y = {} (polar stereographic)", x, y)
                logger.warning("idx_x = {}, idx_y = {}", idx_x, idx_y)
                logger.warning("lat_xy = {}, lon_xy = {} (from SIC dataset)", lat_xy, lon_xy)
        elif data_source == 'interp':
            idx_x = np.abs(self.xgrid_interp - x).argmin()
            idx_y = np.abs(self.ygrid_interp - y).argmin()
            alpha = self.alpha_interp[idx_x][idx_y]
        else:
            logger.error('Invalid value for data_source: {}'.format(data_source))
            raise ValueError('Invalid value for data_source: {}'.format(data_source))

        # TODO: Properly check for masked values.
        if alpha > 1:
            return np.nan

        if alpha > 0 and alpha < 0.15:
            return 0

        return alpha
