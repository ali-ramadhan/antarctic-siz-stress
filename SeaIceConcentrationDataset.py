from os import path
import numpy as np
import netCDF4

import logging
logger = logging.getLogger(__name__)


class SeaIceConcentrationDataset(object):
    from constants import data_dir_path, output_dir_path

    sic_data_dir_path = path.join(data_dir_path, 'NOAA_NSIDC_G02202_V3_SEA_ICE_CONCENTRATION', 'south', 'daily')
    sic_interp_dir = path.join(output_dir_path, 'NOAA_NSIDC_G02202_V3_SEA_ICE_CONCENTRATION', 'south', 'daily')

    def __init__(self, date):
        self.alpha_dataset = None
        self.date = date

        self.lats = None
        self.lons = None
        self.xgrid = None
        self.ygrid = None
        self.alpha = None

        self.alpha_interp = None
        self.xgrid_interp = None
        self.ygrid_interp = None

        logger.info('SeaIceConcentrationDataset object initializing for date {}...'.format(self.date))
        self.load_alpha_dataset()
        self.interpolate_alpha_field()

    def date_to_alpha_dataset_filepath(self, date):
        filename = 'seaice_conc_daily_sh_f17_' + str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2) \
                   + '_v03r00.nc'
        return path.join(self.sic_data_dir_path, str(date.year), filename)

    def load_alpha_dataset(self):
        from utils import log_netCDF_dataset_metadata

        dataset_filepath = self.date_to_alpha_dataset_filepath(self.date)
        logger.info('Loading sea ice concentration dataset: {}'.format(dataset_filepath))
        self.alpha_dataset = netCDF4.Dataset(dataset_filepath)
        logger.info('Successfully loaded sea ice concentration dataset: {}'.format(dataset_filepath))
        log_netCDF_dataset_metadata(self.alpha_dataset)

        self.lats = np.array(self.alpha_dataset.variables['latitude'])
        self.lons = np.array(self.alpha_dataset.variables['longitude'])
        self.xgrid = np.array(self.alpha_dataset.variables['xgrid'])
        self.ygrid = np.array(self.alpha_dataset.variables['ygrid'])
        self.alpha = np.array(self.alpha_dataset.variables['goddard_nt_seaice_conc'][0])

    def interpolate_alpha_field(self):
        from utils import interpolate_scalar_field
        from constants import n_x, n_y, alpha_interp_method

        interp_filename_prefix = 'seaice_conc_daily_sh_f17_' + str(self.date.year) \
                                 + str(self.date.month).zfill(2) + str(self.date.day).zfill(2) \
                                 + '_v03r00'

        interp_filename_suffix = str(n_x) + 'x_' + str(n_y) + 'y.pickle'

        alpha_interp_filename = interp_filename_prefix + '_interp_alpha_' + interp_filename_suffix
        alpha_interp_filepath = path.join(self.sic_interp_dir, str(self.date.year), alpha_interp_filename)

        # TODO: Properly check for masked/filled values.
        mask_value_cond = lambda x: x > 1

        alpha_interp, xgrid_interp, ygrid_interp = interpolate_scalar_field(data=self.alpha, x=self.xgrid, y=self.ygrid,
                                                                            pickle_filepath=alpha_interp_filepath,
                                                                            mask_value_cond=mask_value_cond,
                                                                            grid_type='polar_stereographic_xy',
                                                                            interp_method=alpha_interp_method,
                                                                            repeat0tile1=False, convert_lon_range=False)

        # Convert everything to a numpy array otherwise the argmin functions below have to create a new numpy array
        # every time, slowing down lookup considerably.
        self.alpha_interp = np.array(alpha_interp)
        self.xgrid_interp = np.array(xgrid_interp)
        self.ygrid_interp = np.array(ygrid_interp)

    def sea_ice_concentration(self, lat, lon, data_source):
        from utils import latlon_to_polar_stereographic_xy

        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert -180 <= lon <= 180, "Longitude value {} out of bounds!".format(lon)

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
        elif 0 < alpha < 0.15:
            return 0  # Treating gridpoints with SIC < 0.15 as basically free ice.
        else:
            return alpha
