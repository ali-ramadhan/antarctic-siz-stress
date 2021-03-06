# TODO: Looks like there's a small discontinuity in the absolute u_geo field dataset itself at the 0 longitude line.

from os import path
import numpy as np
import netCDF4

import logging
logger = logging.getLogger(__name__)


class GeostrophicVelocityDataset(object):
    from constants import data_dir_path, output_dir_path

    u_geo_data_dir = path.join(data_dir_path, 'SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047',
                               'dataset-duacs-rep-global-merged-allsat-phy-l4-v3')
    u_geo_interp_dir = path.join(output_dir_path, 'SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047',
                                 'dataset-duacs-rep-global-merged-allsat-phy-l4-v3')

    def __init__(self, date):
        self.u_geo_dataset = None
        self.date = date

        self.lats = None
        self.lons = None
        self.u_geo = None
        self.v_geo = None

        self.u_geo_interp = None
        self.v_geo_interp = None
        self.lats_interp = None
        self.lons_interp = None

        logger.info('GeostrophicVelocityDataset object initializing for date {}...'.format(self.date))
        # self.load_u_geo_dataset()
        # self.interpolate_u_geo_field()

    def date_to_u_geo_dataset_filepath(self, date):
        # FIXME: Must pattern match the ending!!! https://docs.python.org/3.6/library/fnmatch.html ? glob?
        filename = 'dt_global_allsat_msla_h_' + str(date.year) + str(date.month).zfill(2) \
                   + str(date.day).zfill(2) + '_20170110.nc'
        return path.join(self.u_geo_data_dir, str(date.year), filename)

    def load_u_geo_dataset(self):
        from utils import log_netCDF_dataset_metadata

        dataset_filepath = self.date_to_u_geo_dataset_filepath(self.date)
        logger.info('Loading geostrophic velocity dataset: {}'.format(dataset_filepath))
        self.u_geo_dataset = netCDF4.Dataset(dataset_filepath)
        logger.info('Successfully loaded geostrophic velocity dataset: {}'.format(dataset_filepath))
        log_netCDF_dataset_metadata(self.u_geo_dataset)

        self.lats = np.array(self.u_geo_dataset.variables['latitude'])
        self.lons = np.array(self.u_geo_dataset.variables['longitude'])
        self.u_geo = np.array(self.u_geo_dataset.variables['ugos'][0])
        self.v_geo = np.array(self.u_geo_dataset.variables['vgos'][0])

    def interpolate_u_geo_field(self):
        from utils import interpolate_scalar_field
        from constants import lat_min, lat_max, n_lat, lon_min, lon_max, n_lon, u_geo_interp_method

        interp_filename_prefix = 'dt_global_allsat_msla_h_20150701_' + str(self.date.year) \
                                 + str(self.date.month).zfill(2) + str(self.date.day).zfill(2) + '_'
        interp_filename_suffix = 'lat' + str(lat_min) + '-' + str(lat_max) + '_n' + str(n_lat) + '_' \
                                 + 'lon' + str(lon_min) + '-' + str(lon_max) + '_n' + str(n_lon) + '.pickle'

        u_geo_interp_filename = interp_filename_prefix + '_interp_u_geo_' + interp_filename_suffix
        v_geo_interp_filename = interp_filename_prefix + '_interp_v_geo_' + interp_filename_suffix
        u_geo_interp_filepath = path.join(self.u_geo_interp_dir, str(self.date.year), u_geo_interp_filename)
        v_geo_interp_filepath = path.join(self.u_geo_interp_dir, str(self.date.year), v_geo_interp_filename)

        # TODO: Properly check for masked/filled values.
        mask_value_cond = lambda x: x < -100

        u_geo_interp, lats_interp, lons_interp = interpolate_scalar_field(data=self.u_geo, x=self.lats, y=self.lons,
                                                                          pickle_filepath=u_geo_interp_filepath,
                                                                          mask_value_cond=mask_value_cond,
                                                                          grid_type='latlon',
                                                                          interp_method=u_geo_interp_method,
                                                                          repeat0tile1=True, convert_lon_range=True)
        v_geo_interp, lats_interp, lons_interp = interpolate_scalar_field(data=self.v_geo, x=self.lats, y=self.lons,
                                                                          pickle_filepath=v_geo_interp_filepath,
                                                                          mask_value_cond=mask_value_cond,
                                                                          grid_type='latlon',
                                                                          interp_method=u_geo_interp_method,
                                                                          repeat0tile1=True, convert_lon_range=True)

        # Convert everything to a numpy array otherwise the argmin functions below have to create a new numpy array
        # every time, slowing down lookup considerably.
        self.u_geo_interp = np.array(u_geo_interp)
        self.v_geo_interp = np.array(v_geo_interp)
        self.lats_interp = np.array(lats_interp)
        self.lons_interp = np.array(lons_interp)

    def absolute_geostrophic_velocity(self, lat, lon, data_source):
        if lon < 0:
            lon = lon + 360  # Change from our convention lon = [-180, 180] to [0, 360]

        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert 0 <= lon <= 360, "Longitude value {} out of bounds!".format(lon)

        if data_source == 'product':
            idx_lat = np.abs(self.lats - lat).argmin()
            idx_lon = np.abs(self.lons - lon).argmin()
            u_geo_ll = self.u_geo[idx_lat][idx_lon]
            v_geo_ll = self.v_geo[idx_lat][idx_lon]

            if u_geo_ll < -100 or v_geo_ll < -100:
                return np.array([np.nan, np.nan])

        elif data_source == 'interp':
            idx_lat = np.abs(self.lats_interp - lat).argmin()
            idx_lon = np.abs(self.lons_interp - lon).argmin()
            u_geo_ll = self.u_geo_interp[idx_lat][idx_lon]
            v_geo_ll = self.v_geo_interp[idx_lat][idx_lon]
        else:
            logger.error('Invalid value for data_source: {}'.format(data_source))
            raise ValueError('Invalid value for data_source: {}'.format(data_source))

        return np.array([u_geo_ll, v_geo_ll])
