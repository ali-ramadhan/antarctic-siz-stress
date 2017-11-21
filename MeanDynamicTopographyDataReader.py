from os import path
import numpy as np
import netCDF4

import logging
logger = logging.getLogger(__name__)


class MeanDynamicTopographyDataReader(object):
    from constants import data_dir_path
    MDT_file_path = path.join(data_dir_path, 'mdt_cnes_cls2013_global', 'mdt_cnes_cls2013_global.nc')

    def __init__(self):
        from utils import log_netCDF_dataset_metadata

        logger.info('MeanDynamicTopographyDataReader initializing. Loading MDT dataset: {}'.format(self.MDT_file_path))
        self.MDT_dataset = netCDF4.Dataset(self.MDT_file_path)
        logger.info('Successfully loaded MDT dataset: {}'.format(self.MDT_file_path))
        log_netCDF_dataset_metadata(self.MDT_dataset)

        self.lats = np.array(self.MDT_dataset.variables['lat'])
        self.lons = np.array(self.MDT_dataset.variables['lon'])

        self.mdt = np.array(self.MDT_dataset.variables['mdt'][0])
        self.u_geo = np.array(self.MDT_dataset.variables['u'][0])
        self.v_geo = np.array(self.MDT_dataset.variables['v'][0])

        self.latgrid_interp = None
        self.longrid_interp = None

        # Not even using the MDT dataset anymore.
        self.mdt_interp = None
        # self.interpolate_mdt_field()

        self.ugeo_interp = None
        self.vgeo_interp = None
        self.interpolate_u_geo_field()

    def interpolate_mdt_field(self):
        from utils import interpolate_scalar_field
        from constants import data_dir_path
        from constants import lat_min, lat_max, n_lat, lon_min, lon_max, n_lon

        mdt_interp_filename = 'mdt_cnes_cls2013_global' + '_interp_mdt_' \
                              + 'lat' + str(lat_min) + '-' + str(lat_max) + '_n' + str(n_lat) + '_' \
                              + 'lon' + str(lon_min) + '-' + str(lon_max) + '_n' + str(n_lon) + '.pickle'
        mdt_interp_filepath = path.join(data_dir_path, 'mdt_cnes_cls2013_global', mdt_interp_filename)

        # TODO: Properly check for masked/filled values.
        mask_value_cond = lambda x: x < -100

        repeat0tile1 = True
        convert_lon_range = False
        mdt_interp, latgrid_interp, longrid_interp = interpolate_scalar_field(
            self.mdt, self.lats, self.lons, mdt_interp_filepath, mask_value_cond, 'latlon', 'cubic',
            repeat0tile1, convert_lon_range)

        self.mdt_interp = mdt_interp
        self.latgrid_interp = latgrid_interp
        self.longrid_interp = longrid_interp

    def interpolate_u_geo_field(self):
        from utils import interpolate_scalar_field
        from constants import data_dir_path
        from constants import lat_min, lat_max, n_lat, lon_min, lon_max, n_lon

        interp_filename_suffix = 'lat' + str(lat_min) + '-' + str(lat_max) + '_n' + str(n_lat) + '_' \
                               + 'lon' + str(lon_min) + '-' + str(lon_max) + '_n' + str(n_lon) + '.pickle'

        ugeo_interp_filename = 'mdt_cnes_cls2013_global' + '_interp_ugeo_' + interp_filename_suffix
        vgeo_interp_filename = 'mdt_cnes_cls2013_global' + '_interp_vgeo_' + interp_filename_suffix
        ugeo_interp_filepath = path.join(data_dir_path, 'mdt_cnes_cls2013_global', ugeo_interp_filename)
        vgeo_interp_filepath = path.join(data_dir_path, 'mdt_cnes_cls2013_global', vgeo_interp_filename)

        # TODO: Properly check for masked/filled values.
        mask_value_cond = lambda x: x < -100

        repeat0tile1 = True
        convert_lon_range = True
        ugeo_interp, latgrid_interp, longrid_interp = interpolate_scalar_field(
            self.u_geo, self.lats, self.lons, ugeo_interp_filepath, mask_value_cond, 'latlon', 'cubic',
            repeat0tile1, convert_lon_range)
        vgeo_interp, latgrid_interp, longrid_interp = interpolate_scalar_field(
            self.v_geo, self.lats, self.lons, vgeo_interp_filepath, mask_value_cond, 'latlon', 'cubic',
            repeat0tile1, convert_lon_range)

        self.ugeo_interp = ugeo_interp
        self.vgeo_interp = vgeo_interp
        self.latgrid_interp = latgrid_interp
        self.longrid_interp = longrid_interp

    def mean_dynamic_topography(self, lat: float, lon: float, data_source: str) -> float:
        if lon < 0:
            lon = lon + 360  # Change from our convention lon = [-180, 180] to [0, 360]

        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert 0 <= lon <= 360, "Longitude value {} out of bounds!".format(lon)

        if data_source == 'product':
            # Nearest neighbour interpolation
            # Find index of closest matching latitude and longitude
            idx_lat = np.abs(self.lats - lat).argmin()
            idx_lon = np.abs(self.lons - lon).argmin()
            mdt_value = self.mdt[idx_lat][idx_lon]
        elif data_source == 'interp':
            # Technically doing a nearest neighbour interpolation, but the interpolation should coincide exactly with
            # the lat, lon values we use.
            # Find index of closest matching latitude and longitude
            idx_lat = np.abs(self.latgrid_interp - lat).argmin()
            idx_lon = np.abs(self.longrid_interp - lon).argmin()
            mdt_value = self.mdt_interp[idx_lat][idx_lon]
        else:
            logger.error('Invalid value for data_source: {}'.format(data_source))
            raise ValueError('Invalid value for data_source: {}'.format(data_source))

        return mdt_value

    def u_geo_mean(self, lat: float, lon: float, data_source: str) -> np.ndarray:
        if lon < 0:
            lon = lon + 360  # Change from our convention lon = [-180, 180] to [0, 360]

        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert 0 <= lon <= 360, "Longitude value {} out of bounds!".format(lon)

        if data_source == 'product':
            # Nearest neighbour interpolation
            # Find index of closest matching latitude and longitude
            idx_lat = np.abs(self.lats - lat).argmin()
            idx_lon = np.abs(self.lons - lon).argmin()

            u_geo_mean = self.u_geo[idx_lat][idx_lon]
            v_geo_mean = self.v_geo[idx_lat][idx_lon]
        elif data_source == 'interp':
            # Technically doing a nearest neighbour interpolation, but the interpolation should coincide exactly with
            # the lat, lon values we use.
            # Find index of closest matching latitude and longitude
            idx_lat = np.abs(self.latgrid_interp - lat).argmin()
            idx_lon = np.abs(self.longrid_interp - lon).argmin()

            u_geo_mean = self.ugeo_interp[idx_lat][idx_lon]
            v_geo_mean = self.vgeo_interp[idx_lat][idx_lon]
        else:
            logger.error('Invalid value for data_source: {}'.format(data_source))
            raise ValueError('Invalid value for data_source: {}'.format(data_source))

        # TODO: Properly check for masked values.
        if u_geo_mean < -100 or v_geo_mean < -100:
            return np.array([np.nan, np.nan])
        else:
            return np.array([u_geo_mean, v_geo_mean])
