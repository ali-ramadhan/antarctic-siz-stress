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

        self.u = np.array(self.MDT_dataset.variables['u'][0])
        self.v = np.array(self.MDT_dataset.variables['v'][0])

        # self.interpolate_MDT_dataset()

    def interpolate_MDT_dataset(self):
        from scipy.interpolate import griddata

        logger.info('Interpolating MDT dataset...')

        lat = np.array(self.MDT_dataset.variables['lat'])
        lon = np.array(self.MDT_dataset.variables['lon'])

        min_lat = np.min(lat)
        max_lat = np.max(lat)
        min_lon = np.min(lon)
        max_lon = np.max(lon)

        values = np.array(self.MDT_dataset.variables['mdt'])
        values = np.reshape(np.transpose(values), (len(lat)*len(lon), ))
        values[values < -100] = np.nan

        lat2 = np.tile(lat, len(lon))
        lon2 = np.repeat(lon, len(lat))

        lat2 = lat2[~np.isnan(values)]
        lon2 = lon2[~np.isnan(values)]
        values = values[~np.isnan(values)]

        # TODO: Document how you do this cartesian product. Also maybe switch to the faster version you bookmarked.
        # points = np.transpose([np.tile(lat, len(lon)), np.repeat(lon, len(lat))])

        nan = 0
        notnan = 0
        for index, x in np.ndenumerate(values):
            if not np.isnan(x):
                notnan += 1
            else:
                nan += 1

        logger.debug('nan = {}'.format(nan))
        logger.debug('notnan = {}'.format(notnan))

        grid_x, grid_y = np.mgrid[min_lat:max_lat:1000j, min_lon:max_lon:1000j]
        gridded_data = griddata((lat2, lon2), values, (grid_x, grid_y), method='cubic')

        logger.info('Interpolating MDT dataset... DONE!')

        nan = 0
        notnan = 0
        for index, x in np.ndenumerate(gridded_data):
            if not np.isnan(x):
                notnan += 1
            else:
                nan += 1

        logger.debug('nan = {}'.format(nan))
        logger.debug('notnan = {}'.format(notnan))

        import matplotlib.pyplot as plt
        plt.contourf(grid_x, grid_y, gridded_data, 60)
        plt.colorbar()
        plt.show()

        return gridded_data

    def get_MDT(self, lat, lon):
        if lon < 0:
            lon = lon + 360  # Change from our convention lon = [-180, 180] to [0, 360]

        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert 0 <= lon <= 360, "Longitude value {} out of bounds!".format(lon)

        # Nearest neighbour interpolation
        # Find index of closest matching latitude and longitude
        idx_lat = np.abs(np.array(self.MDT_dataset.variables['lat']) - lat).argmin()
        idx_lon = np.abs(np.array(self.MDT_dataset.variables['lon']) - lon).argmin()

        logger.debug("lat = {}, lon = {}".format(lat, lon))
        logger.debug("idx_lat = {}, idx_lon = {}".format(idx_lat, idx_lon))
        logger.debug("lat[idx_lat] = {}, lon[idx_lon] = {}".format(self.lats[idx_lat], self.lons[idx_lon]))

        MDT_value = self.MDT_dataset.variables['mdt'][0][idx_lat][idx_lon]

        return MDT_value

    def u_geo_mean(self, lat, lon):
        if lon < 0:
            lon = lon + 360  # Change from our convention lon = [-180, 180] to [0, 360]

        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert 0 <= lon <= 360, "Longitude value {} out of bounds!".format(lon)

        # Nearest neighbour interpolation
        # Find index of closest matching latitude and longitude
        idx_lat = np.abs(self.lats - lat).argmin()
        idx_lon = np.abs(self.lons - lon).argmin()

        logger.debug("lat = {}, lon = {}".format(lat, lon))
        logger.debug("idx_lat = {}, idx_lon = {}".format(idx_lat, idx_lon))
        logger.debug("lat[idx_lat] = {}, lon[idx_lon] = {}".format(self.lats[idx_lat], self.lons[idx_lon]))

        u_geo_mean = self.u[idx_lat][idx_lon]
        v_geo_mean = self.v[idx_lat][idx_lon]

        if u_geo_mean < -100 or v_geo_mean < -100:
            return np.array([np.nan, np.nan])

        return np.array([u_geo_mean, v_geo_mean])
