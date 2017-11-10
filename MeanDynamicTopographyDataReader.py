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
        self.u = np.array(self.MDT_dataset.variables['u'][0])
        self.v = np.array(self.MDT_dataset.variables['v'][0])

        self.interpolate_mdt_dataset()

    def interpolate_mdt_dataset(self):
        from scipy.interpolate import griddata
        from constants import lat_min, lat_max, lat_step, n_lat, lon_min, lon_max, lon_step, n_lon

        logger.info('Interpolating MDT dataset...')

        # TODO: Properly check for masked/filled values.
        mdt_values = np.ma.array(self.mdt, mask=(self.mdt < -100), copy=True)
        mdt_values = np.reshape(mdt_values, (len(self.lats)*len(self.lons), ))

        lat2 = np.repeat(self.lats, len(self.lons))
        lon2 = np.tile(self.lons, len(self.lats))

        lat2 = np.ma.masked_where(np.ma.getmask(mdt_values), lat2)
        lon2 = np.ma.masked_where(np.ma.getmask(mdt_values), lon2)

        print(mdt_values.shape)

        lat2 = lat2[~lat2.mask]
        lon2 = lon2[~lon2.mask]
        mdt_values = mdt_values[~mdt_values.mask]

        # TODO: Properly convert lat = -180:180 to lat = 0:360. List comprehension then sort?
        grid_x, grid_y = np.mgrid[lat_min:lat_max:n_lat*1j, 0:360:n_lon*1j]
        gridded_data = griddata((lat2, lon2), mdt_values, (grid_x, grid_y), method='linear')

        logger.info('Interpolating MDT dataset... DONE!')

        # Loop through and mask values that are supposed to be land, etc.?
        # Then plot residual field to check that interpolation matches

        import matplotlib.pyplot as plt
        plt.pcolormesh(grid_x, grid_y, gridded_data)
        plt.colorbar()
        plt.show()

        return gridded_data

    def mean_dynamic_topography(self, lat: float, lon: float) -> float:
        if lon < 0:
            lon = lon + 360  # Change from our convention lon = [-180, 180] to [0, 360]

        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert 0 <= lon <= 360, "Longitude value {} out of bounds!".format(lon)

        # Nearest neighbour interpolation
        # Find index of closest matching latitude and longitude
        idx_lat = np.abs(self.lats - lat).argmin()
        idx_lon = np.abs(self.lons - lon).argmin()

        logger.debug("lat = {:f}, lon = {:f}".format(lat, lon))
        logger.debug("idx_lat = {:d}, idx_lon = {:d}".format(idx_lat, idx_lon))
        logger.debug("lat[idx_lat] = {:f}, lon[idx_lon] = {:f}".format(self.lats[idx_lat], self.lons[idx_lon]))

        MDT_value = self.mdt[idx_lat][idx_lon]

        return MDT_value

    def u_geo_mean(self, lat: float, lon: float) -> np.ndarray:
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

        # TODO: Properly check for masked values.
        if u_geo_mean < -100 or v_geo_mean < -100:
            return np.array([np.nan, np.nan])

        return np.array([u_geo_mean, v_geo_mean])
