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

        logger.info('MeanDynamicTopographyDataReader initializing. Loading MDT dataset: %s', self.MDT_file_path)
        self.MDT_dataset = netCDF4.Dataset(self.MDT_file_path)
        logger.info('Successfully loaded MDT dataset: %s', self.MDT_file_path)
        log_netCDF_dataset_metadata(self.MDT_dataset)

        self.interpolate_MDT_dataset()

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
        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert 0 <= lon <= 360, "Longitude value {} out of bounds!".format(lon)

        # Nearest neighbour interpolation
        # Find index of closest matching latitude and longitude
        idx_lat = np.abs(np.array(self.MDT_dataset.variables['lat']) - lat).argmin()
        idx_lon = np.abs(np.array(self.MDT_dataset.variables['lon']) - lon).argmin()

        logger.debug("lat = %f, lon = %f", lat, lon)
        logger.debug("idx_lat = %d, idx_lon = %d", idx_lat, idx_lon)
        logger.debug("lat[idx_lat] = %f, lon[idx_lon] = %f", self.MDT_dataset.variables['lat'][idx_lat],
                     self.MDT_dataset.variables['lon'][idx_lon])

        MDT_value = self.MDT_dataset.variables['mdt'][0][idx_lat][idx_lon]

        return MDT_value

    def u_geo_mean(self, lat, lon):
        from constants import Omega, g
        from utils import distance

        # Calculate the x and y derivatives of MDT at the grid point ij using a second-order centered finite difference
        # approximation.

        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert 0 <= lon <= 360, "Longitude value {} out of bounds!".format(lon)

        dLat = self.MDT_dataset.variables['lat'][1] - self.MDT_dataset.variables['lat'][0]
        dLon = self.MDT_dataset.variables['lon'][1] - self.MDT_dataset.variables['lon'][0]

        logger.debug("dLat = %f, dLon = %f", dLat, dLon)

        # TODO: Set up wrapping/periodicity in the horizontal for calculating those edge cases.
        # TODO: Check that you're not getting back land values or something.
        # TODO: Use scipy interpolation function for this?
        MDT_ip1j = self.get_MDT(lat+dLat, lon)
        MDT_im1j = self.get_MDT(lat-dLat, lon)
        MDT_ijp1 = self.get_MDT(lat, lon+dLon)
        MDT_ijm1 = self.get_MDT(lat, lon-dLon)

        logger.debug("MDT_i+1,j = %f, MDT_i-1,j = %f, MDT_i,j+1 = %f, MDT_i,j-1 = %f,",
                     MDT_ip1j, MDT_im1j, MDT_ijp1, MDT_ijm1)

        # TODO: Make sure I'm calculating this correctly.
        dx = distance(lat-dLat, lon, lat+dLat, lon)
        dy = distance(lat, lon-dLon, lat, lon+dLon)

        # We are using a second-order centered finite difference approximation for the derivative which usually have
        # 2dx and dy in the denominator but here the dx and dy I've calculated are across two cells so there's no need
        # for the factor of 2.
        dMDTdx = (MDT_ip1j - MDT_im1j) / dx
        dMDTdy = (MDT_ijp1 - MDT_ijm1) / dy

        f = 2*Omega*np.sin(np.deg2rad(lat))
        u_geo_u = -(g/f) * dMDTdy
        u_geo_v = (g/f) * dMDTdx

        return np.array([u_geo_u, u_geo_v])