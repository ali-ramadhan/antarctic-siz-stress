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

        self.mdt_interp = None
        self.latgrid_interp = None
        self.longrid_interp = None
        self.interpolate_mdt_dataset()

    def interpolate_mdt_dataset(self):
        import pickle
        from os.path import isfile
        from constants import data_dir_path
        from constants import lat_min, lat_max, n_lat, lon_min, lon_max, n_lon

        mdt_interp_filename = 'mdt_cnes_cls2013_global' + '_interp_' \
                              + 'lat' + str(lat_min) + '-' + str(lat_max) + '_n' + str(n_lat) + '_' \
                              + 'lon' + str(lon_min) + '-' + str(lon_max) + '_n' + str(n_lon) + '.pickle'
        mdt_interp_filepath = path.join(data_dir_path, 'mdt_cnes_cls2013_global', mdt_interp_filename)

        if isfile(mdt_interp_filepath):
            logger.info('Interpolated grid found. Unpickling: {:s}'.format(mdt_interp_filepath))
            with open(mdt_interp_filepath, 'rb') as f:
                mdt_interp_dict = pickle.load(f)
                self.mdt_interp = mdt_interp_dict['mdt_interp']
                self.latgrid_interp = mdt_interp_dict['latgrid_interp']
                self.longrid_interp = mdt_interp_dict['longrid_interp']
            return

        from scipy.interpolate import griddata
        logger.info('Interpolating MDT dataset...')

        # TODO: Properly check for masked/filled values.
        # Mask land values and reshape into a 1D array in preparation for griddata.
        mdt_masked = np.ma.array(self.mdt, mask=(self.mdt < -100), copy=True)
        mdt_masked = np.reshape(mdt_masked, (len(self.lats)*len(self.lons), ))

        # Repeat the latitudes and tile the longitudes so that lat2[i], lon2[i] corresponds to mdt_values[i].
        lat_masked = np.repeat(self.lats, len(self.lons))
        lon_masked = np.tile(self.lons, len(self.lats))

        # Mask the latitudes and longitudes that correspond to land values.
        lat_masked = np.ma.masked_where(np.ma.getmask(mdt_masked), lat_masked)
        lon_masked = np.ma.masked_where(np.ma.getmask(mdt_masked), lon_masked)

        # Use the mask to remove all masked elements as griddata ignores masked data and cannot deal with NaN values.
        lat_masked = lat_masked[~lat_masked.mask]
        lon_masked = lon_masked[~lon_masked.mask]
        mdt_masked = mdt_masked[~mdt_masked.mask]

        # TODO: Properly convert lat = -180:180 to lat = 0:360. List comprehension then sort?
        # f = lambda lon: lon+360 if lon < 0 else lon
        # Create grid of points we wish to evaluate the interpolation on.
        latgrid_interp, longrid_interp = np.mgrid[lat_min:lat_max:n_lat*1j, 0:360:n_lon*1j]

        mdt_interp = griddata((lat_masked, lon_masked), mdt_masked, (latgrid_interp, longrid_interp), method='cubic')

        logger.info('Interpolating MDT dataset... DONE!')

        # Since we get back interpolated values over the land, we must mask them or get rid of them. We do this by
        # looping through the interpolated values and mask values that are supposed to be land by setting their value to
        # np.nan. We do this by comparing each interpolated value mdt_interp[i][j] with the mdt_values value that is
        # closest in latitude and longitude.
        # We can also compute the residuals, that is the error between the interpolated values and the actual values
        # which should be zero where an interpolation gridpoint coincides with an original gridpoint, and should be
        # pretty small everywhere else.
        residual = np.zeros((n_lat, n_lon))
        for i in range(n_lat):
            for j in range(n_lon):
                lat = latgrid_interp[i][j]
                lon = longrid_interp[i][j]
                closest_lat_idx = np.abs(self.lats - lat).argmin()
                closest_lon_idx = np.abs(self.lons - lon).argmin()
                closest_mdt = self.mdt[closest_lat_idx][closest_lon_idx]
                residual[i][j] = (closest_mdt - mdt_interp[i][j])/closest_mdt
                if self.mdt[closest_lat_idx][closest_lon_idx] < -100:  # TODO: Properly check for masked/filled values.
                    mdt_interp[i][j] = np.nan
                    residual[i][j] = np.nan

        # Plot residual field to check that the interpolation matches
        # import matplotlib.pyplot as plt
        # plt.pcolormesh(latgrid_interp, longrid_interp, np.log10(np.abs(residual)))
        # plt.colorbar()
        # plt.show()

        # Pickle the interpolated grid as a form of memoization to avoid having to recompute it again for the same
        # gridpoints.
        with open(mdt_interp_filepath, 'wb') as f:
            logger.info('Pickling interpolated grid: {:s}'.format(mdt_interp_filepath))
            mdt_interp_dict = {
                'mdt_interp': mdt_interp,
                'latgrid_interp': latgrid_interp,
                'longrid_interp': longrid_interp
            }
            pickle.dump(mdt_interp_dict, f, pickle.HIGHEST_PROTOCOL)

        self.mdt_interp = mdt_interp
        self.latgrid_interp = latgrid_interp
        self.longrid_interp = longrid_interp

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
