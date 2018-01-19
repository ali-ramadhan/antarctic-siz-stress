from os import path
from enum import Enum
import numpy as np
import netCDF4

import logging
logger = logging.getLogger(__name__)


class OSWVProduct(Enum):
    NCEP = 1  # NCEP/NCAR Reanalysis 1 project wind data
    CCMP = 2  # Cross-Calibrated Multi-Platform Ocean Surface Wind Vector L3.0 First-Look Analyses


class SurfaceWindDataset(object):
    from constants import data_dir_path, output_dir_path

    data_dir_path = path.join(data_dir_path, 'ncep.reanalysis.dailyavgs', 'surface_gauss')
    interp_dir = path.join(output_dir_path, 'ncep.reanalysis.dailyavgs', 'surface_gauss')

    def __init__(self, date):
        self.u_wind_dataset = None
        self.v_wind_dataset = None
        self.date = date
        self.day_of_year = date.timetuple().tm_yday

        self.lats = None
        self.lons = None
        self.u_wind = None
        self.v_wind = None

        self.u_wind_interp = None
        self.v_wind_interp = None
        self.latgrid_interp = None
        self.longrid_interp = None

        logger.info('SurfaceWindDataset object initializing for date {}...'.format(self.date))
        self.load_surface_wind_dataset()
        self.interpolate_wind_field()

    def date_to_dataset_filepath(self, date):
        uwind_filepath = path.join(self.data_dir_path, 'uwnd.10m.gauss.' + str(self.date.year) + '.nc')
        vwind_filepath = path.join(self.data_dir_path, 'vwnd.10m.gauss.' + str(self.date.year) + '.nc')

        return uwind_filepath, vwind_filepath

    def load_surface_wind_dataset(self):
        from utils import log_netCDF_dataset_metadata

        uwind_dataset_filepath, vwind_dataset_filepath = self.date_to_dataset_filepath(self.date)

        logger.info('Loading NCEP u_wind dataset: {}'.format(uwind_dataset_filepath))
        self.u_wind_dataset = netCDF4.Dataset(uwind_dataset_filepath)
        logger.info('Successfully loaded NCEP u_wind dataset: {}'.format(uwind_dataset_filepath))
        log_netCDF_dataset_metadata(self.u_wind_dataset)

        logger.info('Loading NCEP v_wind dataset: {}'.format(vwind_dataset_filepath))
        self.v_wind_dataset = netCDF4.Dataset(vwind_dataset_filepath)
        logger.info('Successfully loaded NCEP v_wind dataset: {}'.format(vwind_dataset_filepath))
        log_netCDF_dataset_metadata(self.v_wind_dataset)

        self.lats = np.array(self.u_wind_dataset.variables['lat'])
        self.lons = np.array(self.u_wind_dataset.variables['lon'])

        self.lons = np.append(self.lons, 360.0)

        # Numbering starts from 0 so we minus 1 to get the right index.
        self.u_wind = np.array(self.u_wind_dataset.variables['uwnd'][self.day_of_year - 1])
        self.v_wind = np.array(self.v_wind_dataset.variables['vwnd'][self.day_of_year - 1])

        self.u_wind = np.c_[self.u_wind, self.u_wind[:, 0]]
        self.v_wind = np.c_[self.v_wind, self.v_wind[:, 0]]

    def interpolate_wind_field(self):
        from utils import interpolate_scalar_field
        from constants import u_wind_interp_method
        from constants import lat_min, lat_max, n_lat, lon_min, lon_max, n_lon

        interp_filename_suffix = 'lat' + str(lat_min) + '-' + str(lat_max) + '_n' + str(n_lat) + '_' \
                                 + 'lon' + str(lon_min) + '-' + str(lon_max) + '_n' + str(n_lon) + '.pickle'

        day_of_year = str(self.date.timetuple().tm_yday)
        u_wind_interp_filename = 'uwnd.10m.gauss.' + str(self.date.year) + 'day' + day_of_year + \
                                 '_interp_uwind_' + interp_filename_suffix
        v_wind_interp_filename = 'vwnd.10m.gauss.' + str(self.date.year) + 'day' + day_of_year + \
                                 '_interp_vwind_' + interp_filename_suffix

        u_wind_interp_filepath = path.join(self.interp_dir, str(self.date.year), u_wind_interp_filename)
        v_wind_interp_filepath = path.join(self.interp_dir, str(self.date.year), v_wind_interp_filename)

        # TODO: Properly check for masked/filled values.
        mask_value_cond = lambda x: np.full(x.shape, False, dtype=bool)

        logger.info('lats.shape={}, lons.shape={}, u_wind.shape={}'.format(self.lats.shape, self.lons.shape,
                                                                           self.u_wind.shape))

        u_wind_interp, latgrid_interp, longrid_interp = interpolate_scalar_field(data=self.u_wind, x=self.lats,
                                                                                 y=self.lons,
                                                                                 pickle_filepath=u_wind_interp_filepath,
                                                                                 mask_value_cond=mask_value_cond,
                                                                                 grid_type='latlon',
                                                                                 interp_method=u_wind_interp_method,
                                                                                 repeat0tile1=True,
                                                                                 convert_lon_range=True)
        v_wind_interp, latgrid_interp, longrid_interp = interpolate_scalar_field(data=self.v_wind, x=self.lats,
                                                                                 y=self.lons,
                                                                                 pickle_filepath=v_wind_interp_filepath,
                                                                                 mask_value_cond=mask_value_cond,
                                                                                 grid_type='latlon',
                                                                                 interp_method=u_wind_interp_method,
                                                                                 repeat0tile1=True,
                                                                                 convert_lon_range=True)

        # Convert everything to a numpy array otherwise the argmin functions below have to create a new numpy array
        # every time, slowing down lookup considerably.
        self.u_wind_interp = np.array(u_wind_interp)
        self.v_wind_interp = np.array(v_wind_interp)
        self.latgrid_interp = np.array(latgrid_interp)
        self.longrid_interp = np.array(longrid_interp)

    def ocean_surface_wind_vector(self, lat, lon, data_source):
        # lon = 180 - lon  # Change from our convention lon = [-180, 180] to [0, 360]

        # This is the proper conversion from my longitude convention (-=W, +=E) to NCEP's.
        if lon < 0:
            lon = lon + 360

        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert 0 <= lon <= 360, "Longitude value {} out of bounds!".format(lon)

        if data_source == 'product':
            idx_lat = np.abs(self.lats - lat).argmin()
            idx_lon = np.abs(self.lons - lon).argmin()
            u_wind = self.u_wind[idx_lat][idx_lon]
            v_wind = self.v_wind[idx_lat][idx_lon]

            # logger.debug("lat = {}, lon = {}".format(lat, lon))
            # logger.debug("idx_lat = {}, idx_lon = {}".format(idx_lat, idx_lon))
            # logger.debug("lat[idx_lat] = {}, lon[idx_lon] = {}".format(self.lats[idx_lat], self.lons[idx_lon]))
            # logger.debug('time = {}'.format(self.u_wind_dataset.variables['time'][day_of_year]))

        elif data_source == 'interp':
            idx_lat = np.abs(self.latgrid_interp - lat).argmin()
            idx_lon = np.abs(self.longrid_interp - lon).argmin()
            u_wind = self.u_wind_interp[idx_lat][idx_lon]
            v_wind = self.v_wind_interp[idx_lat][idx_lon]

        return np.array([u_wind, v_wind])
