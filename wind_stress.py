# TODO: Use the typing module.
# TODO: Add test case units?
# TODO: Split into multiple files.
# TODO: Switch from printf style logging to Python3 style formatting.
# TODO: Use propoer docstrings for functions.
# TODO: Choose a lat/lon convention for the frontend and convert as required for each product.

# Python standard library
import datetime
from enum import Enum

# Third party libraries
import numpy as np
import netCDF4

from os import path
import logging.config
import logging

from utils import *

import MeanDynamicTopographyDataReader

cwd = path.dirname(path.abspath(__file__))  # Current Working Directory

logging_config_path = path.join(cwd, 'logging.ini')
logging.config.fileConfig(logging_config_path)
logger = logging.getLogger(__name__)

# Data directory string constants
# Everything under a ./data/ softlink?
data_dir_path = path.join(cwd, 'data')

# Physical constants
# Could use a theoretical gravity model: https://en.wikipedia.org/wiki/Theoretical_gravity
g = 9.80665  # standard acceleration due to gravity [m/s^2]
Omega = 7.292115e-5  # rotation rate of the Earth [rad/s]

# Earth radius R could be upgraded to be location-dependent using a simple formula.
# See: https://en.wikipedia.org/wiki/Earth_radius#Location-dependent_radii
R = 6371.228e3  # average radius of the earth [m]


class SeaIceConcentrationDataReader(object):
    sic_data_dir_path = path.join(data_dir_path, 'NOAA_NSIDC_G02202_V3_SEA_ICE_CONCENTRATION', 'south', 'daily')

    def date_to_SIC_dataset_filepath(self, date):
        filename = 'seaice_conc_daily_sh_f17_' + str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)\
                   + '_v03r00.nc'
        return path.join(self.sic_data_dir_path, str(date.year), filename)

    def load_SIC_dataset(self, date):
        dataset_filepath = self.date_to_SIC_dataset_filepath(date)
        logger.info('Loading sea ice concentration dataset: %s', dataset_filepath)
        dataset = netCDF4.Dataset(dataset_filepath)
        logger.info('Successfully loaded sea ice concentration dataset: %s', dataset_filepath)
        log_netCDF_dataset_metadata(dataset)
        return dataset

    def __init__(self, date=None):
        if date is None:
            logger.info('SeaIceConcentrationDataReader object initialized but no dataset was loaded.')
            self.current_SIC_dataset = None
            self.current_date = None
        else:
            logger.info('SeaIceConcentrationDataReader object initializing...')
            self.current_SIC_dataset = self.load_SIC_dataset(date)
            self.current_date = date

    def sea_ice_concentration(self, lat, lon, date):
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
        idx_x = np.abs(np.array(self.current_SIC_dataset.variables['xgrid']) - x).argmin()
        idx_y = np.abs(np.array(self.current_SIC_dataset.variables['ygrid']) - y).argmin()
        lat_xy = self.current_SIC_dataset.variables['latitude'][idx_y][idx_x]
        lon_xy = self.current_SIC_dataset.variables['longitude'][idx_y][idx_x]

        if np.abs(lat - lat_xy) > 0.5 or np.abs(lon - lon_xy) > 0.5:
            logger.warning('Lat or lon obtained from SIC dataset differ by more than 0.5 deg from input lat/lon!')
            logger.debug("lat = %f, lon = %f (input)", lat, lon)
            logger.debug("x = %f, y = %f (polar stereographic)", x, y)
            logger.debug("idx_x = %d, idx_y = %d", idx_x, idx_y)
            logger.debug("lat_xy = %f, lon_xy = %f (from SIC dataset)", lat_xy, lon_xy)

        # TODO: check for masked values, etc.
        return self.current_SIC_dataset.variables['goddard_nt_seaice_conc'][0][idx_y][idx_x]


class OceanSurfaceWindVectorDataReader(object):
    oswv_ccmp_data_dir_path = path.join(data_dir_path, 'CCMP_MEASURES_ATLAS_L4_OW_L3_0_WIND_VECTORS_FLK')
    oswv_ncep_data_dir_path = path.join(data_dir_path, 'ncep.reanalysis.dailyavgs', 'surface_gauss')

    class OSWVProduct(Enum):
        NCEP = 1  # NCEP/NCAR Reanalysis 1 project wind data
        CCMP = 2  # Cross-Calibrated Multi-Platform Ocean Surface Wind Vector L3.0 First-Look Analyses

    def date_to_OSWV_dataset_filepath(self, date):
        if self.current_product is OceanSurfaceWindVectorDataReader.OSWVProduct.NCEP:
            uwind_filepath = path.join(self.oswv_ncep_data_dir_path, 'uwnd.10m.gauss.' + str(date.year) + '.nc')
            vwind_filepath = path.join(self.oswv_ncep_data_dir_path, 'vwnd.10m.gauss.' + str(date.year) + '.nc')
            return uwind_filepath, vwind_filepath
        elif self.current_product is OceanSurfaceWindVectorDataReader.OSWVProduct.CCMP:
            filename = 'analysis_' + str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)\
                       + '_v11l30flk.nc'
            return path.join(self.oswv_ccmp_data_dir_path, str(date.year), str(date.month).zfill(2), filename)
        else:
            logger.error('Invalid Enum value for current_product: {}'.format(self.current_product))
            raise ValueError('Invalid Enum value for current_product: {}'.format(self.current_product))

    def load_OSWV_dataset(self, date):
        if self.current_product is OceanSurfaceWindVectorDataReader.OSWVProduct.NCEP:
            uwind_dataset_filepath, vwind_dataset_filepath = self.date_to_OSWV_dataset_filepath(date)
            logger.info('Loading ocean surface wind vector NCEP datasets...', )

            logger.info('Loading NCEP uwind dataset: {}'.format(uwind_dataset_filepath))
            uwind_dataset = netCDF4.Dataset(uwind_dataset_filepath)
            logger.info('Successfully loaded NCEP uwind dataset: %s', uwind_dataset_filepath)
            log_netCDF_dataset_metadata(uwind_dataset)

            logger.info('Loading NCEP vwind dataset: {}'.format(vwind_dataset_filepath))
            vwind_dataset = netCDF4.Dataset(vwind_dataset_filepath)
            logger.info('Successfully loaded NCEP vwind dataset: %s', vwind_dataset_filepath)
            log_netCDF_dataset_metadata(vwind_dataset)

            return uwind_dataset, vwind_dataset
        elif self.current_product is OceanSurfaceWindVectorDataReader.OSWVProduct.CCMP:
            dataset_filepath = self.date_to_OSWV_dataset_filepath(date)
            logger.info('Loading ocean surface wind vector CCMP dataset: %s', dataset_filepath)
            dataset = netCDF4.Dataset(dataset_filepath)
            dataset.set_auto_mask(False)  # TODO: Why is most of the CCMP wind data masked?
            logger.info('Successfully ocean surface wind vector CCMP dataset: %s', dataset_filepath)
            log_netCDF_dataset_metadata(dataset)
            return dataset
        else:
            logger.error('Invalid value for current_product: {}'.format(self.current_product))
            raise ValueError('Invalid value for current_product: {}'.format(self.current_product))

    def __init__(self, date=None, product=OSWVProduct.NCEP):
        if date is None:
            logger.info('OceanSurfaceWindVectorDataReader object initialized but no dataset was loaded.')
            self.current_product = product
            self.current_date = None
            self.current_OSWV_dataset = None
        else:
            logger.info('OceanSurfaceWindVectorDataReader object initializing...')
            self.current_product = product
            self.current_date = date
            if self.current_product is OceanSurfaceWindVectorDataReader.OSWVProduct.NCEP:
                self.current_uwind_dataset, self.current_vwind_dataset = self.load_OSWV_dataset(date)
            elif self.current_product is OceanSurfaceWindVectorDataReader.OSWVProduct.CCMP:
                self.current_OSWV_dataset = self.load_OSWV_dataset(date)
            else:
                logger.error('Invalid value for current_product: {}'.format(self.current_product))
                raise ValueError('Invalid value for current_product: {}'.format(self.current_product))

    def ocean_surface_wind_vector(self, lat, lon, date):
        if self.current_OSWV_dataset is None:
            logger.info('ocean_surface_wind_vector called with no current dataset loaded.')
            logger.info('Loading ocean surface wind vector dataset for date requested: {}'.format(date))
            self.current_date = date

            if self.current_product is OceanSurfaceWindVectorDataReader.OSWVProduct.NCEP:
                self.current_uwind_dataset, self.current_vwind_dataset = self.load_OSWV_dataset(date)
            elif self.current_product is OceanSurfaceWindVectorDataReader.OSWVProduct.CCMP:
                self.current_OSWV_dataset = self.load_OSWV_dataset(date)
            else:
                logger.error('Invalid value for current_product: {}'.format(self.current_product))
                raise ValueError('Invalid value for current_product: {}'.format(self.current_product))

        if date != self.current_date:
            logger.info('OSWV at different date requested: {} -> {}.'.format(self.current_date, date))
            logger.info('Changing OSWV dataset...')
            self.current_date = date

            if self.current_product is OceanSurfaceWindVectorDataReader.OSWVProduct.NCEP:
                self.current_uwind_dataset, self.current_vwind_dataset = self.load_OSWV_dataset(date)
            elif self.current_product is OceanSurfaceWindVectorDataReader.OSWVProduct.CCMP:
                self.current_OSWV_dataset = self.load_OSWV_dataset(date)
            else:
                logger.error('Invalid value for current_product: {}'.format(self.current_product))
                raise ValueError('Invalid value for current_product: {}'.format(self.current_product))

        if self.current_product is OceanSurfaceWindVectorDataReader.OSWVProduct.NCEP:
            assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
            assert 0 <= lon <= 360, "Longitude value {} out of bounds!".format(lon)

            day_of_year = date.timetuple().tm_yday
            idx_lat = np.abs(np.array(self.current_uwind_dataset.variables['lat']) - lat).argmin()
            idx_lon = np.abs(np.array(self.current_uwind_dataset.variables['lon']) - lon).argmin()

            logger.debug("lat = %f, lon = %f", lat, lon)
            logger.debug("idx_lat = %d, idx_lon = %d", idx_lat, idx_lon)
            logger.debug("lat[idx_lat] = %f, lon[idx_lon] = %f", self.current_uwind_dataset.variables['lat'][idx_lat],
                         self.current_uwind_dataset.variables['lon'][idx_lon])
            logger.debug('time = {}'.format(self.current_uwind_dataset.variables['time'][day_of_year]))

            u_wind = self.current_uwind_dataset.variables['uwnd'][day_of_year][idx_lat][idx_lon]
            v_wind = self.current_vwind_dataset.variables['vwnd'][day_of_year][idx_lat][idx_lon]

            return np.array([u_wind, v_wind])
        elif self.current_product is OceanSurfaceWindVectorDataReader.OSWVProduct.CCMP:
            assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
            assert -180 <= lon <= 180, "Longitude value {} out of bounds!".format(lon)

            idx_lat = np.abs(np.array(self.current_OSWV_dataset.variables['lat']) - lat).argmin()
            idx_lon = np.abs(np.array(self.current_OSWV_dataset.variables['lon']) - lon).argmin()

            logger.debug("lat = %f, lon = %f", lat, lon)
            logger.debug("idx_lat = %d, idx_lon = %d", idx_lat, idx_lon)
            logger.debug("lat[idx_lat] = %f, lon[idx_lon] = %f", self.current_OSWV_dataset.variables['lat'][idx_lat],
                         self.current_OSWV_dataset.variables['lon'][idx_lon])

            u_wind = self.current_OSWV_dataset.variables['uwnd'][0][idx_lat][idx_lon]
            v_wind = self.current_OSWV_dataset.variables['vwnd'][0][idx_lat][idx_lon]

            return np.array([u_wind, v_wind])
        else:
            logger.error('Invalid value for current_product: {}'.format(self.current_product))
            raise ValueError('Invalid value for current_product: {}'.format(self.current_product))


class SeaIceMotionDataReader(object):
    seaice_drift_path = path.join(data_dir_path, 'nsidc0116_icemotion_vectors_v3', 'data', 'south', 'grid')

    def date_to_SIM_filepath(self, date):
        filename = 'icemotion.grid.daily.' + str(date.year) + str(date.timetuple().tm_yday) + '.s.v3.bin'
        # filename = 'icemotion.grid.daily.' + str(date.year) + '170.s.v3.bin'
        return path.join(self.seaice_drift_path, str(date.year), filename)

    def load_south_grid(self):
        # Load in the NSIDC grid for the southern hemisphere. It stores the (x,y) polar stereographic coordinates for
        # each grid point and its corresponding (lat,lon).
        grid_filename = path.join(data_dir_path, 'nsidc0116_icemotion_vectors_v3', 'tools', 'south_x_y_lat_lon.txt')
        with open(grid_filename) as f:
            south_grid = f.readlines()
            south_grid = [s.strip().split() for s in south_grid]
            for i in range(len(south_grid)):
                south_grid[i] = [int(south_grid[i][0]), int(south_grid[i][1]),  # x, y
                                 float(south_grid[i][2]), float(south_grid[i][3])]  # lat, lon

        return south_grid

    def __init__(self):
        self.south_grid = self.load_south_grid()
        self.south_grid_lats = 321
        self.south_grid_lons = 321

        test_dataset_date = datetime.date(2015, 6, 29)
        test_dataset_filepath = self.date_to_SIM_filepath(test_dataset_date)

        # logger.debug('Reading in sea ice drift dataset: {}'.format(test_dataset_filepath))
        # with open(test_dataset_filepath, mode='rb') as file:
        #     file_contents = file.read()
        #
        # logger.info('Successfully read sea ice drift data.')

        # I thought we had to uninterleave the NSIDC ice motion binary data files according to the dataset's user guide.
        # logger.info('Uninterleaving NSIDC ice motion binary data file...')
        # xdim = 321
        # ydim = 321
        # ngrids = 3
        # valsize = 2
        # file_contents_uninterleaved = bytearray(len(file_contents))
        # for i in range(xdim*ydim):
        #     for j in range(ngrids):
        #         for k in range(valsize):
        #             file_contents_uninterleaved[valsize*(xdim*ydim*j+i) + k] = file_contents[valsize*(i*ngrids+j) + k]
        #
        # logger.info('Uninterleaving done.')

        # Apparently this way of unpacking binary data does not work. Had to use np.fromfile below.
        # logger.debug('Unpacking binary (sea ice drift) data...')
        # total = 0
        # valid = 0
        # coast = 0
        # large = 0
        # import struct
        # for i in range(int(len(file_contents)/6)):
        #     total = total+1
        #     # x = list(struct.unpack("<hhh", file_contents[i:i+6]))
        #     x = list(struct.unpack("hhh", file_contents_uninterleaved[i:i+6]))
        #     if x[0] < 0:
        #         coast = coast+1
        #         continue
        #     if np.abs(x[0])/10 > 700 or np.abs(x[1])/10 > 70 or np.abs(x[2])/10 > 70:
        #         large = large+1
        #         continue
        #     if x[0] > 0:
        #         valid = valid+1
        #         x[0] = x[0]/10
        #         x[1] = x[1]/10
        #         x[2] = x[2]/10
        #         print("{} -> {}".format(i, x))
        #
        # logger.debug('total = {}, coast = {}, large = {}, valid = {}'.format(total, coast, large, valid))

        logger.debug('Reading in sea ice motion dataset: {}'.format(test_dataset_filepath))
        data = np.fromfile(test_dataset_filepath, dtype='<i2').reshape(321, 321, 3)
        logger.info('Successfully read sea ice motion data.')

        u_wind = data[..., 1]/10  # [cm/s]
        v_wind = data[..., 2]/10  # [cm/s]
        wind_error = data[..., 0]/10  # [???]

        logger.debug('Building 2D arrays for sea ice motion with lat,lon lookup...')
        self.u_wind = np.zeros((self.south_grid_lats, self.south_grid_lons), dtype=float)
        self.v_wind = np.zeros((self.south_grid_lats, self.south_grid_lons))
        self.wind_error = np.zeros((self.south_grid_lats, self.south_grid_lons))
        self.x = np.zeros((self.south_grid_lats, self.south_grid_lons))
        self.y = np.zeros((self.south_grid_lats, self.south_grid_lons))
        self.lat = np.zeros((self.south_grid_lats, self.south_grid_lons))
        self.lon = np.zeros((self.south_grid_lats, self.south_grid_lons))

        for i in range(self.south_grid_lats):
            for j in range(self.south_grid_lons):
                self.u_wind[i][j] = u_wind[i][j]
                self.v_wind[i][j] = v_wind[i][j]
                self.wind_error[i][j] = wind_error[i][j]
                self.lat[i][j] = self.south_grid[i * self.south_grid_lats + j][2]
                self.lon[i][j] = self.south_grid[i * self.south_grid_lats + j][3]
                self.x[i][j] = self.south_grid[i * self.south_grid_lats + j][0]
                self.y[i][j] = self.south_grid[i * self.south_grid_lats + j][1]

        logger.debug('Lookup arrays built.')

        # import matplotlib.pyplot as plt
        # self.u_wind[self.u_wind == 0] = np.nan
        # self.v_wind[self.u_wind == 0] = np.nan
        # plt.quiver(self.x[::4, ::4], self.y[::4, ::4], self.u_wind[::4, ::4], self.v_wind[::4, ::4], units='width', width=0.001, scale=1000)
        # plt.gca().invert_yaxis()
        # plt.show()

    def seaice_drift_vector(self, lat, lon, day):
        C = 25e3  # nominal cell size [m]
        r0 = 160.0  # map origin column
        s0 = 160.0  # map origin row

        logger.debug('lat = {}, lon = {}'.format(lat, lon))
        lat, lon = np.deg2rad([lat, lon])

        # EASE-Grid coordinate transformation
        # http: // nsidc.org / data / ease / ease_grid.html
        # h = np.cos(np.pi/4 - lat/2)  # Particular scale along meridians
        # k = np.csc(np.pi/4 - lat/2)  # Particular scale along parallels
        col = 2*R/C * np.sin(lon) * np.cos(np.pi/4 - lat/2) + r0
        row = -2*R/C * np.cos(lon) * np.cos(np.pi/4 - lat/2) + s0

        row, col = int(row), int(col)
        u_wind = self.u_wind[row][col]
        v_wind = self.v_wind[row][col]
        lat_rc = self.lat[row][col]
        lon_rc = self.lon[row][col]

        logger.debug('row = {}, col = {}'.format(row, col))
        logger.debug('lat_rc = {}, lon_rc = {}'.format(lat_rc, lon_rc))
        logger.debug('u_wind = {}, v_wind = {}'.format(u_wind, v_wind))


class SeaSurfaceHeightAnomalyDataReader(object):
    # Empty until I need SSH' to compute dynamic ocean topography and dynamic geostrophic currents.
    pass


class WindStressDataWriter(object):
    # Such an object should mainly compute daily (averaged) wind stress and wind stress curl fields and write them out
    # to netCDF files. Computing monthly means makes sense here. But plotting should go elsewhere.
    pass


if __name__ == '__main__':
    # dist = distance(24, 25, 26, 27)
    # logger.info("That distance is %f m or %f km.", dist, dist/1000)

    MDT = MeanDynamicTopographyDataReader.MeanDynamicTopographyDataReader()
    print(MDT.get_MDT(-60, 135+180))
    print(MDT.u_geo_mean(-60, 135+180))

    # sea_ice = SeaIceConcentrationDataReader()
    # print(sea_ice.sea_ice_concentration(-60.0, 133.0, datetime.date(2015, 7, 31)))
    # print(sea_ice.sea_ice_concentration(-71.4, 24.5, datetime.date(2015, 7, 31)))
    # print(sea_ice.sea_ice_concentration(-70, 180, datetime.date(2015, 7, 31)))

    wind_vectors = OceanSurfaceWindVectorDataReader()
    print(wind_vectors.ocean_surface_wind_vector(-60, 20, datetime.date(2015, 10, 10)))

    seaice_drift = SeaIceMotionDataReader()
    print(seaice_drift.seaice_drift_vector(-60, 20, datetime.date(2015, 1, 1)))
