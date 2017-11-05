from os import path
import datetime
import numpy as np

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging
logger = logging.getLogger(__name__)


class SeaIceMotionDataReader(object):
    from constants import data_dir_path
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
        from constants import R

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