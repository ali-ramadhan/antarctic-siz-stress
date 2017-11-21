from os import path
import datetime
import numpy as np

import logging
logger = logging.getLogger(__name__)


class SeaIceMotionDataReader(object):
    from constants import data_dir_path

    seaice_motion_path = path.join(data_dir_path, 'nsidc0116_icemotion_vectors_v3', 'data', 'south', 'grid')

    def __init__(self, date=None):
        logger.info('Loading south grid...')
        self.south_grid = None
        self.load_south_grid()
        self.south_grid_lats = 321
        self.south_grid_lons = 321

        self.dataset_loaded = False
        self.u_wind = None
        self.v_wind = None
        self.wind_error = None
        self.x = None
        self.y = None
        self.lat = None
        self.lon = None

        if date is None:
            logger.info('SeaIceConcentrationDataReader object initialized but no dataset was loaded.')
            self.current_date = None
        else:
            self.current_date = date
            self.load_SIM_dataset(date)
            self.dataset_loaded = True

    def date_to_SIM_filepath(self, date):
        filename = 'icemotion.grid.daily.' + str(date.year) + str(date.timetuple().tm_yday).zfill(3) + '.s.v3.bin'
        return path.join(self.seaice_motion_path, str(date.year), filename)

    def load_south_grid(self):
        from constants import data_dir_path

        # Load in the NSIDC grid for the southern hemisphere. It stores the (x,y) polar stereographic coordinates for
        # each grid point and its corresponding (lat,lon).
        grid_filename = path.join(data_dir_path, 'nsidc0116_icemotion_vectors_v3', 'tools', 'south_x_y_lat_lon.txt')
        with open(grid_filename) as f:
            south_grid = f.readlines()
            south_grid = [s.strip().split() for s in south_grid]
            for i in range(len(south_grid)):
                south_grid[i] = [int(south_grid[i][0]), int(south_grid[i][1]),  # x, y
                                 float(south_grid[i][2]), float(south_grid[i][3])]  # lat, lon

        self.south_grid = south_grid

    def load_SIM_dataset(self, date):
        dataset_filepath = self.date_to_SIM_filepath(date)

        logger.debug('Reading in sea ice motion dataset: {}'.format(dataset_filepath))
        data = np.fromfile(dataset_filepath, dtype='<i2').reshape(321, 321, 3)
        logger.info('Successfully read sea ice motion data.')

        # u_wind = data[..., 1]/10  # [cm/s]
        u_wind = data[..., 1]/1000  # [m/s]
        # v_wind = data[..., 2]/10  # [cm/s]
        v_wind = data[..., 2]/1000  # [cm/s]
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

        logger.debug('Building 2D arrays for sea ice motion with lat,lon lookup... DONE.')

        # import matplotlib.pyplot as plt
        # self.u_wind[self.u_wind == 0] = np.nan
        # self.v_wind[self.u_wind == 0] = np.nan
        # plt.quiver(self.x[::4, ::4], self.y[::4, ::4], self.u_wind[::4, ::4], self.v_wind[::4, ::4], units='width',
        #            width=0.001, scale=1000)
        # plt.gca().invert_yaxis()
        # plt.show()

    def seaice_motion_vector(self, lat, lon, date):
        from constants import R

        if self.dataset_loaded is False:
            logger.info('seaice_motion_vector called with no current dataset loaded.')
            logger.info('Loading sea ice concentration dataset for date requested: {}'.format(date))
            self.current_date = date
            self.load_SIM_dataset(date)

        if date != self.current_date:
            logger.info('SIM at different date requested: {} -> {}.'.format(self.current_date, date))
            logger.info('Changing SIC dataset...')
            self.current_date = date
            self.load_SIM_dataset(date)

        C = 25e3  # nominal cell size [m]
        r0 = 160.0  # map origin column
        s0 = 160.0  # map origin row

        logger.debug('lat = {}, lon = {}'.format(lat, lon))
        lat, lon = np.deg2rad([lat, lon])

        # EASE-Grid coordinate transformation
        # http://nsidc.org/data/ease/ease_grid.html
        # h = np.cos(np.pi/4 - lat/2)  # Particular scale along meridians
        # k = np.csc(np.pi/4 - lat/2)  # Particular scale along parallels
        col = +2*R/C * np.sin(lon) * np.cos(np.pi/4 - lat/2) + r0  # column coordinate
        row = -2*R/C * np.cos(lon) * np.cos(np.pi/4 - lat/2) + s0  # row coordinate

        row, col = int(row), int(col)
        u_motion = self.u_wind[row][col]
        v_motion = self.v_wind[row][col]
        lat_rc = self.lat[row][col]
        lon_rc = self.lon[row][col]

        logger.debug('row = {}, col = {}'.format(row, col))
        logger.debug('lat_rc = {}, lon_rc = {}'.format(lat_rc, lon_rc))
        logger.debug('u_motion = {}, v_motion = {}'.format(u_motion, v_motion))

        return np.array([u_motion, v_motion])
