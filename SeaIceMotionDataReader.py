# TODO: Properly take care of masked values and errors?
# TODO: Convert vectors from polar stereographic (EASE-Grid) to zonal-meridional.

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
        self.u_ice = None
        self.v_ice = None
        self.error = None
        self.x = None
        self.y = None
        self.lat = None
        self.lon = None

        self.row_interp = None
        self.col_interp = None
        self.u_ice_interp = None
        self.v_ice_interp = None

        if date is None:
            logger.info('SeaIceConcentrationDataReader object initialized but no dataset was loaded.')
            self.current_date = None
        else:
            self.current_date = date
            self.load_SIM_dataset(date)
            self.dataset_loaded = True
            self.interpolate_seaice_motion_field()

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

        u_ice = data[..., 1]/10  # [cm/s]
        v_ice = data[..., 2]/10  # [cm/s]
        # u_ice = data[..., 1]/1000  # [m/s]
        # v_ice = data[..., 2]/1000  # [cm/s]
        error = data[..., 0]/10  # square root of the estimated error variance

        logger.debug('Building 2D arrays for sea ice motion with lat,lon lookup...')
        self.u_ice = np.zeros((self.south_grid_lats, self.south_grid_lons), dtype=float)
        self.v_ice = np.zeros((self.south_grid_lats, self.south_grid_lons))
        self.error = np.zeros((self.south_grid_lats, self.south_grid_lons))
        self.x = np.zeros((self.south_grid_lats, self.south_grid_lons))
        self.y = np.zeros((self.south_grid_lats, self.south_grid_lons))
        self.lat = np.zeros((self.south_grid_lats, self.south_grid_lons))
        self.lon = np.zeros((self.south_grid_lats, self.south_grid_lons))

        for i in range(self.south_grid_lats):
            for j in range(self.south_grid_lons):
                self.u_ice[i][j] = u_ice[i][j]
                self.v_ice[i][j] = v_ice[i][j]
                self.error[i][j] = error[i][j]
                self.lat[i][j] = self.south_grid[i * self.south_grid_lats + j][2]
                self.lon[i][j] = self.south_grid[i * self.south_grid_lats + j][3]
                self.x[i][j] = self.south_grid[i * self.south_grid_lats + j][0]
                self.y[i][j] = self.south_grid[i * self.south_grid_lats + j][1]

        # A pixel value of 0 in the third variable indicates no vectors at that location.
        self.u_ice[self.error == 0] = np.nan
        self.v_ice[self.error == 0] = np.nan

        # Negative value indicate vectors that are near coastlines (within 25 km), which may contain false ice as
        # interpolation was applied to a surface map from passive microwave data.
        # At least that's what the documentation says. But half the vectors have sqrt(error)<0 so who knows?
        # self.u_ice[self.error < 0] = np.nan
        # self.v_ice[self.error < 0] = np.nan

        logger.debug('Building 2D arrays for sea ice motion with lat,lon lookup... DONE.')

        # import matplotlib.pyplot as plt
        # plt.quiver(self.x[::4, ::4], self.y[::4, ::4], self.u_ice[::4, ::4], self.v_ice[::4, ::4], units='width',
        #            width=0.001, scale=1000)
        # plt.gca().invert_yaxis()
        # plt.show()
        # plt.pcolormesh(self.x, self.y, self.error)
        # plt.colorbar()
        # plt.show()
        # exit(44)

    def interpolate_seaice_motion_field(self):
        from utils import interpolate_scalar_field
        from constants import data_dir_path
        from constants import lat_min, lat_max, n_lat, lon_min, lon_max, n_lon

        interp_filename_prefix = 'icemotion.grid.daily.' + str(self.current_date.year) \
                                 + str(self.current_date.timetuple().tm_yday).zfill(3) + '.s.v3'
        interp_filename_suffix = 'lat' + str(lat_min) + '-' + str(lat_max) + '_n' + str(n_lat) + '_' \
            + 'lon' + str(lon_min) + '-' + str(lon_max) + '_n' + str(n_lon) + '.pickle'

        u_ice_interp_filename = interp_filename_prefix + '_interp_u_ice_' + interp_filename_suffix
        v_ice_interp_filename = interp_filename_prefix + '_interp_v_ice_' + interp_filename_suffix
        u_ice_interp_filepath = path.join(data_dir_path, 'mdt_cnes_cls2013_global', u_ice_interp_filename)
        v_ice_interp_filepath = path.join(data_dir_path, 'mdt_cnes_cls2013_global', v_ice_interp_filename)

        # TODO: Properly check for masked/filled values.
        mask_value_cond = lambda x: np.isnan(x)

        repeat0tile1 = True
        convert_lon_range = False
        u_ice_interp, row_interp, col_interp = interpolate_scalar_field(
            self.u_ice, self.x[0], self.y[:, 0], u_ice_interp_filepath, mask_value_cond, 'ease_rowcol', 'linear',
            repeat0tile1, convert_lon_range)
        v_ice_interp, row_interp, col_interp = interpolate_scalar_field(
            self.v_ice, self.x[0], self.y[:, 0], v_ice_interp_filepath, mask_value_cond, 'ease_rowcol', 'linear',
            repeat0tile1, convert_lon_range)

        self.u_ice_interp = u_ice_interp
        self.v_ice_interp = v_ice_interp
        self.row_interp = row_interp
        self.col_interp = col_interp

        # import matplotlib.pyplot as plt
        # plt.quiver(self.row_interp, self.col_interp, self.u_ice_interp, self.v_ice_interp, units='width',
        #            width=0.001, scale=1000)
        # plt.gca().invert_yaxis()
        # plt.show()

    def seaice_motion_vector(self, lat, lon, date, data_source):
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
        u_motion = self.u_ice[row][col]
        v_motion = self.v_ice[row][col]
        lat_rc = self.lat[row][col]
        lon_rc = self.lon[row][col]

        logger.debug('row = {}, col = {}'.format(row, col))
        logger.debug('lat_rc = {}, lon_rc = {}'.format(lat_rc, lon_rc))
        logger.debug('u_motion = {}, v_motion = {}'.format(u_motion, v_motion))

        return np.array([u_motion, v_motion])
