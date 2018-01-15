# TODO: Properly take care of masked values and errors?
# TODO: Is u_ice > 0.5 m/s really super crazy? What was the outlier anomaly again?

from os import path
import numpy as np

from utils import polar_stereographic_velocity_vector_to_latlon

import logging
logger = logging.getLogger(__name__)


class SeaIceMotionDataset(object):
    from constants import data_dir_path, output_dir_path

    seaice_motion_path = path.join(data_dir_path, 'nsidc0116_icemotion_vectors_v3', 'data', 'south')
    seaice_motion_interp_dir = path.join(output_dir_path, 'nsidc0116_icemotion_vectors_v3', 'data', 'south', 'grid')

    def __init__(self, date, monthly=False):
        self.date = date
        self.monthly = monthly

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

        logger.info('SeaIceMotionDataset object initializing for date {}...'.format(self.date))
        logger.info('Loading south grid...')
        self.south_grid = None
        self.load_south_grid()
        self.south_grid_lats = 321
        self.south_grid_lons = 321

        self.load_u_ice_dataset()
        self.interpolate_seaice_motion_field()

    def load_south_grid(self):
        from constants import data_dir_path

        # Load in the NSIDC grid for the southern hemisphere. It stores the (x,y) polar stereographic coordinates for
        # each grid point and its corresponding (lat,lon).
        grid_filename = path.join(data_dir_path, 'nsidc0116_icemotion_vectors_v3', 'tools', 'south_x_y_lat_lon.txt')
        with open(grid_filename) as f:
            south_grid = f.readlines()
            south_grid = [s.strip().split() for s in south_grid]
            for i in range(len(south_grid)):
                south_grid[i] = [int(south_grid[i][0]), int(south_grid[i][1]),      # x, y
                                 float(south_grid[i][2]), float(south_grid[i][3])]  # lat, lon

        self.south_grid = south_grid

    def date_to_u_ice_dataset_filepath(self, date):
        if self.monthly:
            filename = 'icemotion.grid.month.' + str(date.year) + '.' + str(date.month).zfill(2) + '.s.v3.bin'
            return path.join(self.seaice_motion_path, 'means', str(date.year), filename)
        else:
            filename = 'icemotion.grid.daily.' + str(date.year) + str(date.timetuple().tm_yday).zfill(3) + '.s.v3.bin'
            return path.join(self.seaice_motion_path, 'grid', str(date.year), filename)

    def load_u_ice_dataset(self):
        dataset_filepath = self.date_to_u_ice_dataset_filepath(self.date)

        logger.info('Reading in sea ice motion dataset: {}'.format(dataset_filepath))
        data = np.fromfile(dataset_filepath, dtype='<i2').reshape(321, 321, 3)
        logger.info('Successfully read sea ice motion data.')

        u_ice = data[..., 0]/1000  # [m/s]
        v_ice = data[..., 1]/1000  # [m/s]
        error = data[..., 2]/10  # square root of the estimated error variance

        logger.info('Building 2D arrays for sea ice motion with lat,lon lookup for date {}...'.format(self.date))
        self.u_ice = np.zeros((self.south_grid_lats, self.south_grid_lons), dtype=float)
        self.v_ice = np.zeros((self.south_grid_lats, self.south_grid_lons), dtype=float)
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
                self.x[i][j] = self.south_grid[i * self.south_grid_lats + j][0]
                self.y[i][j] = self.south_grid[i * self.south_grid_lats + j][1]
                self.lat[i][j] = self.south_grid[i * self.south_grid_lats + j][2]
                self.lon[i][j] = self.south_grid[i * self.south_grid_lats + j][3]

        # A pixel value of 0 in the third variable indicates no vectors at that location.
        self.u_ice[self.error == 0] = np.nan
        self.v_ice[self.error == 0] = np.nan

        # Negative value indicate vectors that are near coastlines (within 25 km), which may contain false ice as
        # interpolation was applied to a surface map from passive microwave data.
        # At least that's what the documentation says. But half the vectors have sqrt(error)<0 so who knows?
        # self.u_ice[self.error < 0] = np.nan
        # self.v_ice[self.error < 0] = np.nan

    def interpolate_seaice_motion_field(self):
        from utils import interpolate_scalar_field
        from constants import n_row, n_col, u_ice_interp_method

        if self.monthly:
            interp_filename_prefix = 'icemotion.grid.month.' + str(self.date.year) + '.' \
                                     + str(self.date.month).zfill(2) + '.s.v3'
        else:
            interp_filename_prefix = 'icemotion.grid.daily.' + str(self.date.year) \
                                     + str(self.date.timetuple().tm_yday).zfill(3) + '.s.v3'

        interp_filename_suffix = str(n_row) + 'rows_' + str(n_col) + 'cols.pickle'

        u_ice_interp_filename = interp_filename_prefix + '_interp_u_ice_' + interp_filename_suffix
        v_ice_interp_filename = interp_filename_prefix + '_interp_v_ice_' + interp_filename_suffix
        u_ice_interp_filepath = path.join(self.seaice_motion_interp_dir, str(self.date.year), u_ice_interp_filename)
        v_ice_interp_filepath = path.join(self.seaice_motion_interp_dir, str(self.date.year), v_ice_interp_filename)

        # Throw out NaN values and extreme anomalies (u_ice or v_ice > 0.5 m/s).
        mask_value_cond = lambda x: np.isnan(x) | (np.abs(x) > 0.5)

        u_ice_interp, row_interp, col_interp = interpolate_scalar_field(data=self.u_ice, x=self.x[0], y=self.y[:, 0],
                                                                        pickle_filepath=u_ice_interp_filepath,
                                                                        mask_value_cond=mask_value_cond,
                                                                        grid_type='ease_rowcol',
                                                                        interp_method=u_ice_interp_method,
                                                                        repeat0tile1=True, convert_lon_range=False)
        v_ice_interp, row_interp, col_interp = interpolate_scalar_field(data=self.v_ice, x=self.x[0], y=self.y[:, 0],
                                                                        pickle_filepath=v_ice_interp_filepath,
                                                                        mask_value_cond=mask_value_cond,
                                                                        grid_type='ease_rowcol',
                                                                        interp_method=u_ice_interp_method,
                                                                        repeat0tile1=True, convert_lon_range=False)

        # Convert everything to a numpy array otherwise the argmin functions below have to create a new numpy array
        # every time, slowing down lookup considerably.
        self.u_ice_interp = np.array(u_ice_interp)
        self.v_ice_interp = np.array(v_ice_interp)
        self.row_interp = np.array(row_interp)
        self.col_interp = np.array(col_interp)

    def plot_sea_ice_motion_vector_field(self):
        import matplotlib.pyplot as plt
        import cartopy
        import cartopy.crs as ccrs

        from constants import lat_min, lat_max, n_lat, lon_min, lon_max, n_lon

        # Using matplotlib.
        logger.info('Plotting u_ice_interp...')
        plt.pcolormesh(self.x, self.y, self.u_ice, cmap='seismic', vmin=-0.2, vmax=0.2)
        plt.colorbar()
        plt.quiver(self.x[::3, ::3], self.y[::3, ::3], self.u_ice[::3, ::3], self.v_ice[::3, ::3], units='width',
                   width=0.001, scale=4)
        plt.gca().invert_yaxis()
        plt.show()

        logger.info('Plotting v_ice_interp...')
        plt.pcolormesh(self.x, self.y, self.v_ice, cmap='seismic', vmin=-0.2, vmax=0.2)
        plt.colorbar()
        plt.quiver(self.x[::3, ::3], self.y[::3, ::3], self.u_ice[::3, ::3], self.v_ice[::3, ::3], units='width',
                   width=0.001, scale=4)
        plt.gca().invert_yaxis()
        plt.show()

        # Using Cartopy
        lats = np.linspace(lat_min, lat_max, n_lat)
        lons = np.linspace(lon_min, lon_max, n_lon)

        u_ice_interp_latlon = np.zeros((len(lats), len(lons)))
        v_ice_interp_latlon = np.zeros((len(lats), len(lons)))

        for i in range(len(lats)):
            lat = lats[i]
            logger.info('{:f}'.format(lat))
            for j in range(len(lons)):
                lon = lons[j]
                u_ice_vec = self.seaice_motion_vector(lat, lon, 'interp')
                u_ice_interp_latlon[i][j] = u_ice_vec[0]
                v_ice_interp_latlon[i][j] = u_ice_vec[1]

        logger.info('Plotting u_ice_interp...')
        ax = plt.axes(projection=ccrs.SouthPolarStereo())
        land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face',
                                                       facecolor='dimgray', linewidth=0)
        ax.add_feature(land_50m)
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        vector_crs = ccrs.PlateCarree()
        im = ax.pcolormesh(lons, lats, u_ice_interp_latlon, transform=vector_crs, cmap='seismic', vmin=-0.2, vmax=0.2)
        plt.colorbar(im)

        ax.quiver(lons[::5], lats[::5], u_ice_interp_latlon[::5, ::5], v_ice_interp_latlon[::5, ::5],
                  transform=vector_crs, units='width', width=0.002, scale=3)
        plt.show()

        logger.info('Plotting v_ice_interp...')
        ax = plt.axes(projection=ccrs.SouthPolarStereo())
        land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face',
                                                       facecolor='dimgray', linewidth=0)
        ax.add_feature(land_50m)
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        vector_crs = ccrs.PlateCarree()
        im = ax.pcolormesh(lons, lats, v_ice_interp_latlon, transform=vector_crs, cmap='seismic', vmin=-0.2, vmax=0.2)
        plt.colorbar(im)

        ax.quiver(lons[::5], lats[::5], u_ice_interp_latlon[::5, ::5], v_ice_interp_latlon[::5, ::5],
                  transform=vector_crs, units='width', width=0.002, scale=3)
        plt.show()

    def seaice_motion_vector(self, lat, lon, data_source):
        from constants import R

        lat, lon = np.deg2rad([lat, lon])

        # EASE-Grid constants
        C = 25e3    # nominal cell size [m]
        r0 = 160.0  # map origin column
        s0 = 160.0  # map origin row

        # EASE-Grid coordinate transformation
        # http://nsidc.org/data/ease/ease_grid.html
        # h = np.cos(np.pi/4 - lat/2)  # Particular scale along meridians
        # k = np.csc(np.pi/4 - lat/2)  # Particular scale along parallels
        col = +2*R/C * np.sin(lon) * np.cos(np.pi/4 - lat/2) + r0  # column coordinate
        row = -2*R/C * np.cos(lon) * np.cos(np.pi/4 - lat/2) + s0  # row coordinate

        if data_source == 'product':
            row, col = int(row), int(col)

            try:
                # TODO: Check if the below is actually correct
                # We minus 1 since rows and cols start counting from 1, but indices count from 0 of course.
                # Actually you don't minus 1 as the south grid does start from 0.
                u_ice_rc = self.u_ice[row][col]
                v_ice_rc = self.v_ice[row][col]
                lat_rc = self.lat[row][col]
                lon_rc = self.lon[row][col]
            except IndexError:
                # This will happen if we're outside the rectangular region of the EASE-Grid so just return NaN.
                return np.array([np.nan, np.nan])

            # logger.debug('row = {}, col = {}'.format(row, col))
            # logger.debug('lat_rc = {}, lon_rc = {}'.format(lat_rc, lon_rc))
            # logger.debug('u_motion = {}, v_motion = {}'.format(u_ice_rc, v_ice_rc))
        elif data_source == 'interp':
            idx_row = np.abs(self.row_interp - row).argmin()
            idx_col = np.abs(self.col_interp - col).argmin()
            u_ice_rc = self.u_ice_interp[idx_row][idx_col]
            v_ice_rc = self.v_ice_interp[idx_row][idx_col]

        lat, lon = np.rad2deg([lat, lon])

        u_ice_vec_xy = np.array([u_ice_rc, v_ice_rc])
        u_ice_vec_latlon = polar_stereographic_velocity_vector_to_latlon(u_ice_vec_xy, lat, lon)

        # Make sure not to return any crazy sea ice motion vectors.
        if np.abs(u_ice_vec_xy[0]) > 0.5 or np.abs(u_ice_vec_xy[1]) > 0.5:
            return np.array([np.nan, np.nan])

        return u_ice_vec_latlon
