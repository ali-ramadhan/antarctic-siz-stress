from os import path
import numpy as np
import netCDF4

import logging
logger = logging.getLogger(__name__)


class GeostrophicCurrentDataset(object):
    from constants import data_dir_path, output_dir_path

    dataset_filepath = path.join(data_dir_path, 'CS2_combined_Southern_Ocean_2011-2016.nc')
    interp_dir = path.join(output_dir_path, 'CS2_combined_Southern_Ocean_2011-2016')

    def __init__(self, date):
        self.CS2_dataset = None  # CryoSat-2 dataset
        self.month_idx = None

        self.x = None
        self.y = None
        self.lats = None
        self.lons = None
        self.area = None

        self.dot = None
        self.mdt = None
        self.sla = None

        self.dot_interp = None
        self.row_interp = None
        self.col_interp = None

        self.u_geo_interp = None
        self.v_geo_interp = None

        if 2011 <= date.year <= 2016:
            self.month_idx = (date.year-2011) * 12 + date.month
        else:
            logger.error('Geostrophic current dataset only available for 2011-2016!')

        logger.info('GeostrophicCurrentDataset object initializing with month_idx={:d}...'.format(self.month_idx))

        self.load_CS2_dataset()
        self.interpolate_geostrophic_current_field()

    def load_CS2_dataset(self):
        logger.info('Loading CryoSat-2 dynamic ocean topography (DOT) dataset: {}'.format(self.dataset_filepath))
        self.CS2_dataset = netCDF4.Dataset(self.dataset_filepath)
        logger.info('Successfully loaded CryoSat-2 dynamic ocean topography (DOT) dataset: {}'.format(self.dataset_filepath))

        self.x = np.array(self.CS2_dataset.variables['X'])
        self.y = np.array(self.CS2_dataset.variables['Y'])
        self.lats = np.array(self.CS2_dataset.variables['Latitude'])
        self.lons = np.array(self.CS2_dataset.variables['Longitude'])
        self.area = np.array(self.CS2_dataset.variables['Area'])

        self.dot = np.array(self.CS2_dataset.variables['DOT'])[self.month_idx]
        self.mdt = np.array(self.CS2_dataset.variables['MDT'])[self.month_idx]
        self.sla = np.array(self.CS2_dataset.variables['SLA'])[self.month_idx]

    def interpolate_geostrophic_current_field(self):
        from utils import interpolate_scalar_field
        from constants import n_row, n_col, dot_interp_method

        interp_filename_prefix = 'CS2_combined_Southern_Ocean' + '_DOT_' + str(self.month_idx)
        interp_filename_suffix = str(n_row) + 'rows_' + str(n_col) + 'cols.pickle'
        interp_filename = interp_filename_prefix + interp_filename_suffix
        interp_filepath = path.join(self.interp_dir, interp_filename)

        # Mask NaN values.
        mask_value_cond = lambda x: np.isnan(x)

        dot_interp, row_interp, col_interp = interpolate_scalar_field(data=np.rot90(self.dot, k=1, axes=(1, 0)),
                                                                      x=np.arange(len(self.y)),
                                                                      y=np.arange(len(self.x)),
                                                                      pickle_filepath=interp_filepath,
                                                                      mask_value_cond=mask_value_cond,
                                                                      grid_type='ease_rowcol',
                                                                      interp_method=dot_interp_method,
                                                                      repeat0tile1=False, convert_lon_range=False,
                                                                      debug_plots=False)

        # Convert everything to a numpy array otherwise the argmin functions below have to create a new numpy array
        # every time, slowing down lookup considerably.
        self.dot_interp = np.array(dot_interp)
        self.row_interp = np.array(row_interp)
        self.col_interp = np.array(col_interp)

        # print('row_interp={:}'.format(row_interp))
        # print('col_interp={:}'.format(col_interp))

    def dynamic_ocean_topography(self, lat, lon):
        from constants import R

        lat, lon = np.deg2rad([lat, lon])

        # EASE-Grid constants
        C = 50e3    # nominal cell size [m]
        s0 = 214-99.534884  # map origin column, calculated as 214 - 214*(5000/(5750+5000))
        r0 = 89.560976  # map origin row, calculated as 204 * (4500/(5750+4500))

        # EASE-Grid coordinate transformation
        # http://nsidc.org/data/ease/ease_grid.html
        # h = np.cos(np.pi/4 - lat/2)  # Particular scale along meridians
        # k = np.csc(np.pi/4 - lat/2)  # Particular scale along parallels
        col = +2*R/C * np.sin(lon) * np.cos(np.pi/4 - lat/2) + r0  # column coordinate
        row = -2*R/C * np.cos(lon) * np.cos(np.pi/4 - lat/2) + s0  # row coordinate

        idx_row = np.abs(self.row_interp - row).argmin()
        idx_col = np.abs(self.col_interp - col).argmin()
        dot_latlon = self.dot_interp[idx_row][idx_col]

        # logger.info('row={:.2f}, col={:.2f}, idx_row={:d}, idx_col={:d}, dot(lat={:.2f}, lon={:.2f}) = {:.2f}'
        #             .format(row, col, idx_row, idx_col, np.rad2deg(lat), np.rad2deg(lon), dot_latlon))

        return dot_latlon

    def geostrophic_current_velocity(self, lat, lon):
        from constants import g, Omega, lat_step, lon_step
        from utils import distance, polar_stereographic_velocity_vector_to_latlon

        f = 2 * Omega * np.sin(np.deg2rad(lat))  # Coriolis parameter [s^-1]

        # Dividing by 100 to convert [cm] -> [m].
        dot_ip1_j = self.dynamic_ocean_topography(lat + lat_step, lon) / 100
        dot_im1_j = self.dynamic_ocean_topography(lat - lat_step, lon) / 100
        dot_i_jp1 = self.dynamic_ocean_topography(lat, lon + lon_step) / 100
        dot_i_jm1 = self.dynamic_ocean_topography(lat, lon - lon_step) / 100

        if not np.isnan(dot_im1_j) and not np.isnan(dot_ip1_j) and not np.isnan(dot_i_jm1) and not np.isnan(dot_i_jp1):
            dx = distance(lat, lon - 0.5*lon_step, lat, lon + 0.5*lon_step)
            dy = distance(lat - 0.5*lat_step, lon, lat + 0.5*lat_step, lon)

            dHdx = (dot_ip1_j - dot_im1_j) / (2*dx)
            dHdy = (dot_i_jp1 - dot_i_jm1) / (2*dy)

            u_geo = -(g/f) * dHdx
            v_geo = (g/f) * dHdy

            u_geo_vec = np.array([u_geo, v_geo])

            return u_geo_vec
        else:
            return np.array([np.nan, np.nan])
