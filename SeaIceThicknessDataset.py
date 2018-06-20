from os import path
import csv
import bisect
from scipy.spatial.distance import cdist

import numpy as np

import logging
logger = logging.getLogger(__name__)


class SeaIceThicknessDataset(object):
    from constants import data_dir_path, output_dir_path

    h_ice_data_dir_path = path.join(data_dir_path, 'ICESat_sea_ice_thickness')

    def __init__(self, date):
        self.date = date
        self.season = None

        self.lats = np.zeros(104913)
        self.lons = np.zeros(104913)
        self.h_ice = np.zeros(104913)
        self.latlon = np.zeros((104913, 2))

        self.h_ice_seasonal = {
            'summer': np.zeros(104913),
            'fall': np.zeros(104913),
            'spring': np.zeros(104913)
        }

        # Lots of "interpolation" happening here. I am using the Feb/Mar field for JFM, the May/Jun field for AMJ, and
        # the Nov/Dec field for OND. Also, since no JAS data exists, I am using the OND field for JAS, my argument
        # being that the JAS h_ice field looks similar to the OND h_ice field in model data published by Holland
        # et al., "Modeled Trends in Antarctic Sea Ice Thickness", Journal of Climate (2014).
        if 1 <= date.month <= 3:
            self.season = 'summer'
        elif 4 <= date.month <= 6:
            self.season = 'fall'
        elif 7 <= date.month <= 12:
            self.season = 'spring'
        else:
            logger.error('No sea ice thickness data for month {:d}!'.format(date.month))

        self.dataset_filename = self.season + '_ICESat_gridded_mean_thickness_sorted.txt'
        self.dataset_filepath = path.join(self.h_ice_data_dir_path, self.dataset_filename)

        logger.info('SeaIceThicknessDataset object initializing for {:s} season...'.format(self.season))
        self.load_h_ice_dataset()

        for season in ['summer', 'fall', 'spring']:
            logger.info('Loading season {:s}...'.format(season))
            dataset_filename = season + '_ICESat_gridded_mean_thickness_sorted.txt'
            dataset_filepath = path.join(self.h_ice_data_dir_path, dataset_filename)

            with open(dataset_filepath, 'rt') as f:
                reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
                for i, line in enumerate(reader):
                    sea_ice_freeboard = line[2]
                    sea_ice_thickness = line[3]

                    if sea_ice_thickness == '-999':
                        self.h_ice_seasonal[season][i] = np.nan
                    else:
                        self.h_ice_seasonal[season][i] = float(sea_ice_thickness)

        from constants import lat_min, lat_max, n_lat, lon_min, lon_max, n_lon
        lats_array = np.linspace(lat_min, lat_max, n_lat)
        lons_array = np.linspace(lon_min, lon_max, n_lon)

        self.closest_point_idx = np.zeros((len(lats_array), len(lons_array)))

        logger.info('Computing (lat, lon) -> closest_point_idx(lat, lon) map...')
        for i in range(len(lats_array)):
            lat = lats_array[i]
            for j in range(len(lons_array)):
                lon = lons_array[j]

                if lon < 0:
                    lon = lon + 360

                lat_start_idx = np.searchsorted(self.lats, lat)

                delta_idx = 250
                idx1 = max(0, lat_start_idx - delta_idx)
                idx2 = min(lat_start_idx + delta_idx, len(self.lats))

                point = np.array([lat, lon])
                points = np.column_stack((self.lats[idx1:idx2], self.lons[idx1:idx2]))

                self.closest_point_idx[i][j] = idx1 + cdist([point], points).argmin()

    def load_h_ice_dataset(self):
        logger.info('Loading sea ice concentration dataset: {}'.format(self.dataset_filepath))

        with open(self.dataset_filepath, 'rt') as f:
            reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
            for i, line in enumerate(reader):
                self.lats[i] = float(line[0])
                self.lons[i] = float(line[1])

                sea_ice_freeboard = line[2]
                sea_ice_thickness = line[3]

                if sea_ice_thickness == '-999':
                    self.h_ice[i] = np.nan
                else:
                    self.h_ice[i] = float(sea_ice_thickness)

        logger.info('Successfully loaded sea ice thickness dataset: {}'.format(self.dataset_filepath))

    def sea_ice_thickness(self, lat, lon, date=None):
        if date is not None:
            if 1 <= date.month <= 3:
                season = 'summer'
            elif 4 <= date.month <= 6:
                season = 'fall'
            elif 7 <= date.month <= 12:
                season = 'spring'

            closest_idx = int(self.closest_point_idx[lat][lon])
            return self.h_ice_seasonal[season][closest_idx]

        if lon < 0:
            lon = lon + 360

        lat_start_idx = np.searchsorted(self.lats, lat)

        delta_idx = 250
        idx1 = max(0, lat_start_idx - delta_idx)
        idx2 = min(lat_start_idx + delta_idx, len(self.lats))

        point = np.array([lat, lon])
        points = np.column_stack((self.lats[idx1:idx2], self.lons[idx1:idx2]))

        closest_point_idx = idx1 + cdist([point], points).argmin()

        return self.h_ice[closest_point_idx]
