import numpy as np
import netCDF4

import datetime

from os import path
import logging.config
import logging

cwd = path.dirname(path.abspath(__file__))  # Current Working Directory

logging_config_path = path.join(cwd, 'logging.ini')
logging.config.fileConfig(logging_config_path)
logger = logging.getLogger(__name__)

# Data directory string constants
# Everything under a ./data/ softlink?
data_dir_path = path.join(cwd, 'data/')

# Physical constants
# Could use a theoretical gravity model: https://en.wikipedia.org/wiki/Theoretical_gravity
g = 9.80665  # standard acceleration due to gravity [m/s^2]
Omega = 7.292115e-5  # rotation rate of the Earth [rad/s]

# Earth radius R could be upgraded to be location-dependent using a simple formula.
# See: https://en.wikipedia.org/wiki/Earth_radius#Location-dependent_radii
R = 6371e3  # average radius of the earth [m]


def distance(ϕ1, λ1, ϕ2, λ2):
    # Calculate the distance between two points on the Earth (ϕ1, λ1) and (ϕ1, λ1) using the haversine formula.
    # See: http://www.movable-type.co.uk/scripts/latlong.html
    # Latitudes are denoted by ϕ while longitudes are denoted by λ.

    ϕ1, λ1, ϕ2, λ2 = np.deg2rad([ϕ1, λ1, ϕ2, λ2])
    Δϕ = ϕ2 - ϕ1
    Δλ = λ2 - λ1

    a = np.sin(Δϕ/2)**2 + np.cos(ϕ1) * np.cos(ϕ2) * np.sin(Δλ/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c


class MDTDataset(object):
    MDT_directory = 'mdt_cnes_cls2013_global'
    MDT_filename = 'mdt_cnes_cls2013_global.nc'
    MDT_file_path = path.join(data_dir_path, MDT_directory, MDT_filename)

    def __init__(self):
        logger.info('MDT class initializing. Loading MDT dataset: %s', self.MDT_filename)
        self.MDT_dataset = netCDF4.Dataset(self.MDT_file_path)

        logger.info('Successfully loaded MDT dataset: %s', self.MDT_filename)
        logger.info('Title: %s', self.MDT_dataset.title)
        logger.info('Data model: %s', self.MDT_dataset.data_model)

        # Nicely display dimension names and sizes in log.
        dim_string = ""
        for dim in self.MDT_dataset.dimensions:
            dim_name = self.MDT_dataset.dimensions[dim].name
            dim_size = self.MDT_dataset.dimensions[dim].size
            dim_string = dim_string + dim_name + '(' + str(dim_size) + ') '
        logger.info('Dimensions: %s', dim_string)

    def get_MDT(self, lat, lon):
        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert 0 <= lon <= 360, "Longitude value {} out of bounds!".format(lon)

        # Nearest neighbour interpolation
        # Find index of closest matching latitude and longitude
        idx_lat = np.abs(np.array(self.MDT_dataset.variables['lat']) - lat).argmin()
        idx_lon = np.abs(np.array(self.MDT_dataset.variables['lon']) - lon).argmin()

        logger.debug("lat = %f, lon = %f", lat, lon)
        logger.debug("idx_lat = %d, idx_lon = %d", idx_lat, idx_lon)
        logger.debug("lat[idx_lat] = %f, lon[idx_lon] = %f", self.MDT_dataset.variables['lat'][idx_lat], self.MDT_dataset.variables['lon'][idx_lon])

        MDT_value = self.MDT_dataset.variables['mdt'][0][idx_lat][idx_lon]

        return MDT_value

    def u_geo_mean(self, lat, lon):
        # Calculate the x and y derivatives of MDT at the grid point ij using a second-order centered finite difference
        # approximation.

        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert 0 <= lon <= 360, "Longitude value {} out of bounds!".format(lon)

        dLat = self.MDT_dataset.variables['lat'][1] - self.MDT_dataset.variables['lat'][0]
        dLon = self.MDT_dataset.variables['lon'][1] - self.MDT_dataset.variables['lon'][0]

        logger.debug("dLat = %f, dLon = %f", dLat, dLon)

        # TODO: Set up wrapping/periodicity in the horizontal for calculating those edge cases.
        # TODO: Check that you're not getting back land values or something.
        MDT_ip1j = self.get_MDT(lat+dLat, lon)
        MDT_im1j = self.get_MDT(lat-dLat, lon)
        MDT_ijp1 = self.get_MDT(lat, lon+dLon)
        MDT_ijm1 = self.get_MDT(lat, lon-dLon)

        logger.debug("MDT_i+1,j = %f, MDT_i-1,j = %f, MDT_i,j+1 = %f, MDT_i,j-1 = %f,",
                     MDT_ip1j, MDT_im1j, MDT_ijp1, MDT_ijm1)

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


class SeaIceDataset(object):
    sea_ice_dir_path = path.join(data_dir_path, 'NOAA_NSIDC_G02202_V3_SEA_ICE_CONCENTRATION', 'south', 'daily')

    def date2filepath(self, date):
        filename = 'seaice_conc_daily_sh_f17_' + str(date.year) + str(date.month).zfill(2) + str(date.day)\
                   + '_v03r00.nc'
        return path.join(self.sea_ice_dir_path, str(date.year), filename)

    def latlon2xy(self, lat, lon):
        lat, lon = np.deg2rad([lat, lon])

        # k_0 = 1.0  # scale factor in all directions apparently
        # lon_0 = 0.0  # location of the prime meridian apparently?
        #
        # RR = 6378.273e3
        # e = 0.081816153
        #
        # chi = 2*np.arctan(np.tan(np.pi/4 + lat/2) * ((1 - e*np.sin(lat)) / (1 + e*np.sin(lat)))**(e/2)) - np.pi/2
        #
        # k = 2*k_0 / (1 - np.sin(chi))
        # x = 2*RR*k * np.tan(np.pi/4 + chi/2) * np.sin(lon - lon_0)
        # y = 2*RR*k * np.tan(np.pi/4 + chi/2) * np.cos(lon - lon_0)

        sgn = -1
        e = 0.081816153
        R_E = 6378.273e3
        slat = 70

        t = np.tan(np.pi/4 - lat/2) / ((1 - e*np.sin(lat)) / (1 + e*np.sin(lat)))**(e/2)

        if np.abs(90 - lat) < 1e-5:
            rho = 2*R_E*t / np.sqrt((1+e)**(1+e) * (1-e)**(1-e))
        else:
            sl = slat * np.pi/180
            t_c = np.tan(np.pi/4 - sl/2) / ((1 - e*np.sin(sl)) / (1 + e*np.sin(sl)))**(e/2)
            m_c = np.cos(sl) / np.sqrt(1 - e*e * (np.sin(sl)**2))
            rho = R_E * m_c * (t/t_c)
            logger.debug("rho = %f, m_c = %f, t = %f, t_c = %f", rho, m_c, t, t_c)

        x = rho * sgn * np.sin(sgn * lon)
        y = -rho * sgn * np.cos(sgn * lon)

        return x, y

    def __init__(self):
        test_dataset_date = datetime.date(2015, 7, 31)
        test_dataset_filepath = self.date2filepath(test_dataset_date)
        logger.info('Sea ice class initializing. Loading sample sea ice dataset: %s', test_dataset_filepath)
        self.sea_ice_dataset = netCDF4.Dataset(test_dataset_filepath)

        logger.info('Successfully loaded sample sea ice dataset: %s', test_dataset_filepath)
        logger.info('Title: %s', self.sea_ice_dataset.title)
        logger.info('Data model: %s', self.sea_ice_dataset.data_model)

        # Nicely display dimension names and sizes in log.
        dim_string = ""
        for dim in self.sea_ice_dataset.dimensions:
            dim_name = self.sea_ice_dataset.dimensions[dim].name
            dim_size = self.sea_ice_dataset.dimensions[dim].size
            dim_string = dim_string + dim_name + '(' + str(dim_size) + ') '
        logger.info('Dimensions: %s', dim_string)

    def sea_ice_concentration(self, lat, lon, day):
        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert -180 <= lon <= 180, "Longitude value {} out of bounds!".format(lon)

        # max_i = min(self.sea_ice_dataset.variables['latitude'].shape)
        # lat_diag = np.array([self.sea_ice_dataset.variables['latitude'][i][i] for i in range(max_i)])
        #
        # lat_idx = np.abs(lat_diag - lat).argmin()
        # lon_idx = np.abs(np.array(self.sea_ice_dataset.variables['longitude'][lat_idx]) - lon).argmin()
        #
        # lat_ij = self.sea_ice_dataset.variables['latitude'][lon_idx][lat_idx]
        # lon_ij = self.sea_ice_dataset.variables['longitude'][lat_idx][lon_idx]
        #
        logger.debug("lat = %f, lon = %f", lat, lon)
        # logger.debug("lat_idx = %d, lon_idx = %d", lat_idx, lon_idx)
        # logger.debug("lat_ij = %f, lon_ij = %f", lat_ij, lon_ij)

        x, y = self.latlon2xy(lat, lon)
        idx_x = np.abs(np.array(self.sea_ice_dataset.variables['xgrid']) - x).argmin()
        idx_y = np.abs(np.array(self.sea_ice_dataset.variables['ygrid']) - y).argmin()
        lat_xy = self.sea_ice_dataset.variables['latitude'][idx_y][idx_x]
        lon_xy = self.sea_ice_dataset.variables['longitude'][idx_y][idx_x]

        logger.debug("x = %f, y = %f", x, y)
        logger.debug("idx_x = %d, idx_y = %d", idx_x, idx_y)
        logger.debug("lat_xy = %f, lon_xy = %f", lat_xy, lon_xy)


if __name__ == '__main__':
    dist = distance(24, 25, 26, 27)
    logger.info("That distance is %f m or %f km.", dist, dist/1000)

    MDT = MDTDataset()
    print(MDT.get_MDT(-60, 135+180))
    print(MDT.u_geo_mean(-60, 135+180))

    sea_ice = SeaIceDataset()
    print(sea_ice.sea_ice_concentration(60.0, 133.0, datetime.date(2015, 7, 31)))
