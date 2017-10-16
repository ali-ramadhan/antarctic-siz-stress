# TODO: Use the typing module.
# TODO: Add test case units?
# TODO: Split into multiple files.
# TODO: Switch from printf style logging to Python3 style formatting.

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

        # TODO: log variables as well.

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
        filename = 'seaice_conc_daily_sh_f17_' + str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)\
                   + '_v03r00.nc'
        return path.join(self.sea_ice_dir_path, str(date.year), filename)

    def latlon2xy(self, lat, lon):
        # TODO: Move to module-level or to util?

        # This subroutine converts from geodetic latitude and longitude to Polar
        # Stereographic (X,Y) coordinates for the polar regions.  The equations
        # are from Snyder, J. P., 1982,  Map Projections Used by the U.S.
        # Geological Survey, Geological Survey Bulletin 1532, U.S. Government
        # Printing Office.  See JPL Technical Memorandum 3349-85-101 for further
        # details.

        # The original equations are from Snyder, J. P., 1982,  Map Projections Used by the U.S. Geological Survey,
        # Geological Survey Bulletin 1532, U.S. Government Printing Office. See JPL Technical Memorandum 3349-85-101 for
        # further details.

        # Original FORTRAN program written by C. S. Morris, April 1985, Jet Propulsion Laboratory, California
        # Institute of Technology

        # The polar stereographic formulae for converting between latitude/longitude and x-y grid coordinates have been
        # taken from map projections used by the U.S. Geological Survey (Snyder 1982).

        # http://nsidc.org/data/polar-stereo/ps_grids.html
        # http://nsidc.org/data/polar-stereo/tools_geo_pixel.html
        # https://nsidc.org/data/docs/daac/nsidc0001_ssmi_tbs/ff.html

        # SSM/I: Special Sensor Microwave Imager

        # WARNING: lat must be positive for the southern hemisphere! Or take absolute value like below.

        sgn = -1  # Sign of the latitude (use +1 for northern hemisphere, -1 for southern)
        e = 0.081816153  # Eccentricity of the Hughes ellipsoid
        R_E = 6378.273e3  # Radius of the Hughes ellipsode [m]
        slat = 70  # Standard latitude for the SSM/I grids is 70 degrees.

        # delta is the meridian offset for the SSM/I grids (0 degrees for the South Polar grids; 45 degrees for the
        # North Polar grids).
        delta = 45 if sgn == 1 else 0

        lat, lon = np.deg2rad([abs(lat), lon+delta])

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
        # TODO: time-dependent lookup.
        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert -180 <= lon <= 180, "Longitude value {} out of bounds!".format(lon)

        x, y = self.latlon2xy(lat, lon)
        idx_x = np.abs(np.array(self.sea_ice_dataset.variables['xgrid']) - x).argmin()
        idx_y = np.abs(np.array(self.sea_ice_dataset.variables['ygrid']) - y).argmin()
        lat_xy = self.sea_ice_dataset.variables['latitude'][idx_y][idx_x]
        lon_xy = self.sea_ice_dataset.variables['longitude'][idx_y][idx_x]

        # TODO: Add a logger.warning if lat_xy, lon_xy don't match lat, lon very closely.
        logger.debug("lat = %f, lon = %f", lat, lon)
        logger.debug("x = %f, y = %f", x, y)
        logger.debug("idx_x = %d, idx_y = %d", idx_x, idx_y)
        logger.debug("lat_xy = %f, lon_xy = %f", lat_xy, lon_xy)

        sea_ice_alpha = self.sea_ice_dataset.variables['goddard_nt_seaice_conc'][0][idx_y][idx_x]

        return sea_ice_alpha


class WindDataset(object):
    wind_dir_path = path.join(data_dir_path, 'CCMP_MEASURES_ATLAS_L4_OW_L3_0_WIND_VECTORS_FLK')

    def date2filepath(self, date):
        filename = 'analysis_' + str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2) + '_v11l30flk.nc'
        return path.join(self.wind_dir_path, str(date.year), str(date.month).zfill(2), filename)

    def __init__(self):
        test_dataset_date = datetime.date(2011, 12, 31)
        test_dataset_filepath = self.date2filepath(test_dataset_date)
        logger.info('WindDataset class initializing. Loading sample wind vector dataset: %s', test_dataset_filepath)
        self.wind_dataset = netCDF4.Dataset(test_dataset_filepath)
        self.wind_dataset.set_auto_mask(False)  # TODO: Why is most of the data masked?

        logger.info('Successfully loaded sample wind vector dataset: %s', test_dataset_filepath)
        logger.info('Title: %s', self.wind_dataset.title)
        logger.info('Data model: %s', self.wind_dataset.data_model)

        # Nicely display dimension names and sizes in log.
        dim_string = ""
        for dim in self.wind_dataset.dimensions:
            dim_name = self.wind_dataset.dimensions[dim].name
            dim_size = self.wind_dataset.dimensions[dim].size
            dim_string = dim_string + dim_name + '(' + str(dim_size) + ') '
        logger.info('Dimensions: %s', dim_string)

    def surface_wind_vector(self, lat, lon, day):
        idx_lat = np.abs(np.array(self.wind_dataset.variables['lat']) - lat).argmin()
        idx_lon = np.abs(np.array(self.wind_dataset.variables['lon']) - lon).argmin()

        logger.debug("lat = %f, lon = %f", lat, lon)
        logger.debug("idx_lat = %d, idx_lon = %d", idx_lat, idx_lon)
        logger.debug("lat[idx_lat] = %f, lon[idx_lon] = %f", self.wind_dataset.variables['lat'][idx_lat],
                     self.wind_dataset.variables['lon'][idx_lon])

        u_wind = self.wind_dataset.variables['uwnd'][0][idx_lat][idx_lon]
        v_wind = self.wind_dataset.variables['vwnd'][0][idx_lat][idx_lon]

        return np.array([u_wind, v_wind])


if __name__ == '__main__':
    dist = distance(24, 25, 26, 27)
    logger.info("That distance is %f m or %f km.", dist, dist/1000)

    MDT = MDTDataset()
    print(MDT.get_MDT(-60, 135+180))
    print(MDT.u_geo_mean(-60, 135+180))

    sea_ice = SeaIceDataset()
    print(sea_ice.sea_ice_concentration(-60.0, 133.0, datetime.date(2015, 7, 31)))
    print(sea_ice.sea_ice_concentration(-71.4, 24.5, datetime.date(2015, 7, 31)))
    print(sea_ice.sea_ice_concentration(-70, 180, datetime.date(2015, 7, 31)))

    wind_vectors = WindDataset()
    print(wind_vectors.surface_wind_vector(-60, 20, datetime.date(2011, 12, 31)))