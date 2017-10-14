import numpy as np
import netCDF4

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
g = 9.80665 # standard acceleration due to gravity [m/s^2]
Omega = 7.292115e-5 # rotation rate of the Earth [rad/s]


def distance(ϕ1, λ1, ϕ2, λ2):
    # Calculate the distance between two points on the Earth (ϕ1, λ1) and (ϕ1, λ1) using the haversine formula.
    # See: http://www.movable-type.co.uk/scripts/latlong.html
    # Latitudes are denoted by ϕ while longitudes are denoted by λ.

    # Earth radius R could be upgraded to be location-dependent using a simple formula.
    # See: https://en.wikipedia.org/wiki/Earth_radius#Location-dependent_radii
    R = 6371e3  # average radius of the earth [m]

    ϕ1, λ1, ϕ2, λ2 = np.deg2rad([ϕ1, λ1, ϕ2, λ2])
    Δϕ = ϕ2 - ϕ1
    Δλ = λ2 - λ1

    a = np.sin(Δϕ/2)**2 + np.cos(ϕ1) * np.cos(ϕ2) * np.sin(Δλ/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c


class MDTDataset(object):
    MDT_filename = 'mdt_cnes_cls2013_global.nc'
    MDT_file_path = path.join(data_dir_path, MDT_filename)

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
        idx_lon = np.abs(np.array(self.MDT_dataset.variables['lon']) - lat).argmin()

        MDT_value = self.MDT_dataset.variables['mdt'][0][idx_lat][idx_lon]

        return MDT_value

    def u_geo(self, lat, lon):
        # Calculate the x and y derivatives of MDT at the grid point ij using a second-order centered finite difference
        # approximation.
        MDT_ip1j = self.get_MDT(lat, lon)

        dMDTdx = (MDT_ip1j - MDTim1j) / (2*dx)
        dMDTdy = (MDT_ijp1 - MDT_ijm1) / (2*dy)

        f = 2*Omega*np.sin(np.deg2rad(lat))
        u_geo_u = -(g/f) * dMDTdy
        u_geo_v = (g/f) * dMDTdx

        return np.array([u_geo_u, u_geo_v])


if __name__ == '__main__':
    logger.info("Logger works I guess!")
    d = distance(24, 25, 26, 27)
    logger.info("That distance is %f m or %f km.", d, d/1000)
    MDT = MDTDataset()
    print(MDT.get_MDT(40, 50))
    print(MDT.get_MDT(-60, 50))
