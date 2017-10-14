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
        logger.info('Dimensions: \n %s', self.MDT_dataset.dimensions)


if __name__ == '__main__':
    logger.info("Logger works I guess!")
    d = distance(24, 25, 26, 27)
    logger.info("That distance is %f m or %f km.", d, d/1000)
    MDT_dataset = MDTDataset()
