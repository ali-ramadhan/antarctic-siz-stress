import numpy as np
from joblib import Parallel, delayed

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig('logging.ini')
logger = logging.getLogger(__name__)

np.set_printoptions(precision=4)


def process_neutral_density_field(time_span, avg_period, grid_size, field_type, depth_level):
    from NeutralDensityDataset import NeutralDensityDataset

    try:
        NeutralDensityDataset(time_span, avg_period, grid_size, field_type, depth_level)
    except Exception as e:
        logger.error('Failed to process neutral density ({}, {}, {}, {}, {}). Returning.'
                     .format(time_span, avg_period, grid_size, field_type, depth_level))
        logger.error('{}'.format(e), exc_info=True)
        return


def process_neutral_density_fields_multiple_depths(time_span, avg_period, grid_size, field_type, depth_levels):
    n_jobs = len(depth_levels)
    Parallel(n_jobs=n_jobs)(delayed(process_neutral_density_field)(time_span, avg_period, grid_size, field_type, lvl)
                            for lvl in depth_levels)


if __name__ == '__main__':
    process_neutral_density_fields_multiple_depths(time_span='A5B2', avg_period='00', grid_size='04', field_type='an',
                                                   depth_levels=range(8))
    process_neutral_density_fields_multiple_depths(time_span='95A4', avg_period='00', grid_size='04', field_type='an',
                                                   depth_levels=range(8))
    process_neutral_density_fields_multiple_depths(time_span='A5B2', avg_period='13', grid_size='04', field_type='an',
                                                   depth_levels=range(8))
    process_neutral_density_fields_multiple_depths(time_span='A5B2', avg_period='14', grid_size='04', field_type='an',
                                                   depth_levels=range(8))
    process_neutral_density_fields_multiple_depths(time_span='A5B2', avg_period='15', grid_size='04', field_type='an',
                                                   depth_levels=range(8))
    process_neutral_density_fields_multiple_depths(time_span='A5B2', avg_period='16', grid_size='04', field_type='an',
                                                   depth_levels=range(8))