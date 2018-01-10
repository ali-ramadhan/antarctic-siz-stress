# TODO: Use the typing module.
# TODO: Use propoer docstrings for functions.
# TODO: Estimate tau_error? Can you? NCEP Reanalysis doesn't really provide a "measurement error".
# TODO: Output more statistics during the analysis?
# TODO: Plot everything but draw the ice line where alpha drops below 0.15.
# TODO: Plot the zero stress line. We expect a unique position for it right?

# Conventions
# Latitude = -90 to +90
# Longitude = -180 to 180

import datetime
import calendar

import numpy as np
from joblib import Parallel, delayed

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config
logging.config.fileConfig('logging.ini')
logger = logging.getLogger(__name__)

np.set_printoptions(precision=4)


def process_day(date):
    from SurfacetressDataWriter import SurfaceStressDataWriter
    surface_stress_dataset = SurfaceStressDataWriter(date)

    surface_stress_dataset.compute_daily_surface_stress_field()
    surface_stress_dataset.compute_daily_ekman_pumping_field()
    surface_stress_dataset.write_fields_to_netcdf()
    surface_stress_dataset.plot_diagnostic_fields(plot_type='daily')


if __name__ == '__main__':
    from SurfacetressDataWriter import SurfaceStressDataWriter
    from utils import date_range

    """ Making sure that sea ice motion fields interpolate properly. """
    # from SeaIceMotionDataReader import SeaIceMotionDataReader
    # sic = SeaIceMotionDataReader(datetime.date(2015, 7, 1))
    # sic.plot_sea_ice_motion_vector_field()

    """ Process only July 16, 2015 """
    # process_day(datetime.date(2015, 7, 16))

    """ Process July 1-31, 2015 and produce a monthly average. """
    # date_in_month = datetime.date(2015, 8, 1)
    # n_days = calendar.monthrange(date_in_month.year, date_in_month.month)[1]
    #
    # Parallel(n_jobs=5)(delayed(process_day)(datetime.date(date_in_month.year, date_in_month.month, day))
    #                    for day in range(1, n_days+1))
    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.compute_monthly_mean_fields(date_in_month, method='partial_data_ok')

    """ Process all of 2015 """
    # for month in [12]:
    #     date_in_month = datetime.date(2015, month, 1)
    #     n_days = calendar.monthrange(date_in_month.year, date_in_month.month)[1]
    #
    #     Parallel(n_jobs=4)(delayed(process_day)(datetime.date(date_in_month.year, date_in_month.month, day))
    #                        for day in range(1, n_days+1))

    # Monthly means
    # for month in [12]: # range(1, 13):
    #     date1 = datetime.date(2015, month, 1)
    #     n_days = calendar.monthrange(date1.year, date1.month)[1]
    #     date2 = datetime.date(2015, month, n_days-1)
    #     dates = date_range(date1, date2)
    #
    #     surface_stress_dataset = SurfaceStressDataWriter(None)
    #     surface_stress_dataset.date = date1
    #     surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
    #     surface_stress_dataset.plot_diagnostic_fields(plot_type='monthly')
    #     # surface_stress_dataset.write_fields_to_netcdf()  # TODO: Add support for mean field netCDF files.

    # Annual mean
    # dates = date_range(datetime.date(2015, 1, 1), datetime.date(2015, 12, 30))
    #
    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.date = dates[0]
    # surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
    # surface_stress_dataset.plot_diagnostic_fields(plot_type='annual')

    # JJA, SON, DJF, and MAM seasonal averages
    seasons = [
        # {'date1': datetime.date(2015, 3, 1), 'date2': datetime.date(2015, 5, 31), 'label': 'Autumn (MAM) 2015 average'},
        {'date1': datetime.date(2015, 6, 1), 'date2': datetime.date(2015, 8, 31), 'label': 'Winter (JJA) 2015 average'},
        # {'date1': datetime.date(2015, 9, 1), 'date2': datetime.date(2015, 11, 30), 'label': 'Spring (SON) 2015 average'}
    ]
    for season in seasons:
        dates = date_range(season['date1'], season['date2'])

        surface_stress_dataset = SurfaceStressDataWriter(None)
        surface_stress_dataset.date = dates[0]
        surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
        surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=season['label'])

    # July climo mean (1995-2015?)
