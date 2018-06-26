# TODO: Use the typing module.
# TODO: Use propoer docstrings for functions.
# TODO: Estimate tau_error? Can you? NCEP Reanalysis doesn't really provide a "measurement error".

import datetime
import calendar

import numpy as np
from joblib import Parallel, delayed

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig('logging.ini')
logger = logging.getLogger(__name__)

np.set_printoptions(precision=4)


def check_distances():
    """ Double checking some distances. """
    from utils import distance
    logger.info('distance(-89.999, 0, -89.999, 180) = {}'.format(distance(-89.999, 0, -89.999, 180)))
    logger.info('2*distance(0, 0, 0, 180) = {}'.format(2 * distance(0, 0, 0, 180)))
    logger.info('distance(-80, 74, -80, 74.5) = {}'.format(distance(-80, 74, -80, 74.5)))
    logger.info('distance(-80, 74.5, -80, 74) = {}'.format(distance(-80, 74.5, -80, 74)))
    logger.info('distance(-80, 74, -80.5, 74) = {}'.format(distance(-80, 74, -80.5, 74)))
    logger.info('distance(-80.5, 74, -80, 74) = {}'.format(distance(-80.5, 74, -80, 74)))


def check_sea_ice_motion_field():
    """ Making sure that sea ice motion fields interpolate properly. """
    from SeaIceMotionDataset import SeaIceMotionDataset
    sic = SeaIceMotionDataset(datetime.date(2015, 7, 16))
    sic.plot_sea_ice_motion_vector_field()


def process_day(date):
    """ Process for only one day. """
    from SurfaceStressDataWriter import SurfaceStressDataWriter

    try:
        surface_stress_dataset = SurfaceStressDataWriter(field_type='daily', date=date)

        surface_stress_dataset.compute_daily_surface_stress_field(u_geo_source='CS2')
        surface_stress_dataset.compute_daily_auxillary_fields()

        surface_stress_dataset.write_fields_to_netcdf()
    except Exception as e:
        logger.error('Failed to process day {}. Returning.'.format(date))
        logger.error('{}'.format(e), exc_info=True)
        return


def process_and_plot_day(date):
    """ Process and plot fields for only one day. """

    from SurfaceStressDataWriter import SurfaceStressDataWriter
    surface_stress_dataset = SurfaceStressDataWriter(field_type='daily', date=date)

    surface_stress_dataset.compute_daily_surface_stress_field(u_geo_source='CS2')
    surface_stress_dataset.compute_daily_auxillary_fields()

    surface_stress_dataset.write_fields_to_netcdf()
    surface_stress_dataset.plot_diagnostic_fields(plot_type='daily', custom_label='no_custom_label_')


def plot_day(date):
    from SurfaceStressDataWriter import SurfaceStressDataWriter
    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.date = date
    surface_stress_dataset.compute_mean_fields([date], avg_method='full_data_only')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='daily')


def process_month(date_in_month):
    """ Process one month and produce a monthly average. """
    from utils import date_range

    year = date_in_month.year
    month = date_in_month.month

    n_days = calendar.monthrange(year, month)[1]
    dates = date_range(datetime.date(year, month, 1), datetime.date(year, month, n_days))

    Parallel(n_jobs=5)(delayed(process_day)(datetime.date(year, month, day)) for day in range(1, n_days + 1))

    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.date = date_in_month
    surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='monthly')
    surface_stress_dataset.write_fields_to_netcdf(field_type='monthly')


def process_months_multiple_years(months, year_start, year_end):
    for year in range(year_end, year_start - 1, -1):
        for month in months:
            n_days = calendar.monthrange(year, month)[1]
            Parallel(n_jobs=16)(delayed(process_day)(datetime.date(year, month, day)) for day in range(1, n_days + 1))


def process_year(date_in_year):
    """ Process an entire year, month by month. """

    year = date_in_year.year

    for month in range(1, 13):
        process_month(datetime.date(year, month, 1))


def process_multiple_years(year_start, year_end):
    for year in range(year_end, year_start - 1, -1):
        for month in range(1, 13):
            n_days = calendar.monthrange(year, month)[1]
            Parallel(n_jobs=12)(delayed(process_day)(datetime.date(year, month, day)) for day in range(1, n_days + 1))

            # try:
            #     Parallel(n_jobs=12)(delayed(process_day)(datetime.date(date_in_month.year, date_in_month.month, day))
            #                         for day in range(1, n_days+1))
            # except Exception as e:
            #     logger.error('{}'.format(e), exc_info=True)
            #     continue


if __name__ == '__main__':
    from SurfaceStressDataWriter import SurfaceStressDataWriter
    from utils import date_range

    process_and_plot_day(datetime.date(2015, 2, 16))
    process_and_plot_day(datetime.date(2015, 6, 30))
    process_and_plot_day(datetime.date(2015, 10, 1))
    # process_day(datetime.date(2015, 7, 16))
    # plot_day(datetime.date(2015, 7, 16))
    # process_day(datetime.date(2015, 1, 1))

    # process_multiple_years(1995, 1995)
