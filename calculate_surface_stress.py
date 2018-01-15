# TODO: Use the typing module.
# TODO: Use propoer docstrings for functions.
# TODO: Estimate tau_error? Can you? NCEP Reanalysis doesn't really provide a "measurement error".
# TODO: Output more statistics during the analysis?
# TODO: Plot everything but draw the ice line where alpha drops below 0.15.
# TODO: Plot the zero stress line. We expect a unique position for it right?

# Conventions
# Latitude = -90 (90 S) to +90 (90 N)
# Longitude = -180 (180 W) to 180 (180 E)

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

    """ Double checking some distances """
    # from utils import distance
    # logger.info('distance(-89.999, 0, -89.999, 180) = {}'.format(distance(-89.999, 0, -89.999, 180)))
    # logger.info('2*distance(0, 0, 0, 180) = {}'.format(2*distance(0, 0, 0, 180)))
    # logger.info('distance(-80, 74, -80, 74.5) = {}'.format(distance(-80, 74, -80, 74.5)))
    # logger.info('distance(-80, 74.5, -80, 74) = {}'.format(distance(-80, 74.5, -80, 74)))
    # logger.info('distance(-80, 74, -80.5, 74) = {}'.format(distance(-80, 74, -80.5, 74)))
    # logger.info('distance(-80.5, 74, -80, 74) = {}'.format(distance(-80.5, 74, -80, 74)))

    """ Making sure that sea ice motion fields interpolate properly. """
    # from SeaIceMotionDataset import SeaIceMotionDataset
    # sic = SeaIceMotionDataset(datetime.date(2015, 7, 1))
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
    # for month in [6, 7, 8]:
    #     date_in_month = datetime.date(2015, month, 1)
    #     n_days = calendar.monthrange(date_in_month.year, date_in_month.month)[1]
    #
    #     Parallel(n_jobs=4)(delayed(process_day)(datetime.date(date_in_month.year, date_in_month.month, day))
    #                        for day in range(1, n_days+1))

    # Monthly means
    # for month in [2]:
    #     date1 = datetime.date(2014, month, 1)
    #     n_days = calendar.monthrange(date1.year, date1.month)[1]
    #     date2 = datetime.date(2014, month, n_days)
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
    # for year in [2015]:
    #     seasons = [
    #         # {'date1': datetime.date(year, 3, 1),
    #         #  'date2': datetime.date(year, 5, 31),
    #         #  'label': 'Autumn_MAM_' + str(year) + '_average'},
    #         {'date1': datetime.date(year, 6, 1),
    #          'date2': datetime.date(year, 8, 31),
    #          'label': 'Winter_JJA_' + str(year) + '_average'},
    #         # {'date1': datetime.date(year, 9, 1),
    #         #  'date2': datetime.date(year, 11, 30),
    #         #  'label': 'Spring_SON_' + str(year) + '_average'}
    #     ]
    #
    #     for season in seasons:
    #         dates = date_range(season['date1'], season['date2'])
    #
    #         surface_stress_dataset = SurfaceStressDataWriter(None)
    #         surface_stress_dataset.date = dates[0]
    #         surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
    #         surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=season['label'])

    # jan = date_range(datetime.date(year+1, 1, 1), datetime.date(year+1, 1, 31))
    # novdec = date_range(datetime.date(year, 11, 1), datetime.date(year, 12, 31))
    # dates = jan + novdec
    # custom_label = 'Summer_DJF_' + str(year) + '-' + str(year+1) + '_average'
    #
    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.date = dates[0]
    # surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
    # surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=custom_label)

    """ Process all of 1995-2014 """
    # for year in range(2014, 1994, -1):
    #     for month in range(1, 13):
    #         date_in_month = datetime.date(year, month, 1)
    #         n_days = calendar.monthrange(date_in_month.year, date_in_month.month)[1]
    #
    #         Parallel(n_jobs=8)(delayed(process_day)(datetime.date(date_in_month.year, date_in_month.month, day))
    #                            for day in range(1, n_days+1))

    """ Climatological seasonal averages"""
    # fall_days = []
    # winter_days = []
    # spring_days = []
    # summer_days = []
    #
    # for year in range(2006, 2015):
    #     fall_days = fall_days + date_range(datetime.date(year, 3, 1), datetime.date(year, 5, 31))
    #     winter_days = winter_days + date_range(datetime.date(year, 6, 1), datetime.date(year, 8, 31))
    #     spring_days = spring_days + date_range(datetime.date(year, 9, 1), datetime.date(year, 11, 30))
    #
    #     summer_days = summer_days + date_range(datetime.date(year, 12, 1), datetime.date(year, 12, 31)) \
    #                   + date_range(datetime.date(year+1, 1, 1), datetime.date(year+1, 2, 28))

    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.date = winter_days[-1]
    # surface_stress_dataset.compute_mean_fields(winter_days, avg_method='partial_data_ok')
    # surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label='Winter_JJA_2006-15_average')

    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.date = summer_days[-1]
    # surface_stress_dataset.compute_mean_fields(summer_days, avg_method='partial_data_ok')
    # surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label='Summer_DJF_2006-15_average')

    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.date = fall_days[-1]
    # surface_stress_dataset.compute_mean_fields(fall_days, avg_method='partial_data_ok')
    # surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label='Fall_MAM_2006-15_average')
    #
    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.date = spring_days[-1]
    # surface_stress_dataset.compute_mean_fields(spring_days, avg_method='partial_data_ok')
    # surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label='Spring_SON_2006-15_average')

    """ Monthly climatologies """
    # dec_days = []
    # jan_days = []
    # feb_days = []
    # sep_days = []
    # for year in range(2006, 2016):
    #     dec_days = dec_days + date_range(datetime.date(year, 12, 1), datetime.date(year, 12, 31))
    #     jan_days = jan_days + date_range(datetime.date(year, 1, 1), datetime.date(year, 1, 31))
    #     feb_days = feb_days + date_range(datetime.date(year, 2, 1), datetime.date(year, 2, 28))
    #     sep_days = sep_days + date_range(datetime.date(year, 9, 1), datetime.date(year, 9, 30))
    #
    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.date = dec_days[-1]
    # surface_stress_dataset.compute_mean_fields(dec_days, avg_method='partial_data_ok')
    # surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label='Dec_2006-15_average')
    #
    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.date = jan_days[-1]
    # surface_stress_dataset.compute_mean_fields(jan_days, avg_method='partial_data_ok')
    # surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label='Jan_2006-15_average')

    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.date = feb_days[-1]
    # surface_stress_dataset.compute_mean_fields(feb_days, avg_method='partial_data_ok')
    # surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label='Feb_2006-15_average')

    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.date = sep_days[-1]
    # surface_stress_dataset.compute_mean_fields(sep_days, avg_method='partial_data_ok')
    # surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label='Sep_2006-15_average')
