import datetime
import calendar

import numpy as np

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig('logging.ini')
logger = logging.getLogger(__name__)

from SurfaceStressDataWriter import SurfaceStressDataWriter
from utils import date_range, get_netCDF_filepath

np.set_printoptions(precision=4)


def produce_monthly_mean(date_in_month):
    month = date_in_month.month
    year = date_in_month.year

    date1 = datetime.date(year, month, 1)
    n_days = calendar.monthrange(date1.year, date1.month)[1]
    date2 = datetime.date(year, month, n_days)
    dates = date_range(date1, date2)

    surface_stress_dataset = SurfaceStressDataWriter(field_type='monthly', date=dates[0])
    surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='monthly')
    surface_stress_dataset.write_fields_to_netcdf()


def produce_annual_mean(year):
    dates = date_range(datetime.date(year, 1, 1), datetime.date(year, 12, 31))

    surface_stress_dataset = SurfaceStressDataWriter(field_type='annual', date=dates[0])
    surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='annual')
    surface_stress_dataset.write_fields_to_netcdf()


def produce_seasonal_mean(seasons_to_compute, year):
    seasons = {
        'JFM': {'date1': datetime.date(year, 1, 1),
                'date2': datetime.date(year, 3, 31),
                'label': 'Summer_JFM_' + str(year) + '_average'},
        'AMJ': {'date1': datetime.date(year, 4, 1),
                'date2': datetime.date(year, 6, 30),
                'label': 'Fall_AMJ_' + str(year) + '_average'},
        'JAS': {'date1': datetime.date(year, 7, 1),
                'date2': datetime.date(year, 9, 30),
                'label': 'Winter_JAS_' + str(year) + '_average'},
        'OND': {'date1': datetime.date(year, 10, 1),
                'date2': datetime.date(year, 12, 31),
                'label': 'Spring_JFM_' + str(year) + '_average'}
    }

    for s in seasons_to_compute:
        logger.info('s={:s}'.format(s))
        dates = date_range(seasons[s]['date1'], seasons[s]['date2'])

        surface_stress_dataset = SurfaceStressDataWriter(None)
        surface_stress_dataset.date = dates[0]
        surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
        surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=seasons[s]['label'])
        surface_stress_dataset.write_fields_to_netcdf(field_type='seasonal', season_str=s)

    # Process summer (DJF) separately.
    # dec = date_range(datetime.date(year - 1, 12, 1), datetime.date(year - 1, 12, 31))
    # janfeb = date_range(datetime.date(year, 1, 1), datetime.date(year, 2, 28))  # TODO: Feb 29
    # dates = dec + janfeb
    # custom_label = 'Summer_DJF_' + str(year - 1) + '-' + str(year) + '_average'
    #
    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.date = dates[0]
    # surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
    # surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=custom_label)
    # surface_stress_dataset.write_fields_to_netcdf(field_type='seasonal')


def produce_seasonal_climatology(seasons, year_start, year_end):
    year_range = str(year_start) + '-' + str(year_end)

    labels = {
        'DJF': 'Summer_DJF_' + year_range + '_average',
        'MAM': 'Fall_MAM_' + year_range + '_average',
        'JJA': 'Winter_JJA_' + year_range + '_average',
        'SON': 'Spring_SON_' + year_range + '_average',
        'JFM': 'Summer_JFM_' + year_range + '_average',
        'AMJ': 'Fall_AMJ_' + year_range + '_average',
        'JAS': 'Winter_JAS_' + year_range + '_average',
        'OND': 'Spring_OND_' + year_range + '_average'
    }

    season_start_month = {
        'DJF': 12,
        'MAM': 3,
        'JJA': 6,
        'SON': 9,
        'JFM': 1,
        'AMJ': 4,
        'JAS': 7,
        'OND': 10
    }

    season_end_month = {
        'DJF': 2,
        'MAM': 5,
        'JJA': 8,
        'SON': 11,
        'JFM': 3,
        'AMJ': 6,
        'JAS': 9,
        'OND': 12
    }

    for season in seasons:
        season_days = []

        for year in range(year_start, year_end + 1):
            start_date = datetime.date(year, season_start_month[season], 1)

            end_month_days = calendar.monthrange(year, season_end_month[season])[1]
            end_date = datetime.date(year, season_end_month[season], end_month_days)

            season_days = season_days + date_range(start_date, end_date)

        surface_stress_dataset = SurfaceStressDataWriter(field_type='seasonal_climo', season_str=season,
                                                         year_start=year_start, year_end=year_end)
        surface_stress_dataset.date = season_days[0]

        surface_stress_dataset.compute_mean_fields(season_days, avg_method='partial_data_ok')

        surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=labels[season])
        surface_stress_dataset.write_fields_to_netcdf()


def produce_monthly_climatology(months, year_start, year_end):
    import calendar
    from utils import get_netCDF_filepath

    year_range = str(year_start) + '-' + str(year_end)

    for month in months:
        label = calendar.month_abbr[month] + '_' + year_range + '_average'

        month_days = []
        for year in range(year_start, year_end + 1):
            n_days = calendar.monthrange(year, month)[1]
            month_days = month_days + date_range(datetime.date(year, month, 1), datetime.date(year, month, n_days))

        avg_period = str(month).zfill(2)
        tau_fp = get_netCDF_filepath(field_type='monthly_climo', date=datetime.date(year, month, 1),
                                     year_start=year_start, year_end=year_end)

        surface_stress_dataset = SurfaceStressDataWriter(field_type='monthly_climo', date=month_days[-1],
                                                         year_start=year_start, year_end=year_end)
        surface_stress_dataset.date = month_days[-1]

        surface_stress_dataset.compute_mean_fields(month_days, avg_method='partial_data_ok')

        surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=label)
        surface_stress_dataset.write_fields_to_netcdf()


def produce_climatology(year_start, year_end):
    from utils import get_netCDF_filepath

    climo_label = str(year_start) + '-' + str(year_end) + '_average'

    dates = date_range(datetime.date(year_start, 1, 1), datetime.date(year_end, 12, 31))

    surface_stress_dataset = SurfaceStressDataWriter(field_type='climo', year_start=year_start, year_end=year_end)
    surface_stress_dataset.date = dates[0]

    surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')

    surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=climo_label)
    surface_stress_dataset.write_fields_to_netcdf()


if __name__ == '__main__':
    # produce_monthly_mean(datetime.date(2015, 10, 1))
    # produce_monthly_climatology([2, 9], 2005, 2015)
    # produce_monthly_climatology([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 2005, 2012)

    # produce_seasonal_climatology(['JFM', 'AMJ', 'JAS', 'OND'], 2005, 2015)
    # for year in range(1992, 2016):
    #     produce_seasonal_mean(['JFM'], year)

    # produce_annual_mean(2000)
    # for year in range(2005, 2014):
    #     produce_annual_mean(year)

    produce_climatology(2011, 2016)
