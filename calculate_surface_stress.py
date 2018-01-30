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


def check_distances():
    """ Double checking some distances. """
    from utils import distance
    logger.info('distance(-89.999, 0, -89.999, 180) = {}'.format(distance(-89.999, 0, -89.999, 180)))
    logger.info('2*distance(0, 0, 0, 180) = {}'.format(2*distance(0, 0, 0, 180)))
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
        surface_stress_dataset = SurfaceStressDataWriter(date)

        surface_stress_dataset.compute_daily_surface_stress_field()
        surface_stress_dataset.compute_daily_ekman_pumping_field()
        surface_stress_dataset.write_fields_to_netcdf()

    except Exception as e:
        logger.error('Failed to process day {}. Returning.'.format(date))
        logger.error('{}'.format(e), exc_info=True)
        return


def process_and_plot_day(date):
    """ Process and plot fields for only one day. """
    from SurfaceStressDataWriter import SurfaceStressDataWriter
    surface_stress_dataset = SurfaceStressDataWriter(date)

    surface_stress_dataset.compute_daily_surface_stress_field()
    surface_stress_dataset.compute_daily_ekman_pumping_field()
    surface_stress_dataset.write_fields_to_netcdf()
    surface_stress_dataset.plot_diagnostic_fields(plot_type='daily')


def plot_day(date):
    from SurfaceStressDataWriter import SurfaceStressDataWriter
    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.date = date
    surface_stress_dataset.compute_mean_fields([date], avg_method='partial_data_ok')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='daily')


def process_month(date_in_month):
    """ Process one month and produce a monthly average. """
    from utils import date_range

    year = date_in_month.year
    month = date_in_month.month

    n_days = calendar.monthrange(year, month)[1]
    dates = date_range(datetime.date(year, month, 1), datetime.date(year, month, n_days))

    Parallel(n_jobs=5)(delayed(process_day)(datetime.date(year, month, day)) for day in range(1, n_days+1))

    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.date = date_in_month
    surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='monthly')


def process_year(date_in_year):
    """ Process an entire year and produce an annual mean. """

    year = date_in_year.year

    for month in range(1, 13):
        process_month(datetime.date(year, month, 1))

    produce_annual_mean(date_in_year)


def produce_monthly_mean(date_in_month):
    from utils import date_range

    month = date_in_month.month
    year = date_in_month.year

    date1 = datetime.date(year, month, 1)
    n_days = calendar.monthrange(date1.year, date1.month)[1]
    date2 = datetime.date(year, month, n_days)
    dates = date_range(date1, date2)

    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.date = date1
    surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='monthly')
    # surface_stress_dataset.write_fields_to_netcdf()  # TODO: Add support for mean field netCDF files.


def produce_annual_mean(date_in_year):
    year = date_in_year.year

    dates = date_range(datetime.date(year, 1, 1), datetime.date(year, 12, 31))

    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.date = dates[0]
    surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='annual')


def produce_seasonal_mean(date_in_year):
    year = date_in_year.year

    seasons = [
        {'date1': datetime.date(year, 3, 1),
         'date2': datetime.date(year, 5, 31),
         'label': 'Autumn_MAM_' + str(year) + '_average'},
        {'date1': datetime.date(year, 6, 1),
         'date2': datetime.date(year, 8, 31),
         'label': 'Winter_JJA_' + str(year) + '_average'},
        {'date1': datetime.date(year, 9, 1),
         'date2': datetime.date(year, 11, 30),
         'label': 'Spring_SON_' + str(year) + '_average'}
    ]

    for season in seasons:
        dates = date_range(season['date1'], season['date2'])

        surface_stress_dataset = SurfaceStressDataWriter(None)
        surface_stress_dataset.date = dates[0]
        surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
        surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=season['label'])

    # Process summer (DJF) separately.
    dec = date_range(datetime.date(year-1, 12, 1), datetime.date(year-1, 12, 31))
    janfeb = date_range(datetime.date(year, 1, 1), datetime.date(year, 2, 28))  # TODO: Feb 29
    dates = dec + janfeb
    custom_label = 'Summer_DJF_' + str(year-1) + '-' + str(year) + '_average'

    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.date = dates[0]
    surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=custom_label)


def process_multiple_years(year_start, year_end):
    for year in range(year_end, year_start - 1, -1):
        for month in range(1, 13):
            n_days = calendar.monthrange(year, month)[1]
            Parallel(n_jobs=12)(delayed(process_day)(datetime.date(year, month, day)) for day in range(1, n_days+1))

            # try:
            #     Parallel(n_jobs=12)(delayed(process_day)(datetime.date(date_in_month.year, date_in_month.month, day))
            #                         for day in range(1, n_days+1))
            # except Exception as e:
            #     logger.error('{}'.format(e), exc_info=True)
            #     continue


def produce_seasonal_climatology(year_start, year_end):
    fall_label = 'Fall_MAM_' + str(year_start) + '-' + str(year_end) + '_' + 'average'
    winter_label = 'Winter_JJA_' + str(year_start) + '-' + str(year_end) + '_' + 'average'
    spring_label = 'Spring_SON_' + str(year_start) + '-' + str(year_end) + '_' + 'average'
    summer_label = 'Summer_DJF_' + str(year_start) + '-' + str(year_end) + '_' + 'average'

    fall_days = []
    winter_days = []
    spring_days = []
    summer_days = []

    for year in range(year_start, year_end + 1):
        fall_days = fall_days + date_range(datetime.date(year, 3, 1), datetime.date(year, 5, 31))
        winter_days = winter_days + date_range(datetime.date(year, 6, 1), datetime.date(year, 8, 31))
        spring_days = spring_days + date_range(datetime.date(year, 9, 1), datetime.date(year, 11, 30))

        summer_days = summer_days + date_range(datetime.date(year, 12, 1), datetime.date(year, 12, 31)) \
                      + date_range(datetime.date(year+1, 1, 1), datetime.date(year+1, 2, 28))

    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.date = winter_days[-1]
    surface_stress_dataset.compute_mean_fields(winter_days, avg_method='partial_data_ok')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=winter_label)

    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.date = summer_days[-1]
    # surface_stress_dataset.compute_mean_fields(summer_days, avg_method='partial_data_ok')
    # surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=summer_label)
    #
    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.date = fall_days[-1]
    # surface_stress_dataset.compute_mean_fields(fall_days, avg_method='partial_data_ok')
    # surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=fall_label)
    #
    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.date = spring_days[-1]
    # surface_stress_dataset.compute_mean_fields(spring_days, avg_method='partial_data_ok')
    # surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=spring_label)


def produce_monthly_climatology(year_start, year_end):
    dec_label = 'Dec_' + str(year_start) + '-' + str(year_end) + '_' + 'average'
    jan_label = 'Jan_' + str(year_start) + '-' + str(year_end) + '_' + 'average'
    feb_label = 'Feb_' + str(year_start) + '-' + str(year_end) + '_' + 'average'
    sep_label = 'Sep_' + str(year_start) + '-' + str(year_end) + '_' + 'average'

    dec_days = []
    jan_days = []
    feb_days = []
    sep_days = []

    for year in range(year_start, year_end + 1):
        dec_days = dec_days + date_range(datetime.date(year, 12, 1), datetime.date(year, 12, 31))
        jan_days = jan_days + date_range(datetime.date(year, 1, 1), datetime.date(year, 1, 31))
        feb_days = feb_days + date_range(datetime.date(year, 2, 1), datetime.date(year, 2, 28))
        sep_days = sep_days + date_range(datetime.date(year, 9, 1), datetime.date(year, 9, 30))

    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.date = dec_days[-1]
    # surface_stress_dataset.compute_mean_fields(dec_days, avg_method='partial_data_ok')
    # surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=dec_label)
    #
    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.date = jan_days[-1]
    # surface_stress_dataset.compute_mean_fields(jan_days, avg_method='partial_data_ok')
    # surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=jan_label)

    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.date = feb_days[-1]
    surface_stress_dataset.compute_mean_fields(feb_days, avg_method='partial_data_ok')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=feb_label)

    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.date = sep_days[-1]
    surface_stress_dataset.compute_mean_fields(sep_days, avg_method='partial_data_ok')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=sep_label)


def produce_climatology(year_start, year_end):
    climo_label = str(year_start) + '-' + str(year_end) + '_average'

    dates = date_range(datetime.date(year_start, 1, 1), datetime.date(year_end, 12, 31))

    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.date = dates[0]
    surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=climo_label)


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


def plot_meridional_salinity_profile(time_span, avg_period, grid_size, field_type, lon, split_depth):
    import os

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    import cmocean.cm

    from SalinityDataset import SalinityDataset
    from constants import output_dir_path

    image_filepaths = []

    for avg_period in ['00', '13', '14', '15', '16']:
        salinity_dataset = SalinityDataset(time_span, avg_period, grid_size, field_type)
        lats, depths, salinity_profile = salinity_dataset.meridional_salinity_profile(lon=lon, lat_min=-80, lat_max=0)

        time_span_str = time_span
        if time_span == 'A5B2':
            time_span_str = '2005-12'
        elif time_span == '95A4':
            time_span_str = '1995-2004'

        avg_period_str = avg_period
        if avg_period == '00':
            avg_period_str = 'mean'
        elif avg_period == '13':
            avg_period_str = 'JFM_seasonal'
        elif avg_period == '14':
            avg_period_str = 'AMJ_seasonal'
        elif avg_period == '15':
            avg_period_str = 'JAS_seasonal'
        elif avg_period == '16':
            avg_period_str = 'OND_seasonal'

        title_str = time_span_str + '_' + avg_period_str + '_lon=' + str(int(lon))

        fig, (ax1, ax2) = plt.subplots(2)

        idx_split_depth = np.abs(depths - split_depth).argmin()

        im1 = ax1.pcolormesh(lats, depths[:idx_split_depth], salinity_profile[:idx_split_depth, :],
                             cmap=cmocean.cm.haline, vmin=33.8, vmax=36)
        im2 = ax2.pcolormesh(lats, depths[idx_split_depth:], salinity_profile[idx_split_depth:, :],
                             cmap=cmocean.cm.haline, vmin=33.8, vmax=36)

        idx_40S = np.abs(lats - -40).argmin()
        idx_60S = np.abs(lats - -60).argmin()
        idx_max_salinity = salinity_profile[0, idx_60S:idx_40S].argmin() + idx_60S
        lat_max_salinity = lats[idx_max_salinity]
        ax1.plot([lat_max_salinity, lat_max_salinity], [0, 50], 'red', lw=2)
        ax1.text(lat_max_salinity+0.5, 30, '{:.1f}°'.format(lat_max_salinity), fontsize=10, color='red')

        ax1.set_title(title_str, y=1.1, fontsize=12)

        fig.subplots_adjust(left=0.10, bottom=0.20, right=0.95, top=0.9, hspace=0)
        cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
        clb = fig.colorbar(im1, cax=cbar_ax, extend='both', orientation='horizontal')
        clb.ax.set_title('salinity (g/kg)', fontsize=12)

        ax1.set_ylim(0, depths[idx_split_depth-1])
        ax2.set_ylim(depths[idx_split_depth], 5000)
        ax1.set_xlim(-70, 0)
        ax2.set_xlim(-70, 0)
        ax1.invert_yaxis()
        ax2.invert_yaxis()

        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax2.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        ax1.xaxis.tick_top()
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%d°'))

        # plt.subplot_tool()
        # plt.show()

        png_filename = 'salinity_profile_woa13_' + time_span + '_' + avg_period + '_' + grid_size + '_' + \
                       'lon' + str(int(lon))
        png_filepath = os.path.join(output_dir_path, 'salinity_profiles', png_filename + '.png')

        image_filepaths.append(png_filepath)

        dir = os.path.dirname(png_filepath)
        if not os.path.exists(dir):
            logger.info('Creating directory: {:s}'.format(dir))
            os.makedirs(dir)

        logger.info('Saving salinity profile: {:s}'.format(png_filepath))
        plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')
if __name__ == '__main__':
    from SurfaceStressDataWriter import SurfaceStressDataWriter
    from utils import date_range

    # process_and_plot_day(datetime.date(2015, 7, 16))
    # process_day(datetime.date(2015, 7, 16))
    # plot_day(datetime.date(2015, 7, 16))
    # produce_seasonal_climatology(2011, 2012)
    # process_multiple_years(1995, 1995)
    # process_day(datetime.date(2015, 1, 1))

    # process_neutral_density_fields_multiple_depths(time_span='A5B2', avg_period='00', grid_size='04', field_type='an',
    #                                                depth_levels=range(8))
    # process_neutral_density_fields_multiple_depths(time_span='95A4', avg_period='00', grid_size='04', field_type='an',
    #                                                depth_levels=range(8))
    # process_neutral_density_fields_multiple_depths(time_span='A5B2', avg_period='13', grid_size='04', field_type='an',
    #                                                depth_levels=range(8))
    # process_neutral_density_fields_multiple_depths(time_span='A5B2', avg_period='14', grid_size='04', field_type='an',
    #                                                depth_levels=range(8))
    # process_neutral_density_fields_multiple_depths(time_span='A5B2', avg_period='15', grid_size='04', field_type='an',
    #                                                depth_levels=range(8))
    # process_neutral_density_fields_multiple_depths(time_span='A5B2', avg_period='16', grid_size='04', field_type='an',
    #                                                depth_levels=range(8))

    plot_meridional_salinity_profile(time_span='A5B2', avg_period='15', grid_size='04', field_type='an', lon=80,
                                     split_depth=500)
