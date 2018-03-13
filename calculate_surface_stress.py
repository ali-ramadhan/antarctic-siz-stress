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
        surface_stress_dataset = SurfaceStressDataWriter(date)

        surface_stress_dataset.compute_daily_surface_stress_field()
        surface_stress_dataset.compute_daily_ekman_pumping_field()
        surface_stress_dataset.write_fields_to_netcdf(field_type='daily')

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
    surface_stress_dataset.write_fields_to_netcdf(field_type='daily')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='daily', custom_label='no_custom_label_', avg_period='00')


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
    surface_stress_dataset.write_fields_to_netcdf(field_type='monthly')


def produce_annual_mean(year):
    dates = date_range(datetime.date(year, 1, 1), datetime.date(year, 12, 31))

    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.date = dates[0]
    surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='annual')
    surface_stress_dataset.write_fields_to_netcdf(field_type='annual')


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
    dec = date_range(datetime.date(year - 1, 12, 1), datetime.date(year - 1, 12, 31))
    janfeb = date_range(datetime.date(year, 1, 1), datetime.date(year, 2, 28))  # TODO: Feb 29
    dates = dec + janfeb
    custom_label = 'Summer_DJF_' + str(year - 1) + '-' + str(year) + '_average'

    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.date = dates[0]
    surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=custom_label)
    surface_stress_dataset.write_fields_to_netcdf(field_type='seasonal')


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
                      + date_range(datetime.date(year + 1, 1, 1), datetime.date(year + 1, 2, 28))

    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.date = winter_days[-1]
    surface_stress_dataset.compute_mean_fields(winter_days, avg_method='partial_data_ok')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=winter_label)
    surface_stress_dataset.write_fields_to_netcdf(field_type='seasonal_climo')

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
    surface_stress_dataset.write_fields_to_netcdf(field_type='monthly_climo')

    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.date = sep_days[-1]
    surface_stress_dataset.compute_mean_fields(sep_days, avg_method='partial_data_ok')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=sep_label)
    surface_stress_dataset.write_fields_to_netcdf(field_type='monthly_climo')


def produce_climatology(year_start, year_end):
    climo_label = str(year_start) + '-' + str(year_end) + '_average'

    dates = date_range(datetime.date(year_start, 1, 1), datetime.date(year_end, 12, 31))

    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.date = dates[0]
    surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
    surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=climo_label)
    surface_stress_dataset.write_fields_to_netcdf(field_type='climo')


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


def plot_meridional_salinity_profiles(time_span, grid_size, field_type, lon, split_depth):
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
            avg_period_str = 'JFM-seasonal'
        elif avg_period == '14':
            avg_period_str = 'AMJ-seasonal'
        elif avg_period == '15':
            avg_period_str = 'JAS-seasonal'
        elif avg_period == '16':
            avg_period_str = 'OND-seasonal'

        title_str = time_span_str + '-' + avg_period_str + '-lon=' + str(int(lon))

        fig, (ax1, ax2) = plt.subplots(2)

        levels = np.linspace(33.8, 36, 21)
        idx_split_depth = np.abs(depths - split_depth).argmin()

        im1 = ax1.contourf(lats, depths[:idx_split_depth], salinity_profile[:idx_split_depth, :],
                             cmap=cmocean.cm.haline, colors=None, vmin=33.8, vmax=36, levels=levels, extend='both')
        im2 = ax2.contourf(lats, depths[idx_split_depth:], salinity_profile[idx_split_depth:, :],
                             cmap=cmocean.cm.haline, colors=None, vmin=33.8, vmax=36, levels=levels, extend='both')

        # plt.xticks(list(plt.xticks()[0]) + [split_depth])

        idx_40S = np.nanargmin(np.abs(lats - -40))
        idx_80S = np.nanargmin(np.abs(lats - -80))
        idx_min_salinity = np.nanargmin(salinity_profile[0, idx_80S:idx_40S]) + idx_80S
        lat_min_salinity = lats[idx_min_salinity]
        ax1.plot([lat_min_salinity, lat_min_salinity], [0, 50], 'red', lw=2)
        ax1.text(lat_min_salinity + 0.5, 30, '{:.1f}°'.format(lat_min_salinity), fontsize=10, color='red')

        ax1.set_title(title_str, y=1.15, fontsize=12)

        fig.subplots_adjust(left=0.10, bottom=0.20, right=0.95, top=0.9, hspace=0)
        cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
        clb = fig.colorbar(im1, cax=cbar_ax, extend='both', orientation='horizontal')
        clb.ax.set_title('salinity (g/kg)', fontsize=12)

        ax1.set_ylim(0, depths[idx_split_depth - 1])
        ax2.set_ylim(depths[idx_split_depth], 5000)
        ax1.set_xlim(-75, 0)
        ax2.set_xlim(-75, 0)
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

    from PIL import Image

    images = []
    for fp in image_filepaths:
        images.append(Image.open(fp, 'r'))

    widths, heights = zip(*(i.size for i in images))

    w = widths[0]
    h = heights[0]

    new_im = Image.new('RGB', (3*w, 2*h), color=(255, 255, 255))

    new_im.paste(images[1], (0, 0))
    new_im.paste(images[2], (w, 0))
    new_im.paste(images[3], (0, h))
    new_im.paste(images[4], (w, h))
    new_im.paste(images[0], (2*w, int(np.ceil(0.5*h))))

    all_filename = 'salinity_profile_woa13_' + time_span + '_all_' + grid_size + '_' + 'lon' + str(int(lon))
    all_filepath = os.path.join(output_dir_path, 'salinity_profiles', all_filename + '.png')

    logger.info('Saving combined salinity profiles: {:s}'.format(all_filepath))
    new_im.save(all_filepath)


def plot_meridional_temperature_profiles(time_span, grid_size, field_type, lon, split_depth):
    import os

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    import cmocean.cm

    from TemperatureDataset import TemperatureDataset
    from constants import output_dir_path

    image_filepaths = []

    for avg_period in ['00', '13', '14', '15', '16']:
        temperature_dataset = TemperatureDataset(time_span, avg_period, grid_size, field_type)
        lats, depths, temperature_profile = temperature_dataset.meridional_temperature_profile(lon=lon, lat_min=-80, lat_max=0)

        time_span_str = time_span
        if time_span == 'A5B2':
            time_span_str = '2005-12'
        elif time_span == '95A4':
            time_span_str = '1995-2004'

        avg_period_str = avg_period
        if avg_period == '00':
            avg_period_str = 'mean'
        elif avg_period == '13':
            avg_period_str = 'JFM-seasonal'
        elif avg_period == '14':
            avg_period_str = 'AMJ-seasonal'
        elif avg_period == '15':
            avg_period_str = 'JAS-seasonal'
        elif avg_period == '16':
            avg_period_str = 'OND-seasonal'

        title_str = time_span_str + '-' + avg_period_str + '-lon=' + str(int(lon))

        fig, (ax1, ax2) = plt.subplots(2)

        idx_split_depth = np.abs(depths - split_depth).argmin()

        im1 = ax1.pcolormesh(lats, depths[:idx_split_depth], temperature_profile[:idx_split_depth, :],
                             cmap=cmocean.cm.thermal, vmin=-2, vmax=2)
        im2 = ax2.pcolormesh(lats, depths[idx_split_depth:], temperature_profile[idx_split_depth:, :],
                             cmap=cmocean.cm.thermal, vmin=-2, vmax=2)

        # idx_40S = np.abs(lats - -40).argmin()
        # idx_60S = np.abs(lats - -60).argmin()
        # idx_max_temperature = temperature_profile[0, idx_60S:idx_40S].argmin() + idx_60S
        # lat_max_temperature = lats[idx_max_temperature]
        # ax1.plot([lat_max_temperature, lat_max_temperature], [0, 50], 'red', lw=2)
        # ax1.text(lat_max_temperature + 0.5, 30, '{:.1f}°'.format(lat_max_temperature), fontsize=10, color='red')

        ax1.set_title(title_str, y=1.15, fontsize=12)

        fig.subplots_adjust(left=0.10, bottom=0.20, right=0.95, top=0.9, hspace=0)
        cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
        clb = fig.colorbar(im1, cax=cbar_ax, extend='both', orientation='horizontal')
        clb.ax.set_title('temperature (g/kg)', fontsize=12)

        ax1.set_ylim(0, depths[idx_split_depth - 1])
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

        png_filename = 'temperature_profile_woa13_' + time_span + '_' + avg_period + '_' + grid_size + '_' + \
                       'lon' + str(int(lon))
        png_filepath = os.path.join(output_dir_path, 'temperature_profiles', png_filename + '.png')

        image_filepaths.append(png_filepath)

        dir = os.path.dirname(png_filepath)
        if not os.path.exists(dir):
            logger.info('Creating directory: {:s}'.format(dir))
            os.makedirs(dir)

        logger.info('Saving temperature profile: {:s}'.format(png_filepath))
        plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')

    from PIL import Image

    images = []
    for fp in image_filepaths:
        images.append(Image.open(fp, 'r'))

    widths, heights = zip(*(i.size for i in images))

    w = widths[0]
    h = heights[0]

    new_im = Image.new('RGB', (3*w, 2*h), color=(255, 255, 255))

    new_im.paste(images[1], (0, 0))
    new_im.paste(images[2], (w, 0))
    new_im.paste(images[3], (0, h))
    new_im.paste(images[4], (w, h))
    new_im.paste(images[0], (2*w, int(np.ceil(0.5*h))))

    all_filename = 'temperature_profile_woa13_' + time_span + '_all_' + grid_size + '_' + 'lon' + str(int(lon))
    all_filepath = os.path.join(output_dir_path, 'temperature_profiles', all_filename + '.png')

    logger.info('Saving combined temperature profiles: {:s}'.format(all_filepath))
    new_im.save(all_filepath)


def plot_meridional_gamma_profiles(time_span, grid_size, field_type, lon, split_depth):
    import os

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    import cmocean.cm

    from NeutralDensityDataset import NeutralDensityDataset
    from constants import output_dir_path

    image_filepaths = []

    for avg_period in ['00']:
        gamma_dataset = NeutralDensityDataset(time_span, avg_period, grid_size, field_type, depth_levels=np.arange(100))
        lats, depths, gamma_profile = gamma_dataset.meridional_gamma_profile(lon=lon, lat_min=-80, lat_max=-40)

        depths = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                  125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500,
                  550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400,
                  1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000,
                  2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700,
                  3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300]
        depths = np.array(depths)

        time_span_str = time_span
        if time_span == 'A5B2':
            time_span_str = '2005-12'
        elif time_span == '95A4':
            time_span_str = '1995-2004'

        avg_period_str = avg_period
        if avg_period == '00':
            avg_period_str = 'mean'
        elif avg_period == '13':
            avg_period_str = 'JFM-seasonal'
        elif avg_period == '14':
            avg_period_str = 'AMJ-seasonal'
        elif avg_period == '15':
            avg_period_str = 'JAS-seasonal'
        elif avg_period == '16':
            avg_period_str = 'OND-seasonal'

        title_str = time_span_str + '-' + avg_period_str + '-lon=' + str(int(lon))

        fig, (ax1, ax2) = plt.subplots(2)

        levels = np.linspace(26.6, 28, 21)
        idx_split_depth = np.abs(depths - split_depth).argmin()

        im1 = ax1.contourf(lats, depths[:idx_split_depth], gamma_profile[:idx_split_depth, :],
                             cmap=cmocean.cm.dense, colors=None, vmin=26.6, vmax=28, levels=levels, extend='both')
        im2 = ax2.contourf(lats, depths[idx_split_depth:], gamma_profile[idx_split_depth:, :],
                             cmap=cmocean.cm.dense, colors=None, vmin=26.6, vmax=28, levels=levels, extend='both')

        # plt.xticks(list(plt.xticks()[0]) + [split_depth])

        idx_40S = np.nanargmin(np.abs(lats - -40))
        idx_80S = np.nanargmin(np.abs(lats - -80))
        idx_min_gamma = np.nanargmax(gamma_profile[0, idx_80S:idx_40S]) + idx_80S
        lat_min_gamma = lats[idx_min_gamma]
        ax1.plot([lat_min_gamma, lat_min_gamma], [0, 50], 'red', lw=2)
        ax1.text(lat_min_gamma + 0.5, 30, '{:.1f}°'.format(lat_min_gamma), fontsize=10, color='red')

        ax1.set_title(title_str, y=1.15, fontsize=12)

        fig.subplots_adjust(left=0.10, bottom=0.20, right=0.95, top=0.9, hspace=0)
        cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
        clb = fig.colorbar(im1, cax=cbar_ax, extend='both', orientation='horizontal')
        clb.ax.set_title('gamma_n (kg/m$^3$)', fontsize=12)

        ax1.set_ylim(0, depths[idx_split_depth - 1])
        ax2.set_ylim(depths[idx_split_depth], 5000)
        ax1.set_xlim(-75, -40)
        ax2.set_xlim(-75, -40)
        ax1.invert_yaxis()
        ax2.invert_yaxis()

        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax2.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        ax1.xaxis.tick_top()
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%d°'))

        # plt.subplot_tool()
        # plt.show()

        png_filename = 'gamma_profile_woa13_' + time_span + '_' + avg_period + '_' + grid_size + '_' + \
                       'lon' + str(int(lon))
        png_filepath = os.path.join(output_dir_path, 'gamma_profiles', png_filename + '.png')

        image_filepaths.append(png_filepath)

        dir = os.path.dirname(png_filepath)
        if not os.path.exists(dir):
            logger.info('Creating directory: {:s}'.format(dir))
            os.makedirs(dir)

        logger.info('Saving gamma profile: {:s}'.format(png_filepath))
        plt.savefig(png_filepath, dpi=300, format='png', transparent=False, bbox_inches='tight')

    # from PIL import Image
    #
    # images = []
    # for fp in image_filepaths:
    #     images.append(Image.open(fp, 'r'))
    #
    # widths, heights = zip(*(i.size for i in images))
    #
    # w = widths[0]
    # h = heights[0]
    #
    # new_im = Image.new('RGB', (3*w, 2*h), color=(255, 255, 255))
    #
    # new_im.paste(images[1], (0, 0))
    # new_im.paste(images[2], (w, 0))
    # new_im.paste(images[3], (0, h))
    # new_im.paste(images[4], (w, h))
    # new_im.paste(images[0], (2*w, int(np.ceil(0.5*h))))
    #
    # all_filename = 'gamma_profile_woa13_' + time_span + '_all_' + grid_size + '_' + 'lon' + str(int(lon))
    # all_filepath = os.path.join(output_dir_path, 'gamma_profiles', all_filename + '.png')
    #
    # logger.info('Saving combined gamma profiles: {:s}'.format(all_filepath))
    # new_im.save(all_filepath)


def look_at_neutral_density_contours(year_start, year_end):
    # Just looking at the surface neutral density for A5B2.
    for avg_period in ['00']:  # '['13', '14', '15', '16', '00']:
        dates = []
        for year in range(year_start, year_end+1):
            if avg_period == '00' or avg_period == '13':
                dates = dates + date_range(datetime.date(year, 1, 1), datetime.date(year, 3, 31))
            if avg_period == '00' or avg_period == '14':
                dates = dates + date_range(datetime.date(year, 4, 1), datetime.date(year, 6, 30))
            if avg_period == '00' or avg_period == '15':
                dates = dates + date_range(datetime.date(year, 7, 1), datetime.date(year, 9, 30))
            if avg_period == '00' or avg_period == '16':
                dates = dates + date_range(datetime.date(year, 10, 1), datetime.date(year, 12, 31))

        custom_label = ''
        if avg_period == '00':
            custom_label = str(year_start) + '-' + str(year_end) + '_climo'
        elif avg_period == '13':
            custom_label = str(year_start) + '-' + str(year_end) + '_seasonal_JFM'
        elif avg_period == '14':
            custom_label = str(year_start) + '-' + str(year_end) + '_seasonal_AMJ'
        elif avg_period == '15':
            custom_label = str(year_start) + '-' + str(year_end) + '_seasonal_JAS'
        elif avg_period == '16':
            custom_label = str(year_start) + '-' + str(year_end) + '_seasonal_OND'

        surface_stress_dataset = SurfaceStressDataWriter(None)
        surface_stress_dataset.date = dates[-1]
        surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
        surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label=custom_label, avg_period=avg_period)


def analyze_zero_zonal_stress_line():
    import os
    import csv
    from constants import output_dir_path
    surface_stress_dir = os.path.join(output_dir_path, 'surface_stress')

    import matplotlib.pyplot as plt
    import cartopy
    import cartopy.crs as ccrs

    years = np.arange(1992, 2016, 1)

    lon_bins = np.arange(-180, 180, 1)
    lat_northward = np.empty((len(lon_bins), len(years)))
    lat_northward[:] = -180

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    NUM_COLORS = len(years)
    cm = plt.get_cmap('gist_rainbow')
    ax.set_color_cycle([cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    for i in range(len(years)):
        year = years[i]

        csv_filename = 'zero_zonal_stress_line_contour' + str(year) + '.csv'
        csv_filepath = os.path.join(surface_stress_dir, str(year), csv_filename)

        with open(csv_filepath, 'r') as f:
            reader = csv.reader(f)

            lons = np.array(reader.__next__())
            reader.__next__()  # Not sure why csv writer skipped a line...
            lats = np.array(reader.__next__())

        lons = lons.astype(float)
        lats = lats.astype(float)

        for j in range(len(lons)):
            lon = lons[j]
            lat = lats[j]

            closest_lon_idx = np.abs(lon_bins - lon).argmin()
            closest_lon = lons[closest_lon_idx]

            if lat > lat_northward[closest_lon_idx][i]:
                lat_northward[closest_lon_idx][i] = lat

        logger.info('{:d} max lat: {:f}'.format(year, np.max(lat_northward[:, i])))

        # Mask values that we don't have data for.
        lat_northward[lat_northward[:, i] == -180, i] = np.nan

        plt.plot(lon_bins, lat_northward[:, i], label=str(year))
        # plt.scatter(lon_bins, lat_northward, marker=',', s=2, label=str(year))

    plt.legend()
    # plt.savefig('tau_x_zero_line.png', dpi=600, format='png', transparent=False, bbox_inches='tight')
    plt.show()

    from scipy import stats

    slopes = np.zeros(len(lon_bins))
    intercepts = np.zeros(len(lon_bins))
    r_values = np.zeros(len(lon_bins))
    p_values = np.zeros(len(lon_bins))
    std_errs = np.zeros(len(lon_bins))

    # Do linear regression for the N/S position of the line at each longitude.
    for i in range(len(lon_bins)):
        logger.info(lon_bins[i])
        lat_time_series = lat_northward[i, :]

        slope, intercept, r_value, p_value, std_err = stats.linregress(years, lat_time_series)
        slopes[i] = slope
        intercepts[i] = intercept
        r_values[i] = r_value**2
        p_values[i] = p_value
        std_errs[i] = std_err

    plt.plot(lon_bins, slopes, label='slopes')
    # plt.plot(lon_bins, intercepts, label='intercepts')
    # plt.plot(lon_bins, r_values, label='r_values')
    # plt.plot(lon_bins, p_values, label='p_values')
    # plt.plot(lon_bins, std_errs, label='std_errs')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    from SurfaceStressDataWriter import SurfaceStressDataWriter
    from utils import date_range

    process_and_plot_day(datetime.date(2015, 7, 16))
    # process_day(datetime.date(2015, 7, 16))
    # plot_day(datetime.date(2015, 7, 16))
    # produce_monthly_mean(datetime.date(2015, 7, 1))
    # produce_seasonal_climatology(2011, 2012)
    # process_multiple_years(1995, 1995)
    # process_day(datetime.date(2015, 1, 1))

    # for year in range(2013, 2016):
    #     produce_annual_mean(year)

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

    # plot_meridional_salinity_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=-135, split_depth=250)
    # plot_meridional_salinity_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=-30, split_depth=250)
    # plot_meridional_salinity_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=75, split_depth=250)

    # plot_meridional_temperature_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=-135, split_depth=500)
    # plot_meridional_temperature_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=-30, split_depth=500)
    # plot_meridional_temperature_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=75, split_depth=500)
    #
    # dates = []
    # for year in range(2005, 2006):
    #     dates = dates + date_range(datetime.date(year, 7, 1), datetime.date(year, 9, 30))
    #
    # surface_stress_dataset = SurfaceStressDataWriter(None)
    # surface_stress_dataset.date = dates[-1]
    # surface_stress_dataset.compute_mean_fields(dates, avg_method='partial_data_ok')
    # surface_stress_dataset.plot_diagnostic_fields(plot_type='custom', custom_label='2005-2012_JAS')

    # look_at_neutral_density_contours(2005, 2012)
    # look_at_neutral_density_contours(2014, 2015)
    # look_at_neutral_density_contours(1992, 1993)
    #
    # plot_meridional_gamma_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=-135, split_depth=250)
    # plot_meridional_gamma_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=-30, split_depth=250)
    # plot_meridional_gamma_profiles(time_span='A5B2', grid_size='04', field_type='an', lon=75, split_depth=250)

    # analyze_zero_zonal_stress_line()
