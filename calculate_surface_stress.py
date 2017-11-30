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
    surface_stress_dataset.plot_diagnostic_fields(type='daily')


if __name__ == '__main__':
    # from joblib import Parallel, delayed

    # from SeaIceMotionDataReader import SeaIceMotionDataReader
    # sic = SeaIceMotionDataReader(datetime.date(2015, 7, 16))
    # sic.plot_sea_ice_motion_vector_field()
    # exit()

    date_in_month = datetime.date(2015, 7, 1)
    n_days = calendar.monthrange(date_in_month.year, date_in_month.month)[1]

    # Parallel(n_jobs=8)(delayed(process_day)(datetime.date(date_in_month.year, date_in_month.month, day))
    #                    for day in range(1, n_days+1))

    for day in range(1, n_days+1):
        current_date = datetime.date(date_in_month.year, date_in_month.month, day)
        process_day(current_date)

    from SurfacetressDataWriter import SurfaceStressDataWriter

    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.compute_monthly_mean_fields(date_in_month, method='partial_data_ok')
