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

if __name__ == '__main__':
    # from utils import latlon_to_polar_stereographic_xy
    # logger.info('{}'.format(latlon_to_polar_stereographic_xy(-75, -150)))
    #
    # from utils import polar_stereographic_velocity_vector_to_latlon
    # v_latlon = polar_stereographic_velocity_vector_to_latlon(np.array([1, -1]), -70, 100)
    # logger.info('v_latlon={}'.format(v_latlon))
    # exit(66)

    from SurfacetressDataWriter import SurfaceStressDataWriter

    date_in_month = datetime.date(2015, 7, 1)
    n_days = calendar.monthrange(date_in_month.year, date_in_month.month)[1]

    for day in range(1, n_days+1):
        date = datetime.date(date_in_month.year, date_in_month.month, day)

        surface_stress_dataset = SurfaceStressDataWriter(date)

        surface_stress_dataset.compute_daily_surface_stress_field()
        surface_stress_dataset.compute_daily_ekman_pumping_field()
        surface_stress_dataset.write_fields_to_netcdf()
        surface_stress_dataset.plot_diagnostic_fields(type='daily')
        exit()

    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.compute_monthly_mean_fields(date_in_month, method='full_data_only')
