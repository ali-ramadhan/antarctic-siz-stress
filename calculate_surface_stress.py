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
import numpy as np

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config
logging.config.fileConfig('logging.ini')
logger = logging.getLogger(__name__)

np.set_printoptions(precision=4)

if __name__ == '__main__':
    from SurfacetressDataWriter import SurfaceStressDataWriter

    # for i in range(1, 32):
    #     date = datetime.date(2015, 7, i)
    #
    #     surface_stress_dataset = SurfaceStressDataWriter(date)
    #
    #     surface_stress_dataset.compute_daily_surface_stress_field()
    #     surface_stress_dataset.compute_daily_ekman_pumping_field()
    #     surface_stress_dataset.write_fields_to_netcdf()
    #     surface_stress_dataset.plot_diagnostic_fields()

    surface_stress_dataset = SurfaceStressDataWriter(None)
    surface_stress_dataset.compute_monthly_mean_field()
