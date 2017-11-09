import numpy as np

import logging
logger = logging.getLogger(__name__)


def distance(ϕ1, λ1, ϕ2, λ2):
    from constants import R

    # Calculate the distance between two points on the Earth (ϕ1, λ1) and (ϕ1, λ1) using the haversine formula.
    # See: http://www.movable-type.co.uk/scripts/latlong.html
    # Latitudes are denoted by ϕ while longitudes are denoted by λ.

    ϕ1, λ1, ϕ2, λ2 = np.deg2rad([ϕ1, λ1, ϕ2, λ2])
    Δϕ = ϕ2 - ϕ1
    Δλ = λ2 - λ1

    a = np.sin(Δϕ/2)**2 + np.cos(ϕ1) * np.cos(ϕ2) * np.sin(Δλ/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c


def log_netCDF_dataset_metadata(dataset):
    # Nicely format dimension names and sizes.
    dim_string = ""
    for dim in dataset.dimensions:
        dim_name = dataset.dimensions[dim].name
        dim_size = dataset.dimensions[dim].size
        dim_string = dim_string + dim_name + '(' + str(dim_size) + ') '

    # Nicely format variable information.
    var_string = ""
    for var in dataset.variables:
        var_type = dataset.variables[var].dtype
        var_name = dataset.variables[var].name

        var_dim_str = '('
        for dim in dataset.variables[var].dimensions:
            var_dim_str = var_dim_str + str(dim) + ', '
        var_dim_str = var_dim_str[:-2] + ')'

        var_string = var_string + str(var_type) + ' ' + var_name + var_dim_str + ', '

    logger.info('Title: {:s}'.format(dataset.title))
    logger.info('Data model: {:s}'.format(dataset.data_model))
    logger.info('Dimensions: {:s}'.format(dim_string))
    logger.info('Variables: {:s}'.format(var_string[:-2]))


def latlon_to_polar_stereographic_xy(lat, lon):
    # This function converts from geodetic latitude and longitude to polar stereographic (x,y) coordinates for the polar
    # regions. The original equations are from Snyder, J. P., 1982,  Map Projections Used by the U.S. Geological Survey,
    # Geological Survey Bulletin 1532, U.S. Government Printing Office.  See JPL Technical Memorandum 3349-85-101 for
    # further details.
    #
    # The original FORTRAN program written by C. S. Morris, April 1985, Jet Propulsion Laboratory, California
    # Institute of Technology
    #
    # More information:
    # http://nsidc.org/data/polar-stereo/ps_grids.html
    # http://nsidc.org/data/polar-stereo/tools_geo_pixel.html
    # https://nsidc.org/data/docs/daac/nsidc0001_ssmi_tbs/ff.html
    #
    # SSM/I: Special Sensor Microwave Imager
    # Note: lat must be positive for the southern hemisphere! Or take absolute value like below.

    sgn = -1  # Sign of the latitude (use +1 for northern hemisphere, -1 for southern)
    e = 0.081816153  # Eccentricity of the Hughes ellipsoid
    R_E = 6378.273e3  # Radius of the Hughes ellipsode [m]
    slat = 70  # Standard latitude for the SSM/I grids is 70 degrees.

    # delta is the meridian offset for the SSM/I grids (0 degrees for the South Polar grids; 45 degrees for the
    # North Polar grids).
    delta = 45 if sgn == 1 else 0

    lat, lon = np.deg2rad([abs(lat), lon+delta])

    t = np.tan(np.pi/4 - lat/2) / ((1 - e*np.sin(lat)) / (1 + e*np.sin(lat)))**(e/2)

    if np.abs(90 - lat) < 1e-5:
        rho = 2*R_E*t / np.sqrt((1+e)**(1+e) * (1-e)**(1-e))
    else:
        sl = slat * np.pi/180
        t_c = np.tan(np.pi/4 - sl/2) / ((1 - e*np.sin(sl)) / (1 + e*np.sin(sl)))**(e/2)
        m_c = np.cos(sl) / np.sqrt(1 - e*e * (np.sin(sl)**2))
        rho = R_E * m_c * (t/t_c)
        logger.debug('rho = {:f}, m_c = {:f}, t = {:f}, t_c = {:f}'.format(rho, m_c, t, t_c))

    x = rho * sgn * np.sin(sgn * lon)
    y = -rho * sgn * np.cos(sgn * lon)

    return x, y
