import numpy as np

import logging
logger = logging.getLogger(__name__)


def date_range(date1, date2):
    from datetime import timedelta
    dates = []
    for n in range(int((date2 - date1).days) + 1):
        dates.append(date1 + timedelta(n))
    return dates


def distance(lat1, lon1, lat2, lon2):
    from constants import R

    # Calculate the distance between two points on the Earth (lat1, lon1) and (lat2, lon2) using the haversine formula.
    # See: http://www.movable-type.co.uk/scripts/latlong.html

    lat1, lon1, lat2, lon2 = np.deg2rad([lat1, lon1, lat2, lon2])
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = np.sin(delta_lat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon/2)**2
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

    logger.debug('Title: {:s}'.format(dataset.title))
    logger.debug('Data model: {:s}'.format(dataset.data_model))
    logger.debug('Dimensions: {:s}'.format(dim_string))
    logger.debug('Variables: {:s}'.format(var_string[:-2]))


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
        # logger.debug('rho = {:f}, m_c = {:f}, t = {:f}, t_c = {:f}'.format(rho, m_c, t, t_c))

    x = rho * sgn * np.sin(sgn * lon)
    y = -rho * sgn * np.cos(sgn * lon)

    return x, y


def polar_stereographic_velocity_vector_to_latlon(v_xy, lat, lon):
    # if lon < 0:
    #     lon = lon + 360  # Change from our convention lon = [-180, 180] to [0, 360]

    vx, vy = v_xy
    lat, lon = np.deg2rad([lat, lon])

    u = vx * np.cos(lon) - vy * np.sin(lon)
    v = vx * np.sin(lon) + vy * np.cos(lon)

    return np.array([u, v])


def convert_lon_range_to_0360(old_lon_min, old_lon_max):
    # TODO: Properly convert lat = -180:180 to lat = 0:360. List comprehension then sort?
    if old_lon_min == -180 and old_lon_max == 180:
        return 0, 360


def interp_weights(xyz, uvw, dim):
    import scipy.spatial.qhull as qhull

    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, dim]
    bary = np.einsum('njk,nk->nj', temp[:, :dim, :], delta)

    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)


def interpolate_scalar_field(data, x, y, pickle_filepath, mask_value_cond, grid_type, interp_method,
                             repeat0tile1, convert_lon_range, debug_plots=False):
    import pickle
    from os.path import isfile

    # Check if the data has already been interpolated for the same grid points before doing the interpolation again. If
    # so, load the file, unpickle it and return the interpolated grid.
    if (pickle_filepath is not None) and isfile(pickle_filepath):
        logger.info('Interpolated grid already computed and saved. Unpickling: {:s}'.format(pickle_filepath))
        with open(pickle_filepath, 'rb') as f:
            data_interp_dict = pickle.load(f)
            data_interp = data_interp_dict['data_interp']
            x_interp = data_interp_dict['x_interp']
            y_interp = data_interp_dict['y_interp']
            # residual_interp = data_interp_dict['residual_interp']
        return data_interp, x_interp, y_interp

    from scipy.interpolate import griddata
    from constants import lat_min, lat_max, n_lat, lon_min, lon_max, n_lon

    if convert_lon_range:
        lon_min, lon_max = convert_lon_range_to_0360(lon_min, lon_max)

    logger.info('Options: grid_type={:s}, interp_method={:s}, repeat0tile1={}, convert_lon_range={}'
                .format(grid_type, interp_method, repeat0tile1, convert_lon_range))

    logger.info('Data information:')
    logger.info('x.min={:.2f}, x.max={:.2f}, x.shape={}'.format(x.min(), x.max(), x.shape))
    logger.info('y.min={:.2f}, y.max={:.2f}, y.shape={}'.format(y.min(), y.max(), y.shape))
    logger.info('data.min={:.2f}, data.max={:.2f}, data.shape={}'.format(data.min(), data.max(), data.shape))

    logger.info('Interpolation grid information:')
    logger.info('x_min={:.2f}, x_max={:.2f}, n_x={:d}'.format(lat_min, lat_max, n_lat))
    logger.info('y_min={:.2f}, y_max={:.2f}, n_y={:d}'.format(lon_min, lon_max, n_lon))

    # Mask certain values (e.g. land, missing data) according to the mask value condition and reshape into a 1D array
    # in preparation for griddata.
    data_masked = np.ma.array(data, mask=mask_value_cond(data))

    if debug_plots:
        logger.info('Plotting masked data.')
        if repeat0tile1:
            plot_scalar_field(y, x, data_masked, grid_type)
        else:
            plot_scalar_field(x, y, data_masked, grid_type)

    data_masked = np.reshape(data_masked, (len(x) * len(y),))

    # Repeat the x-coordinate and tile the y-coordinate so that x_masked[i], y_masked[i] corresponds to data[i].
    if repeat0tile1:
        x_masked = np.repeat(x, len(y))
        y_masked = np.tile(y, len(x))
    else:
        x_masked = np.tile(x, len(y))
        y_masked = np.repeat(y, len(x))

    # Mask the latitudes and longitudes that correspond to masked data values.
    x_masked = np.ma.masked_where(np.ma.getmask(data_masked), x_masked)
    y_masked = np.ma.masked_where(np.ma.getmask(data_masked), y_masked)

    # Use the mask to remove all masked elements as griddata ignores masked data and cannot deal with NaN values.
    x_masked = x_masked[~x_masked.mask]
    y_masked = y_masked[~y_masked.mask]
    data_masked = data_masked[~data_masked.mask]

    # Create grid of points we wish to evaluate the interpolation on.
    if grid_type == 'latlon':
        x_interp, y_interp = np.mgrid[lat_min:lat_max:n_lat * 1j, lon_min:lon_max:n_lon * 1j]
    elif grid_type == 'polar_stereographic_xy':
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        x_interp, y_interp = np.mgrid[x_min:x_max:1000*1j, y_min:y_max:1000*1j]
    elif grid_type == 'ease_rowcol':
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        x_interp, y_interp = np.mgrid[x_min:x_max:1000*1j, y_min:y_max:1000*1j]
    else:
        logger.error('Invalid value for grid_type: {}'.format(grid_type))
        raise ValueError('Invalid value for grid_type: {}'.format(grid_type))

    # TODO: Why do I even need this hack, especially for the NCEP wind field?
    if x_masked.shape[0] == 1:
        print(x_masked.shape[1])
        logger.warning('x_masked has wrong shape. Reshaping: {} -> ({:d},)'.format(x_masked.shape, x_masked.shape[1]))
        x_masked = np.reshape(x_masked, (x_masked.shape[1],))
    if y_masked.shape[0] == 1:
        logger.warning('y_masked has wrong shape. Reshaping: {} -> ({:d},)'.format(y_masked.shape, y_masked.shape[1]))
        y_masked = np.reshape(y_masked, (y_masked.shape[1],))

    logger.info('Data masked in preparation for interpolation.')
    logger.info('x_masked: min={:.2f}, max={:.2f}, shape={}'
                .format(x_masked.min(), x_masked.max(), x_masked.shape))
    logger.info('y_masked: min={:.2f}, max={:.2f}, shape={}'
                .format(y_masked.min(), y_masked.max(), y_masked.shape))
    logger.info('data_masked: min={:.2f}, max={:.2f}, shape={}'
                .format(data_masked.min(), data_masked.max(), data_masked.shape))
    logger.info('x_interp: min={:.2f}, max={:.2f}, shape={}'
                .format(x_interp.min(), x_interp.max(), x_interp.shape))
    logger.info('y_interp: min={:.2f}, max={:.2f}, shape={}'
                .format(y_interp.min(), y_interp.max(), y_interp.shape))

    logger.info('Interpolating dataset...')
    data_interp = griddata((x_masked, y_masked), data_masked, (x_interp, y_interp), method=interp_method)

    # TODO: Save and reuse the Delaunay triangulation!
    # vtx, wts = interp_weights(xyz, uvw)

    if debug_plots:
        logger.info('Plotting interpolated data.')
        plot_scalar_field(y_interp, x_interp, data_interp, grid_type)

    # Since we get back interpolated values over the land, we must mask them or get rid of them. We do this by
    # looping through the interpolated values and mask values that are supposed to be land by setting their value to
    # np.nan. We do this by comparing each interpolated value mdt_interp[i][j] with the mdt_values value that is
    # closest in latitude and longitude.
    # We can also compute the residuals, that is the error between the interpolated values and the actual values
    # which should be zero where an interpolation gridpoint coincides with an original gridpoint, and should be
    # pretty small everywhere else.
    logger.info('Masking invalid values in the interpolated grid...')
    residual_interp = np.zeros(data_interp.shape)
    for i in range(x_interp.shape[0]):
        for j in range(x_interp.shape[1]):
            x_ij = x_interp[i][j]
            y_ij = y_interp[i][j]
            closest_x_idx = np.abs(x - x_ij).argmin()
            closest_y_idx = np.abs(y - y_ij).argmin()

            if repeat0tile1:
                closest_data = data[closest_x_idx][closest_y_idx]
            else:
                closest_data = data[closest_y_idx][closest_x_idx]

            if mask_value_cond(closest_data) or mask_value_cond(data_interp[i][j]):
                data_interp[i][j] = np.nan
                residual_interp[i][j] = np.nan
            else:
                residual_interp[i][j] = data_interp[i][j] - closest_data

    if debug_plots:
        logger.info('Plotting masked interpolated data.')
        x_interp = np.ma.masked_where(np.isnan(x_interp), x_interp)
        y_interp = np.ma.masked_where(np.isnan(y_interp), y_interp)
        data_interp = np.ma.masked_where(np.isnan(data_interp), data_interp)
        plot_scalar_field(y_interp, x_interp, data_interp, grid_type)

        logger.info('Plotting interpolated data residual.')
        residual_interp = np.ma.masked_where(np.isnan(residual_interp), residual_interp)
        plot_scalar_field(y_interp, x_interp, residual_interp, grid_type)

    logger.info('Interpolating dataset... DONE!')

    # We only need to store the list of x's and y's used.
    x_interp = x_interp[:, 0]
    y_interp = y_interp[0]

    # Pickle the interpolated grid as a form of memoization to avoid having to recompute it again for the same
    # gridpoints.
    if pickle_filepath is not None:
        # Create directory if it does not exist already.
        import os
        pickle_dir = os.path.dirname(pickle_filepath)
        if not os.path.exists(pickle_dir):
            logger.info('Creating directory: {:s}'.format(pickle_dir))
            os.makedirs(pickle_dir)

        with open(pickle_filepath, 'wb') as f:
            logger.info('Pickling interpolated grid: {:s}'.format(pickle_filepath))
            data_interp_dict = {
                'data_interp': data_interp,
                'x_interp': x_interp,
                'y_interp': y_interp,
                'residual_interp': residual_interp
            }
            pickle.dump(data_interp_dict, f, pickle.HIGHEST_PROTOCOL)

    return data_interp, x_interp, y_interp


def plot_scalar_field(lons, lats, data, grid_type):
    import matplotlib.pyplot as plt
    import cartopy
    import cartopy.crs as ccrs

    if grid_type == 'latlon':
        ax = plt.axes(projection=ccrs.SouthPolarStereo())

        land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face',
                                                       facecolor='dimgray', linewidth=0)
        ax.add_feature(land_50m)
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

        vector_crs = ccrs.PlateCarree()
        im = ax.pcolormesh(lons, lats, data, transform=vector_crs)

        plt.colorbar(im)
        plt.show()
    elif grid_type == 'polar_stereographic_xy' or grid_type == 'ease_rowcol':
        plt.pcolormesh(lons, lats, data)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.show()


def plot_vector_field(self):
    import matplotlib.pyplot as plt
    plt.quiver(self.x[::3, ::3], self.y[::3, ::3], self.u_ice[::3, ::3], self.v_ice[::3, ::3], units='width',
               width=0.001, scale=10)
    plt.gca().invert_yaxis()
    plt.show()
    # plt.pcolormesh(self.x, self.y, self.error)
    # plt.colorbar()
    # plt.show()

    self.u_ice = self.u_ice[~np.isnan(self.u_ice)]
    self.v_ice = self.v_ice[~np.isnan(self.v_ice)]
    plt.hist(self.u_ice, bins=50)
    plt.hist(self.v_ice, bins=250)
    plt.show()


def get_netCDF_filepath(field_type, date=None, season_str=None, year_start=None, year_end=None):
    from os import path
    from constants import output_dir_path

    surface_stress_dir = path.join(output_dir_path, 'surface_stress')

    if date is not None:
        year = str(date.year)
        month = str(date.month).zfill(2)
        day = str(date.day).zfill(2)

    if year_start is not None and year_end is not None:
        year_range = str(year_start) + '-' + str(year_end)

    if field_type == 'daily':
        filename = 'surface_stress_' + year + month + day + '.nc'
        filepath = path.join(surface_stress_dir, year, filename)
    elif field_type == 'monthly':
        filename = 'surface_stress_' + year + month + '_monthly_avg.nc'
        filepath = path.join(surface_stress_dir, year, filename)
    elif field_type == 'annual':
        filename = 'surface_stress_' + year + '_annual_avg.nc'
        filepath = path.join(surface_stress_dir, year, filename)
    elif field_type == 'seasonal':
        filename = 'surface_stress_' + year + '_' + season_str + '_seasonal_avg.nc'
        filepath = path.join(surface_stress_dir, year, filename)
    elif field_type == 'monthly_climo':
        filename = 'surface_stress_' + month + '_' + year_range + '_monthly_climo.nc'
        filepath = path.join(surface_stress_dir, 'monthly_climo', filename)
    elif field_type == 'seasonal_climo':
        filename = 'surface_stress_' + season_str + '_' + year_range + '_seasonal_climo.nc'
        filepath = path.join(surface_stress_dir, 'seasonal_climo', filename)
    elif field_type == 'climo':
        filename = 'surface_stress_' + year_range + '_climo.nc'
        filepath = path.join(surface_stress_dir, 'climo', filename)

    return filepath


def get_field_from_netcdf(tau_filepath, var):
    import sys
    import netCDF4

    try:
        tau_dataset = netCDF4.Dataset(tau_filepath)
        log_netCDF_dataset_metadata(tau_dataset)

        lats = np.array(tau_dataset.variables['lat'])
        lons = np.array(tau_dataset.variables['lon'])
        field = np.array(tau_dataset.variables[var])

        return lons, lats, field

    except Exception as e:
        logger.error('{}'.format(e))
        logger.error('Dataset not found: {:s}'.format(tau_filepath))

        return None

        # sys.exit('Dataset not found: {:s}'.format(tau_filepath))


def get_contour_from_netcdf(tau_filepath, var, contour_level):
    import netCDF4
    import matplotlib.pyplot as plt
    import cartopy
    import cartopy.crs as ccrs

    try:
        tau_dataset = netCDF4.Dataset(tau_filepath)
        log_netCDF_dataset_metadata(tau_dataset)
    except OSError as e:
        logger.error('{}'.format(e))

    lats = np.array(tau_dataset.variables['lat'])
    lons = np.array(tau_dataset.variables['lon'])
    var_field = np.array(tau_dataset.variables[var])
    alpha_field = np.array(tau_dataset.variables['alpha'])

    # Contour the zonal stress field so we can then just extract the zero contour.
    # vector_crs = ccrs.PlateCarree()
    # fig = plt.figure()
    # ax = plt.subplot(111, projection=ccrs.SouthPolarStereo())
    # cs = ax.contour(lons, lats, np.ma.array(var_field, mask=np.isnan(alpha_field)), levels=[contour_level],
    #                 transform=vector_crs)

    ax = plt.subplot(111)
    n = len(lons)
    # lons = lons + 180
    var_field = np.append(var_field[:int(n/2), :], var_field[int(n/2):, :], axis=0)
    cs = ax.contour(lons, lats, np.ma.array(var_field, mask=np.isnan(alpha_field)), levels=[contour_level])

    # Extract the contour piece by piece.
    contour_lons = np.array([])
    contour_lats = np.array([])
    for path in cs.collections[0].get_paths():
        vertices = path.vertices
        contour_lons = np.append(contour_lons, vertices[:, 0])
        contour_lats = np.append(contour_lats, vertices[:, 1])

    plt.close()

    # return contour_lons-180, contour_lats
    return contour_lons, contour_lats


def get_northward_zero_zonal_stress_line(tau_filepath):
    from constants import lon_min, lon_max, n_lon
    lon_bins = np.linspace(lon_min, lon_max, n_lon)

    lat_northward = np.empty(len(lon_bins))
    lat_northward[:] = -180

    lons, lats = get_contour_from_netcdf(tau_filepath, 'tau_x', 0)

    for j in range(len(lons)):
        lon = lons[j]
        lat = lats[j]

        closest_lon_idx = np.abs(lon_bins - lon).argmin()

        if lat > lat_northward[closest_lon_idx]:
            lat_northward[closest_lon_idx] = lat

    lat_northward[lat_northward == -180] = np.nan

    return lon_bins, lat_northward


def get_northward_zero_zonal_wind_line(tau_filepath):
    from constants import lon_min, lon_max, n_lon
    lon_bins = np.linspace(lon_min, lon_max, n_lon)

    lat_northward = np.empty(len(lon_bins))
    lat_northward[:] = -180

    lons, lats = get_contour_from_netcdf(tau_filepath, 'wind_', 0)

    for j in range(len(lons)):
        lon = lons[j]
        lat = lats[j]

        closest_lon_idx = np.abs(lon_bins - lon).argmin()

        if lat > lat_northward[closest_lon_idx]:
            lat_northward[closest_lon_idx] = lat

    lat_northward[lat_northward == -180] = np.nan

    return lon_bins, lat_northward


def get_northward_ice_edge(tau_filepath):
    from constants import lon_min, lon_max, n_lon
    lon_bins = np.linspace(lon_min, lon_max, n_lon)

    lat_northward = np.empty(len(lon_bins))
    lat_northward[:] = -180

    lons, lats = get_contour_from_netcdf(tau_filepath, 'alpha', 0.15)

    for j in range(len(lons)):
        lon = lons[j]
        lat = lats[j]

        closest_lon_idx = np.abs(lon_bins - lon).argmin()

        if lat > lat_northward[closest_lon_idx]:
            lat_northward[closest_lon_idx] = lat

    lat_northward[lat_northward == -180] = np.nan

    return lon_bins, lat_northward


def get_coast_coordinates(tau_filepath):
    lons, lats, alpha_field = get_field_from_netcdf(tau_filepath, 'alpha')

    lat_northward = np.zeros(len(lons))

    for i in range(len(lons)):
        lon = lons[i]

        # Bit of a weird range to get around the fact that lats[-0] = lats[0].
        for j in range(1, len(lats)+1):
            lat = lats[-j]

            if np.isnan(alpha_field[-j][i]) and lat < -60:
                lat_northward[i] = lat
                break

    lat_northward[lat_northward == 0] = np.nan

    return lons, lat_northward


def date_to_WOA_time_span(date):
    year = date.year

    if 2005 <= year:
        return 'A5B2'
    elif 1995 <= year <= 2004:
        return '95A4'
    elif 1985 <= year <= 1994:
        return '8594'
    elif 1975 <= year <= 1984:
        return '7584'
    elif 1965 <= year <= 1974:
        return '6574'
    elif 1955 <= year <= 1964:
        return '5564'
    else:
        logger.error('Pre-1955 data not available from WOA. Input date: {}'.format(date))


def year_range_to_WOA_time_span(year_start, year_end):
    import datetime

    # TODO: Actually implement this function rather than just cheap out.
    return date_to_WOA_time_span(datetime.date(year_end, 1, 1))


def season_str_to_WOA_avg_period(season):
    if season == 'JFM':
        return '13'
    elif season == 'AMJ':
        return '14'
    elif season == 'JAS':
        return '15'
    elif season == 'OND':
        return '16'
    else:
        logger.error('Input season ({}) not available from WOA.'.format(season))


def get_WOA_parameters(field_type, date, season_str, year_start, year_end):
    if field_type == 'daily':
        time_span = date_to_WOA_time_span(date)
        avg_period = str(date.month).zfill(2)
    elif field_type == 'monthly':
        time_span = date_to_WOA_time_span(date)
        avg_period = str(date.month).zfill(2)
    elif field_type == 'annual':
        time_span = date_to_WOA_time_span(date)
        avg_period = '00'
    elif field_type == 'seasonal':
        time_span = date_to_WOA_time_span(date)
        avg_period = season_str_to_WOA_avg_period(season_str)
    elif field_type == 'monthly_climo':
        time_span = year_range_to_WOA_time_span(year_start, year_end)
        avg_period = str(date.month).zfill(2)
    elif field_type == 'seasonal_climo':
        time_span = year_range_to_WOA_time_span(year_start, year_end)
        avg_period = season_str_to_WOA_avg_period(season_str)
    elif field_type == 'climo':
        time_span = year_range_to_WOA_time_span(year_start, year_end)
        avg_period = '00'
    else:
        logger.error('Invalid field_type: {:s}'.format(field_type))
        return {}

    WOA_parameters = {
        'time_span': time_span,
        'avg_period': avg_period,
        'grid_size': '04',
        'field_type': 'an'
    }

    return WOA_parameters
