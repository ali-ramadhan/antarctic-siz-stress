from os import path

from matplotlib import cm
import cmocean.cm

""" Data directory constants """
# Current Working Directory
cwd = path.dirname(path.abspath(__file__))

# Data directory string constants
# data_dir_path = path.join(cwd, 'data')
data_dir_path = 'D:\\data\\'
# data_dir_path = '/d1/alir/data/'

# Output data directory
# output_dir_path = path.join(cwd, 'output')
# output_dir_path = 'D:\\output\\'
# output_dir_path = '/d1/alir/output/'
output_dir_path = 'C:\\Users\\Ali\\Downloads\\output\\'

""" Physical constants """
# Could use a theoretical gravity model: https://en.wikipedia.org/wiki/Theoretical_gravity
g = 9.80665  # standard acceleration due to gravity [m/s^2]

Omega = 7.292115e-5  # rotation rate of the Earth [rad/s]

# Earth radius R could be upgraded to be location-dependent using a simple formula.
# See: https://en.wikipedia.org/wiki/Earth_radius#Location-dependent_radii
# Well, actually some of the projections rely on this value being constant so I would be careful. Plus, it's probably
# a TINY correction that's dwarfed by satellite measurement errors.
R = 6371.228e3  # average radius of the earth [m]

rho_air = 1.225  # [kg/m^3]
rho_seawater = 1025  # [kg/m^3]
C_air = 0.00125  # Drag coefficient
C_seawater = 0.0055  # Drag coefficient

rho_0 = 1027.5  # "reference" density [kg/m^3]
D_e = 20  # Ekman layer depth [m]

""" Latitude-longitude interpolation grid constants """
lat_min = -80
lat_max = -40
lon_min = -180
lon_max = 180

lat_step = 0.25
lon_step = 0.25

n_lat = int((lat_max - lat_min) / lat_step) + 1
n_lon = int((lon_max - lon_min) / lon_step) + 1

""" Polar stereographic interpolation grid constants """
n_x = 1000
n_y = 1000

""" EASE-Grid interpolation constants """
n_row = 1000
n_col = 1000

""" Interpolation methods for each dataset """
mdt_interp_method = 'cubic'
u_geo_interp_method = 'linear'
alpha_interp_method = 'linear'
u_ice_interp_method = 'cubic'
u_wind_interp_method = 'cubic'

""" Constants for diagnostic field plots """
# Don't think I need these lists as I can just use SurfaceStressDataWriter.var_fields.keys() and figure_fields.keys().
# The name of each field stored in the netCDF files.
# netcdf_var_names = ['tau_air_x', 'tau_air_y', 'tau_ice_x', 'tau_ice_y', 'tau_x', 'tau_y', 'tau_SIZ_x', 'tau_SIZ_y',
#                     'wind_stress_curl', 'Ekman_w', 'Ekman_u', 'Ekman_v', 'Ekman_SIZ_u', 'Ekman_SIZ_v',
#                     'geo_u', 'geo_v', 'wind_u', 'wind_v', 'alpha', 'ice_u', 'ice_v',
#                     'dtauydx', 'dtauxdy']

# The name of each field that will be plotted.
# plot_var_names = ['tau_air_x', 'tau_air_y', 'tau_ice_x', 'tau_ice_y', 'tau_x', 'tau_y',
#                   'Ekman_u', 'Ekman_v', 'wind_stress_curl', 'Ekman_w',
#                   'geo_u', 'geo_v', 'wind_u', 'wind_v', 'alpha', 'ice_u', 'ice_v',
#                   'dtauydx', 'dtauxdy']

var_units = {
    'tau_air_x': 'N/m^2',
    'tau_air_y': 'N/m^2',
    'tau_ice_x': 'N/m^2',
    'tau_ice_y': 'N/m^2',
    'tau_x': 'N/m^2',
    'tau_y': 'N/m^2',
    'tau_SIZ_x': 'N/m^2',
    'tau_SIZ_y': 'N/m^2',
    'wind_stress_curl': 'N/m^3',
    'Ekman_w': 'm/s',  # TODO: Save as [m/year].
    'Ekman_u': 'm/s',
    'Ekman_v': 'm/s',
    'Ekman_SIZ_u': 'm/s',
    'Ekman_SIZ_v': 'm/s',
    'geo_u': 'm/s',
    'geo_v': 'm/s',
    'wind_u': 'm/s',
    'wind_v': 'm/s',
    'alpha': 'fractional',
    'ice_u': 'm/s',
    'ice_v': 'm/s',
    'dtauydx': 'N/m^3',
    'dtauxdy': 'N/m^3'
}

var_positive = {
    'tau_air_x': 'up',
    'tau_air_y': 'up',
    'tau_ice_x': 'up',
    'tau_ice_y': 'up',
    'tau_x': 'up',
    'tau_y': 'up',
    'tau_SIZ_x': 'up',
    'tau_SIZ_y': 'up',
    'wind_stress_curl': 'up',
    'Ekman_w': 'up',
    'Ekman_u': 'up',
    'Ekman_v': 'up',
    'Ekman_SIZ_u': 'up',
    'Ekman_SIZ_v': 'up',
    'geo_u': 'up',
    'geo_v': 'up',
    'wind_u': 'up',
    'wind_v': 'up',
    'alpha': 'up',
    'ice_u': 'up',
    'ice_v': 'up',
    'dtauydx': 'up',
    'dtauxdy': 'up'
}

var_long_names = {
    'geo_u': 'Zonal geostrophic velocity',
    'geo_v': 'Meridional geostrophic velocity',
    'wind_u': 'Zonal wind velocity',
    'wind_v': 'Meridional wind velocity',
    'ice_u': 'Zonal sea ice motion',
    'ice_v': 'Meridional sea ice motion',
    'alpha': 'Sea ice concentration',
    'tau_air_x': 'Zonal air surface stress',
    'tau_air_y': 'Meridional air surface stress',
    'tau_ice_x': 'Zonal sea ice surface stress',
    'tau_ice_y': 'Meridional sea ice surface stress',
    'tau_SIZ_x': 'Zonal surface stress in the SIZ',
    'tau_SIZ_y': 'Meridional surface stress in the SIZ',
    'tau_x': 'Zonal surface stress',
    'tau_y': 'Meridional surface stress',
    'Ekman_u': 'Zonal Ekman velocity',
    'Ekman_v': 'Meridional Ekman velocity',
    'Ekman_SIZ_u': 'Zonal Ekman velocity in the SIZ',
    'Ekman_SIZ_v': 'Meridional Ekman velocity in the SIZ',
    'dtauxdy': 'd/dy (tau_x)',
    'dtauydx': 'd/dx (tau_y)',
    'wind_stress_curl': 'Vertical wind stress curl',
    'Ekman_w': 'Ekman pumping'
}

titles = {
    'u_geo': 'u_geo',
    'v_geo': 'v_geo',
    'u_wind': 'u_wind',
    'v_wind': 'v_wind',
    'u_ice': 'u_ice',
    'v_ice': 'v_ice',
    'alpha': 'alpha',
    'tau_air_x': '(1-alpha)*tau_air_x',
    'tau_air_y': '(1-alpha)*tau_air_y',
    'tau_ice_x': 'alpha*tau_ice_x',
    'tau_ice_y': 'alpha*tau_ice_y',
    'tau_x': 'tau_x',
    'tau_y': 'tau_y',
    'u_Ekman': 'u_Ekman',
    'v_Ekman': 'v_Ekman',
    'dtauydx': 'd/dx (tau_y)',
    'dtauxdy': '-d/dy (tau_x)',
    # 'tau_SIZ_x': 'Zonal surface stress in the SIZ',
    # 'tau_SIZ_y': 'Meridional surface stress in the SIZ',
    'curl_tau': 'curl_tau',
    'w_Ekman': 'w_Ekman',
    'freshwater_flux': 'u_Ek · grad(S)',
    'ice_div': 'div(h*u_ice)',
    'temperature': 'surface temperature',
    'salinity': 'surface salinity',
    'neutral_density': 'neutral density'
}

gs_coords = {
    'u_geo': (0, 0),
    'v_geo': (1, 0),
    'u_wind': (0, 1),
    'v_wind': (1, 1),
    'u_ice': (0, 2),
    'v_ice': (1, 2),
    'alpha': (0, 3),
    'tau_air_x': (0, 4),
    'tau_air_y': (1, 4),
    'tau_ice_x': (0, 5),
    'tau_ice_y': (1, 5),
    'tau_x': (slice(2, 5), slice(0, 3)),
    'tau_y': (slice(2, 5), slice(3, 6)),
    'u_Ekman': (0, 6),
    'v_Ekman': (1, 6),
    'dtauydx': (0, 7),
    'dtauxdy': (1, 7),
    # 'tau_SIZ_x': (1, 7),
    # 'tau_SIZ_y': (1, 7),
    'curl_tau': (0, 8),
    'w_Ekman': (slice(2, 5), slice(6, 9)),
    'freshwater_flux': (0, 9),
    'ice_div': (1, 9),
    'temperature': (0, 10),
    'salinity': (1, 10),
    'neutral_density': (slice(2, 4), slice(9, 11))
}

scale_factor = {
    'u_geo': 1,
    'v_geo': 1,
    'u_wind': 1,
    'v_wind': 1,
    'u_ice': 1,
    'v_ice': 1,
    'alpha': 1,
    'tau_air_x': 1,
    'tau_air_y': 1,
    'tau_ice_x': 1,
    'tau_ice_y': 1,
    'tau_x': 1,
    'tau_y': 1,
    'u_Ekman': 1,
    'v_Ekman': 100,
    'dtauydx': 1e7,
    'dtauxdy': -1e7,
    'curl_tau': 1e7,
    # 'tau_SIZ_x': 1,
    # 'tau_SIZ_y': 1,
    'w_Ekman': 3600 * 365 * 24,  # [m/s] -> [m/year]
    'freshwater_flux': 1,
    'ice_div': 3600 * 365 * 24,  # [m/s] -> [m/year]
    'temperature': 1,
    'salinity': 1,
    'neutral_density': 1
}

colorbar_label = {
    'u_geo': 'm/s',
    'v_geo': 'm/s',
    'u_wind': 'm/s',
    'v_wind': 'm/s',
    'u_ice': 'm/s',
    'v_ice': 'm/s',
    'alpha': '',
    'tau_air_x': r'N/m$^2$',
    'tau_air_y': r'N/m$^2$',
    'tau_ice_x': r'N/m$^2$',
    'tau_ice_y': r'N/m$^2$',
    'tau_x': r'N/m$^2$',
    'tau_y': r'N/m$^2$',
    'u_Ekman': 'm/s',
    'v_Ekman': 'cm/s',
    'dtauydx': r'10$^{-7}$ N/m$^3$',
    'dtauxdy': r'10$^{-7}$ N/m$^3$',
    'curl_tau': r'10$^{-7}$ N/m$^3$',
    # 'tau_SIZ_x': r'N/m$^2$',
    # 'tau_SIZ_y': r'N/m$^2$',
    'w_Ekman': r'm/year',
    'freshwater_flux': 'm*g/(s*kg)',
    'ice_div': 'm/year',
    'temperature': '°C',
    'salinity': 'g/kg',
    'neutral_density': 'kg/m$^3$'
}

# 'seismic' seems like a stronger version of 'RdBu_r', although I like the colors of 'RdBu_r' better.
cmaps = {
    'u_geo': 'seismic',
    'v_geo': 'seismic',
    'u_wind': 'seismic',
    'v_wind': 'seismic',
    'u_ice': 'seismic',
    'v_ice': 'seismic',
    'alpha': cmocean.cm.ice,
    'tau_air_x': 'seismic',
    'tau_air_y': 'seismic',
    'tau_ice_x': 'seismic',
    'tau_ice_y': 'seismic',
    'tau_x': 'seismic',
    'tau_y': 'seismic',
    'u_Ekman': 'seismic',
    'v_Ekman': 'seismic',
    'dtauydx': 'seismic',
    'dtauxdy': 'seismic',
    'curl_tau': 'seismic',
    # 'tau_SIZ_x': 'seismic',
    # 'tau_SIZ_y': 'seismic',
    'w_Ekman': 'seismic',
    'freshwater_flux': 'seismic',
    'ice_div': 'seismic',
    'temperature': cmocean.cm.thermal,
    'salinity': cmocean.cm.haline,
    'neutral_density': cm.get_cmap('viridis', 10)
}

cmap_ranges = {
    'u_geo': (-1, 1),
    'v_geo': (-1, 1),
    'u_wind': (-20, 20),
    'v_wind': (-20, 20),
    'u_ice': (-0.2, 0.2),
    'v_ice': (-0.2, 0.2),
    'alpha': (0, 1),
    'tau_air_x': (-0.15, 0.15),
    'tau_air_y': (-0.15, 0.15),
    'tau_ice_x': (-0.15, 0.15),
    'tau_ice_y': (-0.15, 0.15),
    'tau_x': (-0.15, 0.15),
    'tau_y': (-0.15, 0.15),
    'u_Ekman': (-0.05, 0.05),
    'v_Ekman': (-2, 2),
    'dtauydx': (-5, 5),
    'dtauxdy': (-5, 5),
    'curl_tau': (-5, 5),
    # 'tau_SIZ_x': (-0.5, 0.5),
    # 'tau_SIZ_y': (-0.5, 0.5),
    'w_Ekman': (-100, 100),
    'freshwater_flux': (-1e-7, 1e-7),
    'ice_div': (-1, 1),
    'temperature': (-2.5, 2.5),
    'salinity': (33.75, 35),
    'neutral_density': (27, 28)
}
