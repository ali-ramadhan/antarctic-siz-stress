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
output_dir_path = 'C:\\Users\\Ali\\Downloads\\output\\'
# output_dir_path = '/d1/alir/output/'

figure_dir_path = 'C:\\Users\\Ali\\Downloads\\output\\figures\\'

""" Physical constants """
# Could use a theoretical gravity model: https://en.wikipedia.org/wiki/Theoretical_gravity
g = 9.80665  # standard acceleration due to gravity [m/s^2]

Omega = 7.292115e-5  # rotation rate of the Earth [rad/s]

# Earth radius R could be upgraded to be location-dependent using a simple formula.
# See: https://en.wikipedia.org/wiki/Earth_radius#Location-dependent_radii
# Well, actually some of the projections rely on this value being constant so I would be careful. Plus, it's probably
# a TINY correction that's dwarfed by satellite measurement errors.
R = 6371.228e3  # average radius of the earth [m]

rho_air = 1.25  # [kg/m^3]
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

""" Constants for saving fields to netCDF file. """
var_units = {
    'geo_u': 'm/s',
    'geo_v': 'm/s',
    'wind_u': 'm/s',
    'wind_v': 'm/s',
    'alpha': 'fractional',
    'ice_u': 'm/s',
    'ice_v': 'm/s',
    'tau_air_x': 'N/m^2',
    'tau_air_y': 'N/m^2',
    'tau_ice_x': 'N/m^2',
    'tau_ice_y': 'N/m^2',
    'tau_SIZ_x': 'N/m^2',
    'tau_SIZ_y': 'N/m^2',
    'tau_x': 'N/m^2',
    'tau_y': 'N/m^2',
    'Ekman_u': 'm/s',
    'Ekman_v': 'm/s',
    'Ekman_SIZ_u': 'm/s',
    'Ekman_SIZ_v': 'm/s',
    'dtau_x_dy': 'N/m^3',
    'dtau_y_dx': 'N/m^3',
    'curl_stress': 'N/m^3',
    'Ekman_w': 'm/s',
    'salinity': 'g/kg',
    'temperature': 'degC',
    'neutral_density': 'kg/m^3',
    'dSdx': '1/m',
    'dSdy': '1/m',
    'uEk_S_ddx': 'm/s',
    'uEk_S_ddy': 'm/s',
    'fwf_uEk_S': 'm/s',
    'ice_flux_div_x': 'm/s',
    'ice_flux_div_y': 'm/s',
    'ice_flux_div': 'm/s',
    'psi_delta': 'Sv',
    'melt_rate_x': 'm/s',
    'melt_rate_y': 'm/s',
    'melt_rate': 'm/s',
    'h_ice': 'm'
}

var_positive = {
    'geo_u': 'east',
    'geo_v': 'north',
    'wind_u': 'east',
    'wind_v': 'north',
    'alpha': '',
    'ice_u': 'east',
    'ice_v': 'north',
    'tau_air_x': 'east',
    'tau_air_y': 'north',
    'tau_ice_x': 'east',
    'tau_ice_y': 'north',
    'tau_SIZ_x': 'east',
    'tau_SIZ_y': 'north',
    'tau_x': 'east',
    'tau_y': 'north',
    'Ekman_u': 'east',
    'Ekman_v': 'north',
    'Ekman_SIZ_u': 'east',
    'Ekman_SIZ_v': 'north',
    'dtau_x_dy': 'east',
    'dtau_y_dx': 'north',
    'curl_stress': 'up',
    'Ekman_w': 'up',
    'salinity': '',
    'temperature': '',
    'neutral_density': '',
    'dSdx': 'east',
    'dSdy': 'north',
    'uEk_S_ddx': 'east',
    'uEk_S_ddy': 'north',
    'fwf_uEk_S': 'up',
    'ice_flux_div_x': 'east',
    'ice_flux_div_y': 'north',
    'ice_flux_div': 'up',
    'psi_delta': '',
    'melt_rate_x': 'east',
    'melt_rate_y': 'north',
    'melt_rate': 'up',
    'h_ice': 'up'
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
    'dtau_x_dy': 'd/dy (tau_x)',
    'dtau_y_dx': 'd/dx (tau_y)',
    'curl_stress': 'Wind stress curl',
    'Ekman_w': 'Ekman pumping',
    'salinity': 'Salinity (may be surface or depth-averaged)',
    'temperature': 'Temperature (may be surface or depth-averaged)',
    'neutral_density': 'Neutral density (may be surface or depth-averaged)',
    'dSdx': 'dS/dx',
    'dSdy': 'dS/dy',
    'uEk_S_ddx': 'd/dx (u_Ek * S)',
    'uEk_S_ddy': 'd/dy (v_Ek * S)',
    'fwf_uEk_S': 'Freshwater Ekman advection flux',
    'ice_flux_div_x': 'Zonal ice flux divergence',
    'ice_flux_div_y': 'Meridional ice flux divergence',
    'ice_flux_div': 'Ice flux divergence',
    'psi_delta': 'Meridional streamfunction ',
    'melt_rate_x': 'Zonal melting/freezing rate?',
    'melt_rate_y': 'Meridional melting/freezing rate?',
    'melt_rate': 'Melting/freezing rate',
    'h_ice': 'Sea ice thickness'
}

""" Constants for diagnostic field plots """
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
    'ice_div': 'div(alpha*h*u_ice)',
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
    'u_geo': 100,
    'v_geo': 100,
    'u_wind': 1,
    'v_wind': 1,
    'u_ice': 100,
    'v_ice': 100,
    'alpha': 1,
    'tau_air_x': 1,
    'tau_air_y': 1,
    'tau_ice_x': 1,
    'tau_ice_y': 1,
    'tau_x': 1,
    'tau_y': 1,
    'u_Ekman': 100,
    'v_Ekman': 100,
    'dtauydx': 1e7,
    'dtauxdy': -1e7,
    'curl_tau': 1e7,
    # 'tau_SIZ_x': 1,
    # 'tau_SIZ_y': 1,
    'w_Ekman': 3600 * 365 * 24,  # [m/s] -> [m/year]
    'freshwater_flux': 3600 * 365 * 24,  # [m/s] -> [m/year]
    'ice_div': 3600 * 365 * 24,  # [m/s] -> [m/year]
    'temperature': 1,
    'salinity': 1,
    'neutral_density': 1
}

colorbar_label = {
    'u_geo': 'cm/s',
    'v_geo': 'cm/s',
    'u_wind': 'm/s',
    'v_wind': 'm/s',
    'u_ice': 'cm/s',
    'v_ice': 'cm/s',
    'alpha': '',
    'tau_air_x': r'N/m$^2$',
    'tau_air_y': r'N/m$^2$',
    'tau_ice_x': r'N/m$^2$',
    'tau_ice_y': r'N/m$^2$',
    'tau_x': r'N/m$^2$',
    'tau_y': r'N/m$^2$',
    'u_Ekman': 'cm/s',
    'v_Ekman': 'cm/s',
    'dtauydx': r'10$^{-7}$ N/m$^3$',
    'dtauxdy': r'10$^{-7}$ N/m$^3$',
    'curl_tau': r'10$^{-7}$ N/m$^3$',
    # 'tau_SIZ_x': r'N/m$^2$',
    # 'tau_SIZ_y': r'N/m$^2$',
    'w_Ekman': r'm/year',
    'freshwater_flux': 'm/year',
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
    'u_geo': (-10, 10),
    'v_geo': (-10, 10),
    'u_wind': (-20, 20),
    'v_wind': (-20, 20),
    'u_ice': (-10, 10),
    'v_ice': (-10, 10),
    'alpha': (0, 1),
    'tau_air_x': (-0.15, 0.15),
    'tau_air_y': (-0.15, 0.15),
    'tau_ice_x': (-0.15, 0.15),
    'tau_ice_y': (-0.15, 0.15),
    'tau_x': (-0.15, 0.15),
    'tau_y': (-0.15, 0.15),
    'u_Ekman': (-3, 3),
    'v_Ekman': (-3, 3),
    'dtauydx': (-5, 5),
    'dtauxdy': (-5, 5),
    'curl_tau': (-5, 5),
    # 'tau_SIZ_x': (-0.5, 0.5),
    # 'tau_SIZ_y': (-0.5, 0.5),
    'w_Ekman': (-100, 100),
    'freshwater_flux': (-5, 5),
    'ice_div': (-3, 3),
    'temperature': (-2.5, 2.5),
    'salinity': (33.75, 35),
    'neutral_density': (27, 28)
}
