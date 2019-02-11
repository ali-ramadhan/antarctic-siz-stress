from os import path

from matplotlib import cm
import cmocean.cm

""" Data directory constants """
# Current Working Directory
cwd = path.dirname(path.abspath(__file__))

# Data directory string constants
# data_dir_path = path.join(cwd, 'data')
# data_dir_path = 'E:\\data\\'
data_dir_path = '/d1/alir/data/'

# Output data directory
# output_dir_path = path.join(cwd, 'output')
# output_dir_path = 'E:\\output\\'
# output_dir_path = 'C:\\Users\\Ali\\Downloads\\output\\'
output_dir_path = '/d1/alir/output/'

# figure_dir_path = 'E:\\figures\\antarctic-siz-stress\\'
figure_dir_path = "/d1/alir/figures/"

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

rho_ice = 925  # [kg/m^3] nominal sea ice density
rho_fw = 1000  # [kg/m^3] reference freshwater density
s_ice = 6    # [g/kg] sea oce salinity
s_sw = 34.7  # [g/kg] reference seawater salinity
C_fw = rho_ice * (1 - s_ice/s_sw) / rho_fw  # conversion factor between sea ice volume flux to a freshwater equivalent

# Estimate of the Osborne-Cox diffusivity in the Southern Ocean from Abernathey & Marhsall, "Global surface eddy
# diffusivities derived from satellite altimetry", JGR Oceans (2013)
kappa = 400  # [m^2/s]

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
dot_interp_method = 'linear'

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
    'tau_nogeo_air_x': 'N/m^2',
    'tau_nogeo_air_y': 'N/m^2',
    'tau_nogeo_ice_x': 'N/m^2',
    'tau_nogeo_ice_y': 'N/m^2',
    'tau_nogeo_SIZ_x': 'N/m^2',
    'tau_nogeo_SIZ_y': 'N/m^2',
    'tau_nogeo_x': 'N/m^2',
    'tau_nogeo_y': 'N/m^2',
    'tau_ig_x': 'N/m^2',
    'tau_ig_y': 'N/m^2',
    'tau_ice_dot_u_geo': 'N/m*s',
    'tau_ice_dot_u_Ekman': 'N/m*s',
    'tau_ice_dot_u_ocean': 'N/m*s',
    'Ekman_u': 'm/s',
    'Ekman_v': 'm/s',
    'Ekman_SIZ_u': 'm/s',
    'Ekman_SIZ_v': 'm/s',
    'Ekman_U': 'm^2/s',
    'Ekman_V': 'm^2/s',
    'Ekman_SIZ_U': 'm^2/s',
    'Ekman_SIZ_V': 'm^2/s',
    'dtau_x_dy': 'N/m^3',
    'dtau_y_dx': 'N/m^3',
    'curl_stress': 'N/m^3',
    'Ekman_w': 'm/s',
    'ddy_tau_nogeo_x': 'N/m^3',
    'ddx_tau_nogeo_y': 'N/m^3',
    'stress_curl_nogeo': 'N/m^3',
    'w_Ekman_nogeo': 'm/s',
    'w_a': 'm/s',
    'w_i': 'm/s',
    'w_i0': 'm/s',
    'w_ig': 'm/s',
    'w_A': 'm/s',
    'gamma_metric': 'unitless',
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
    'tau_nogeo_air_x': 'east',
    'tau_nogeo_air_y': 'north',
    'tau_nogeo_ice_x': 'east',
    'tau_nogeo_ice_y': 'north',
    'tau_nogeo_SIZ_x': 'east',
    'tau_nogeo_SIZ_y': 'north',
    'tau_nogeo_x': 'east',
    'tau_nogeo_y': 'north',
    'tau_ig_x': 'east',
    'tau_ig_y': 'north',
    'tau_ice_dot_u_geo': '',
    'tau_ice_dot_u_Ekman': '',
    'tau_ice_dot_u_ocean': '',
    'Ekman_u': 'east',
    'Ekman_v': 'north',
    'Ekman_SIZ_u': 'east',
    'Ekman_SIZ_v': 'north',
    'Ekman_U': 'east',
    'Ekman_V': 'north',
    'Ekman_SIZ_U': 'east',
    'Ekman_SIZ_V': 'north',
    'dtau_x_dy': 'east',
    'dtau_y_dx': 'north',
    'curl_stress': 'up',
    'Ekman_w': 'up',
    'ddy_tau_nogeo_x': 'east',
    'ddx_tau_nogeo_y': 'north',
    'stress_curl_nogeo': 'up',
    'w_Ekman_nogeo': 'up',
    'w_a': 'up',
    'w_i': 'up',
    'w_i0': 'up',
    'w_ig': 'up',
    'w_A': 'up',
    'gamma_metric': 'unitless',
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
    'tau_nogeo_air_x': 'Zonal air surface stress (neglecting geostrophic currents)',
    'tau_nogeo_air_y': 'Meridional air surface stress (neglecting geostrophic currents)',
    'tau_nogeo_ice_x': 'Zonal sea ice surface stress (neglecting geostrophic currents)',
    'tau_nogeo_ice_y': 'Meridional sea ice surface stress (neglecting geostrophic currents)',
    'tau_nogeo_SIZ_x': 'Zonal surface stress in the SIZ (neglecting geostrophic currents)',
    'tau_nogeo_SIZ_y': 'Meridional surface stress in the SIZ (neglecting geostrophic currents)',
    'tau_nogeo_x': 'Zonal surface stress (neglecting geostrophic currents)',
    'tau_nogeo_y': 'Meridional surface stress (neglecting geostrophic currents)',
    'tau_ig_x': 'Difference between zonal surface stress with and without geostrophic currents',
    'tau_ig_y': 'Difference between meridional surface stress with and without geostrophic currents',
    'tau_ice_dot_u_geo': 'Ice-ocean surface stress dotted with geostrophic velocity',
    'tau_ice_dot_u_Ekman': 'Ice-ocean surface stress dotted with Ekman surface velocity',
    'tau_ice_dot_u_ocean': 'Ice-ocean surface stress dotted with ocean (geostrophic + Ekman) velocity',
    'Ekman_u': 'Zonal surface Ekman velocity',
    'Ekman_v': 'Meridional surface Ekman velocity',
    'Ekman_SIZ_u': 'Zonal surface Ekman velocity in the SIZ',
    'Ekman_SIZ_v': 'Meridional surface Ekman velocity in the SIZ',
    'Ekman_U': 'Zonal Ekman volume transport',
    'Ekman_V': 'Meridional Ekman volume transport',
    'Ekman_SIZ_U': 'Zonal Ekman volume transport in the SIZ',
    'Ekman_SIZ_V': 'Meridional Ekman volume transport in the SIZ',
    'dtau_x_dy': 'd/dy (tau_x)',
    'dtau_y_dx': 'd/dx (tau_y)',
    'curl_stress': 'Wind stress curl',
    'Ekman_w': 'Ekman pumping',
    'ddy_tau_nogeo_x': 'd/dy (tau_nogeo_x)',
    'ddx_tau_nogeo_y': 'd/dx (tau_nogeo_y)',
    'stress_curl_nogeo': 'Wind stress curl (neglecting geostrophic currents)',
    'w_Ekman_nogeo': 'Ekman pumping (neglecting geostrophic currents)',
    'w_a': 'Ekman pumping due to air-ocean stresses (wind+geo)',
    'w_i': 'Ekman pumping due to ice-ocean stresses (ice+geo)',
    'w_i0': 'Ekman pumping due to ice-ocean stresses (ice, i.e. neglecting geostrophic currents)',
    'w_ig': 'Ekman pumping due to the inclusion of geostrophic currents (geo)',
    'w_A': 'Ekman pumping in an ice-free ocean',
    'gamma_metric': 'Metric of importance of the geostrophic current relative to the total Ekman pumping',
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
    'U_Ekman': 'U_Ekman',
    'V_Ekman': 'V_Ekman',
    'dtauydx': 'd/dx (tau_y)',
    'dtauxdy': '-d/dy (tau_x)',
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
    'U_Ekman': (0, 6),
    'V_Ekman': (1, 6),
    'dtauydx': (0, 7),
    'dtauxdy': (1, 7),
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
    'U_Ekman': 1,
    'V_Ekman': 1,
    'dtauydx': 1e7,
    'dtauxdy': -1e7,
    'curl_tau': 1e7,
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
    'U_Ekman': r'm$^2$/s',
    'V_Ekman': r'm$^2$/s',
    'dtauydx': r'10$^{-7}$ N/m$^3$',
    'dtauxdy': r'10$^{-7}$ N/m$^3$',
    'curl_tau': r'10$^{-7}$ N/m$^3$',
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
    'U_Ekman': 'seismic',
    'V_Ekman': 'seismic',
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
    'U_Ekman': (-2, 2),
    'V_Ekman': (-2, 2),
    'dtauydx': (-5, 5),
    'dtauxdy': (-5, 5),
    'curl_tau': (-5, 5),
    'w_Ekman': (-100, 100),
    'freshwater_flux': (-5, 5),
    'ice_div': (-3, 3),
    'temperature': (-2.5, 2.5),
    'salinity': (33.75, 35),
    'neutral_density': (27, 28)
}
