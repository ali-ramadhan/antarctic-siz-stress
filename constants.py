from os import path

""" Data directory constants """
# Current Working Directory
cwd = path.dirname(path.abspath(__file__))

# Data directory string constants
# Everything under a ./data/ softlink?
data_dir_path = path.join(cwd, 'data')
# data_dir_path = 'D:\\data\\'

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
C_seawater = 0.0055  # Draf coefficient

rho_0 = 1027.5  # "reference" density [kg/m^3]
D_e = 20  # Ekman layer depth [m]

""" Latitude-longitude interpolation grid constants """
lat_min = -75
lat_max = -55
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
alpha_interp_method = 'linear'
u_ice_interp_method = 'linear'
u_wind_interp_method = 'linear'
