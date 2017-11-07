from os import path

# Current Working Directory
cwd = path.dirname(path.abspath(__file__))

# Data directory string constants
# Everything under a ./data/ softlink?
data_dir_path = path.join(cwd, 'data')

# Physical constants
# Could use a theoretical gravity model: https://en.wikipedia.org/wiki/Theoretical_gravity
g = 9.80665  # standard acceleration due to gravity [m/s^2]
Omega = 7.292115e-5  # rotation rate of the Earth [rad/s]

# Earth radius R could be upgraded to be location-dependent using a simple formula.
# See: https://en.wikipedia.org/wiki/Earth_radius#Location-dependent_radii
R = 6371.228e3  # average radius of the earth [m]

lat_min = -50
lat_max = 70
lon_min = -180
lon_max = 180

lat_step = 0.25
lon_step = 0.25