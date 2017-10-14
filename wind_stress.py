import numpy as np

def distance(lat1, lon1, lat2, lon2):
    R = 6371e3  # average radius of the earth [m]
                # Could be upgraded to be location-dependent using a simple formula

    dLat = np.deg2rad(lat2 - lat1)
    dLon = np.deg2rad(lon2 - lon1)
    a = np.sin(dLat/2)*np.sin(dLat/2) + np.cos(np.deg2rad(lat1))*np.cos(np.deg2rad(lat2))*np.sin(dLon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R*c

    return d

