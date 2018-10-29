varz = ['tau_air_x_field', 'tau_air_y_field', 'tau_ice_x_field', 'tau_ice_y_field',
        'tau_SIZ_x_field', 'tau_SIZ_y_field', 'tau_x_field', 'tau_y_field', 'u_Ekman_field',
        'v_Ekman_field', 'u_Ekman_SIZ_field', 'v_Ekman_SIZ_field', 'u_geo_field', 'v_geo_field', 'u_wind_field',
        'v_wind_field', 'alpha_field', 'u_ice_field', 'v_ice_field', 'stress_curl_field', 'w_Ekman_field']

# for s in varz:
#     print('{:s}_avg = {:s}_avg + np.nan_to_num({:s})'.format(s, s, s))
#     print('{:s}[~np.isnan({:s})] = 1'.format(s, s))
#     print('{:s}[np.isnan({:s})] = 0'.format(s, s))
#     print('{:s}_days = {:s}_days + {:s}\n'.format(s, s, s))
#
#     print('{:s}_days = np.zeros((len(self.lats), len(self.lons)))'.format(s))
#
#     print('self.{:s} = np.divide({:s}_avg, {:s}_days)'.format(s, s, s))

varz2 = ['alpha', 'u_ice', 'v_ice', 'h_ice', 'zonal_div', 'merid_div', 'div',
         'hu_dadx', 'au_dhdx', 'ah_dudx', 'hv_dady', 'av_dhdy', 'ah_dvdy', 'div2',
         'salinity']

varz2prime = ['alpha', 'tau_io_x_nogeo', 'tau_io_y_nogeo', 'tau_io_x_geo', 'tau_io_y_geo', 'tau_ig_x', 'tau_ig_y',
              'tau_ao_x', 'tau_ao_y', 'w_Ek_nogeo', 'w_Ek_geo', 'w_a', 'w_i', 'w_i0', 'w_ig', 'w_A', 'gamma']

# for var in varz2prime:
#     print('{:s}_avg_field = {:s}_avg_field + np.nan_to_num({:s}_daily_field)'.format(var, var, var))
#     print('{:s}_daily_field[~np.isnan({:s}_daily_field)] = 1'.format(var, var))
#     print('{:s}_daily_field[np.isnan({:s}_daily_field)] = 0'.format(var, var))
#     print('{:s}_day_field = {:s}_day_field + {:s}_daily_field'.format(var, var, var))
#     print('')
#
for var in varz2prime:
    print('{:s}_avg_field = np.divide({:s}_avg_field, {:s}_day_field)'.format(var, var, var))

# varz3 = ['alpha', 'tau_ao_x', 'tau_ao_y', 'tau_io_x_nogeo', 'tau_io_y_nogeo', 'tau_io_x_geo', 'tau_io_y_geo',
#          'tau_ig_x', 'tau_ig_y']
#
# for var in varz3:
#     print('{:s}_ij = {:s}_daily_field[i][j]'.format(var, var))
#     print('{:s}_i_jp1 = {:s}_daily_field[i][jp1]'.format(var, var))
#     print('{:s}_i_jm1 = {:s}_daily_field[i][jm1]'.format(var, var))
#     print('{:s}_ip1_j = {:s}_daily_field[i+1][j]'.format(var, var))
#     print('{:s}_im1_j = {:s}_daily_field[i-1][j]'.format(var, var))
#     print("")