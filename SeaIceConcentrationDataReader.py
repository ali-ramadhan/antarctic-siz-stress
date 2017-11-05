class SeaIceConcentrationDataReader(object):
    sic_data_dir_path = path.join(data_dir_path, 'NOAA_NSIDC_G02202_V3_SEA_ICE_CONCENTRATION', 'south', 'daily')

    def date_to_SIC_dataset_filepath(self, date):
        filename = 'seaice_conc_daily_sh_f17_' + str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)\
                   + '_v03r00.nc'
        return path.join(self.sic_data_dir_path, str(date.year), filename)

    def load_SIC_dataset(self, date):
        dataset_filepath = self.date_to_SIC_dataset_filepath(date)
        logger.info('Loading sea ice concentration dataset: %s', dataset_filepath)
        dataset = netCDF4.Dataset(dataset_filepath)
        logger.info('Successfully loaded sea ice concentration dataset: %s', dataset_filepath)
        log_netCDF_dataset_metadata(dataset)
        return dataset

    def __init__(self, date=None):
        if date is None:
            logger.info('SeaIceConcentrationDataReader object initialized but no dataset was loaded.')
            self.current_SIC_dataset = None
            self.current_date = None
        else:
            logger.info('SeaIceConcentrationDataReader object initializing...')
            self.current_SIC_dataset = self.load_SIC_dataset(date)
            self.current_date = date

    def sea_ice_concentration(self, lat, lon, date):
        assert -90 <= lat <= 90, "Latitude value {} out of bounds!".format(lat)
        assert -180 <= lon <= 180, "Longitude value {} out of bounds!".format(lon)

        if self.current_SIC_dataset is None:
            logger.info('sea_ice_concentration called with no current dataset loaded.')
            logger.info('Loading sea ice concentration dataset for date requested: {}'.format(date))
            self.current_SIC_dataset = self.load_SIC_dataset(date)
            self.current_date = date

        if date != self.current_date:
            logger.info('SIC at different date requested: {} -> {}.'.format(self.current_date, date))
            logger.info('Changing SIC dataset...')
            self.current_SIC_dataset = self.load_SIC_dataset(date)
            self.current_date = date

        x, y = latlon_to_polar_stereographic_xy(lat, lon)
        idx_x = np.abs(np.array(self.current_SIC_dataset.variables['xgrid']) - x).argmin()
        idx_y = np.abs(np.array(self.current_SIC_dataset.variables['ygrid']) - y).argmin()
        lat_xy = self.current_SIC_dataset.variables['latitude'][idx_y][idx_x]
        lon_xy = self.current_SIC_dataset.variables['longitude'][idx_y][idx_x]

        if np.abs(lat - lat_xy) > 0.5 or np.abs(lon - lon_xy) > 0.5:
            logger.warning('Lat or lon obtained from SIC dataset differ by more than 0.5 deg from input lat/lon!')
            logger.debug("lat = %f, lon = %f (input)", lat, lon)
            logger.debug("x = %f, y = %f (polar stereographic)", x, y)
            logger.debug("idx_x = %d, idx_y = %d", idx_x, idx_y)
            logger.debug("lat_xy = %f, lon_xy = %f (from SIC dataset)", lat_xy, lon_xy)

        # TODO: check for masked values, etc.
        return self.current_SIC_dataset.variables['goddard_nt_seaice_conc'][0][idx_y][idx_x]