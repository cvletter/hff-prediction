
weer_data_sl = weer_data[['temperatuur_gem', 'zonuren', 'neerslag_mm']]
weer_data_last_w = weer_data_sl.shift(-1)
weer_data_last_w.columns = ['temp_lw', 'zonuren_lw', 'neerslag_lw']
weer_data_next_w = weer_data_sl.shift(1)
weer_data_next_w.columns = ['temp_nw', 'zonuren_nw', 'neerslag_nw']

order_data_weer_1 = order_data_wk.join(weer_data_sl, how='left') \
    .join(weer_data_last_w, how='left') \
    .join(weer_data_next_w, how='left')

order_data_weer_1.dropna(how='any', inplace=True)


def prep_weather_features():
    pass


def prep_holiday_features():
    pass


def prep_covid_features():
    pass


def prep_exogenous_features():
    pass
