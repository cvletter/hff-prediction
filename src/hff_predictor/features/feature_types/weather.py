import hff_predictor.config.column_names as cn


def prep_weather_features(input_weer_data, index_col=cn.FIRST_DOW):
    if not input_weer_data.index.name == index_col:
        input_weer_data.reset_index(inplace=True, drop=True)
        input_weer_data.set_index(index_col, inplace=True)

    input_weer_data.sort_index(ascending=False, inplace=True)

    cols = [cn.TEMP_GEM, cn.ZONUREN]

    weer_data_a = input_weer_data[cols]
    weer_data_d = weer_data_a.diff(periods=-1)
    weer_data_d.columns = ["d_temperatuur_gem", "d_zonuren"]

    return weer_data_a.join(weer_data_d, how="left").dropna(how="any")