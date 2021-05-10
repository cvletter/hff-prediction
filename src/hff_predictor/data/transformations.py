import pandas as pd


def first_difference_data(undifferenced_data, delta=1, scale=True):
    undifferenced_data.sort_index(ascending=True, inplace=True)
    differenced_data = undifferenced_data.diff(periods=delta)
    differenced_data.sort_index(ascending=False, inplace=True)
    undifferenced_data.sort_index(ascending=False, inplace=True)

    if scale:
        differenced_data = differenced_data / undifferenced_data.shift(-1)

    return differenced_data[:-delta]


def fill_missing_values(data):
    data.fillna(value=0, inplace=True)


def create_lags(data, lag_range):
    data_lags = pd.DataFrame(index=data.index)

    if type(lag_range) is int:
        lag_range = list(reversed(range(-lag_range, 1)))

    data_columns = data.columns

    for i in data_columns:
        for l in lag_range:
            if l <= 0:
                _temp_name = "{}_last{}w".format(i, abs(l))
            else:
                _temp_name = "{}_next{}w".format(i, abs(l))

            data_lags[_temp_name] = data[i].shift(l)

    return data_lags
