import pandas as pd
import hff_predictor.config.column_names as cn


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

    # Verzekeren dat data in juiste volgorde staat gesorteerd
    data_temp = data.sort_index(ascending=False, inplace=False)
    data_lags = pd.DataFrame(index=data_temp.index)

    if type(lag_range) is int:
        lag_range = list(reversed(range(-lag_range, 1)))

    data_columns = data.columns

    for i in data_columns:
        for l in lag_range:
            if l <= 0:
                _temp_name = "{}_last{}w".format(i, abs(l))
            else:
                _temp_name = "{}_next{}w".format(i, abs(l))

            data_lags[_temp_name] = data_temp[i].shift(l)

    return data_lags


def find_rol_products(data, consumentgroep_nrs):

    if type(data) == pd.DataFrame:
        data_cols = data.columns
    else:
        data_cols = data

    # Bepalen welke van de actieve producten rol producten zijn
    rol_products = consumentgroep_nrs[consumentgroep_nrs[cn.CONSUMENT_GROEP_NR] == 16].index
    return list(set.intersection(set(rol_products), set(data_cols)))