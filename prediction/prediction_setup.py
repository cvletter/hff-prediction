import pandas as pd
import datetime
import prediction.general_purpose_functions as gf


def select_products_to_predict(active_sold_products, min_obs=70, eval_week='2020-08-24'):
    eval_date = datetime.datetime.strptime(eval_week, "%Y-%m-%d")
    end_date = eval_date - datetime.timedelta(weeks=min_obs)
    fitting_window = active_sold_products.loc[eval_date:end_date]
    obs_count = pd.DataFrame(fitting_window.count())
    obs_count.columns = ['count']

    series_to_model = obs_count[obs_count['count'] >= min_obs].index
    series_not_to_model = obs_count[obs_count['count'] < min_obs].index

    return active_sold_products[series_to_model], active_sold_products[series_not_to_model]


def add_exogenous_features():
    pass


def split_train_test(data, eval_week='2020-08-24', test_size=10, train_size=60):

    eval_week = datetime.datetime.strptime(eval_week, "%Y-%m-%d")
    split_date = eval_week - datetime.timedelta(weeks=test_size)

    first_train_date = split_date - datetime.timedelta(weeks=1)
    last_train_date = first_train_date - datetime.timedelta(weeks=train_size)

    test_data = data.loc[:split_date]
    train_data = data.loc[split_date:]
    train_data = train_data.iloc[1:]
    train_data = train_data.loc[:last_train_date]

    return train_data, test_data


def fill_missing_values(data):
    data.fillna(value=0, inplace=True)


def create_lags(input_data, n_lags=2):
    data_lags = pd.DataFrame(index=input_data.index)

    data_lags.drop('week_jaar', axis=1, inplace=True, errors='ignore')
    data_lags.sort_index(ascending=False, inplace=True)

    for lag in range(1, n_lags+1):
        for product in input_data.columns:
            lag_name = "{}_lag_{}".format(product, lag)
            data_lags[lag_name] = input_data[product].shift(-lag)

    return data_lags[data_lags.columns.sort_values()][:-n_lags]


def first_difference_data(undifferenced_data, delta=1, scale=True):

    undifferenced_data.sort_index(ascending=True, inplace=True)
    differenced_data = undifferenced_data.diff(periods=delta)
    differenced_data.sort_index(ascending=False, inplace=True)
    undifferenced_data.sort_index(ascending=False, inplace=True)

    if scale:
        differenced_data = differenced_data / undifferenced_data.shift(-1)

    return differenced_data[:-delta]


# TODO: Add exogenous factors
def create_model_setup(y, difference=True, lags=2):
    if difference:
        y = first_difference_data(undifferenced_data=y, delta=1, scale=False)

    return y[:-lags], create_lags(input_data=y, n_lags=lags)


if __name__ == '__main__':

    DATA_LOC = '/Users/cornelisvletter/Google Drive/HFF/Data/Prepared'
    FILE_NAME = 'actieve_halffabricaten_wk_2020916-1128.csv'
    import_name = '{}/{}'.format(DATA_LOC, FILE_NAME)

    order_data = pd.read_csv(import_name, sep=";", decimal=",")
    order_data['eerste_dag_week'] = pd.to_datetime(order_data['eerste_dag_week'], format='%Y-%m-%d')
    order_data.set_index('eerste_dag_week', inplace=True)

    order_data_pred, order_data_npred = select_products_to_predict(active_sold_products=order_data)
    order_train, order_test = split_train_test(data=order_data_pred)
    fill_missing_values(order_train)

    exog_data, ar_components = create_model_setup(y=order_train, difference=True, lags=2)

    gf.save_to_csv(data=exog_data, file_name='producten_pred_diff', folder=DATA_LOC)
    gf.save_to_csv(data=ar_components, file_name='producten_pred_ar_diff', folder=DATA_LOC)




