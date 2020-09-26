import pandas as pd
import datetime
import prediction.general_purpose_functions as gf
import prediction.file_management as fm
import prediction.column_names as cn


def split_products(active_products, min_obs=cn.TRAIN_OBS, prediction_date=cn.PREDICTION_DATE, hold_out=cn.PREDICTION_WINDOW):

    last_train_date = prediction_date - datetime.timedelta(weeks=hold_out)
    first_train_date = last_train_date - datetime.timedelta(weeks=min_obs)
    fitting_window = active_products.loc[last_train_date:first_train_date]

    active_products = active_products.loc[last_train_date:first_train_date]

    obs_count = pd.DataFrame(fitting_window.count())
    obs_count.columns = ['count']

    series_to_model = obs_count[obs_count['count'] >= min_obs].index
    print("Number of products able to model: {}".format(len(series_to_model)))

    series_not_to_model = obs_count[obs_count['count'] < min_obs].index
    print("Number of products not able to model: {}".format(len(series_not_to_model)))

    products_model = active_products[series_to_model]
    products_model[cn.MOD_PROD_SUM] = products_model.sum(axis=1)
    products_no_model = active_products[series_not_to_model]

    return products_model, products_no_model


def add_exogenous_features():
    pass

def fill_missing_values(data):
    data.fillna(value=0, inplace=True)


def create_lags(input_data, n_lags=cn.N_LAGS):
    data_lags = input_data.copy(deep=True)
    first_lag_cols = []
    for product in input_data.columns:
        first_lag_cols.append("{}_lag_{}".format(product, 1))

    data_lags.columns = first_lag_cols
    data_lags.sort_index(ascending=False, inplace=True)

    for lag in range(2, n_lags+1):
        for product in input_data.columns:
            lag_name = "{}_lag_{}".format(product, lag)
            data_lags[lag_name] = input_data[product].shift(-lag+1)

    data_lags['prediction_date'] = data_lags.index + datetime.timedelta(days=cn.PREDICTION_WINDOW * 7)
    data_lags.rename(index=data_lags['prediction_date'], inplace=True)

    data_lags.drop('prediction_date', axis=1, inplace=True)

    return data_lags[:-n_lags]



def first_difference_data(undifferenced_data, delta=1, scale=True):

    undifferenced_data.sort_index(ascending=True, inplace=True)
    differenced_data = undifferenced_data.diff(periods=delta)
    differenced_data.sort_index(ascending=False, inplace=True)
    undifferenced_data.sort_index(ascending=False, inplace=True)

    if scale:
        differenced_data = differenced_data / undifferenced_data.shift(-1)

    return differenced_data[:-delta]


# TODO: Add exogenous factors
def create_model_setup(y_m, y_nm, X_exog, difference=True, lags=cn.N_LAGS, prediction_date=cn.PREDICTION_DATE, hold_out=cn.PREDICTION_WINDOW):

    last_train_date = prediction_date - datetime.timedelta(weeks=hold_out)

    fill_missing_values(data=y_m)
    fill_missing_values(data=y_nm)

    if difference:
        y_m = first_difference_data(undifferenced_data=y_m, delta=1, scale=False)
        y_nm = first_difference_data(undifferenced_data=y_nm, delta=1, scale=False)

    y_ar_m = create_lags(input_data=y_m, n_lags=lags)
    y_ar_nm = create_lags(input_data=y_nm, n_lags=lags)

    y_ar_m_fit = y_ar_m.loc[last_train_date:]
    X_exog_fit = X_exog.loc[y_ar_m_fit.index]
    y_true_fit = y_m.loc[y_ar_m_fit.index]

    yl_ar_m_prd = y_ar_m.loc[prediction_date]
    yl_ar_nm_prd = y_ar_nm.loc[prediction_date]
    X_exog_prd = X_exog.loc[prediction_date]

    model_fitting = {'y_true': y_true_fit,
                    'y_ar': y_ar_m_fit,
                    'X_exog': X_exog_fit}

    model_prediction = {'y_ar_m': yl_ar_m_prd,
                        'y_ar_mm': yl_ar_nm_prd,
                        'X_exog': X_exog_prd}

    return model_fitting, model_prediction


if __name__ == '__main__':

    # NEW

    active_products = gf.import_temp_file(file_name=fm.ORDER_DATA_ACT,
                                         data_loc=fm.SAVE_LOC,
                                         set_index=True)

    inactive_products = gf.import_temp_file(file_name=fm.ORDER_DATA_INACT,
                                           data_loc=fm.SAVE_LOC,
                                           set_index=True)

    exog_features = gf.import_temp_file(file_name=fm.EXOG_FEATURES,
                                       data_loc=fm.SAVE_LOC,
                                       set_index=True)

    products_model, products_nmodel = split_products(active_products=active_products,
                                                     min_obs=cn.TRAIN_OBS,
                                                     prediction_date=cn.PREDICTION_DATE,
                                                     hold_out=cn.PREDICTION_WINDOW)

    data_fitting, data_prediction = create_model_setup(y_m=products_model,
                                                       y_nm=products_nmodel,
                                                       X_exog=exog_features
                                                       )

    gf.save_to_pkl(data=data_fitting, file_name='fit_data', folder=fm.SAVE_LOC)
    gf.save_to_pkl(data=data_prediction, file_name='prediction_data', folder=fm.SAVE_LOC)
