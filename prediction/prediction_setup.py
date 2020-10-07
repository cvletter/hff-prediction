import pandas as pd
import datetime
import prediction.general_purpose_functions as gf
import prediction.file_management as fm
import prediction.column_names as cn


def split_products(active_products, min_obs=cn.TRAIN_OBS, prediction_date=cn.PREDICTION_DATE,
                   hold_out=cn.PREDICTION_WINDOW):
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

    products_model = active_products[series_to_model].copy(deep=True)
    products_model[cn.MOD_PROD_SUM] = products_model.sum(axis=1)
    products_no_model = active_products[series_not_to_model]

    return products_model, products_no_model


def fill_missing_values(data):
    data.fillna(value=0, inplace=True)


def get_top_correlations(y, y_lags, top_correl=5):
    all_correlations = pd.DataFrame(columns=y_lags.columns, index=y.columns)

    for i in y.columns:
        for j in y_lags.columns:
            if j[:-6] == i:
                all_correlations.loc[i, j] = -1e9
            else:
                all_correlations.loc[i, j] = abs(y[i].corr(y_lags[j]))

    top_correlations = {}
    for p in all_correlations.index:
        top_correlations[p] = all_correlations.loc[p].sort_values(ascending=False)[:top_correl].index

    return top_correlations


def create_lags(input_data, n_lags=cn.N_LAGS, prediction_window=cn.PREDICTION_WINDOW):
    data_lags = input_data.copy(deep=True)
    first_lag_cols = []
    for product in input_data.columns:
        first_lag_cols.append("{}_lag_{}".format(product, prediction_window))

    data_lags.columns = first_lag_cols
    data_lags.sort_index(ascending=False, inplace=True)

    for lag in range(1, n_lags):
        for product in input_data.columns:
            lag_name = "{}_lag_{}".format(product, lag + prediction_window)
            data_lags[lag_name] = input_data[product].shift(-lag)

    data_lags['prediction_date'] = data_lags.index + datetime.timedelta(days=prediction_window * 7)
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


def create_model_setup(y_m, y_nm, X_exog, difference=True, lags=cn.N_LAGS, prediction_date=cn.PREDICTION_DATE,
                       hold_out=cn.PREDICTION_WINDOW):
    last_train_date = prediction_date - datetime.timedelta(weeks=hold_out)

    fill_missing_values(data=y_m)
    fill_missing_values(data=y_nm)

    y_m_ltd = y_m.loc[last_train_date]
    y_nm_ltd = y_nm.loc[last_train_date]

    if difference:
        y_m = first_difference_data(undifferenced_data=y_m, delta=1, scale=False)
        y_nm = first_difference_data(undifferenced_data=y_nm, delta=1, scale=False)

    # TODO CREATE CORRELATION FEATURE SELECTION HERE

    y_ar_m = create_lags(input_data=y_m, n_lags=lags, prediction_window=hold_out)
    y_ar_nm = create_lags(input_data=y_nm, n_lags=lags, prediction_window=hold_out)

    top_corr = get_top_correlations(y=y_m, y_lags=y_ar_m, top_correl=5)

    y_ar_m_fit = y_ar_m.loc[last_train_date:]
    X_exog_fit = X_exog.loc[y_ar_m_fit.index]
    y_true_fit = y_m.loc[y_ar_m_fit.index]

    yl_ar_m_prd = y_ar_m.loc[prediction_date]
    yl_ar_nm_prd = y_ar_nm.loc[prediction_date]
    X_exog_prd = X_exog.loc[prediction_date]

    model_fitting = {cn.Y_TRUE: y_true_fit,
                     cn.Y_AR: y_ar_m_fit,
                     cn.X_EXOG: X_exog_fit,
                     "correlations": top_corr}

    model_prediction = {cn.Y_AR_M: yl_ar_m_prd,
                        cn.Y_AR_NM: yl_ar_nm_prd,
                        cn.X_EXOG: X_exog_prd,
                        cn.Y_M_UNDIF: y_m_ltd,
                        cn.Y_NM_UNDIF: y_nm_ltd}

    return model_fitting, model_prediction


def prediction_setup_wrapper(prediction_date, prediction_window, train_obs,
                             nlags, difference,
                             act_products, exog_features,
                             save_to_pkl=False):
    if type(prediction_date) == str:
        prediction_date = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")

    products_model, products_nmodel = split_products(active_products=act_products,
                                                     min_obs=train_obs,
                                                     prediction_date=prediction_date,
                                                     hold_out=prediction_window)

    data_fitting, data_prediction = create_model_setup(y_m=products_model,
                                                       y_nm=products_nmodel,
                                                       prediction_date=prediction_date,
                                                       hold_out=prediction_window,
                                                       X_exog=exog_features,
                                                       difference=difference,
                                                       lags=nlags)

    if save_to_pkl:
        gf.save_to_pkl(data=data_fitting, file_name='fit_data', folder=fm.SAVE_LOC)
        gf.save_to_pkl(data=data_prediction, file_name='prediction_data', folder=fm.SAVE_LOC)

    return data_fitting, data_prediction


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
                                                     hold_out=1)

    data_fitting, data_prediction = create_model_setup(y_m=products_model,
                                                       y_nm=products_nmodel,
                                                       X_exog=exog_features,
                                                       lags=2,
                                                       prediction_date=cn.PREDICTION_DATE,
                                                       difference=False,
                                                       hold_out=1)

    gf.save_to_pkl(data=data_fitting, file_name='fit_data', folder=fm.SAVE_LOC)
    gf.save_to_pkl(data=data_prediction, file_name='prediction_data', folder=fm.SAVE_LOC)

    Y_raw = products_model
    fill_missing_values(Y_raw)

    Y_ar_corr = create_lags(input_data=Y_raw, n_lags=5, prediction_window=2)

    def get_top_correlations(Y, Y_lags, top_correl=5):
        all_correlations = pd.DataFrame(columns=Y_lags.columns, index=Y.columns)

        for i in Y.columns:
            for j in Y_lags.columns:
                if j[:-6] == i:
                    all_correlations.loc[i, j] = -1e9
                else:
                    all_correlations.loc[i, j] = abs(Y[i].corr(Y_lags[j]))

        top_correlations = {}
        for p in all_correlations.index:
            top_correlations[p] = all_correlations.loc[p].sort_values(ascending=False)[:top_correl].index

        return top_correlations, all_correlations

    top_corr, all_correl = get_top_correlations(Y=Y_raw, Y_lags=Y_ar)
