import pandas as pd
import numpy as np
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


def create_lags(input_data, n_lags=cn.N_LAGS):
    data_lags = pd.DataFrame(index=input_data.index)

    for i in input_data.columns:
        for k in range(0, n_lags):
            _temp_name = "{}_last{}w".format(i, k)
            data_lags[_temp_name] = input_data[i].shift(-k)

    return data_lags[:-n_lags]


def first_difference_data(undifferenced_data, delta=1, scale=True):
    undifferenced_data.sort_index(ascending=True, inplace=True)
    differenced_data = undifferenced_data.diff(periods=delta)
    differenced_data.sort_index(ascending=False, inplace=True)
    undifferenced_data.sort_index(ascending=False, inplace=True)

    if scale:
        differenced_data = differenced_data / undifferenced_data.shift(-1)

    return differenced_data[:-delta]


def create_model_setup(y_m, y_nm, X_exog, difference=False, lags=cn.N_LAGS, prediction_date=cn.PREDICTION_DATE,
                       hold_out=cn.PREDICTION_WINDOW):

    def create_predictive_context(mod, non_mod, exog_f, hold_out=hold_out):

        exog_f = exog_f.loc[mod.index]

        return mod.shift(-hold_out)[:-hold_out], non_mod.shift(-hold_out)[:-hold_out], \
               exog_f.shift(-hold_out)[:-hold_out], exog_f

    last_train_date = prediction_date - datetime.timedelta(weeks=hold_out)

    fill_missing_values(data=y_m)
    fill_missing_values(data=y_nm)

    y_m_ltd = y_m.loc[last_train_date]
    y_nm_ltd = y_nm.loc[last_train_date]

    if difference:
        y_m = first_difference_data(undifferenced_data=y_m, delta=1, scale=False)
        y_nm = first_difference_data(undifferenced_data=y_nm, delta=1, scale=False)

    y_m_lags = create_lags(input_data=y_m, n_lags=lags)
    y_nm_lags = create_lags(input_data=y_nm, n_lags=lags)

    X_exog_nl = X_exog[cn.SEASONAL_COLS]
    X_exog_ml = X_exog.drop(cn.SEASONAL_COLS, axis=1)

    y_ar_m, y_ar_nm, X_exog_mll, X_exog_mlt = create_predictive_context(mod=y_m_lags, non_mod=y_nm_lags, exog_f=X_exog_ml,
                                                          hold_out=hold_out)

    X_exog_l = pd.concat([X_exog_mll, X_exog_nl], axis=1)
    X_exog_t = pd.concat([X_exog_mlt, X_exog_nl.shift(hold_out)], axis=1)

    y_ar_m_fit = y_ar_m.loc[last_train_date:]
    X_exog_fit = X_exog_l.loc[y_ar_m_fit.index]
    y_true_fit = y_m.loc[y_ar_m_fit.index]

    yl_ar_m_prd = y_m_lags.loc[last_train_date]
    yl_ar_nm_prd = y_nm_lags.loc[last_train_date]
    X_exog_prd = X_exog_t.loc[last_train_date]

    yl_ar_m_prd.name += datetime.timedelta(days=hold_out * 7)
    yl_ar_nm_prd.name += datetime.timedelta(days=hold_out * 7)
    X_exog_prd.name += datetime.timedelta(days=hold_out * 7)


    model_fitting = {cn.Y_TRUE: y_true_fit,
                     cn.Y_AR: y_ar_m_fit,
                     cn.X_EXOG: X_exog_fit,
                     cn.MOD_PROD: y_m.columns,
                     cn.NON_MOD_PROD: y_nm.columns}

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

    if products_model.shape[1] == 1:
        train_obs_reduced = train_obs - 10
        products_model, products_nmodel = split_products(active_products=act_products,
                                                         min_obs=train_obs_reduced,
                                                         prediction_date=prediction_date,
                                                         hold_out=prediction_window)

        print("Reduced train obs to have modelable products.")

    data_fitting, data_prediction = create_model_setup(y_m=products_model,
                                                       y_nm=products_nmodel,
                                                       prediction_date=prediction_date,
                                                       hold_out=prediction_window,
                                                       X_exog=exog_features,
                                                       difference=difference,
                                                       lags=nlags)

    if save_to_pkl:
        gf.save_to_pkl(data=data_fitting, file_name='fit_data', folder=fm.SAVE_LOC)
        gf.save_to_pkl(data=data_prediction, file_name='predict_data', folder=fm.SAVE_LOC)

    return data_fitting, data_prediction


if __name__ == '__main__':

    active_products_t = gf.import_temp_file(file_name=fm.ORDER_DATA_ACT,
                                            data_loc=fm.SAVE_LOC,
                                            set_index=True)

    inactive_products_t = gf.import_temp_file(file_name=fm.ORDER_DATA_INACT,
                                              data_loc=fm.SAVE_LOC,
                                              set_index=True)

    exog_features_t = gf.import_temp_file(file_name=fm.EXOG_FEATURES,
                                          data_loc=fm.SAVE_LOC,
                                          set_index=True)

    data_fitting_t, data_prediction_t = prediction_setup_wrapper(prediction_date='2020-10-05',
                                                                 prediction_window=2,
                                                                 train_obs=cn.TRAIN_OBS,
                                                                 nlags=3,
                                                                 difference=False,
                                                                 act_products=active_products_t,
                                                                 exog_features=exog_features_t,
                                                                 save_to_pkl=True)

    gf.save_to_pkl(data=data_fitting_t, file_name='fit_data', folder=fm.SAVE_LOC)
    gf.save_to_pkl(data=data_prediction_t, file_name='predict_data', folder=fm.SAVE_LOC)


