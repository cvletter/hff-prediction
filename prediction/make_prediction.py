from prediction.data_preparation import data_prep_wrapper
from prediction.create_features import prep_all_features
from prediction.prediction_setup import prediction_setup_wrapper
from prediction.fit_model import fit_and_predict
from prediction.prediction_evaluation import in_sample_plot
import prediction.file_management as fm
import prediction.column_names as cn
import prediction.general_purpose_functions as gf
import pandas as pd
import time
import numpy as np


def run_prediction_bootstrap(date_to_predict=cn.PREDICTION_DATE, prediction_window=cn.PREDICTION_WINDOW,
                             train_obs=cn.TRAIN_OBS,
                             difference=False, lags=cn.N_LAGS, order_data=fm.RAW_DATA, weather_data=fm.WEER_DATA,
                             campaign_data=fm.CAMPAIGN_DATA,
                             product_data=fm.PRODUCT_STATUS, model_type='OLS', feature_threshold=None,
                             bootstrap_iter=None):
    if feature_threshold is None:
        feature_threshold = [0.2, 15]

    if bootstrap_iter is None:
        do_bootstrap = False

    else:
        do_bootstrap = True

    def convert_series_to_dataframe(input_series, date_val, index_name=cn.FIRST_DOW):
        input_df = pd.DataFrame(input_series).T
        input_df[index_name] = date_val
        return input_df.set_index(index_name, drop=True, inplace=False)

    def in_sample_error(all_fits, all_true_values):
        fit_error = abs(all_fits.subtract(all_true_values[all_fits.columns], axis='index'))
        avg_fit_error = fit_error.mean(axis=0)
        corr_true_values = all_true_values.replace(0, 1)
        pct_fit_error = fit_error / corr_true_values
        avg_pct_fit_error = pct_fit_error.mean(axis=0)

        avg_fit_error_df = convert_series_to_dataframe(input_series=avg_fit_error, date_val=date_to_predict)
        avg_pct_fit_error_df = convert_series_to_dataframe(input_series=avg_pct_fit_error, date_val=date_to_predict)

        return avg_fit_error_df, avg_pct_fit_error_df

    #Catch all output
    all_output = {}

    # Import and prepare data
    active_products, inactive_products, weather_data_processed, order_data_su, campaign_data_pr = data_prep_wrapper(
        prediction_date=date_to_predict,
        prediction_window=prediction_window,
        reload_data=False,
        campaign_data_loc=campaign_data,
        order_data_loc=order_data,
        weer_data_loc=weather_data,
        product_data_loc=product_data,
        agg_weekly=True, exclude_su=True,
        save_to_csv=False)

    exogenous_features = prep_all_features(weather_data_processed=weather_data_processed,
                                           order_data_su=order_data_su,
                                           campaign_data_su=campaign_data_pr,
                                           prediction_date=date_to_predict,
                                           train_obs=train_obs,
                                           save_to_csv=False)

    fit_data, predict_data = prediction_setup_wrapper(
        prediction_date=date_to_predict,
        prediction_window=prediction_window,
        train_obs=train_obs,
        nlags=lags,
        difference=difference,
        act_products=active_products,
        exog_features=exogenous_features,
        save_to_pkl=False)

    in_sample_fits, all_predictions, all_pars = fit_and_predict(
        fit_dict=fit_data, predict_dict=predict_data,
        model_type=model_type,
        feature_threshold=[feature_threshold[0],
                           feature_threshold[1]])

    all_output[date_to_predict] = {}
    all_output[date_to_predict][cn.MOD_PROD] = fit_data[cn.MOD_PROD]
    all_output[date_to_predict][cn.NON_MOD_PROD] = fit_data[cn.NON_MOD_PROD]

    all_output[date_to_predict][cn.SELECTED_FEATURES] = all_pars

    avg_fit_err, avg_pct_err = in_sample_error(all_fits=in_sample_fits, all_true_values=fit_data[cn.Y_TRUE])

    all_output[date_to_predict][cn.FIT_ERROR_ABS] = avg_fit_err
    all_output[date_to_predict][cn.FIT_ERROR_PCT] = avg_pct_err

    if do_bootstrap:
        all_predictions[cn.BOOTSTRAP_ITER] = 0

        for i in range(1, bootstrap_iter):
            print("Running iteration {} of {}".format(i, bootstrap_iter))
            fits, temp_os, pars = fit_and_predict(fit_dict=fit_data, predict_dict=predict_data, bootstrap=True,
                                                  model_type=model_type, feature_threshold=[feature_threshold[0],
                                                                                            feature_threshold[1]])
            temp_os[cn.BOOTSTRAP_ITER] = i

            all_predictions = pd.concat([all_predictions, temp_os])

            na_values = all_predictions.isna().sum().sum()
            print("In {} there are {} na_values".format(date_to_predict, na_values))

    all_output[date_to_predict][cn.PREDICTION_OS] = all_predictions

    return all_output


def run_prediction(date_to_predict=cn.PREDICTION_DATE, prediction_window=cn.PREDICTION_WINDOW, train_obs=cn.TRAIN_OBS,
                   difference=False, lags=cn.N_LAGS, order_data=fm.RAW_DATA, weather_data=fm.WEER_DATA,
                   product_data=fm.PRODUCT_STATUS, model_type='OLS', feature_threshold=None):
    if feature_threshold is None:
        feature_threshold = [0.2, 15]

    def convert_series_to_dataframe(input_series, date_val, index_name=cn.FIRST_DOW):
        input_df = pd.DataFrame(input_series).T
        input_df[index_name] = date_val
        return input_df.set_index(index_name, drop=True, inplace=False)

    def in_sample_error(all_fits, all_true_values):
        fit_error = abs(all_fits.subtract(all_true_values[all_fits.columns], axis='index'))
        avg_fit_error = fit_error.mean(axis=0)
        corr_true_values = all_true_values.replace(0, 1)
        pct_fit_error = fit_error / corr_true_values
        avg_pct_fit_error = pct_fit_error.mean(axis=0)

        avg_fit_error_df = convert_series_to_dataframe(input_series=avg_fit_error, date_val=date_to_predict)
        avg_pct_fit_error_df = convert_series_to_dataframe(input_series=avg_pct_fit_error, date_val=date_to_predict)

        return avg_fit_error_df, avg_pct_fit_error_df

    # Import and prepare data
    active_products, inactive_products, weather_data_processed, order_data_su = data_prep_wrapper(
        prediction_date=date_to_predict,
        prediction_window=prediction_window,
        reload_data=False,
        order_data_loc=order_data,
        weer_data_loc=weather_data,
        product_data_loc=product_data,
        agg_weekly=True, exclude_su=True,
        save_to_csv=False)

    exogenous_features = prep_all_features(weather_data_processed=weather_data_processed,
                                           order_data_su=order_data_su,
                                           prediction_date=date_to_predict,
                                           train_obs=train_obs,
                                           save_to_csv=False)

    fit_data, predict_data = prediction_setup_wrapper(
        prediction_date=date_to_predict,
        prediction_window=prediction_window,
        train_obs=train_obs,
        nlags=lags,
        difference=difference,
        act_products=active_products,
        exog_features=exogenous_features,
        save_to_pkl=False)

    in_sample_fit, prediction_os, __ = fit_and_predict(
        fit_dict=fit_data, predict_dict=predict_data,
        model_type=model_type,
        feature_threshold=[feature_threshold[0],
                           feature_threshold[1]])

    fit_data['avg_fit_error'], fit_data['avg_pct_fit_error'] = in_sample_error(all_fits=in_sample_fit,
                                                                               all_true_values=fit_data['y_true'])

    return in_sample_fit, prediction_os, fit_data, predict_data


def batch_prediction(prediction_dates, model_settings):
    p_window = model_settings['prediction_window']
    train_size = model_settings['train_size']
    differencing = model_settings['differencing']
    ar_lags = model_settings['ar_lags']
    fit_model = model_settings['fit_model']
    feature_threshold = model_settings['feature_threshold']

    all_is_abs_errors = pd.DataFrame([])
    all_is_pct_errors = pd.DataFrame([])
    all_los_predictions = pd.DataFrame([])
    all_hos_predictions = pd.DataFrame([])
    all_os_predictions = pd.DataFrame([])
    all_mod_prod = {}
    all_non_mod_prod = {}

    for p_date in prediction_dates[cn.FIRST_DOW]:
        _fit, _predict, _fitdata, _predictdata = run_prediction(
            date_to_predict=p_date, prediction_window=p_window, train_obs=train_size,
            difference=differencing, lags=ar_lags, order_data=fm.RAW_DATA, weather_data=fm.WEER_DATA,
            product_data=fm.PRODUCT_STATUS, model_type=fit_model, feature_threshold=[feature_threshold[0],
                                                                                     feature_threshold[1]])

        all_is_abs_errors = pd.concat([all_is_abs_errors, _fitdata['avg_fit_error']], axis=0)
        all_is_pct_errors = pd.concat([all_is_pct_errors, _fitdata['avg_pct_fit_error']], axis=0)
        all_os_predictions = pd.concat([all_os_predictions, _predict], axis=0)
        all_mod_prod[p_date] = _fitdata[cn.MOD_PROD]
        all_non_mod_prod[p_date] = _fitdata[cn.NON_MOD_PROD]

    return all_os_predictions, all_is_abs_errors, all_is_pct_errors, all_mod_prod, all_non_mod_prod


if __name__ == '__main__':
    start = time.time()
    # In sample testing of 2020-31-8
    is_fit1, os_pr1, fit_data1, predict_data1 = run_prediction(date_to_predict='2020-08-10',
                                                                         prediction_window=1,
                                                                         train_obs=cn.TRAIN_OBS,
                                                                         difference=False, lags=4,
                                                                         order_data=fm.RAW_DATA,
                                                                         weather_data=fm.WEER_DATA,
                                                                         product_data=fm.PRODUCT_STATUS,
                                                                         model_type='OLS',
                                                                         feature_threshold=[0.2, 15])

    elapsed = round((time.time() - start), 2)
    print("It takes {} seconds to run a prediction.".format(elapsed))

    active_products_act = gf.import_temp_file(file_name=fm.ORDER_DATA_ACT, data_loc=fm.SAVE_LOC)
    inactive_products_act = gf.import_temp_file(file_name=fm.ORDER_DATA_INACT, data_loc=fm.SAVE_LOC)
    all_products_act = active_products_act.join(inactive_products_act, how='outer')

    is_performance1 = in_sample_plot(y_true=fit_data1, y_fit=is_fit1,
                                     title="test")

    date_to_predict = '2020-10-05'
    prediction_window = 2
    train_obs = cn.TRAIN_OBS
    difference = False
    lags = cn.N_LAGS
    order_data = fm.RAW_DATA
    weather_data = fm.WEER_DATA
    product_data = fm.PRODUCT_STATUS
    model_type = 'OLS'
    feature_threshold = None
    bootstrap_iter = 2

    test = run_prediction_bootstrap(date_to_predict='2020-06-22', prediction_window=2, train_obs=cn.TRAIN_OBS,
                                    difference=False, lags=cn.N_LAGS, order_data=fm.RAW_DATA, campaign_data=fm.CAMPAIGN_DATA,
                                    weather_data=fm.WEER_DATA, product_data=fm.PRODUCT_STATUS,
                                    model_type='OLS', feature_threshold=None, bootstrap_iter=2)

    test2 = run_prediction_bootstrap(date_to_predict='2020-09-28', prediction_window=2, train_obs=cn.TRAIN_OBS,
                                    difference=False, lags=cn.N_LAGS, order_data=fm.RAW_DATA,
                                    weather_data=fm.WEER_DATA, product_data=fm.PRODUCT_STATUS,
                                    model_type='OLS', feature_threshold=None, bootstrap_iter=None)