import datetime
import time

import hff_predictor.generic.files
import pandas as pd

import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
from hff_predictor.features.create import prep_all_features
from hff_predictor.data.prepare import data_prep_wrapper
from hff_predictor.model.fit import fit_and_predict
from hff_predictor.evaluation.prediction import in_sample_plot
from hff_predictor.predict.setup import prediction_setup_wrapper


def run_prediction_bootstrap(
    date_to_predict=cn.PREDICTION_DATE,
    prediction_window=cn.PREDICTION_WINDOW,
    train_obs=cn.TRAIN_OBS,
    difference=False,
    lags=cn.N_LAGS,
    order_data=fm.RAW_DATA,
    weather_data=fm.WEER_DATA,
    campaign_data=fm.CAMPAIGN_DATA,
    product_data=fm.PRODUCT_STATUS,
    model_type="OLS",
    feature_threshold=None,
    bootstrap_iter=None,
):
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
        fit_error = abs(
            all_fits.subtract(all_true_values[all_fits.columns], axis="index")
        )
        avg_fit_error = fit_error.mean(axis=0)
        corr_true_values = all_true_values.replace(0, 1)
        pct_fit_error = fit_error / corr_true_values
        avg_pct_fit_error = pct_fit_error.mean(axis=0)

        avg_fit_error_df = convert_series_to_dataframe(
            input_series=avg_fit_error, date_val=date_to_predict
        )
        avg_pct_fit_error_df = convert_series_to_dataframe(
            input_series=avg_pct_fit_error, date_val=date_to_predict
        )

        return avg_fit_error_df, avg_pct_fit_error_df

    # Catch all output
    all_output = {}

    # Import and prepare data
    (
        active_products,
        inactive_products,
        weather_data_processed,
        order_data_su,
        campaign_data_pr,
    ) = data_prep_wrapper(
        prediction_date=date_to_predict,
        prediction_window=prediction_window,
        reload_data=False,
        campaign_data_loc=campaign_data,
        order_data_loc=order_data,
        weer_data_loc=weather_data,
        product_data_loc=product_data,
        agg_weekly=True,
        exclude_su=True,
        save_to_csv=False,
    )

    exogenous_features = prep_all_features(
        weather_data_processed=weather_data_processed,
        order_data_su=order_data_su,
        campaign_data_su=campaign_data_pr,
        prediction_date=date_to_predict,
        train_obs=train_obs,
        save_to_csv=False,
    )

    fit_data, predict_data = prediction_setup_wrapper(
        prediction_date=date_to_predict,
        prediction_window=prediction_window,
        train_obs=train_obs,
        nlags=lags,
        difference=difference,
        act_products=active_products,
        exog_features=exogenous_features,
        save_to_pkl=False,
    )

    in_sample_fits, all_predictions, all_pars = fit_and_predict(
        fit_dict=fit_data,
        predict_dict=predict_data,
        model_type=model_type,
        feature_threshold=[feature_threshold[0], feature_threshold[1]],
    )

    all_output[date_to_predict] = {}
    all_output[date_to_predict][cn.MOD_PROD] = fit_data[cn.MOD_PROD]
    all_output[date_to_predict][cn.NON_MOD_PROD] = fit_data[cn.NON_MOD_PROD]

    all_output[date_to_predict][cn.SELECTED_FEATURES] = all_pars

    avg_fit_err, avg_pct_err = in_sample_error(
        all_fits=in_sample_fits, all_true_values=fit_data[cn.Y_TRUE]
    )

    all_output[date_to_predict][cn.FIT_ERROR_ABS] = avg_fit_err
    all_output[date_to_predict][cn.FIT_ERROR_PCT] = avg_pct_err

    if do_bootstrap:
        all_predictions[cn.BOOTSTRAP_ITER] = 0

        for i in range(1, bootstrap_iter):
            print("Running iteration {} of {}".format(i, bootstrap_iter))
            fits, temp_os, pars = fit_and_predict(
                fit_dict=fit_data,
                predict_dict=predict_data,
                bootstrap=True,
                model_type=model_type,
                feature_threshold=[feature_threshold[0], feature_threshold[1]],
            )
            temp_os[cn.BOOTSTRAP_ITER] = i

            all_predictions = pd.concat([all_predictions, temp_os])

            na_values = all_predictions.isna().sum().sum()
            print("In {} there are {} na_values".format(date_to_predict, na_values))

    all_output[date_to_predict][cn.PREDICTION_OS] = all_predictions

    return all_output


def init_predict(date):
    start = time.time()
    # In sample testing of 2020-31-8
    test = run_prediction_bootstrap(
        date_to_predict=date,
        prediction_window=2,
        train_obs=cn.TRAIN_OBS,
        difference=False,
        lags=cn.N_LAGS,
        order_data=fm.RAW_DATA,
        campaign_data=fm.CAMPAIGN_DATA,
        weather_data=fm.WEER_DATA,
        product_data=fm.PRODUCT_STATUS,
        model_type="OLS",
        feature_threshold=None,
        bootstrap_iter=2,
    )


    elapsed = round((time.time() - start), 2)
    print("It takes {} seconds to run a prediction.".format(elapsed))

