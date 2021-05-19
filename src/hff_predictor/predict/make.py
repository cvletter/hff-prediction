import time
import hff_predictor.generic.files as fl
import pandas as pd
import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm

from hff_predictor.features.create import prep_all_features
from hff_predictor.data.prepare import data_prep_wrapper
from hff_predictor.model.fit import fit_and_predict
from hff_predictor.predict.setup import prediction_setup_wrapper
from hff_predictor.model.benchmark import moving_average
from hff_predictor.model.bootstrapper import bootstrap

import logging
LOGGER = logging.getLogger(__name__)


def run_prediction_bootstrap(
    date_to_predict=cn.PREDICTION_DATE,
    prediction_window=cn.PREDICTION_WINDOW,
    train_obs=cn.TRAIN_OBS,
    difference=False,
    lags=cn.N_LAGS,
    model_type="OLS",
    feature_threshold=None,
    bootstrap_iter=None,
    reload_data=False,
    save_predictions=False
):
    start_prediction = time.time()

    LOGGER.info("Maak voorspelling voor: {}".format(date_to_predict))

    if feature_threshold is None:
        feature_threshold = [0.2, 30]

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
        reload_data=reload_data,
        agg_weekly=True,
        exclude_su=True,
        save_to_csv=True,
    )

    if reload_data:
        data_load_log = "Data is voorbereid en opnieuw ingeladen"
    else:
        data_load_log = "Data is voorbereid en maar niet opnieuw ingeladen"

    LOGGER.debug(data_load_log)

    exogenous_features = prep_all_features(
        weather_data_processed=weather_data_processed,
        order_data_su=order_data_su,
        campaign_data_su=campaign_data_pr,
        prediction_date=date_to_predict,
        train_obs=train_obs,
        save_file=False,
    )

    LOGGER.debug("Features zijn voorbereid")

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

    LOGGER.debug("Data is voorbereid voor voorspelling")

    in_sample_fits, all_predictions, all_pars = fit_and_predict(
        fit_dict=fit_data,
        predict_dict=predict_data,
        model_type=model_type,
        feature_threshold=[feature_threshold[0], feature_threshold[1]],
    )

    ma_predictions = moving_average(active_products=active_products,
                                    prediction_window=prediction_window,
                                    prediction_date=date_to_predict)

    LOGGER.debug("Model is gefit en voorspellingen zijn gemaakt")

    all_output[date_to_predict] = {}
    all_output[date_to_predict][cn.MOD_PROD] = fit_data[cn.MOD_PROD]
    all_output[date_to_predict][cn.NON_MOD_PROD] = fit_data[cn.NON_MOD_PROD]

    all_output[date_to_predict][cn.SELECTED_FEATURES] = all_pars

    avg_fit_err, avg_pct_err = in_sample_error(
        all_fits=in_sample_fits, all_true_values=fit_data[cn.Y_TRUE]
    )

    all_output[date_to_predict][cn.FIT_ERROR_ABS] = avg_fit_err
    all_output[date_to_predict][cn.FIT_ERROR_PCT] = avg_pct_err

    all_output[date_to_predict][cn.MA_BENCHMARK] = ma_predictions

    prediction_output = all_predictions

    if do_bootstrap:
        LOGGER.debug("Bootstrap wordt uitgevoerd met {} iteraties".format(bootstrap_iter))

        boundaries = bootstrap(
            prediction=all_predictions,
            fit_dict=fit_data,
            predict_dict=predict_data,
            bootstrap=True,
            iterations=bootstrap_iter,
            model_type=model_type,
            feature_threshold=feature_threshold,
        )

        prediction_output = pd.concat([all_predictions, boundaries.T, ma_predictions]).T
        prediction_output.columns = ["voorspelling", "ondergrens", "bovengrens", "5weeks_gemiddelde"]

        # all_predictions.drop(cn.BOOTSTRAP_ITER, axis=1, inplace=True)

        na_values = all_predictions.isna().sum().sum()
        logging.debug("In {} there are {} na_values".format(date_to_predict, na_values))

    all_output[date_to_predict][cn.PREDICTION_OS] = all_predictions

    if save_predictions:

        save_name = "predictions_p{}_d{}".format(prediction_window, date_to_predict)
        fl.save_to_csv(data=prediction_output,
                       file_name=save_name,
                       folder=fm.PREDICTIONS_FOLDER)

    elapsed = round((time.time() - start_prediction), 2)
    LOGGER.info("De voorspelling is klaar en duurde {} seconden".format(elapsed))

    return all_output


def init_predict(date, window, reload):

    test = run_prediction_bootstrap(
        date_to_predict=date,
        prediction_window=window,
        train_obs=cn.TRAIN_OBS,
        difference=False,
        lags=cn.N_LAGS,
        model_type="OLS",
        feature_threshold=None,
        bootstrap_iter=2,
        reload_data=reload,
        save_predictions=True
    )