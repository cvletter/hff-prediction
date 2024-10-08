import time
import sys
import hff_predictor.generic.files as fl
import pandas as pd
import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
import hff_predictor.config.prediction_settings as ps

from hff_predictor.features.create import prep_all_features
from hff_predictor.data.prepare import data_prep_wrapper
from hff_predictor.model.fit import fit_and_predict
from hff_predictor.predict.setup import prediction_setup_wrapper
from hff_predictor.model.benchmark import moving_average
from hff_predictor.model.bootstrapper import bootstrap
from hff_predictor.data.transformations import add_product_number
from hff_predictor.generic.dates import first_day_of_week

import logging
LOGGER = logging.getLogger(__name__)


def run_prediction_bootstrap(date_to_predict: str, prediction_window: int,
                             train_obs: int, difference: bool, lags: int, weather_forecast: bool,
                             model_type: str, feature_threshold: list = None, bootstrap_iter: int = None,
                             reload_data: bool = False, su_member: str = None, save_predictions: bool = False):
    """
    Deze functie brengt alles bij elkaar, van data inladen tot maken van voorspellingen

    :param date_to_predict: Voorspeldatum
    :param prediction_window: Aantal weken dat vooruit wordt voorspeld
    :param train_obs: Aantal observaties waar mee wordt getraind
    :param difference: Eerste verschillen
    :param lags: Aantal weken waarin wordt teruggekeken
    :param weather_forecast: Optie om weersfactoren als voorspelling mee te nemen
    :param model_type: Type model voor voorspellingen
    :param feature_threshold: Feature optimalisatie parameter
    :param bootstrap_iter: Aantal bootstrap iteraties
    :param reload_data: Optie om data opnieuw in te laden
    :param save_predictions: Optie om resultaten op te slaan
    :return: Voorspellingen
    """
    # Start timer
    start_total = time.time()

    LOGGER.info("Maak voorspelling voor: {}".format(date_to_predict))

    if feature_threshold is None:
        feature_threshold = ps.FEATURE_OPT

    if bootstrap_iter is None:
        do_bootstrap = False

    else:
        do_bootstrap = True

    def convert_series_to_dataframe(input_series: pd.Series, date_val: str,
                                    index_name: str = cn.FIRST_DOW) -> pd.DataFrame:
        """
        Vertaal een pandas Series object naar een DataFrame

        :param input_series: Series object
        :param date_val: Datum kolom
        :param index_name: Index naam
        :return: Dataframe van input Series
        """
        input_df = pd.DataFrame(input_series).T
        input_df[index_name] = date_val
        return input_df.set_index(index_name, drop=True, inplace=False)

    def in_sample_error(all_fits: pd.DataFrame, all_true_values: pd.DataFrame) -> tuple:
        """
        Hulpfunctie om de in-sample voorspelfout te bepalen, m.a.w. hoe goed de modelfit is

        :param all_fits: Gefitte waarden
        :param all_true_values: Werkelijke waarden
        :return: Combinatie van absolute en procentuele voorspelfout
        """
        fit_error = abs(all_fits.subtract(all_true_values[all_fits.columns], axis="index"))
        avg_fit_error = fit_error.mean(axis=0)
        corr_true_values = all_true_values.replace(0, 1)
        pct_fit_error = fit_error / corr_true_values
        avg_pct_fit_error = pct_fit_error.mean(axis=0)

        avg_fit_error_df = convert_series_to_dataframe(input_series=avg_fit_error, date_val=date_to_predict)
        avg_pct_fit_error_df = convert_series_to_dataframe(input_series=avg_pct_fit_error, date_val=date_to_predict)

        return avg_fit_error_df, avg_pct_fit_error_df

    # Verzamel dict voor alle output
    all_output = {}

    # Importeren en voorbereiden van data
    start_prep = time.time()
    active_products, inactive_products, weather_data_processed, order_data_su, campaign_data_pr = data_prep_wrapper(
        prediction_date=date_to_predict, prediction_window=prediction_window, reload_data=reload_data,
        agg_weekly=True, exclude_su=True, su_member=su_member, save_to_csv=True)

    elapsed_prep = round((time.time() - start_prep), 2)

    if reload_data:
        data_load_log = "Data is voorbereid en opnieuw ingeladen, dit duurde {} seconden".format(elapsed_prep)
    else:
        data_load_log = "Data is voorbereid maar niet opnieuw ingeladen, dit duurde {} seconden".format(elapsed_prep)

    LOGGER.debug(data_load_log)

    # Features voorbereiden
    start_features = time.time()
    exogenous_features = prep_all_features(weather_data_processed=weather_data_processed, order_data_su=order_data_su,
                                           active_products=active_products, campaign_data_su=campaign_data_pr,
                                           prediction_date=date_to_predict, train_obs=train_obs, save_file=False)

    elapsed_features = round((time.time() - start_features), 2)

    LOGGER.debug("Features zijn voorbereid, dit duurde {} seconden".format(elapsed_features))

    # Bereid voorspelsetup voor
    start_setup = time.time()
    fit_data, predict_data = prediction_setup_wrapper(prediction_date=date_to_predict,
                                                      prediction_window=prediction_window, train_obs=train_obs,
                                                      nlags=lags, difference=difference, act_products=active_products,
                                                      exog_features=exogenous_features, save_to_pkl=False)

    elapsed_setup = round((time.time() - start_setup), 2)

    LOGGER.debug("Data is voorbereid voor modelleren, dit duurde {} seconden".format(elapsed_setup))

    # Schat hier de modellen en maak de voorspellingen
    start_fit = time.time()
    in_sample_fits, all_predictions, all_pars, all_wpredictions = fit_and_predict(
        fit_dict=fit_data, predict_dict=predict_data,
        model_type=model_type,
        prediction_window=prediction_window,
        feature_threshold=[feature_threshold[0],
                           feature_threshold[1]],
        weather_forecast=weather_forecast,
        standardize=ps.STANDARDIZE)

    # Maak hier de voorspellingen op basis van top-down predictie
    all_products_m = predict_data['y_ar_m'][['last0w' in x for x in predict_data['y_ar_m'].index]]
    all_products_nm = predict_data['y_ar_nm'][['last0w' in x for x in predict_data['y_ar_nm'].index]]
    all_products = pd.DataFrame(pd.concat([all_products_m, all_products_nm]))
    all_products = all_products.T
    all_products.columns = [x[:-7] for x in all_products.columns]

    # Verwijder de geaggregeerde kolommen
    aggregated_cols = [cn.MOD_PROD_SUM, cn.ALL_PROD_SUM, cn.ALL_ROL_SUM]
    products_tosum = all_products.drop(aggregated_cols, axis=1, inplace=False)

    # Bepaalde verdeelsleutel
    product_distribution = all_products.div(products_tosum.sum(axis=1), axis=0)
    total_prediction = all_predictions[cn.ALL_PROD_SUM]

    # Maak de verdeelde voorspellingen
    prediction_distributed = product_distribution.mul(total_prediction, axis=0).astype(int)
    for i in aggregated_cols:
        prediction_distributed[i] = all_predictions[i]

    # Maak de moving average voorspellingen
    ma_predictions, ma_now = moving_average(active_products=active_products, prediction_window=prediction_window,
                                    prediction_date=date_to_predict)

    elapsed_prediction = round((time.time() - start_fit), 2)
    LOGGER.debug("Model is gefit en voorspellingen zijn gemaakt, dit duurde {} seconden".format(elapsed_prediction))

    # Verzamel de voorspellingen in het output object
    all_output[date_to_predict] = {}
    all_output[date_to_predict][cn.MOD_PROD] = fit_data[cn.MOD_PROD].astype(str)
    all_output[date_to_predict][cn.NON_MOD_PROD] = fit_data[cn.NON_MOD_PROD].astype(str)

    # Verzamnel de geselecteerde features
    all_output[date_to_predict][cn.SELECTED_FEATURES] = all_pars

    # Verzamel de fit errors
    avg_fit_err, avg_pct_err = in_sample_error(all_fits=in_sample_fits, all_true_values=fit_data[cn.Y_TRUE])
    all_output[date_to_predict][cn.FIT_ERROR_ABS] = avg_fit_err.astype(float)
    all_output[date_to_predict][cn.FIT_ERROR_PCT] = avg_pct_err.astype(float)

    # Verzamel de MA voorspelling
    all_output[date_to_predict][cn.MA_BENCHMARK] = ma_predictions.astype(int)
    all_output[date_to_predict]["last_week"] = ma_now.astype(int)
    all_output[date_to_predict]["distributed_prediction"] = prediction_distributed

    # Maak een apart object aan voor de voorspellingen, ter voorbereiding op eventuele bootstrap

    prediction_output = all_predictions

    if do_bootstrap:
        start_bootstrap = time.time()
        LOGGER.debug("Bootstrap wordt uitgevoerd met {} iteraties".format(bootstrap_iter))

        boundaries = bootstrap(
            prediction=all_predictions,
            fit_dict=fit_data,
            predict_dict=predict_data,
            prediction_window=prediction_window,
            bootstrap_sample=True,
            iterations=bootstrap_iter,
            model_type=model_type,
            feature_threshold=feature_threshold,
            weather_forecast=weather_forecast
        )

        pred_out = pd.concat([ma_now, all_predictions, boundaries.T, ma_predictions, all_wpredictions.T]).T

        nan_values = pred_out.isna().sum().sum()

        # Onderstaande code aanzetten als ook de top-down voorspellingen moeten worden meegenomen
        """
        pred_out = pd.concat(
            [ma_now, all_predictions, prediction_distributed, boundaries.T, ma_predictions, all_wpredictions.T]).T
        """

        try:
            prediction_output = pred_out.astype(int)
            LOGGER.debug("Er zijn {} ontbrekende voorspellingen".format(nan_values))
        except ValueError:
            LOGGER.critical("Er zijn {} ontbrekende voorspellingen, controleer of data correct is ingeladen,"
                            " de voorspelling is gestopt".format(nan_values))
            sys.exit(1)

        prediction_output['modelleerbaar'] = "Ja"
        prediction_output.loc[fit_data[cn.NON_MOD_PROD], 'modelleerbaar'] = "Nee"

        prediction_output.columns = ["afgelopen_week", "voorspelling", "ondergrens", "bovengrens",
                                     "5weeks_gemiddelde", "beter_weer", "slechter_weer", "modelleerbaar"]

        # "voorspelling_verdeeld" toevoegen als top-down moet worden meegenomen

        elapsed_bootstrap = round((time.time() - start_bootstrap), 2)
        LOGGER.debug("Bootstrap voorspellingen zijn gemaakt, dit duurde {} seconden".format(elapsed_bootstrap))

    # Als de voorspelling top-down is, sla deze dan op ipv de bottom-up voorspellingen
    if ps.TOP_DOWN:
        all_output[date_to_predict][cn.PREDICTION_OS] = prediction_distributed.astype(int)
    else:
        all_output[date_to_predict][cn.PREDICTION_OS] = all_predictions.astype(int)

    if save_predictions:
        pred_out = add_product_number(data=prediction_output)
        save_name = "predictions_p{}_d{}".format(prediction_window, date_to_predict)
        if su_member is not None:
            save_name = "{}_{}".format(save_name, su_member.lower())
        fl.save_to_csv(data=pred_out, file_name=save_name, folder=fm.PREDICTIONS_FOLDER)

    elapsed = round((time.time() - start_total), 2)
    LOGGER.info("De voorspelling is klaar en duurde {} seconden".format(elapsed))

    return all_output


def init_predict(date, window, reload, su_member):

    # Bepaalt hier de week waar een voorspelling voor moet worden gemaakt o.b.v. automatische detectie
    if date == cn.DEFAULT_PRED_DATE:
        current_week, prediction_date = first_day_of_week()
    else:
        prediction_date = date

    test = run_prediction_bootstrap(
        date_to_predict=prediction_date,
        prediction_window=window,
        train_obs=ps.TRAIN_OBS,
        weather_forecast=ps.WEATHER_FORECAST,
        difference=ps.DIFFERENCING,
        lags=ps.N_LAGS,
        model_type=ps.MODEL_TYPE,
        feature_threshold=ps.FEATURE_OPT,
        bootstrap_iter=ps.BOOTSTRAP_ITER,
        reload_data=reload,
        su_member=su_member,
        save_predictions=True
    )

"""
    date_to_predict = "2021-09-13"
    prediction_window = 2
    train_obs = ps.TRAIN_OBS
    weather_forecast = ps.WEATHER_FORECAST
    difference = ps.DIFFERENCING
    lags = ps.N_LAGS
    model_type = ps.MODEL_TYPE
    feature_threshold = ps.FEATURE_OPT
    bootstrap_iter = ps.BOOTSTRAP_ITER
    reload_data = False
    su_member = "Hollander Plus"
    save_predictions = False
"""

