import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
import hff_predictor.generic.dates as gf
import hff_predictor.generic.files
import hff_predictor.config.prediction_settings as ps
from hff_predictor.predict.make import run_prediction_bootstrap
import pandas as pd
import multiprocessing
import time

import logging
LOGGER = logging.getLogger(__name__)


# Vaste settings die door de multiprocessing heen gaan
model_settings = {
    "prediction_window": ps.PREDICTION_WINDOW,
    "train_size": ps.TRAIN_OBS,
    "differencing": ps.DIFFERENCING,
    "ar_lags": ps.N_LAGS,
    "fit_model": ps.MODEL_TYPE,
    "feature_threshold": ps.FEATURE_OPT,
    "bootstraps": 1,
    "su_member": "Hollander Plus",
    "weather_forecast": ps.WEATHER_FORECAST
}


def batch_prediction_bs(prediction_date: str):
    """
    Aparte batch functie met 1 parameter, zodat voorspellingen kunnen worden gegenereerd via multiprocessing

    :param prediction_date: Voorspeldatum
    :return: Multiprocessing object met voorspelling
    """

    # Haalt voorspelsettings op
    p_window = model_settings["prediction_window"]
    train_size = model_settings["train_size"]
    differencing = model_settings["differencing"]
    ar_lags = model_settings["ar_lags"]
    fit_model = model_settings["fit_model"]
    feature_threshold = model_settings["feature_threshold"]
    bootstrap_iterations = model_settings["bootstraps"]
    weather_forecast = model_settings["weather_forecast"]
    su_member = model_settings["su_member"]

    # Maakt voorspelling
    LOGGER.info("Maak voorspelling voor datum: {}".format(prediction_date))

    _predict = run_prediction_bootstrap(
        date_to_predict=prediction_date,
        prediction_window=p_window,
        train_obs=train_size,
        weather_forecast=weather_forecast,
        difference=differencing,
        lags=ar_lags,
        model_type=fit_model,
        feature_threshold=[feature_threshold[0], feature_threshold[1]],
        su_member=su_member,
        bootstrap_iter=bootstrap_iterations,
    )

    return _predict


def init_test(date, periods):

    start = time.time()

    # Genereert hier set met datums
    prediction_dates = pd.DataFrame(pd.date_range(end=date, periods=periods, freq="W-MON").astype(str),
                                    columns=[cn.FIRST_DOW])

    LOGGER.info("Maakt in totaal {} voorspellingen, tussen {} en {}".format(periods,
                                                                            min(prediction_dates[cn.FIRST_DOW]),
                                                                            max(prediction_dates[cn.FIRST_DOW])))

    pred_dates = list(prediction_dates[cn.FIRST_DOW])

    # Aantal cores die tegelijkertijd kunnen worden ingeschakeld
    num_cores = multiprocessing.cpu_count()

    if model_settings["fit_model"] != "OLS":
        LOGGER.warning("Zet aantal cores terug naar 1, optimalisatie loopt anders vast als gevolg van algoritme")

    # Run alle predicties en pool resutlaten
    pool = multiprocessing.Pool(num_cores)
    results = pool.map(batch_prediction_bs, pred_dates)

    pool.close()
    pool.join()

    # Sla resultaten op in pickle bestand
    hff_predictor.generic.files.save_to_pkl(
        data=results, file_name="test_result_bs_2p_2l_70obs", folder=fm.TEST_RESULTS_FOLDER
    )

    elapsed = round((time.time() - start), 2)
    LOGGER.info("Het duurt {} seconden om alle {} voorspellingen te genereren".format(elapsed,
                                                                                      periods))


