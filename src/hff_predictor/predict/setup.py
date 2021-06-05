import datetime
import hff_predictor.generic.files
import pandas as pd
from typing import Union
import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
import hff_predictor.data.transformations as dtr
from hff_predictor.generic.files import import_temp_file
from hff_predictor.data.transformations import fill_missing_values

import logging
LOGGER = logging.getLogger(__name__)


def split_products(active_products: pd.DataFrame, min_obs: int = cn.TRAIN_OBS,
                   prediction_date: str = cn.PREDICTION_DATE, prediction_window: int = cn.PREDICTION_WINDOW) -> tuple:
    """
    Maakt onderscheid tussen modelleerbaar en niet modelleerbare  producten, op basis beschikbare data

    :param active_products: Producten die in de laatste week van voorspellen nog zijn besteld
    :param min_obs: Observaties nodig voor schatten model
    :param prediction_date: Datum voor voorspelling
    :param prediction_window: Aantal weken dat vooruit wordt voorspeld
    :return: Verschillende subsets aan producten, modelleerbaar en niet modelleerbaar
    """

    # Bepaal hier het window waartussen de benodigde observaties beschikbaar moeten zijn
    last_train_date = prediction_date - datetime.timedelta(weeks=prediction_window)
    first_train_date = last_train_date - datetime.timedelta(weeks=min_obs)
    fitting_window = active_products.loc[last_train_date:first_train_date]

    active_products = active_products.loc[last_train_date:first_train_date]

    # Tel per product hoeveel observaties er beschikbaar zijn
    obs_count = pd.DataFrame(fitting_window.count())
    obs_count.columns = ["count"]

    # Splits producten in modelleerbaar en niet modelleerbaar
    series_to_model = obs_count[obs_count["count"] >= min_obs].index
    LOGGER.info("Number of products able to model: {}".format(len(series_to_model)))

    series_not_to_model = obs_count[obs_count["count"] < min_obs].index
    LOGGER.info("Number of products not able to model: {}".format(len(series_not_to_model)))

    # Consumentgroep nummer inladen om rol producten te identificeren
    consumentgroep_nr = import_temp_file(data_loc=fm.ORDER_DATA_CG_PR_FOLDER, set_index=False)
    consumentgroep_nr = consumentgroep_nr[[cn.INKOOP_RECEPT_NM, cn.CONSUMENT_GROEP_NR]]
    consumentgroep_nr.set_index(cn.INKOOP_RECEPT_NM, inplace=True)

    # Vind rolproducten
    active_rol_products = dtr.find_rol_products(data=active_products,
                                                consumentgroep_nrs=consumentgroep_nr)

    # Selecteer modlleerbare producten
    products_model = active_products[series_to_model].copy(deep=True)

    # Hier worden verschillende product groeperingen gemaakt, die als totaal worden voorspeld
    products_model[cn.MOD_PROD_SUM] = products_model.sum(axis=1)  # Modelleerbare producten
    products_model[cn.ALL_PROD_SUM] = active_products.sum(axis=1)  # Alle producten
    products_model[cn.ALL_ROL_SUM] = active_products[active_rol_products].sum(axis=1)  # Alle rol producten
    products_no_model = active_products[series_not_to_model]  # Niet modlleerbare producten

    return products_model, products_no_model


def create_lagged_sets(y_modelable: pd.DataFrame, y_nonmodelable: pd.DataFrame,
                       exogenous_features: pd.DataFrame, prediction_window: int, lags: Union[list, int]) -> tuple:
    """
    Functie om vertraagde en vooruitkijkende varianten te genereren van bestaande set met features

    :param y_modelable: Dataset met modelleerbare producten
    :param y_nonmodelable: Dataset met niet-modelleerbare producten
    :param weather_forecast: Genereer weersvoorspellingen
    :param exogenous_features: Externe factoren
    :param prediction_window: Voorspelwindow, vaak 2
    :param lags: Aantal vertragingen of vooruitkijkende punten
    :return: Set met vertraagde variabelen
    """

    # Subset van variabelen die alleen kunnen terugkijken: Superunie factoren (o.b.v. bestellingen) en weer
    exog_features_lookback = exogenous_features['superunie_n'].join(
        exogenous_features['superunie_pct'], how='left')

    # Subset van variabelen die ook vooruit kunnen kijken, zoals feestdagen, campagnes en COVID features
    exog_features_lookahead = exogenous_features['weather'].join(exogenous_features['covid'], how='left')

    exog_features_lookahead_far = exogenous_features['holidays'].join(exogenous_features['campaigns'], how='left')

    # Genereren van de vertraging voor lookback features
    exog_features_lookback_lags = dtr.create_lags(exog_features_lookback, lag_range=lags)
    y_m_lags = dtr.create_lags(y_modelable, lag_range=lags)
    y_nm_lags = dtr.create_lags(y_nonmodelable, lag_range=lags)

    # Voor seizoenen en structurele breuken wordt nu geen correctie uitgevoerd
    exog_features_no_adj = exogenous_features['seasons'].join(exogenous_features['breaks'], how='left')

    # Hier staat nu hard-coded welke range de faetures vooruit kunnen kijken

    # +3, betekent een range tot (-lags, 5). Dit leidt tot een range met voorspellingen tot + 4 weken vooruit
    # In de volgende functie wordt de voorspelcontext gezet, waardoor er 2 weken vanuaf gaan
    # Hier door wordt er feitelijkt maar 4 - 2 = 2 weken vooruit gekeken, ofwel de week van de voorspelling
    lookahead = prediction_window + 3
    lookahead_range = list(reversed(range(-lags, lookahead)))

    lookahead_far = prediction_window + 5
    lookahead_far_range = list(reversed(range(-lags, lookahead_far)))

    exog_features_lookahead_lags = dtr.create_lags(exog_features_lookahead, lag_range=lookahead_range)

    exog_features_lookahead_far_lags = dtr.create_lags(exog_features_lookahead_far, lag_range=lookahead_far_range)

    exog_features_lookahead_combined_lags = exog_features_lookahead_lags.join(
        exog_features_lookahead_far_lags, how='left')

    return (y_m_lags, y_nm_lags,
            exog_features_lookback_lags,
            exog_features_lookahead_combined_lags,
            exog_features_no_adj)


def create_predictive_context(y_modelable_lag: pd.DataFrame,
                              y_nonmodelable_lag: pd.DataFrame,
                              features_lag_lookback: pd.DataFrame,
                              features_lag_lookahead: pd.DataFrame,
                              features_no_adj: pd.DataFrame,
                              prediction_window: int) -> tuple:

    # Zorg dat niet aangepaste features in juiste volgorde staan en verschuif dan de waarden in de juiste context
    # Deze verschuif ik nu dus 2 weken vooruit, om ze mee te kunnen nemen in het totaal en die vervolgens
    # 2 weken te vertragen, waardoor de niet aangepaste features ook echt niet aangepast worden
    features_na_corr = features_no_adj.sort_index(ascending=False).shift(prediction_window)

    # Breng alle features bij elkaar
    features_total = features_lag_lookback.join(
        features_lag_lookahead, how='left').join(
        features_na_corr, how='left'
    )

    # Breng de juiste voorspelcontext aan en verwijder nu NaN waarden onderaan
    features_total_shift = features_total.shift(-prediction_window)[:-prediction_window]

    features_total2 = features_lag_lookahead.shift(-prediction_window).join(
        features_lag_lookback, how='left').join(features_na_corr, how='left')[:-prediction_window]
    fill_missing_values(features_total2)

    return (y_modelable_lag.shift(-prediction_window)[:-prediction_window],
            y_nonmodelable_lag.shift(-prediction_window)[:-prediction_window],
            features_total2,
            features_total_shift
            )


def create_model_setup(y_modelable: pd.DataFrame, y_nonmodelable: pd.DataFrame, exogenous_features: pd.DataFrame,
                       difference: bool = False, lags: int = cn.N_LAGS,
                       prediction_date: str = cn.PREDICTION_DATE,
                       prediction_window: int = cn.PREDICTION_WINDOW,) -> tuple:
    """
    Verzamelfunctie om alle data klaar te maken voor fit en voorspellingen

    :param y_modelable: Modelleerbare producten
    :param y_nonmodelable: Niet modelleerbare producten
    :param weather_forecast: Optie om weersfactoren als voorspelling mee te nemen
    :param exogenous_features: Externe factoren
    :param difference: Nemen van eerste verschillen of niet
    :param lags: Aantal vertragingen
    :param prediction_date: Voorspeldatum
    :param prediction_window: Voorspelwindow
    :return: Fit en predictie sets
    """

    # Datum vertalen naar datetime object
    logging.info("The prediction date is: {}".format(
        datetime.datetime.strftime(prediction_date, "%Y-%m-%d")))

    # Laatste datum die wordt gebruikt om modellen te trainen
    last_train_date = prediction_date - datetime.timedelta(weeks=prediction_window)

    logging.info("The prediction window is {}, so the last train date is: {}".format(
        prediction_window,
        datetime.datetime.strftime(last_train_date, "%Y-%m-%d")))

    # Invullen van missende waarden tot nullen
    dtr.fill_missing_values(data=y_modelable)
    dtr.fill_missing_values(data=y_nonmodelable)

    # Vang de laatste datum van trainen op (LTD = Last Train Date), dit is ook het voorspelmoment
    y_m_ltd = y_modelable.loc[last_train_date]
    y_nm_ltd = y_nonmodelable.loc[last_train_date]

    # Neem eerste verschillen
    if difference:
        y_modelable = dtr.first_difference_data(undifferenced_data=y_modelable, delta=1, scale=False)
        y_nonmodelable = dtr.first_difference_data(undifferenced_data=y_nonmodelable, delta=1, scale=False)

    # Maak de sets met vertraagde variabelen
    y_m_lags, y_nm_lags, X_lbl, X_lal, X_na = create_lagged_sets(y_modelable=y_modelable,
                                                                 y_nonmodelable=y_nonmodelable,
                                                                 exogenous_features=exogenous_features,
                                                                 prediction_window=prediction_window,
                                                                 lags=lags)

    # Maak de verschillende sets in de juiste context
    # AR features modelleerbaar en niet modelleerbaar, features totaal (exog_t) en features in context (exog_l)
    y_ar_m, y_ar_nm, X_exog_t, X_exog_l = create_predictive_context(y_modelable_lag=y_m_lags,
                                                                    y_nonmodelable_lag=y_nm_lags,
                                                                    features_lag_lookback=X_lbl,
                                                                    features_lag_lookahead=X_lal,
                                                                    features_no_adj=X_na,
                                                                    prediction_window=prediction_window)

    # Zet de fitting window
    max_date = last_train_date
    min_date = y_ar_m.index.min() + datetime.timedelta(days=7*lags) # Adjust for lags

    # Maak de juiste fit sets: AR factoren, externe factoren en werkelijke waarden
    y_ar_m_fit = y_ar_m.loc[max_date: min_date]
    X_exog_fit = X_exog_l.loc[y_ar_m_fit.index]
    y_true_fit = y_modelable.loc[y_ar_m_fit.index]

    # Isoleer de waarden die gaan worden gebruikt voor predictie
    yl_ar_m_prd = y_m_lags.loc[last_train_date]
    yl_ar_nm_prd = y_nm_lags.loc[last_train_date]
    X_exog_prd = X_exog_t.loc[last_train_date]

    # Pas de index aan van de waarden die worden gebruikt voor de voorspelling
    yl_ar_m_prd.name += datetime.timedelta(days=prediction_window * 7)
    yl_ar_nm_prd.name += datetime.timedelta(days=prediction_window * 7)
    X_exog_prd.name += datetime.timedelta(days=prediction_window * 7)

    # Verzamel alle fit elementen in een dict
    model_fitting = {
        cn.Y_TRUE: y_true_fit,
        cn.Y_AR: y_ar_m_fit,
        cn.X_EXOG: X_exog_fit,
        cn.MOD_PROD: y_modelable.columns,
        cn.NON_MOD_PROD: y_nonmodelable.columns,
    }

    # Verzamel alle predict elementen in een dict
    model_prediction = {
        cn.Y_AR_M: yl_ar_m_prd,
        cn.Y_AR_NM: yl_ar_nm_prd,
        cn.X_EXOG: X_exog_prd,
        cn.Y_M_UNDIF: y_m_ltd,
        cn.Y_NM_UNDIF: y_nm_ltd,
    }

    return model_fitting, model_prediction


def prediction_setup_wrapper(prediction_date: str, prediction_window: int, train_obs: int,
                             nlags: int, difference: bool, act_products: pd.DataFrame,
                             exog_features: pd.DataFrame, save_to_pkl: bool = False,) -> tuple:
    """
    Verzamelfunctie om alle predictie setup functies bij elkaar te brengen en achter elkaar te draaien

    :param prediction_date: Voorspeldatum
    :param prediction_window: Voorspelwindow
    :param train_obs: Aantal observastie om model te fitten
    :param weather_forecast: Optie om weersvoorspellingen mee te nemen
    :param nlags: Aantal te gebruiken lags
    :param difference: Nemen van eerste verschillen
    :param act_products: Set met actieve producten
    :param exog_features: Externe factoren
    :param save_to_pkl: Opslaan tot pickle bestand
    :return: Fit en voorspel sets
    """

    # Datum vertalen tot datetime object
    if type(prediction_date) == str:
        prediction_date = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")

    # Scheiden van producten in modelleerbaar en niet modelleerbaar
    products_model, products_nmodel = split_products(active_products=act_products, min_obs=train_obs,
                                                     prediction_date=prediction_date,
                                                     prediction_window=prediction_window)

    # Als er maar 1 te modelleren product is (som van voorspellingen), verlaag dan het aantal train observaties en
    # draai opnieuw
    if products_model.shape[1] == 1:
        train_obs_reduced = train_obs - 10
        products_model, products_nmodel = split_products(active_products=act_products, min_obs=train_obs_reduced,
                                                         prediction_date=prediction_date,
                                                         prediction_window=prediction_window,)

        LOGGER.debug("Aantal fit observaties verlaagd tot {}, om meer modelleerbare producten te hebben".format(
            train_obs_reduced))

    # Maak hier de totale setup
    data_fitting, data_prediction = create_model_setup(y_modelable=products_model, y_nonmodelable=products_nmodel,
                                                       prediction_date=prediction_date,
                                                       prediction_window=prediction_window,
                                                       exogenous_features=exog_features,
                                                       difference=difference, lags=nlags,)
    # Opslaan tot python pickle file
    if save_to_pkl:
        hff_predictor.generic.files.save_to_pkl(data=data_fitting, file_name="fit_data", folder=fm.SAVE_LOC)
        hff_predictor.generic.files.save_to_pkl(data=data_prediction, file_name="predict_data", folder=fm.SAVE_LOC)

    return data_fitting, data_prediction


def init_setup_prediction():
    active_products_t = hff_predictor.generic.files.import_temp_file(
        data_loc=fm.ORDER_DATA_ACT_PR_FOLDER,
        set_index=True
    )

    inactive_products_t = hff_predictor.generic.files.import_temp_file(
        data_loc=fm.ORDER_DATA_INACT_PR_FOLDER,
        set_index=True
    )

    exog_features_t = hff_predictor.generic.files.read_pkl(
        data_loc=fm.FEATURES_PROCESSED_FOLDER
    )

    data_fitting_t, data_prediction_t = prediction_setup_wrapper(
        prediction_date="2021-04-26",
        prediction_window=2,
        train_obs=cn.TRAIN_OBS,
        nlags=3,
        difference=False,
        act_products=active_products_t,
        exog_features=exog_features_t,
        save_to_pkl=True,
    )

    hff_predictor.generic.files.save_to_pkl(
        data=data_fitting_t, file_name="fit_data", folder=fm.SAVE_LOC
    )
    hff_predictor.generic.files.save_to_pkl(
        data=data_prediction_t, file_name="predict_data", folder=fm.SAVE_LOC
    )
