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


def create_predictive_context(y_modelable,
                               y_nonmodelable,
                               exogenous_features,
                               prediction_date,
                               prediction_window):

    current_max_date = y_modelable.index.max()
    required_max_date = prediction_date
    n_dates = int(4 + (required_max_date - current_max_date).days / 7)
    dates_to_add = [current_max_date + datetime.timedelta(days=7*i) for i in range(1, n_dates+1)]
    new_index = list(y_modelable.index) + dates_to_add

    def context_creator(data, required_index, prediction_window):
        data_new = pd.DataFrame(index=required_index).sort_index(ascending=False, inplace=False)
        data_new = data_new.join(data, how='left')
        return data_new.shift(-prediction_window)[:-prediction_window]

    y_mod_context = context_creator(data=y_modelable, required_index=new_index, prediction_window=prediction_window)
    y_nmod_context = context_creator(data=y_nonmodelable, required_index=new_index, prediction_window=prediction_window)
    exogenous_features_context = {}
    for x in exogenous_features.keys():
        exogenous_features_context[x] = context_creator(data=exogenous_features[x], required_index=new_index,
                                                            prediction_window=prediction_window)

    return y_mod_context, y_nmod_context, exogenous_features_context


def create_lagged_sets(y_mod_context, y_nmod_context, exogenous_features_context, lags, prediction_window):

    y_mod_lags = dtr.create_lags(data=y_mod_context, lag_range=lags)
    y_nmod_lags = dtr.create_lags(data=y_nmod_context, lag_range=lags)

    # Subset van variabelen die alleen kunnen terugkijken: Superunie factoren (o.b.v. bestellingen) en weer
    exog_features_lookback = exogenous_features_context['superunie_n'].join(
        exogenous_features_context['superunie_pct'], how='left')

    exog_features_lookback_lags = dtr.create_lags(data=exog_features_lookback, lag_range=lags)

    # Subset van variabelen die ook vooruit kunnen kijken, zoals feestdagen, campagnes en COVID features
    exog_features_lookahead = exogenous_features_context['weather'].join(
        exogenous_features_context['covid'], how='left').join(
        exogenous_features_context['holidays'], how='left').join(
        exogenous_features_context['campaigns'], how='left')

    lookahead_range = list(reversed(range(-lags, prediction_window + 1)))
    exog_features_lookahead_lags = dtr.create_lags(data=exog_features_lookahead, lag_range=lookahead_range)

    # Voor seizoenen en structurele breuken wordt nu geen correctie uitgevoerd
    exog_features_no_adj = exogenous_features_context['seasons'].join(exogenous_features_context['breaks'], how='left')

    exog_features_total = exog_features_lookback_lags.join(
        exog_features_lookahead_lags, how="left").join(
        exog_features_no_adj, how="left"
    )

    return y_mod_lags, y_nmod_lags, exog_features_total


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

    # Maak predictive context
    y_mod_context, y_nmod_context, exogenous_features_context = create_predictive_context(
        y_modelable=y_modelable,
        y_nonmodelable=y_nonmodelable,
        exogenous_features=exogenous_features,
        prediction_date=prediction_date,
        prediction_window=prediction_window)

    y_mod_lags, y_nmod_lags, exog_features_total = create_lagged_sets(
        y_mod_context=y_mod_context,
        y_nmod_context=y_nmod_context,
        exogenous_features_context=exogenous_features_context,
        lags=lags,
        prediction_window=prediction_window)

    # Zet de fitting window
    max_date = last_train_date
    min_date = y_mod_lags.index.min() + datetime.timedelta(days=7*lags) # Adjust for lags

    # Maak de juiste fit sets: AR factoren, externe factoren en werkelijke waarden
    y_ar_m_fit = y_mod_lags.loc[max_date: min_date]
    X_exog_fit = exog_features_total.loc[y_ar_m_fit.index]
    y_true_fit = y_modelable.loc[y_ar_m_fit.index]

    # Isoleer de waarden die gaan worden gebruikt voor predictie
    yl_ar_m_prd = y_mod_lags.loc[prediction_date]
    yl_ar_nm_prd = y_nmod_lags.loc[prediction_date]
    X_exog_prd = exog_features_total.loc[prediction_date]

    # Pas de index aan van de waarden die worden gebruikt voor de voorspelling

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
