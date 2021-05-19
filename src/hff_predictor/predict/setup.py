import datetime
import hff_predictor.generic.files
import pandas as pd
import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
import hff_predictor.data.transformations as dtr
from hff_predictor.generic.files import import_temp_file

import logging
LOGGER = logging.getLogger(__name__)


def split_products(
    active_products,
    min_obs=cn.TRAIN_OBS,
    prediction_date=cn.PREDICTION_DATE,
    prediction_window=cn.PREDICTION_WINDOW,
):
    last_train_date = prediction_date - datetime.timedelta(weeks=prediction_window)
    first_train_date = last_train_date - datetime.timedelta(weeks=min_obs)
    fitting_window = active_products.loc[last_train_date:first_train_date]

    active_products = active_products.loc[last_train_date:first_train_date]

    obs_count = pd.DataFrame(fitting_window.count())
    obs_count.columns = ["count"]

    series_to_model = obs_count[obs_count["count"] >= min_obs].index
    LOGGER.info("Number of products able to model: {}".format(len(series_to_model)))

    series_not_to_model = obs_count[obs_count["count"] < min_obs].index
    LOGGER.info("Number of products not able to model: {}".format(len(series_not_to_model)))

    # Consumentgroep nummer inladen
    consumentgroep_nr = import_temp_file(data_loc=fm.ORDER_DATA_CG_PR_FOLDER, set_index=False)
    consumentgroep_nr = consumentgroep_nr[[cn.INKOOP_RECEPT_NM, cn.CONSUMENT_GROEP_NR]]
    consumentgroep_nr.set_index(cn.INKOOP_RECEPT_NM, inplace=True)

    active_rol_products = dtr.find_rol_products(data=active_products,
                                                consumentgroep_nrs=consumentgroep_nr)

    products_model = active_products[series_to_model].copy(deep=True)

    # Product groupings
    products_model[cn.MOD_PROD_SUM] = products_model.sum(axis=1) # Modelable products
    products_model[cn.ALL_PROD_SUM] = active_products.sum(axis=1) # All products
    products_model[cn.ALL_ROL_SUM] = active_products[active_rol_products].sum(axis=1) # All rol-products
    products_no_model = active_products[series_not_to_model]

    return products_model, products_no_model


def create_lagged_sets(y_modelable, y_nonmodelable, exogenous_features, prediction_window, lags):
    # Look-back only features: Orders, superunie, weather

    exog_features_lookback = exogenous_features['superunie_n'].join(
        exogenous_features['superunie_pct'], how='left')
       # .join(exogenous_features['weather'], how='left') # wee tijd hier weggelaten

    # Generate lags
    exog_features_lookback_lags = dtr.create_lags(exog_features_lookback, lag_range=lags)
    y_m_lags = dtr.create_lags(y_modelable, lag_range=lags)
    y_nm_lags = dtr.create_lags(y_nonmodelable, lag_range=lags)

    # Features that can look forward
    exog_features_lookahead = (exogenous_features['holidays'].join(
        exogenous_features['campaigns'], how='left').join(
        exogenous_features['covid'], how='left').join(
        exogenous_features['weather'], how='left') # tijdelijk weer toegevoegd
    )

    exog_features_no_adj = exogenous_features['seasons'].join(exogenous_features['breaks'], how='left')

    lookahead = prediction_window + 3  # Number of weeks the features can look ahead
    lookahead_range = list(reversed(range(-lags, lookahead)))

    exog_features_lookahead_lags = dtr.create_lags(exog_features_lookahead, lag_range=lookahead_range)

    return (y_m_lags, y_nm_lags,
            exog_features_lookback_lags,
            exog_features_lookahead_lags,
            exog_features_no_adj)


def create_predictive_context(y_modelable_lag,
                              y_nonmodelable_lag,
                              features_lag_lookback,
                              features_lag_lookahead,
                              features_no_adj,
                              prediction_window):

    features_na_corr = features_no_adj.sort_index(ascending=False).shift(prediction_window)
    features_total = features_lag_lookback.join(
        features_lag_lookahead, how='left').join(
        features_na_corr, how='left'
    )

    features_total_shift = features_total.shift(-prediction_window)[:-prediction_window]

    return (y_modelable_lag.shift(-prediction_window)[:-prediction_window],
            y_nonmodelable_lag.shift(-prediction_window)[:-prediction_window],
            features_total,
            features_total_shift
            )


def create_model_setup(
    y_modelable,
    y_nonmodelable,
    exogenous_features,
    difference=False,
    lags=cn.N_LAGS,
    prediction_date=cn.PREDICTION_DATE,
    prediction_window=cn.PREDICTION_WINDOW,
):

    logging.info("The prediction date is: {}".format(
        datetime.datetime.strftime(prediction_date, "%Y-%m-%d")))

    last_train_date = prediction_date - datetime.timedelta(weeks=prediction_window)

    logging.info("The prediction window is {}, so the last train date is: {}".format(
        prediction_window,
        datetime.datetime.strftime(last_train_date, "%Y-%m-%d")))

    dtr.fill_missing_values(data=y_modelable)
    dtr.fill_missing_values(data=y_nonmodelable)

    # LTD = Last Train Date
    y_m_ltd = y_modelable.loc[last_train_date]
    y_nm_ltd = y_nonmodelable.loc[last_train_date]

    if difference:
        y_modelable = dtr.first_difference_data(undifferenced_data=y_modelable, delta=1, scale=False)
        y_nonmodelable = dtr.first_difference_data(undifferenced_data=y_nonmodelable, delta=1, scale=False)

    # Create lags
    y_m_lags, y_nm_lags, X_lbl, X_lal, X_na = create_lagged_sets(y_modelable=y_modelable,
                                                                 y_nonmodelable=y_nonmodelable,
                                                                 exogenous_features=exogenous_features,
                                                                 prediction_window=prediction_window,
                                                                 lags=lags)

    # Create predictive context, X_exog_t for non shifted exogenous features
    y_ar_m, y_ar_nm, X_exog_t, X_exog_l = create_predictive_context(y_modelable_lag=y_m_lags,
                                                                    y_nonmodelable_lag=y_nm_lags,
                                                                    features_lag_lookback=X_lbl,
                                                                    features_lag_lookahead=X_lal,
                                                                    features_no_adj=X_na,
                                                                    prediction_window=prediction_window)

    max_date = last_train_date
    min_date = y_ar_m.index.min() + datetime.timedelta(days=7*lags) # Adjust for lags
    y_ar_m_fit = y_ar_m.loc[max_date: min_date]
    X_exog_fit = X_exog_l.loc[y_ar_m_fit.index]
    y_true_fit = y_modelable.loc[y_ar_m_fit.index]

    yl_ar_m_prd = y_m_lags.loc[last_train_date]
    yl_ar_nm_prd = y_nm_lags.loc[last_train_date]
    X_exog_prd = X_exog_t.loc[last_train_date]

    yl_ar_m_prd.name += datetime.timedelta(days=prediction_window * 7)
    yl_ar_nm_prd.name += datetime.timedelta(days=prediction_window * 7)
    X_exog_prd.name += datetime.timedelta(days=prediction_window * 7)

    model_fitting = {
        cn.Y_TRUE: y_true_fit,
        cn.Y_AR: y_ar_m_fit,
        cn.X_EXOG: X_exog_fit,
        cn.MOD_PROD: y_modelable.columns,
        cn.NON_MOD_PROD: y_nonmodelable.columns,
    }

    model_prediction = {
        cn.Y_AR_M: yl_ar_m_prd,
        cn.Y_AR_NM: yl_ar_nm_prd,
        cn.X_EXOG: X_exog_prd,
        cn.Y_M_UNDIF: y_m_ltd,
        cn.Y_NM_UNDIF: y_nm_ltd,
    }

    return model_fitting, model_prediction


def prediction_setup_wrapper(
    prediction_date,
    prediction_window,
    train_obs,
    nlags,
    difference,
    act_products,
    exog_features,
    save_to_pkl=False,
):

    if type(prediction_date) == str:
        prediction_date = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")

    products_model, products_nmodel = split_products(
        active_products=act_products,
        min_obs=train_obs,
        prediction_date=prediction_date,
        prediction_window=prediction_window,
    )

    if products_model.shape[1] == 1:
        train_obs_reduced = train_obs - 10
        products_model, products_nmodel = split_products(
            active_products=act_products,
            min_obs=train_obs_reduced,
            prediction_date=prediction_date,
            prediction_window=prediction_window,
        )

        print("Reduced train obs to have modelable products.")

    data_fitting, data_prediction = create_model_setup(
        y_modelable=products_model,
        y_nonmodelable=products_nmodel,
        prediction_date=prediction_date,
        prediction_window=prediction_window,
        exogenous_features=exog_features,
        difference=difference,
        lags=nlags,
    )

    if save_to_pkl:
        hff_predictor.generic.files.save_to_pkl(
            data=data_fitting, file_name="fit_data", folder=fm.SAVE_LOC
        )
        hff_predictor.generic.files.save_to_pkl(
            data=data_prediction, file_name="predict_data", folder=fm.SAVE_LOC
        )

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
