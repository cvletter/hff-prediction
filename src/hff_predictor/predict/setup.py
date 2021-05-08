import datetime

import hff_predictor.generic.files
import pandas as pd

import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
import hff_predictor.generic.data_transformations as dtr
import logging


def split_products(
    active_products,
    min_obs=cn.TRAIN_OBS,
    prediction_date=cn.PREDICTION_DATE,
    hold_out=cn.PREDICTION_WINDOW,
):
    last_train_date = prediction_date - datetime.timedelta(weeks=hold_out)
    first_train_date = last_train_date - datetime.timedelta(weeks=min_obs)
    fitting_window = active_products.loc[last_train_date:first_train_date]

    active_products = active_products.loc[last_train_date:first_train_date]

    obs_count = pd.DataFrame(fitting_window.count())
    obs_count.columns = ["count"]

    series_to_model = obs_count[obs_count["count"] >= min_obs].index
    logging.info("Number of products able to model: {}".format(len(series_to_model)))

    series_not_to_model = obs_count[obs_count["count"] < min_obs].index
    logging.info("Number of products not able to model: {}".format(len(series_not_to_model)))

    #TODO: HIER NOG MEER AGGREGATIES TOEVOEGEN
    products_model = active_products[series_to_model].copy(deep=True)
    products_model[cn.MOD_PROD_SUM] = products_model.sum(axis=1)
    products_no_model = active_products[series_not_to_model]

    return products_model, products_no_model


def create_predictive_context(y_m, y_nm, exog_f, hold_out):

    exog_f_restruc= pd.DataFrame(index=y_m.index)



    return (
        y_m.shift(-hold_out)[:-hold_out],
        y_nm.shift(-hold_out)[:-hold_out],
        exog_f.shift(-hold_out)[:-hold_out],
        exog_f,
    )


def create_model_setup(
    y_m,
    y_nm,
    X_exog,
    difference=False,
    lags=cn.N_LAGS,
    prediction_date=cn.PREDICTION_DATE,
    hold_out=cn.PREDICTION_WINDOW,
):

    logging.info("The prediction date is: {}".format(
        datetime.datetime.strftime(prediction_date, "%Y-%m-%d")))

    last_train_date = prediction_date - datetime.timedelta(weeks=hold_out)

    logging.info("The prediction window is {}, so the last train date is: {}".format(
        hold_out,
        datetime.datetime.strftime(last_train_date, "%Y-%m-%d")))

    dtr.fill_missing_values(data=y_m)
    dtr.fill_missing_values(data=y_nm)

    # LTD = Last Train Date
    y_m_ltd = y_m.loc[last_train_date]
    y_nm_ltd = y_nm.loc[last_train_date]

    if difference:
        y_m = dtr.first_difference_data(undifferenced_data=y_m, delta=1, scale=False)
        y_nm = dtr.first_difference_data(undifferenced_data=y_nm, delta=1, scale=False)

    # Create lags
    def create_lagged_sets(y_m, y_nm, exog_features, hold_out, lags):
        # Look-back only features: Orders, superunie, weather

        exog_features_lookback = exog_features['weather'].join(
            exog_features['superunie_pct'], how='left'). join(
            exog_features['superunie_n'], how='left'
        )

        exog_features_lookback_lags = dtr.create_lags(exog_features_lookback, lag_range=lags)
        y_m_lags = dtr.create_lags(y_m, lag_range=lags)
        y_nm_lags = dtr.create_lags(y_nm, lag_range=lags)

        exog_features_lookahead = exog_features['holidays'].join(
            exog_features['campaigns'], how='left'). join(
            exog_features['covid'], how='left'
        )

        lookahead = hold_out + 3
        lookahead_range = list(reversed(range(-lags, lookahead)))

        exog_features_lookahead_lags = dtr.create_lags(exog_features_lookahead, lag_range=lookahead_range)

        exog_features_unadj = exog_features['seasons'].join(exog_features['breaks'], how='left')

        return y_m_lags, y_nm_lags, exog_features_lookback_lags, exog_features_lookahead_lags, exog_features_unadj

    ym_l, y_nm_l, X_lbl, X_lal, X_ua = create_lagged_sets(y_m=y_m,
                                                          y_nm=y_nm,
                                                          exog_features=X_exog,
                                                          hold_out=hold_out,
                                                          lags=nlags)

    # Create predictive context

    def create_predictive_context(y_ml,
                                  y_nml,
                                  X_lbl,
                                  X_lal,
                                  X_ua,
                                  hold_out):

        X_exog_shift = X_lbl.join(X_lal, how='left').shift(-hold_out)[:-hold_out]
        X_exog = X_exog_shift.join(X_ua, how='left')

        return (y_ml.shift(-hold_out)[:-hold_out],
                y_nml.shift(-hold_out)[:-hold_out],
                X_exog
        )

    y_ar_m, y_ar_nm, X_exog_t = create_predictive_context(y_ml=ym_l, y_nml=y_nm_l, X_lbl=X_lbl, X_lal=X_lal, X_ua=X_ua,
                                                          hold_out=hold_out)



    ## TOT HIER GEKOMEN!



    y_m_lags = create_lags(data=y_m, lag_range=lags)

    X_exog_nl = X_exog[cn.SEASONAL_COLS]
    X_exog_ml = X_exog.drop(cn.SEASONAL_COLS, axis=1)

    y_ar_m, y_ar_nm, X_exog_mll, X_exog_mlt = create_predictive_context(
        mod=y_m_lags, non_mod=y_nm_lags, exog_f=X_exog_ml, hold_out=hold_out
    )

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

    model_fitting = {
        cn.Y_TRUE: y_true_fit,
        cn.Y_AR: y_ar_m_fit,
        cn.X_EXOG: X_exog_fit,
        cn.MOD_PROD: y_m.columns,
        cn.NON_MOD_PROD: y_nm.columns,
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
        hold_out=prediction_window,
    )

    if products_model.shape[1] == 1:
        train_obs_reduced = train_obs - 10
        products_model, products_nmodel = split_products(
            active_products=act_products,
            min_obs=train_obs_reduced,
            prediction_date=prediction_date,
            hold_out=prediction_window,
        )

        print("Reduced train obs to have modelable products.")

    data_fitting, data_prediction = create_model_setup(
        y_m=products_model,
        y_nm=products_nmodel,
        prediction_date=prediction_date,
        hold_out=prediction_window,
        X_exog=exog_features,
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
        prediction_date="2020-12-04",
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

# To delete later
prediction_date = "2021-04-12"
prediction_window = 2
train_obs = cn.TRAIN_OBS
nlags = 3
difference = False
act_products = active_products_t
exog_features = exog_features_t
save_to_pkl = False

products_model, products_nmodel = split_products(
    active_products=act_products,
    min_obs=train_obs,
    prediction_date=prediction_date,
    hold_out=prediction_window,
)

y_m = products_model
y_nm = products_nmodel
hold_out = prediction_window
X_exog = exog_features
lags = nlags