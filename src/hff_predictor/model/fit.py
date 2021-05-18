import hff_predictor.generic.files
import pandas as pd
import numpy as np
import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
from hff_predictor.model.preprocessing import fit_model, predictor

import logging
LOGGER = logging.getLogger(__name__)


def get_top_correlations(y, y_lags, top_correl=5):
    # Rowwise mean of input arrays & subtract from input arrays themeselves

    A_mA = y - y.mean()
    B_mB = y_lags - y_lags.mean()

    # Sum of squares across rows
    ssA = (A_mA ** 2).sum()
    ssB = (B_mB ** 2).sum()

    numerator = np.dot(A_mA.T, B_mB)
    denominator = np.sqrt(np.dot(pd.DataFrame(ssA), pd.DataFrame(ssB).T))
    correls = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )
    corrs = pd.DataFrame(correls, index=y.columns, columns=y_lags.columns)

    for i in corrs.index:
        for j in corrs.columns:
            if i == j[:-7]:
                corrs.loc[i, j] = -1e9

    top_correlations = {}
    if len(y.columns) == 1 & top_correl == 1:
        top_name = corrs.T.idxmax()[0]
        top_value = round(corrs[top_name].values[0], 3)
        return top_name, top_value
    else:
        for p in corrs.index:
            top_correlations[p] = (
                corrs.loc[p].sort_values(ascending=False)[:top_correl].index
            )

        # Finally get corr coeff
        return top_correlations, corrs


def optimize_ar_model(y, y_ar, X_exog, constant=True, model="OLS"):
    # Baseline features
    all_level_features = X_exog[cn.STRUCTURAL_BREAK_COLS]
    all_season_features = X_exog[cn.SEASONAL_COLS]
    all_month_features = X_exog[cn.MONTH_COLS]
    all_holiday_features = X_exog[cn.HOLIDAY_COLS]
    sorted_lags = y_ar.columns.sort_values(ascending=True)

    all_baseline_features = all_level_features.join(
        all_season_features, how='left').join(
        all_month_features, how='left').join(
        all_holiday_features, how='left'
    )

    use_baseline_features = all_baseline_features.loc[
        :, (all_baseline_features != 0).any(axis=0)
    ]

    if use_baseline_features.sum(axis=1).sum() == len(y):
        use_baseline_features = use_baseline_features.iloc[:, 1:]

    # season_break_cols = use_baseline_features.join(all_season_features, how="left")
    optimal_lags = 1
    min_fit_val = 1e9

    for lag in range(1, len(sorted_lags) + 1):

        _y_ar = y_ar.iloc[:, :lag]
        X_ar = _y_ar.join(use_baseline_features, how="left")

        if constant:
            X_ar.insert(0, "constant", 1)

        _fit = fit_model(y=y, X=X_ar, model=model)

        _fit_value = round((abs(y - _fit.predict(X_ar)) / y).median(), 5)
        # print("Current fit value {}, with {} lags".format(_fit_value, lag))

        if _fit_value < min_fit_val:
            min_fit_val = _fit_value
            optimal_lags = lag

    lag_values = y_ar.iloc[:, :optimal_lags]
    drop_cols = cn.SEASONAL_COLS + cn.STRUCTURAL_BREAK_COLS + cn.MONTH_COLS + [cn.HOLIDAY_COLS]

    X_exog_rf = X_exog.drop(columns=drop_cols, inplace=False, errors="ignore")

    return lag_values.join(use_baseline_features, how="left"), X_exog_rf


def batch_fit_model(
    Y, Y_ar, X_exog, add_constant=True, model="OLS", feature_threshold=None
):

    if feature_threshold is None:
        feature_threshold = [0.2, 15]

    Y_pred = pd.DataFrame(index=Y.index)
    fitted_models = {}
    all_params = {}
    optimized_ar_features = {}
    optimized_exog_features = {}

    for product in Y.columns:
        y_name = product
        y = Y[y_name]
        lag_index = [y_name in x for x in Y_ar.columns]
        y_ar = Y_ar.iloc[:, lag_index]

        lag_index_other = [y_name not in x for x in Y_ar.columns]
        y_ar_other = Y_ar.iloc[:, lag_index_other]

        ar_baseline, X_exog_rf = optimize_ar_model(
            y=y, y_ar=y_ar, X_exog=X_exog, constant=add_constant, model=model
        )
        baseline_fit = fit_model(y=y, X=ar_baseline, model=model)

        all_possible_features = y_ar_other.join(X_exog_rf, how="left")

        resid = y - baseline_fit.predict(ar_baseline)
        correlation_val = 1
        selected_features = ar_baseline.copy(deep=True)

        if add_constant:
            selected_features.insert(0, "constant", 1)

        while (
            correlation_val > feature_threshold[0]
            and selected_features.shape[1] < feature_threshold[1]
        ):

            corr_name, correlation_val = get_top_correlations(
                y=pd.DataFrame(resid), y_lags=all_possible_features, top_correl=1
            )

            selected_features = selected_features.join(
                all_possible_features[corr_name], how="left"
            )

            all_possible_features.drop(corr_name, axis=1, inplace=True)

            mdl_fit = fit_model(y=y, X=selected_features, model=model)
            resid = y - mdl_fit.predict(selected_features)

        ar_name = "{}_last".format(y_name)
        ar_cols = [ar_name in x for x in selected_features.columns]

        exog_cols = [not x for x in ar_cols]
        ar_features = selected_features.iloc[:, ar_cols]
        exog_features = selected_features.iloc[:, exog_cols]

        Y_pred[y_name] = mdl_fit.predict(selected_features)
        fitted_models[y_name] = mdl_fit
        all_params[y_name] = selected_features.columns
        optimized_ar_features[y_name] = ar_features.columns
        optimized_exog_features[y_name] = exog_features.columns

        # Determine Prediction intervals

    return (
        Y_pred,
        fitted_models,
        all_params,
        optimized_ar_features,
        optimized_exog_features,
    )


def batch_make_prediction(
    Yp_ar_m,
    Yp_ar_nm,
    Xp_exog,
    fitted_models,
    Yf_ar_opt,
    Yf_exog_opt,
    add_constant=True,
    prep_input=True,
    model_type="OLS",
    find_comparable_model=True,
):
    def series_to_dataframe(pd_series):
        return pd.DataFrame(pd_series).transpose()

    if prep_input:
        Yp_ar_m = series_to_dataframe(Yp_ar_m)
        Yp_ar_nm = series_to_dataframe(Yp_ar_nm)
        Xp_exog = series_to_dataframe(Xp_exog)

    Y_pred = pd.DataFrame(index=Yp_ar_m.index)

    Ym_products = list(set([x[:-7] for x in Yp_ar_m.columns]))  # Remove 'lag' tag

    for y_name_m in Ym_products:
        lag_index = [y_name_m in x for x in Yp_ar_m.columns]
        Xp_ar_m = Yp_ar_m.iloc[:, lag_index]

        Xf_ar_m = Yf_ar_opt[y_name_m]
        Xp_ar_m = Xp_ar_m.iloc[:, :Xf_ar_m.shape[0]]

        Xp_all_features = Yp_ar_m.join(Xp_exog, how="left")

        Xf_exog_m = Yf_exog_opt[y_name_m].drop("constant")

        Xp_arx_m = Xp_all_features[Xf_exog_m]

        if add_constant:
            Xp_ar_m.insert(0, "constant", 1)

        Xp_tot = Xp_ar_m.join(Xp_arx_m, how="left")

        Y_pred[y_name_m] = predictor(Xpred=Xp_tot, fitted_model=fitted_models[y_name_m], model=model_type)
        # Y_pred[y_name_m] = fitted_models[y_name_m].predict(Xp_tot)

    Ynm_products = list(set([x[:-7] for x in Yp_ar_nm.columns]))
    for y_name_nm in Ynm_products:

        lag_index = [y_name_nm in x for x in Yp_ar_nm.columns]
        Xp_ar_nm = Yp_ar_nm.iloc[:, lag_index]

        if find_comparable_model:
            # Find product which has similar magnitude absolute sales
            lag_val = "_last0w"
            _y_nm_val = Yp_ar_nm["{}{}".format(y_name_nm, lag_val)][0]

            lag1_index = [lag_val in x for x in Yp_ar_m.columns]
            _Y_m_vals = Yp_ar_m.iloc[:, lag1_index]

            _closest_prod = (abs(_Y_m_vals - _y_nm_val) / _y_nm_val).T

            closest_product_name = _closest_prod.idxmin()[0][:-7]

        else:
            closest_product_name = cn.MOD_PROD_SUM

        Xf_ar_cp = Yf_ar_opt[closest_product_name]
        Xp_ar_nm = Xp_ar_nm.iloc[:, : Xf_ar_cp.shape[0]]

        Xp_all_features = Yp_ar_m.join(Xp_exog, how="left")

        Xf_exog_cp = Yf_exog_opt[closest_product_name].drop("constant")

        Xp_arx_cp = Xp_all_features[Xf_exog_cp]

        if add_constant:
            Xp_ar_nm.insert(0, "constant", 1)

        Xp_tot = Xp_ar_nm.join(Xp_arx_cp, how="left")

        Y_pred[y_name_nm] = predictor(Xpred=Xp_tot, fitted_model=fitted_models[closest_product_name], model=model_type)
        # Y_pred[y_name_nm] = fitted_models[closest_product_name].predict(Xp_tot)

    return Y_pred


def fit_and_predict(
    fit_dict, predict_dict, model_type="OLS", bootstrap=False, feature_threshold=None
):
    def reset_index(data):
        data_new = data.reset_index(drop=True, inplace=False)
        data_new["bootstrap_index"] = np.arange(data.shape[0])
        return data_new.set_index("bootstrap_index", inplace=False, drop=True)

    if feature_threshold is None:
        feature_threshold = [0.2, 15]

    Y_org = fit_dict[cn.Y_TRUE]
    Yar_org = fit_dict[cn.Y_AR]
    X_org = fit_dict[cn.X_EXOG]

    # Y_org.sample(n=Y_org.shape[0], replace=True)

    if bootstrap:
        Y_fit = Y_org.sample(n=Y_org.shape[0], replace=True)
        Yar_fit = Yar_org.loc[Y_fit.index, :]
        X_fit = X_org.loc[Y_fit.index, :]

        Y_fit = reset_index(data=Y_fit)
        Yar_fit = reset_index(data=Yar_fit)
        X_fit = reset_index(data=X_fit)

    else:
        Y_fit = Y_org
        Yar_fit = Yar_org
        X_fit = X_org

    Yis_fit, model_fits, all_pars, Yar_opt, X_opt = batch_fit_model(
        Y=Y_fit,
        Y_ar=Yar_fit,
        add_constant=True,
        X_exog=X_fit,
        model=model_type,
        feature_threshold=[feature_threshold[0], feature_threshold[1]],
    )

    Yos_pred = batch_make_prediction(
        Yp_ar_m=predict_dict[cn.Y_AR_M],
        Yp_ar_nm=predict_dict[cn.Y_AR_NM],
        Xp_exog=predict_dict[cn.X_EXOG],
        fitted_models=model_fits,
        Yf_ar_opt=Yar_opt,
        Yf_exog_opt=X_opt,
        add_constant=True,
        model_type=model_type,
        find_comparable_model=True,
    )

    return Yis_fit, Yos_pred, all_pars


def init_train():
    fit_dict = hff_predictor.generic.files.read_pkl(
        file_name=fm.FIT_DATA, data_loc=fm.SAVE_LOC
    )
    predict_dict = hff_predictor.generic.files.read_pkl(
        file_name=fm.PREDICT_DATA, data_loc=fm.SAVE_LOC
    )

    Yis_fit, model_fits, all_pars, ar_f, exog_f = batch_fit_model(
        Y=fit_dict[cn.Y_TRUE],
        Y_ar=fit_dict[cn.Y_AR],
        X_exog=fit_dict[cn.X_EXOG],
        model="OLS",
        feature_threshold=[0.2, 25],
    )

    Yis_fit, Yos_pred, all_pars = batch_make_prediction(
        Yp_ar_m=predict_dict[cn.Y_AR_M],
        Yp_ar_nm=predict_dict[cn.Y_AR_NM],
        Xp_exog=predict_dict[cn.X_EXOG],
        Yf_ar_opt=ar_f,
        Yf_exog_opt=exog_f,
        fitted_models=model_fits,
        find_comparable_model=True,
    )

    yisfit, yosfit, pars = fit_and_predict(
        fit_dict=fit_dict, predict_dict=predict_dict, feature_threshold=[0.2, 25]
    )
