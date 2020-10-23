import statsmodels.api as sm
import pandas as pd
import numpy as np
import prediction.general_purpose_functions as gf
import prediction.file_management as fm
import prediction.column_names as cn


def get_top_correlations(y, y_lags, top_correl=5):
    # Rowwise mean of input arrays & subtract from input arrays themeselves

    A_mA = y - y.mean()
    B_mB = y_lags - y_lags.mean()

    # Sum of squares across rows
    ssA = (A_mA**2).sum()
    ssB = (B_mB**2).sum()

    numerator = np.dot(A_mA.T, B_mB)
    denominator = np.sqrt(np.dot(pd.DataFrame(ssA), pd.DataFrame(ssB).T))
    correls = abs(numerator / denominator)

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
            top_correlations[p] = corrs.loc[p].sort_values(ascending=False)[:top_correl].index

        # Finally get corr coeff
        return top_correlations, corrs


def optimize_ar_model(y, y_ar, X_exog, constant=True, model='OLS'):
    level_cols = ['period_2', 'period_3', 'trans_period_1', 'trans_period_2']
    all_level_features = X_exog[level_cols]
    sorted_lags = y_ar.columns.sort_values(ascending=True)

    use_level_features = all_level_features.loc[:, (all_level_features != 0).any(axis=0)]

    optimal_lags = 1
    min_fit_val = 1e9

    for lag in range(1, len(sorted_lags)+1):
        _y_ar = y_ar.iloc[:, :lag]
        X_ar = _y_ar.join(use_level_features, how='left')

        if constant:
            X_ar.insert(0, 'constant', 1)

        _fit = fit_model(y=y, X=X_ar, model=model)

        _fit_value = round((abs(y - _fit.predict())/y).median(), 5)
        print("Current fit value {}, with {} lags".format(_fit_value, lag))

        if _fit_value < min_fit_val:
            min_fit_val = _fit_value
            optimal_lags = lag

    lag_values = y_ar.iloc[:, :optimal_lags]

    X_exog_rf = X_exog.drop(columns=level_cols, inplace=False, errors='ignore')

    return lag_values.join(use_level_features, how='left'), X_exog_rf


def fit_model(y, X, model='OLS'):

    if model == 'OLS':
        temp_mdl = sm.OLS(y, X, missing='drop')

    elif model == 'Poisson':
        temp_mdl = sm.GLM(y, X,
                          family=sm.families.Poisson(),
                          missing='drop')

    elif model == 'Negative-Binomial':
        aux_reg_feat = pd.DataFrame(index=y.index)

        temp_mdl_poisson = sm.GLM(y, X,
                                  family=sm.families.Poisson(),
                                  missing='drop')

        temp_poisson_fit = temp_mdl_poisson.fit()

        aux_reg_feat['lambda'] = temp_poisson_fit.mu
        aux_reg_feat['dep_var'] = ((y - temp_poisson_fit.mu) ** 2 - y) / temp_poisson_fit.mu
        aux_reg = sm.OLS(aux_reg_feat['dep_var'], aux_reg_feat['lambda']).fit()

        alpha_fit = aux_reg.params[0]

        temp_mdl = sm.GLM(y, X,
                          family=sm.families.NegativeBinomial(alpha=alpha_fit),
                          missing='drop')

    return temp_mdl.fit()


def batch_fit_model(Y, Y_ar, X_exog, add_constant=True, model='OLS'):
    Y_pred = pd.DataFrame(index=Y.index)
    fitted_models = {}
    optimized_features = {}

    for product in Y.columns:
        y_name = product
        y = Y[y_name]
        lag_index = [y_name in x for x in Y_ar.columns]
        y_ar = Y_ar.iloc[:, lag_index]

        lag_index_other = [y_name not in x for x in Y_ar.columns]
        y_ar_other = Y_ar.iloc[:, lag_index_other]

        ar_baseline, X_exog_rf = optimize_ar_model(y=y, y_ar=y_ar, X_exog=X_exog, constant=add_constant, model=model)
        baseline_fit = fit_model(y=y, X=ar_baseline, model=model)

        all_possible_features = y_ar_other.join(X_exog_rf, how='left')

        resid = y - baseline_fit.predict()
        correlation_val = 1
        selected_features = ar_baseline.copy(deep=True)

        if add_constant:
            selected_features.insert(0, 'constant', 1)

        while correlation_val > 0.10 and selected_features.shape[1] < 20:

            corr_name, correlation_val = get_top_correlations(y=pd.DataFrame(resid), y_lags=all_possible_features,
                                                              top_correl=1)
            selected_features = selected_features.join(all_possible_features[corr_name], how='left')
            mdl_fit = fit_model(y=y, X=selected_features, model=model)
            resid = y - mdl_fit.predict()

        Y_pred[y_name] = mdl_fit.predict()
        fitted_models[y_name] = mdl_fit
        optimized_features[y_name] = selected_features.columns

    return Y_pred, fitted_models, optimized_features


def batch_make_prediction(Yp_ar_m, Yp_ar_nm, Xp_exog, Y_cross_ar, fitted_models, prediction_window,
                          add_constant=True, prep_input=True, find_comparable_model=True):

    def series_to_dataframe(pd_series):
        return pd.DataFrame(pd_series).transpose()

    if prep_input:
        Yp_ar_m = series_to_dataframe(Yp_ar_m)
        Yp_ar_nm = series_to_dataframe(Yp_ar_nm)
        Xp_exog = series_to_dataframe(Xp_exog)

    Y_pred = pd.DataFrame(index=Yp_ar_m.index)

    # product_m = Yp_ar_m.columns[0]
    for product_m in Yp_ar_m.columns:
        y_name_m = product_m[:-6]

        lag_index = [y_name_m in x for x in Yp_ar_m.columns]
        Xp_ar_m = Yp_ar_m.iloc[:, lag_index]

        Xp_arx_m = Yp_ar_m[Y_cross_ar[y_name_m]]

        if add_constant:
            Xp_ar_m.insert(0, 'constant', 1)

        Xp_tot = Xp_ar_m.join(Xp_arx_m, how='left').join(Xp_exog, how='left')

        Y_pred[y_name_m] = fitted_models[y_name_m].predict(Xp_tot)

    # product_nm = Yp_ar_nm.columns[0]
    for product_nm in Yp_ar_nm.columns:
        y_name_nm = product_nm[:-6]  # remove '_lag_1 or 2'

        lag_index = [y_name_nm in x for x in Yp_ar_nm.columns]
        Xp_ar_nm = Yp_ar_nm.iloc[:, lag_index]

        if find_comparable_model:
            # Find product which has similar magnitude absolute sales
            lag_val = '_lag_{}'.format(prediction_window)
            _y_nm_val = Yp_ar_nm['{}{}'.format(y_name_nm, lag_val)][0]

            lag1_index = [lag_val in x for x in Yp_ar_m.columns]
            _Y_m_vals = Yp_ar_m.iloc[:, lag1_index]

            _closest_prod = (abs(_Y_m_vals - _y_nm_val) / _y_nm_val).T

            closest_product_name = _closest_prod.idxmin()[0][:-6]

        else:
            closest_product_name = cn.MOD_PROD_SUM

        if add_constant:
            Xp_ar_nm.insert(0, 'constant', 1)

        Xp_arx_nm = Yp_ar_m[Y_cross_ar[closest_product_name]]

        Xp_tot = Xp_ar_nm.join(Xp_arx_nm, how='left').join(Xp_exog, how='left')

        Y_pred[y_name_nm] = fitted_models[closest_product_name].predict(Xp_tot)

    return Y_pred


def fit_and_predict(fit_dict, predict_dict, prediction_window, model_type='OLS'):
    Yis_fit, model_fits = batch_fit_model(Y=fit_dict[cn.Y_TRUE], Y_ar=fit_dict[cn.Y_AR], X_exog=fit_dict[cn.X_EXOG],
                                          Y_cross_ar=fit_dict['correlations'], model=model_type)

    Yos_pred = batch_make_prediction(Yp_ar_m=predict_dict[cn.Y_AR_M], Yp_ar_nm=predict_dict[cn.Y_AR_NM],
                                     Xp_exog=predict_dict[cn.X_EXOG], Y_cross_ar=fit_dict["correlations"],
                                     fitted_models=model_fits,
                                     find_comparable_model=True, prediction_window=prediction_window)

    return Yis_fit, Yos_pred


if __name__ == '__main__':

    fit_dict = gf.read_pkl(file_name=fm.FIT_DATA, data_loc=fm.SAVE_LOC)
    predict_dict = gf.read_pkl(file_name=fm.PREDICT_DATA, data_loc=fm.SAVE_LOC)
    model_type = 'OLS'
    Y = fit_dict[cn.Y_TRUE]
    Y_ar = fit_dict[cn.Y_AR]
    X_exog = fit_dict[cn.X_EXOG]
    Y_cross_ar = fit_dict['correlations']
    model = model_type

    Yis_fit, model_fits, model_features = batch_fit_model(Y=fit_dict[cn.Y_TRUE], Y_ar=fit_dict[cn.Y_AR],
                                                          X_exog=fit_dict[cn.X_EXOG], model=model_type)

    Yos_pred = batch_make_prediction(Yp_ar_m=predict_dict[cn.Y_AR_M], Yp_ar_nm=predict_dict[cn.Y_AR_NM],
                                     Xp_exog=predict_dict[cn.X_EXOG], Y_cross_ar=fit_dict["correlations"],
                                     fitted_models=model_fits,
                                     find_comparable_model=True, prediction_window=2)

    gf.save_to_csv(data=Yis_fit, file_name="insample_fit", folder=fm.SAVE_LOC)


