import statsmodels.api as sm
import pandas as pd
import numpy as np
import scipy.stats as stats
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
    correls = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
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
        # print("Current fit value {}, with {} lags".format(_fit_value, lag))

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


def batch_fit_model(Y, Y_ar, X_exog, add_constant=True, model='OLS', feature_threshold=None):

    if feature_threshold is None:
        feature_threshold = [0.2, 15]

    Y_pred = pd.DataFrame(index=Y.index)
    Y_pred_se = {}
    sigma_2 = {}
    fitted_models = {}
    optimized_ar_features = {}
    optimized_exog_features = {}

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

        while correlation_val > feature_threshold[0] and selected_features.shape[1] < feature_threshold[1]:

            corr_name, correlation_val = get_top_correlations(y=pd.DataFrame(resid), y_lags=all_possible_features,
                                                              top_correl=1)
            selected_features = selected_features.join(all_possible_features[corr_name], how='left')
            all_possible_features.drop(corr_name, axis=1, inplace=True)

            mdl_fit = fit_model(y=y, X=selected_features, model=model)
            resid = y - mdl_fit.predict()

        ar_name = "{}_last".format(y_name)
        ar_cols = [ar_name in x for x in selected_features.columns]

        exog_cols = [not x for x in ar_cols]
        ar_features = selected_features.iloc[:, ar_cols]
        exog_features = selected_features.iloc[:, exog_cols]

        Y_pred[y_name] = mdl_fit.predict()
        fitted_models[y_name] = mdl_fit
        optimized_ar_features[y_name] = ar_features.columns
        optimized_exog_features[y_name] = exog_features.columns

        # Determine Prediction intervals

        sigma_2[y_name] = np.sum((y - mdl_fit.predict()) ** 2) / (len(y) - 2)

        Xs = selected_features
        Y_pred_se[y_name] = np.linalg.inv(np.dot(np.transpose(Xs), Xs))

    return Y_pred, fitted_models, optimized_ar_features, optimized_exog_features, Y_pred_se, sigma_2


def batch_make_prediction(Yp_ar_m, Yp_ar_nm, Xp_exog, fitted_models, Yf_ar_opt, Yf_exog_opt,
                          Y_pred_se, sigma_2, add_constant=True, prep_input=True, find_comparable_model=True):

    def series_to_dataframe(pd_series):
        return pd.DataFrame(pd_series).transpose()

    if prep_input:
        Yp_ar_m = series_to_dataframe(Yp_ar_m)
        Yp_ar_nm = series_to_dataframe(Yp_ar_nm)
        Xp_exog = series_to_dataframe(Xp_exog)

    Y_pred = pd.DataFrame(index=Yp_ar_m.index)
    Yl_pred = pd.DataFrame(index=Yp_ar_m.index)
    Yh_pred = pd.DataFrame(index=Yp_ar_m.index)
    Ym_width = {}

    Ym_products = list(set([x[:-7] for x in Yp_ar_m.columns]))

    for y_name_m in Ym_products:
        lag_index = [y_name_m in x for x in Yp_ar_m.columns]
        Xp_ar_m = Yp_ar_m.iloc[:, lag_index]

        Xf_ar_m = Yf_ar_opt[y_name_m]
        Xp_ar_m = Xp_ar_m.iloc[:, :Xf_ar_m.shape[0]]

        Xp_all_features = Yp_ar_m.join(Xp_exog, how='left')

        Xf_exog_m = Yf_exog_opt[y_name_m].drop('constant')

        Xp_arx_m = Xp_all_features[Xf_exog_m]

        if add_constant:
            Xp_ar_m.insert(0, 'constant', 1)

        Xp_tot = Xp_ar_m.join(Xp_arx_m, how='left')

        Y_pred[y_name_m] = fitted_models[y_name_m].predict(Xp_tot)

        y_pred_se = np.dot(np.dot(Xp_tot, Y_pred_se[y_name_m]), np.transpose(Xp_tot))
        y_pred_se = np.identity(len(Xp_tot)) + y_pred_se
        y_pred_se = sigma_2[y_name_m] * y_pred_se
        y_pred_se = np.sqrt(np.diag(y_pred_se))
        alpha = 0.05
        Ym_width[y_name_m] = stats.t.ppf(q=1-alpha/2, df=cn.TRAIN_OBS-2) * y_pred_se

        Yl_pred[y_name_m] = Y_pred[y_name_m] - Ym_width[y_name_m]
        Yh_pred[y_name_m] = Y_pred[y_name_m] + Ym_width[y_name_m]

    Ynm_products = list(set([x[:-7] for x in Yp_ar_nm.columns]))
    for y_name_nm in Ynm_products:

        lag_index = [y_name_nm in x for x in Yp_ar_nm.columns]
        Xp_ar_nm = Yp_ar_nm.iloc[:, lag_index]

        if find_comparable_model:
            # Find product which has similar magnitude absolute sales
            lag_val = '_last0w'
            _y_nm_val = Yp_ar_nm['{}{}'.format(y_name_nm, lag_val)][0]

            lag1_index = [lag_val in x for x in Yp_ar_m.columns]
            _Y_m_vals = Yp_ar_m.iloc[:, lag1_index]

            _closest_prod = (abs(_Y_m_vals - _y_nm_val) / _y_nm_val).T

            closest_product_name = _closest_prod.idxmin()[0][:-7]

        else:
            closest_product_name = cn.MOD_PROD_SUM

        Xf_ar_cp = Yf_ar_opt[closest_product_name]
        Xp_ar_nm = Xp_ar_nm.iloc[:, :Xf_ar_cp.shape[0]]

        Xp_all_features = Yp_ar_m.join(Xp_exog, how='left')

        Xf_exog_cp = Yf_exog_opt[closest_product_name].drop('constant')

        Xp_arx_cp = Xp_all_features[Xf_exog_cp]

        if add_constant:
            Xp_ar_nm.insert(0, 'constant', 1)

        Xp_tot = Xp_ar_nm.join(Xp_arx_cp, how='left')

        Y_pred[y_name_nm] = fitted_models[closest_product_name].predict(Xp_tot)
        Yh_pred[y_name_nm] = Y_pred[y_name_nm] + Ym_width[closest_product_name]
        Yl_pred[y_name_nm] = Y_pred[y_name_nm] - Ym_width[closest_product_name]

    return Y_pred, Yh_pred, Yl_pred


def fit_and_predict(fit_dict, predict_dict, model_type='OLS', feature_threshold=None):

    if feature_threshold is None:
        feature_threshold = [0.2, 15]

    Yis_fit, model_fits,Yar_opt, X_opt, Ypred_se, sigma2 = batch_fit_model(
        Y=fit_dict[cn.Y_TRUE], Y_ar=fit_dict[cn.Y_AR],
        add_constant=True, X_exog=fit_dict[cn.X_EXOG],
        model=model_type, feature_threshold=[feature_threshold[0], feature_threshold[1]])

    Yos_pred, Yos_hpred, Yos_lpred = batch_make_prediction(
        Yp_ar_m=predict_dict[cn.Y_AR_M], Yp_ar_nm=predict_dict[cn.Y_AR_NM],
        Y_pred_se=Ypred_se, sigma_2=sigma2,
        Xp_exog=predict_dict[cn.X_EXOG], fitted_models=model_fits, Yf_ar_opt=Yar_opt,
        Yf_exog_opt=X_opt, add_constant=True, find_comparable_model=True)

    return Yis_fit, Yos_pred, Yos_hpred, Yos_lpred


if __name__ == '__main__':

    fit_dict = gf.read_pkl(file_name=fm.FIT_DATA, data_loc=fm.SAVE_LOC)
    predict_dict = gf.read_pkl(file_name=fm.PREDICT_DATA, data_loc=fm.SAVE_LOC)

    fit_dict = fit_data
    predict_dict = predict_data
    model_type = 'OLS'
    Y = fit_dict[cn.Y_TRUE]
    Y_ar = fit_dict[cn.Y_AR]
    X_exog = fit_dict[cn.X_EXOG]
    Y_cross_ar = fit_dict['correlations']
    model = model_type

    Yp_ar_m = predict_dict[cn.Y_AR_M]
    Yp_ar_nm = predict_dict[cn.Y_AR_NM]
    Xp_exog = predict_dict[cn.X_EXOG]
    Yf_ar_opt = ar_f
    Yf_exog_opt = exog_f
    fitted_models = model_fits
    find_comparable_model = True
    prediction_window = 1

    Yis_fit, model_fits, ar_f, exog_f, ypredse, sigma2 = batch_fit_model(Y=fit_dict[cn.Y_TRUE], Y_ar=fit_dict[cn.Y_AR],
                                                        X_exog=fit_dict[cn.X_EXOG], model='OLS',
                                                        feature_threshold=[0.2, 15])

    Yos_pred = batch_make_prediction(Yp_ar_m=predict_dict[cn.Y_AR_M], Yp_ar_nm=predict_dict[cn.Y_AR_NM],
                                     Xp_exog=predict_dict[cn.X_EXOG], Yf_ar_opt=ar_f, Yf_exog_opt=exog_f,
                                     fitted_models=model_fits,
                                     find_comparable_model=True)

    Yis_fit, Yos_pred, Yos_hpred, Yos_lpred = fit_and_predict(fit_dict=fit_dict,
                                                              predict_dict=predict_dict, model_type='OLS')

    gf.save_to_csv(data=Yis_fit, file_name="insample_fit", folder=fm.SAVE_LOC)


