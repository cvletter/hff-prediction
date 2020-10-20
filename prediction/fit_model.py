import statsmodels.api as sm
import pandas as pd
import prediction.general_purpose_functions as gf
import prediction.file_management as fm
import prediction.column_names as cn


def batch_fit_model(Y, Y_ar, X_exog, Y_cross_ar, add_constant=True, model='Negative-Binomial'):
    Y_pred = pd.DataFrame(index=Y.index)

    fitted_models = {}
    # y_name = Y.columns[0]
    for product in Y.columns:
        y_name = product
        y = Y[y_name]
        Y_arx = Y_ar[Y_cross_ar[y_name]]

        lag_index = [y_name in x for x in Y_ar.columns]
        x_ar = Y_ar.iloc[:, lag_index]

        if add_constant:
            x_ar.insert(0, 'constant', 1)

        X_tot = x_ar.join(Y_arx, how='left').join(X_exog, how='left')

        if model == 'OLS':
            temp_mdl = sm.OLS(y, X_tot, missing='drop')

        elif model == 'Poisson':
            temp_mdl = sm.GLM(y, X_tot,
                              family=sm.families.Poisson(),
                              missing='drop')

        elif model == 'Negative-Binomial':
            aux_reg_feat = pd.DataFrame(index=y.index)

            temp_mdl_poisson = sm.GLM(y, X_tot,
                              family=sm.families.Poisson(),
                              missing='drop')

            temp_poisson_fit = temp_mdl_poisson.fit()

            aux_reg_feat['lambda'] = temp_poisson_fit.mu
            aux_reg_feat['dep_var'] = ((y - temp_poisson_fit.mu)**2 - y) / temp_poisson_fit.mu
            aux_reg = sm.OLS(aux_reg_feat['dep_var'], aux_reg_feat['lambda']).fit()

            alpha_fit = aux_reg.params[0]

            temp_mdl = sm.GLM(y, X_tot,
                              family=sm.families.NegativeBinomial(alpha=alpha_fit),
                              missing='drop')

        temp_fit = temp_mdl.fit()

        Y_pred[y_name] = temp_fit.predict()
        fitted_models[y_name] = temp_fit

    return Y_pred, fitted_models


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




    Yos_pred = batch_make_prediction(Yp_ar_m=predict_data[cn.Y_AR_M], Yp_ar_nm=predict_data[cn.Y_AR_NM],
                                     Xp_exog=predict_data[cn.X_EXOG], Y_cross_ar=fit_data["correlations"],
                                     fitted_models=model_fits,
                                     find_comparable_model=True, prediction_window=1)

    gf.save_to_csv(data=Yis_fit, file_name="insample_fit", folder=fm.SAVE_LOC)


