import statsmodels.api as sm
import numpy as np
import pandas as pd
import prediction.general_purpose_functions as gf
import prediction.file_management as fm
import prediction.column_names as cn
# import seaborn as sns


def batch_fit_model(Y, Y_ar, X_exog, add_constant=True):
    Y_pred = pd.DataFrame(index=Y.index)

    fitted_models = {}

    for product in Y.columns:
        y_name = product
        y = Y[y_name]

        lag_index = [y_name in x for x in Y_ar.columns]
        x_ar = Y_ar.iloc[:, lag_index]

        if add_constant:
            x_ar.insert(0, 'constant', 1)

        X_tot = x_ar.join(X_exog, how='left')

        temp_mdl = sm.OLS(y, X_tot, missing='drop')
        temp_fit = temp_mdl.fit()

        Y_pred[y_name] = temp_fit.predict()
        fitted_models[product] = temp_fit

        # print("R2 and R2-adj for product {} are: {} and {}".format(product,
        #                                                           np.round(temp_fit.rsquared, 2),
        #                                                           np.round(temp_fit.rsquared_adj, 2)))
    return Y_pred, fitted_models


def batch_make_prediction(Yp_ar_m, Yp_ar_nm, Xp_exog, fitted_models,
                          add_constant=True, prep_input=True, agg_model_name=cn.MOD_PROD_SUM):

    def series_to_dataframe(pd_series):
        return pd.DataFrame(pd_series).transpose()

    if prep_input:
        Yp_ar_m = series_to_dataframe(Yp_ar_m)
        Yp_ar_nm = series_to_dataframe(Yp_ar_nm)
        Xp_exog = series_to_dataframe(Xp_exog)

    Y_pred = pd.DataFrame(index=Yp_ar_m.index)

    for product_m in Yp_ar_m.columns:
        y_name_m = product_m[:-6]

        lag_index = [y_name_m in x for x in Yp_ar_m.columns]
        Xp_ar_m = Yp_ar_m.iloc[:, lag_index]

        if add_constant:
            Xp_ar_m.insert(0, 'constant', 1)

        Xp_tot = Xp_ar_m.join(Xp_exog, how='left')

        Y_pred[y_name_m] = fitted_models[y_name_m].predict(Xp_tot)


    for product_nm in Yp_ar_nm.columns:
        y_name_nm = product_nm[:-6]

        lag_index = [y_name_nm in x for x in Yp_ar_nm.columns]
        Xp_ar_nm = Yp_ar_nm.iloc[:, lag_index]

        if add_constant:
            Xp_ar_nm.insert(0, 'constant', 1)

        Xp_tot = Xp_ar_nm.join(Xp_exog, how='left')

        Y_pred[y_name_nm] = fitted_models[agg_model_name].predict(Xp_tot)

    return Y_pred


def fit_and_predict(fit_dict, predict_dict):
    Yis_fit, model_fits = batch_fit_model(Y=fit_dict[cn.Y_TRUE], Y_ar=fit_dict[cn.Y_AR], X_exog=fit_dict[cn.X_EXOG])

    Yos_pred = batch_make_prediction(Yp_ar_m=predict_dict[cn.Y_AR_M], Yp_ar_nm=predict_dict[cn.Y_AR_NM],
                                     Xp_exog=predict_dict[cn.X_EXOG], fitted_models=model_fits,
                                     agg_model_name=cn.MOD_PROD_SUM)

    return Yis_fit, Yos_pred



if __name__ == '__main__':
    fit_data = gf.read_pkl(file_name=fm.FIT_DATA, data_loc=fm.SAVE_LOC)
    predict_data = gf.read_pkl(file_name=fm.PREDICT_DATA, data_loc=fm.SAVE_LOC)

    Yf_true = fit_data[cn.Y_TRUE]
    Yf_ar = fit_data[cn.Y_AR]
    Xf_exog = fit_data[cn.X_EXOG]

    Yis_fit, model_fits = batch_fit_model(Y=Yf_true, Y_ar=Yf_ar, X_exog=Xf_exog)

    Yp_ar = predict_data[cn.Y_AR_M]
    Yp_ar_nm = predict_data[cn.Y_AR_NM]
    Xp_exog = predict_data[cn.X_EXOG]

    Yos_pred = batch_make_prediction(Yp_ar_m=Yp_ar, Yp_ar_nm=Yp_ar_nm,
                                     Xp_exog=Xp_exog, fitted_models=model_fits,
                                     )

    gf.save_to_csv(data=Yis_fit, file_name="insample_fit", folder=fm.SAVE_LOC)

    Yt_total = pd.DataFrame(Yf_true[cn.MOD_PROD_SUM])
    Yt_total['fit_model'] = Yis_fit[cn.MOD_PROD_SUM]
    Yt_total['fit_model_sum'] = Yis_fit.sum(axis=1)

   # sns.relplot(data=Yt_total, kind="line")