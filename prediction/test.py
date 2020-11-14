import prediction.column_names as cn
import prediction.file_management as fm
from prediction.create_features import prep_all_features
from prediction.data_preparation import data_prep_wrapper
from prediction.prediction_setup import prediction_setup_wrapper
from prediction.fit_model import fit_and_predict
import prediction.general_purpose_functions as gf
import pandas as pd
import datetime
if __name__ == '__main__':
    # Parameters

    date_to_predict = "2020-10-05"
    prediction_window = 2
    train_obs = cn.TRAIN_OBS
    difference = False
    lags = cn.N_LAGS
    order_data = fm.RAW_DATA
    weather_data = fm.WEER_DATA
    product_data = fm.PRODUCT_STATUS
    model_type = 'OLS'
    feature_threshold = [0.2, 25]

    # Import and prepare data
    active_products, inactive_products, weather_data_processed, order_data_su, campaigns = data_prep_wrapper(
        prediction_date=date_to_predict,
        prediction_window=prediction_window,
        reload_data=False,
        order_data_loc=order_data,
        weer_data_loc=weather_data,
        product_data_loc=product_data,
        agg_weekly=True, exclude_su=True,
        save_to_csv=False)

    exogenous_features = prep_all_features(weather_data_processed=weather_data_processed,
                                           order_data_su=order_data_su,
                                           campaign_data_su=campaigns,
                                           prediction_date=date_to_predict,
                                           train_obs=train_obs,
                                           save_to_csv=False)

    fit_data, predict_data = prediction_setup_wrapper(
        prediction_date=date_to_predict,
        prediction_window=prediction_window,
        train_obs=train_obs,
        nlags=lags,
        difference=difference,
        act_products=active_products,
        exog_features=exogenous_features,
        save_to_pkl=False)


    def prep_level_shifts():
        def str2date(date_str):
            return datetime.datetime.strptime(date_str, "%Y-%m-%d")

        level_shifts = pd.DataFrame(pd.date_range('2018-01-01', periods=1200, freq='D'), columns=['day'])

        # this becomes the new constant
        # level_shifts['period_1'] = [1 if x <= str2date('2019-03-11') else 0 for x in level_shifts['day']]
        level_shifts['a_trans_period_1'] = [1 if (str2date('2019-03-18') <= x <= str2date('2019-04-08')) else 0 for x in
                                            level_shifts['day']]
        level_shifts['b_period_2'] = [1 if str2date('2019-04-15') <= x <= str2date('2020-04-27') else 0 for x in
                                      level_shifts['day']]
        level_shifts['c_trans_period_2'] = [1 if (str2date('2020-05-04') <= x <= str2date('2020-05-25')) else 0 for x in
                                            level_shifts['day']]
        level_shifts['d_trans_period_2b'] = [1 if (str2date('2020-06-01') <= x <= str2date('2020-06-29')) else 0 for x
                                             in
                                             level_shifts['day']]
        level_shifts['e_period_3'] = [1 if x >= str2date('2020-06-01') else 0 for x in level_shifts['day']]

        gf.add_week_year(data=level_shifts, date_name='day')
        gf.add_first_day_week(add_to=level_shifts, week_col_name=cn.WEEK_NUMBER, set_as_index=True)
        level_shifts.drop('day', axis=1, inplace=True)

        return level_shifts.groupby(cn.FIRST_DOW, as_index=True).max()


    from statsmodels.tsa.seasonal import seasonal_decompose
    import statsmodels.api as sm
    import numpy as np
    #import matplotlib.pyplot as plt

    Y_true = fit_data[cn.Y_TRUE]
    y_sum = Y_true[cn.MOD_PROD_SUM]



    features = pd.DataFrame(index=y_sum.index)
    features['constant'] = 1
    features = features.join(levels, how='left')

    levels = prep_level_shifts()

    features['trend'] = sorted(np.arange(1, len(y_sum)+1), reverse=True)
    features['winter'] = [1 if x.month <= 3 else 0 for x in features.index]
    features['lente'] = [1 if 4 <= x.month <= 6 else 0 for x in features.index]
    features['zomer'] = [1 if 7 <= x.month <= 9 else 0 for x in features.index]

    features2 = pd.DataFrame(index=y_sum.index)
    features2['constant'] = 1
    features2['trend'] = sorted(np.arange(1, len(y_sum)+1), reverse=True)

    features['lag1'] = y_sum.shift(-1)
    features['lag2'] = y_sum.shift(-2)

    y_sum_fit = y_sum[:-2]

    features2 = features[:-2]


    for i in range(1, 12):
        name = "month_{}".format(i)
        features2[name] = [1 if x.month == i else 0 for x in features2.index]

    def test_mdl(y_sum, features):
        mdl = sm.OLS(y_sum, features, missing='drop')
        fits = mdl.fit()
        print(fits.summary())

        fitted_vals = fits.predict()
        comp_series = pd.DataFrame(index=y_sum.index)
        comp_series['true'] = y_sum
        comp_series['fit'] = fitted_vals

        # plt.plot(comp_series)
        return comp_series

    constant = test_mdl(y_sum=y_sum, features=features)
    trend_ = test_mdl(y_sum=y_sum, features=features)
    levels = test_mdl(y_sum=y_sum, features=features)
    seasons = test_mdl(y_sum=y_sum, features=features)
    autoreg = test_mdl(y_sum=y_sum_fit, features=features2)


    total = pd.concat((constant['true'], constant['fit'], trend_['fit'], levels['fit'], seasons['fit'], autoreg['fit']), axis=1)
    total.columns = ['true', 'constant', 'trend', 'levels', 'seasons', 'autoreg']
    total.to_csv("decomposition.csv", sep=";", decimal=",")


    result_4 = seasonal_decompose(y_sum, model='additive', period=4)
    result_4.plot()

    trend = np.

    def seasonal_dummies()


    in_sample_fits, all_predictions, all_pars = fit_and_predict(
        fit_dict=fit_data, predict_dict=predict_data,
        model_type=model_type,
        feature_threshold=[feature_threshold[0],
                           feature_threshold[1]])


    bootstrap_iter = 5

    all_predictions

    for i in range(1, bootstrap_iter):
        print("Running iteration {} of {}".format(i, bootstrap_iter))
        fits, temp_os, pars = fit_and_predict(fit_dict=fit_data, predict_dict=predict_data, bootstrap=True,
                                              model_type=model_type, feature_threshold=[feature_threshold[0],
                                                                                        feature_threshold[1]])
        temp_os[cn.BOOTSTRAP_ITER] = i

        all_predictions = pd.concat([all_predictions, temp_os])

        na_values = all_predictions.isna().sum().sum()
        print("In {} there are {} na_values".format(date_to_predict, na_values))

    all_output[date_to_predict][cn.PREDICTION_OS] = all_predictions


