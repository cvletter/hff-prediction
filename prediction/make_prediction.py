import prediction.general_purpose_functions as gf
from prediction.dataprep import data_prep_wrapper
from prediction.create_features import prep_exogenous_features
from prediction.prediction_setup import prediction_setup_wrapper
from prediction.fit_model import fit_and_predict
import prediction.file_management as fm
import prediction.column_names as cn
import pandas as pd
import datetime
import numpy as np


def run_prediction(pred_date=cn.PREDICTION_DATE, prediction_window=cn.PREDICTION_WINDOW, train_obs=cn.TRAIN_OBS,
                   difference=True, lags=cn.N_LAGS, order_data=fm.RAW_DATA, weather_data=fm.WEER_DATA,
                   product_data=fm.PRODUCT_STATUS, model_type='OLS'):

    # Import and prepare data
    active_products, inactive_products, weather_data_processed = data_prep_wrapper(
        prediction_date=pred_date,
        prediction_window=prediction_window,
        order_data_loc=order_data,
        weer_data_loc=weather_data,
        product_data_loc=product_data,
        agg_weekly=True, exclude_su=True,
        save_to_csv=False)

    exogenous_features = prep_exogenous_features(weather_data_processed=weather_data_processed, save_to_csv=False,
                                                 prediction_window=prediction_window)

    fit_data, predict_data = prediction_setup_wrapper(
        prediction_date=pred_date,
        prediction_window=prediction_window,
        train_obs=train_obs,
        nlags=lags,
        difference=difference,
        act_products=active_products,
        exog_features=exogenous_features,
        save_to_pkl=False)

    in_sample_fit, out_of_sample_prediction = fit_and_predict(fit_dict=fit_data, predict_dict=predict_data,
                                                              model_type=model_type)

    return in_sample_fit, out_of_sample_prediction, fit_data, predict_data


def two_step_prediction(final_prediction_date):

    if type(final_prediction_date) == str:
        final_prediction_date = datetime.datetime.strptime(final_prediction_date, "%Y-%m-%d")

    first_prediction_date = final_prediction_date - datetime.timedelta(days=7)
    __, pred1_diff, __, pred1 = run_prediction(pred_date=first_prediction_date, prediction_window=1)
    pred1_raw = pd.DataFrame(pd.concat([pred1[cn.Y_M_UNDIF], pred1[cn.Y_NM_UNDIF]]))
    pred1_diff = pred1_diff.T.set_index(pred1_diff.columns)

    pred1_combined = pred1_diff.join(pred1_raw, how='left')
    pred1_combined['pred1_final'] = (pred1_combined.sum(axis=1)).astype(int)
    # pred1_combined['pred1_final'] = [0 if x < 0 else x for x in pred1_combined['pred1_final']]

    __, pred2_diff, __, __ = run_prediction(pred_date=first_prediction_date, prediction_window=2)
    pred2_diff = pred2_diff.T.set_index(pred2_diff.columns)

    pred2_combined = pred2_diff.join(pred1_combined['pred1_final'], how='left')
    pred2_combined['pred2_final'] = (pred2_combined.sum(axis=1)).astype(int)
    # pred2_combined['pred2_final'] = [0 if x < 0 else x for x in pred2_combined['pred2_final']]

    return pred1_combined['pred1_final'].rename(first_prediction_date), \
           pred2_combined['pred2_final'].rename(final_prediction_date)


def batch_2step_prediction(prediction_dates):
    all_predictions_1stp = pd.DataFrame([])
    all_predictions_2stp = pd.DataFrame([])

    for dt in prediction_dates:
        _1step, _2step = two_step_prediction(final_prediction_date=dt)
        all_predictions_1stp = pd.concat([all_predictions_1stp, _1step], axis=1)
        all_predictions_2stp = pd.concat([all_predictions_2stp, _2step], axis=1)

    return all_predictions_1stp.T, all_predictions_2stp.T


if __name__ == '__main__':
    # Parameter settings
    pred_date_2 = '2020-08-31'
    pred_date_1 = '2020-08-24'
    # prediction_window = 1
    train_obs = 70
    difference = False
    order_data = fm.RAW_DATA
    weather_data = fm.WEER_DATA
    product_data = fm.PRODUCT_STATUS
    model = 'Poisson'

    in_sample_fit, out_of_sample_prediction, fit_data, predict_data = run_prediction(
        pred_date=cn.PREDICTION_DATE, prediction_window=cn.PREDICTION_WINDOW, train_obs=cn.TRAIN_OBS,
                   difference=False, lags=cn.N_LAGS, order_data=fm.RAW_DATA, weather_data=fm.WEER_DATA,
                   product_data=fm.PRODUCT_STATUS, model_type='Poisson')

    is_fit_tot = pd.DataFrame(in_sample_fit[cn.MOD_PROD_SUM])
    is_fit_tot.columns = ['fit']
    is_true_tot = pd.DataFrame(fit_data['y_true'][cn.MOD_PROD_SUM])
    is_true_tot.columns = ['true']
    test = is_fit_tot.join(is_true_tot, how='left')

    import seaborn as sns

    prediction_dates = pd.DataFrame(pd.date_range('2020-07-01', periods=9, freq='W-MON').astype(str), columns=[cn.FIRST_DOW])
    all_1step, all_2step = batch_2step_prediction(prediction_dates=prediction_dates[cn.FIRST_DOW])

    all_1step.index.rename(cn.FIRST_DOW, inplace=True)
    all_2step.index.rename(cn.FIRST_DOW, inplace=True)

    gf.save_to_csv(all_1step, file_name="1step_predictions", folder=fm.SAVE_LOC)
    gf.save_to_csv(all_2step, file_name="2step_predictions", folder=fm.SAVE_LOC)



