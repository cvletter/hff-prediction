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
                   difference=True, lags=cn.N_LAGS,
                   order_data=fm.RAW_DATA, weather_data=fm.WEER_DATA, product_data=fm.PRODUCT_STATUS):

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

    in_sample_fit, out_of_sample_prediction = fit_and_predict(fit_dict=fit_data, predict_dict=predict_data)

    return in_sample_fit, out_of_sample_prediction, fit_data, predict_data

# Parameter settings
pred_date_2 = '2020-08-31'
pred_date_1 = '2020-08-24'
prediction_window = 1
train_obs = 70
difference = True
order_data = fm.RAW_DATA
weather_data = fm.WEER_DATA
product_data = fm.PRODUCT_STATUS

_yhat_1, _yos_1, fit1, pred1 = run_prediction(pred_date=pred_date_1, prediction_window=1)
_yhat_2, _yos_2, fit2, pred2 = run_prediction(pred_date=pred_date_2, prediction_window=2)

all_predictions = pd.DataFrame([])
prediction_dates = pd.DataFrame(pd.date_range('2020-04-01', periods=22, freq='W-MON').astype(str), columns=[cn.FIRST_DOW])

for dt in prediction_dates[cn.FIRST_DOW]:
    _yhat, _yos = run_prediction(pred_date=dt)
    all_predictions = pd.concat([all_predictions, _yos], axis=0, join='outer')


def two_step_prediction(final_prediction_date):

    if type(final_prediction_date) == str:
        final_prediction_date = datetime.datetime.strptime(final_prediction_date, "%Y-%m-%d")

    first_prediction_date = final_prediction_date - datetime.timedelta(days=7)
    __, pred1_diff, __, pred1 = run_prediction(pred_date=first_prediction_date, prediction_window=1)
    pred1_raw = pd.DataFrame(pd.concat([pred1[cn.Y_M_UNDIF], pred1[cn.Y_NM_UNDIF]]))
    pred1_diff = pred1_diff.T.set_index(pred1_diff.columns)
    pred1_combined = pred1_diff.join(pred1_raw, how='left')
    pred1_combined['pred1_final'] = (pred1_combined.sum(axis=1)).astype(int)
    pred1_combined['pred1_final'] = [0 if x < 0 else x for x in pred1_combined['pred1_final']]

    __, pred2_diff, __, __ = run_prediction(pred_date=first_prediction_date, prediction_window=1)
    pred2_diff = pred2_diff.T.set_index(pred2_diff.columns)
    pred2_combined = pred2_diff.join(pred1_combined['pred1_final'], how='left')
    pred2_combined['pred2_final'] = (pred2_combined.sum(axis=1)).astype(int)
    pred2_combined['pred2_final'] = [0 if x < 0 else x for x in pred2_combined['pred2_final']]

    return pred1_combined['pred1_final'].rename(first_prediction_date), \
           pred2_combined['pred2_final'].rename(final_prediction_date)



test1, test2 = two_step_prediction(final_prediction_date=prediction_dates[cn.FIRST_DOW][20])


