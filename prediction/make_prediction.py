import prediction.general_purpose_functions as gf
from prediction.dataprep import data_prep_wrapper
from prediction.create_features import prep_exogenous_features
from prediction.prediction_setup import prediction_setup_wrapper
from prediction.fit_model import fit_and_predict
import prediction.file_management as fm
import prediction.column_names as cn
import pandas as pd

# FIXED PARAMETERS
PREDICTION_DATE = '2020-06-22'
PREDICTION_WINDOW = 1
LAGS = 2
TRAIN_OBS = 70
ORDER_DATA = fm.RAW_DATA
WEATHER_DATA = fm.WEER_DATA
PRODUCT_DATA = fm.PRODUCT_STATUS
DIFFERENCE = True


def run_prediction(pred_date, prediction_window=PREDICTION_WINDOW, train_obs=TRAIN_OBS, difference=DIFFERENCE,
                   lags=LAGS, order_data=ORDER_DATA, weather_data=WEATHER_DATA, product_data=PRODUCT_DATA):

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


pred_date_2 = '2020-08-31'
pred_date_1 = '2020-08-24'
prediction_window = PREDICTION_WINDOW
train_obs = TRAIN_OBS
difference = DIFFERENCE
order_data = ORDER_DATA
weather_data = WEATHER_DATA
product_data = PRODUCT_DATA

_yhat_1, _yos_1, fit1, pred1 = run_prediction(pred_date=pred_date_1, prediction_window=1)
_yhat_2, _yos_2, fit2, pred2 = run_prediction(pred_date=pred_date_2, prediction_window=2)

all_predictions = pd.DataFrame([])
prediction_dates = pd.DataFrame(pd.date_range('2020-04-01', periods=22, freq='W-MON').astype(str), columns=[cn.FIRST_DOW])

for dt in prediction_dates[cn.FIRST_DOW]:
    _yhat, _yos = run_prediction(pred_date=dt)
    all_predictions = pd.concat([all_predictions, _yos], axis=0, join='outer')

