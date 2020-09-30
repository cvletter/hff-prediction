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
PREDICTION_WINDOW = 2
TRAIN_OBS = 70
ORDER_DATA = fm.RAW_DATA
WEATHER_DATA = fm.WEER_DATA
PRODUCT_DATA = fm.PRODUCT_STATUS
DIFFERENCE = True


def run_prediction(pred_date, prediction_window=PREDICTION_WINDOW, train_obs=TRAIN_OBS, difference=DIFFERENCE,
                   order_data=ORDER_DATA, weather_data=WEATHER_DATA, product_data=PRODUCT_DATA):

    # Import and prepare data
    active_products, inactive_products, weather_data_processed = data_prep_wrapper(
        prediction_date=pred_date,
        prediction_window=prediction_window,
        order_data_loc=order_data,
        weer_data_loc=weather_data,
        product_data_loc=product_data,
        agg_weekly=True, exclude_su=True,
        save_to_csv=False)

    exogenous_features = prep_exogenous_features(weather_data_processed=weather_data_processed, save_to_csv=False)

    fit_data, predict_data = prediction_setup_wrapper(
        prediction_date=pred_date,
        prediction_window=prediction_window,
        train_obs=train_obs,
        nlags=prediction_window,
        difference=difference,
        act_products=active_products,
        exog_features=exogenous_features,
        save_to_pkl=False)

    in_sample_fit, out_of_sample_prediction = fit_and_predict(fit_dict=fit_data, predict_dict=predict_data)

    return in_sample_fit, out_of_sample_prediction


pred_date = '2020-06-22'
prediction_window = PREDICTION_WINDOW
train_obs = TRAIN_OBS
difference = DIFFERENCE
order_data = ORDER_DATA
weather_data = WEATHER_DATA
product_data = PRODUCT_DATA

# Y_is_1, Y_os_1 = run_prediction(pred_date='2020-06-22')
# Y_is_2, Y_os_2 = run_prediction(pred_date='2020-06-29')

# all_predictions = pd.DataFrame(index=prediction_dates[cn.FIRST_DOW])

all_predictions = pd.DataFrame([])
prediction_dates = pd.DataFrame(pd.date_range('2020-04-01', periods=22, freq='W-MON').astype(str), columns=[cn.FIRST_DOW])

for dt in prediction_dates[cn.FIRST_DOW]:
    _yhat, _yos = run_prediction(pred_date=dt)
    all_predictions = pd.concat([all_predictions, _yos], axis=0, join='outer')

