from data_preparation import data_prep_wrapper
from create_features import prep_all_features
from prediction_setup import prediction_setup_wrapper
from fit_model import fit_and_predict
# from prediction_evaluation import in_sample_plot
import file_management as fm
import column_names as cn
import general_purpose_functions as gf
import pandas as pd
import time
import multiprocessing
import numpy as np
from functools import partial
import copy

# Parameters
print("kippie")
date_to_predict = cn.PREDICTION_DATE
prediction_window = cn.PREDICTION_WINDOW
train_obs = cn.TRAIN_OBS
difference = False
lags = cn.N_LAGS
order_data = fm.RAW_DATA
weather_data = fm.WEER_DATA
product_data = fm.PRODUCT_STATUS
model_type = 'OLS'
feature_threshold = None
samples = 3

if feature_threshold is None:
    feature_threshold = [0.2, 15]


def convert_series_to_dataframe(input_series, date_val, index_name=cn.FIRST_DOW):
    input_df = pd.DataFrame(input_series).T
    input_df[index_name] = date_val
    return input_df.set_index(index_name, drop=True, inplace=False)


def in_sample_error(all_fits, all_true_values):
    fit_error = abs(all_fits.subtract(all_true_values[all_fits.columns], axis='index'))
    avg_fit_error = fit_error.mean(axis=0)
    corr_true_values = all_true_values.replace(0, 1)
    pct_fit_error = fit_error / corr_true_values
    avg_pct_fit_error = pct_fit_error.mean(axis=0)

    avg_fit_error_df = convert_series_to_dataframe(input_series=avg_fit_error, date_val=date_to_predict)
    avg_pct_fit_error_df = convert_series_to_dataframe(input_series=avg_pct_fit_error, date_val=date_to_predict)

    return avg_fit_error_df, avg_pct_fit_error_df

# Import and prepare data
active_products, inactive_products, weather_data_processed, order_data_su = data_prep_wrapper(
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

"""
in_sample_fit, prediction_os = fit_and_predict(
    fit_dict=fit_data, predict_dict=predict_data,
    model_type=model_type,
    feature_threshold=[feature_threshold[0],
                       feature_threshold[1]])
"""

fits = copy.deepcopy(fit_data)
preds = copy.deepcopy(predict_data)


def bootstrapper(n_samples):
    __, pred_os = fit_and_predict(fit_dict=fits, predict_dict=preds, bootstrap=True)
    print("Finished sample: {}".format(n_samples))
    return pred_os

if __name__ == '__main__':

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(4)
    samps = np.arange(0, samples)
    results = pool.map(bootstrapper, samps)
    pool.close()
    pool.join()

    all_preds = pd.concat(results)
    print(all_preds)