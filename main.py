import prediction.make_prediction as pred
import prediction.prediction_evaluation as eval_pred
import prediction.column_names as cn
import prediction.file_management as fm
import prediction.general_purpose_functions as gf
import pandas as pd
import multiprocessing
import time

# Prediction
model_settings = {'prediction_window': 2, 'train_size': 60, 'differencing': False, 'ar_lags': 4,
                 'fit_model': 'OLS', 'feature_threshold': [0.2, 15]}


def batch_prediction_bs(prediction_date):

    print("Starting for date: {}".format(prediction_date))

    p_window = model_settings['prediction_window']
    train_size = model_settings['train_size']
    differencing = model_settings['differencing']
    ar_lags = model_settings['ar_lags']
    fit_model = model_settings['fit_model']
    feature_threshold = model_settings['feature_threshold']

    _predict = pred.run_prediction_bootstrap(
        date_to_predict=prediction_date, prediction_window=p_window, train_obs=train_size,
        difference=differencing, lags=ar_lags, order_data=fm.RAW_DATA, weather_data=fm.WEER_DATA,
        product_data=fm.PRODUCT_STATUS, model_type=fit_model, feature_threshold=[feature_threshold[0],
                                                                                 feature_threshold[1]])

    return _predict


if __name__ == "__main__":
    start = time.time()

    prediction_dates = pd.DataFrame(pd.date_range(end='2020-10-5', periods=2, freq='W-MON').astype(str),
                                    columns=[cn.FIRST_DOW])

    pred_dates = list(prediction_dates[cn.FIRST_DOW])

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    results = pool.map(batch_prediction_bs, pred_dates)
    pool.close()
    pool.join()

    all_preds = pd.concat(results)

    # print(results)
    print(all_preds)

    elapsed = round((time.time() - start), 2)
    print("It takes {} seconds to run a prediction.".format(elapsed))



