import prediction.make_prediction as pred
import prediction.prediction_evaluation as eval_pred
import prediction.column_names as cn
import prediction.file_management as fm
import prediction.general_purpose_functions as gf
import pandas as pd
import multiprocessing
import time

# Prediction
model_settings = {'prediction_window': 1,
                  'train_size': 60,
                  'differencing': False,
                  'ar_lags': 4,
                  'fit_model': 'OLS',
                  'feature_threshold': [0.2, 15],
                  'bootstraps': 40}


def batch_prediction_bs(prediction_date):

    print("Starting for date: {}".format(prediction_date))

    p_window = model_settings['prediction_window']
    train_size = model_settings['train_size']
    differencing = model_settings['differencing']
    ar_lags = model_settings['ar_lags']
    fit_model = model_settings['fit_model']
    feature_threshold = model_settings['feature_threshold']
    bootstrap_iterations = model_settings['bootstraps']

    _predict = pred.run_prediction_bootstrap(
        date_to_predict=prediction_date, prediction_window=p_window, train_obs=train_size,
        difference=differencing, lags=ar_lags, order_data=fm.RAW_DATA, weather_data=fm.WEER_DATA,
        product_data=fm.PRODUCT_STATUS, model_type=fit_model, feature_threshold=[feature_threshold[0],
                                                                                 feature_threshold[1]],
        bootstrap_iter=bootstrap_iterations)

    return _predict


if __name__ == "__main__":
    start = time.time()

    prediction_dates = pd.DataFrame(pd.date_range(end='2020-10-5', periods=40, freq='W-MON').astype(str),
                                    columns=[cn.FIRST_DOW])

    pred_dates = list(prediction_dates[cn.FIRST_DOW])

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    results = pool.map(batch_prediction_bs, pred_dates)
    pool.close()
    pool.join()

    # all_preds = pd.concat(results)
    # print(results)
    # print(all_preds)

    gf.save_to_pkl(data=results, file_name='test_result_bs_1p', folder=fm.SAVE_LOC)

    elapsed = round((time.time() - start), 2)
    print("It takes {} seconds to run a prediction.".format(elapsed))

    """

    test_rsults = gf.read_pkl(file_name='test_result_bs_20201031_1247.p', data_loc=fm.SAVE_LOC)

    test = pd.read_json(test_rsults)
    test = str(test_rsults)
    test1 = test_rsults[0]
    test2 = test_rsults[1]
    
    test1.update(test2) #join dictstest
    """




