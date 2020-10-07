from prediction.data_preparation import data_prep_wrapper
from prediction.create_features import prep_exogenous_features
from prediction.prediction_setup import prediction_setup_wrapper
from prediction.fit_model import fit_and_predict
from prediction.prediction_evaluation import in_sample_plot
import prediction.file_management as fm
import prediction.column_names as cn
import prediction.general_purpose_functions as gf
import pandas as pd


def run_prediction(pred_date=cn.PREDICTION_DATE, prediction_window=cn.PREDICTION_WINDOW, train_obs=cn.TRAIN_OBS,
                   difference=True, lags=cn.N_LAGS, order_data=fm.RAW_DATA, weather_data=fm.WEER_DATA,
                   product_data=fm.PRODUCT_STATUS, model_type='OLS'):

    def convert_series_to_dataframe(input_series, date_val, index_name=cn.FIRST_DOW):
        input_df = pd.DataFrame(input_series).T
        input_df[index_name] = date_val
        return input_df.set_index(index_name, drop=True, inplace=False)

    def in_sample_error(all_fits, all_true_values):
        fit_error = abs(all_fits.subtract(all_true_values[all_fits.columns], axis='index'))
        avg_fit_error = fit_error.mean(axis=0)
        avg_true_values = all_true_values[all_fits.columns].mean(axis=0)
        avg_pct_fit_error = avg_fit_error / avg_true_values

        avg_fit_error_df = convert_series_to_dataframe(input_series=avg_fit_error, date_val=pred_date)
        avg_pct_fit_error_df = convert_series_to_dataframe(input_series=avg_pct_fit_error, date_val=pred_date)

        return avg_fit_error_df, avg_pct_fit_error_df

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
                                                              model_type=model_type, prediction_window=prediction_window)

    fit_data['avg_fit_error'], fit_data['avg_pct_fit_error'] = in_sample_error(all_fits=in_sample_fit,
                                                                               all_true_values=fit_data['y_true'])

    return in_sample_fit, out_of_sample_prediction, fit_data, predict_data


def batch_prediction(prediction_dates, model_settings):
    p_window = model_settings['prediction_window']
    train_size = model_settings['train_size']
    differencing = model_settings['differencing']
    ar_lags = model_settings['ar_lags']
    fit_model = model_settings['fit_model']

    all_is_abs_errors = pd.DataFrame([])
    all_is_pct_errors = pd.DataFrame([])
    all_os_predictions = pd.DataFrame([])

    for p_date in prediction_dates[cn.FIRST_DOW]:
        _fit, _predict, _fitdata, _predictdata = run_prediction(
            pred_date=p_date, prediction_window=p_window, train_obs=train_size,
            difference=differencing, lags=ar_lags, order_data=fm.RAW_DATA, weather_data=fm.WEER_DATA,
            product_data=fm.PRODUCT_STATUS, model_type=fit_model)

        all_is_abs_errors = pd.concat([all_is_abs_errors, _fitdata['avg_fit_error']], axis=0)
        all_is_pct_errors = pd.concat([all_is_pct_errors, _fitdata['avg_pct_fit_error']], axis=0)
        all_os_predictions = pd.concat([all_os_predictions, _predict], axis=0)

    return all_os_predictions, all_is_abs_errors, all_is_pct_errors


if __name__ == '__main__':

    # In sample testing of 2020-31-8
    is_fit1, os_pr1, fit_data1, predict_data1 = run_prediction(pred_date='2020-08-31',
                                                               prediction_window=1,
                                                               train_obs=cn.TRAIN_OBS,
                                                               difference=False, lags=3,
                                                               order_data=fm.RAW_DATA,
                                                               weather_data=fm.WEER_DATA,
                                                               product_data=fm.PRODUCT_STATUS,
                                                               model_type='Negative-Binomial')

    is_fit2, os_pr2, fit_data2, predict_data2 = run_prediction(pred_date='2020-08-31',
                                                               prediction_window=2,
                                                               train_obs=cn.TRAIN_OBS,
                                                               difference=False, lags=3,
                                                               order_data=fm.RAW_DATA,
                                                               weather_data=fm.WEER_DATA,
                                                               product_data=fm.PRODUCT_STATUS,
                                                               model_type='Negative-Binomial')

    active_products_act = gf.import_temp_file(file_name=fm.ORDER_DATA_ACT, data_loc=fm.SAVE_LOC)
    inactive_products_act = gf.import_temp_file(file_name=fm.ORDER_DATA_INACT, data_loc=fm.SAVE_LOC)
    all_products_act = active_products_act.join(inactive_products_act, how='outer')

    product_name = 'Copparolletjes vijgenrk 81g HF'
    is_performance = in_sample_plot(y_true=fit_data1, y_fit=is_fit1,
                                    title="test")