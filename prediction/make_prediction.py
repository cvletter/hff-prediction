from prediction.data_preparation import data_prep_wrapper
from prediction.create_features import prep_exogenous_features
from prediction.prediction_setup import prediction_setup_wrapper
from prediction.fit_model import fit_and_predict
import prediction.file_management as fm
import prediction.column_names as cn
import prediction.general_purpose_functions as gf
import pandas as pd
import seaborn as sns


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

    prediction_dates = pd.DataFrame(pd.date_range('2020-08-01', periods=3, freq='W-MON').astype(str), columns=[cn.FIRST_DOW])

    model_settings = {}
    model_settings['prediction_window'] = 1
    model_settings['train_size'] = 70
    model_settings['differencing'] = False
    model_settings['ar_lags'] = 2
    model_settings['fit_model'] = 'Negative-Binomial'
    os_pr, is_abs, is_pct = batch_prediction(prediction_dates=prediction_dates, model_settings=model_settings)

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

    active_products_act = gf.import_temp_file(file_name="actieve_halffabricaten_wk_2020926_1610.csv",
                                              data_loc=fm.SAVE_LOC)
    inactive_products_act = gf.import_temp_file(file_name="inactieve_halffabricaten_wk_2020926_1610.csv",
                                                data_loc=fm.SAVE_LOC)
    all_products_act = active_products_act.join(inactive_products_act, how='outer')


    def prediction_evaluation(product_name, Y_true, Y_pred):

        if product_name == 'total':
            Y_pred_t = Y_pred.copy(deep=True)
            Y_pred_t.drop(cn.MOD_PROD_SUM, axis=1, inplace=True)
            y_true = Y_true[Y_pred_t.columns].sum(axis=1)
            y_eval = Y_pred_t.sum(axis=1)
        else:
            y_eval = Y_pred[product_name]
            y_true = Y_true[product_name]

        y_true = pd.DataFrame(y_true)
        y_true.columns = ['actual']
        y_eval = pd.DataFrame(y_eval)
        y_eval.columns = ['prediction']

        y_eval = y_eval.join(y_true, how='left')
        y_eval['prediction_error'] = y_eval['prediction'] - y_eval['actual']

        sns.relplot(data=y_eval[['prediction', 'actual']], kind="line")
        return y_eval

    import matplotlib.pyplot as plt
    import numpy as np

def plot_compare(y_true, y_fit, title, name=cn.MOD_PROD_SUM):
    y_comp = pd.concat([y_fit[name], y_true['y_true'][name]], axis=1)
    y_comp.columns = ['fitted', 'actual']
    y_comp['error'] = y_comp['fitted'] - y_comp['actual']
    y_comp['nullijn'] = 0
    #y_comp['pct_error'] = abs(y_comp['error']) / y_comp['actual']

    #avg_error = abs(y_comp['error']).mean()
    avg_pct_error = np.round((abs(y_comp['error']) / y_comp['actual']).mean(),3)

    graph_title = "{}, foutmarge: {}".format(title, avg_pct_error)

    sns.set_theme(style="darkgrid")
    graph_fit = sns.relplot(data=y_comp, kind='line')
    graph_fit.set(xlabel='week', ylabel='productie (CE)')
    graph_fit.fig.suptitle(graph_title, fontsize=10)
    plt.show()

product_name = 'Copparolletjes vijgenrk 81g HF'

plot_compare(name=product_name,
            y_true=fit_data2,
             y_fit=is_fit2,
             title="Geschat model (2 weken, 31/8), Copparolletjes vijgenrk 81g HF")