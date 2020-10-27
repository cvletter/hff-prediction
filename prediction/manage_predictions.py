import prediction.make_prediction as pred
import prediction.prediction_evaluation as eval_pred
import prediction.column_names as cn
import prediction.file_management as fm
import prediction.general_purpose_functions as gf
import pandas as pd

# Predictions
prediction_dates = pd.DataFrame(pd.date_range(end='2020-10-5', periods=1, freq='W-MON').astype(str),
                                columns=[cn.FIRST_DOW])

ols1_settings = {'prediction_window': 1, 'train_size': 60, 'differencing': False, 'ar_lags': 4,
                 'fit_model': 'OLS', 'feature_threshold': [0.2, 15]}

pred_ols1, is_abs_ols1, is_pct_ols1, mod_prod_ols1, non_mod_prod_ols1 = pred.batch_prediction(
    prediction_dates=prediction_dates, model_settings=ols1_settings)

ols2_settings = {'prediction_window': 2, 'train_size': 70, 'differencing': False, 'ar_lags': 2,
                 'fit_model': 'OLS', 'feature_threshold': [0.2, 10]}

pred_ols2, predh_ols2, predl_ols2, is_abs_ols2, is_pct_ols2, mod_prod_ols2, non_mod_prod_ols2 = pred.batch_prediction(
    prediction_dates=prediction_dates, model_settings=ols2_settings)

# Evaluation
active_products_act = gf.import_temp_file(file_name=fm.ORDER_DATA_ACT, data_loc=fm.SAVE_LOC)
inactive_products_act = gf.import_temp_file(file_name=fm.ORDER_DATA_INACT, data_loc=fm.SAVE_LOC)
all_products_act = active_products_act.join(inactive_products_act, how='outer')

eval_ols1 = eval_pred.prediction_performance_evaluation(Y_true=all_products_act, Y_pred=pred_ols1,
                                                       Y_pred_mod=mod_prod_ols1, Y_pred_non_mod=non_mod_prod_ols1)

eval_ols2 = eval_pred.prediction_performance_evaluation(Y_true=all_products_act, Y_pred=pred_ols2,
                                                       Y_pred_mod=mod_prod_ols2, Y_pred_non_mod=non_mod_prod_ols2)

eval_is_ols1_time, eval_is_ols1_prod = eval_pred.in_sample_evaluation(pct_fits=is_pct_ols1)
