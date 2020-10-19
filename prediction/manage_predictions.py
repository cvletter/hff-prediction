import prediction.make_prediction as pred
import prediction.prediction_evaluation as eval_pred
import prediction.column_names as cn
import prediction.file_management as fm
import prediction.general_purpose_functions as gf
import pandas as pd


prediction_dates = pd.DataFrame(pd.date_range(end='2020-10-5', periods=40, freq='W-MON').astype(str),
                                columns=[cn.FIRST_DOW])

ols1_settings = {'prediction_window': 1, 'train_size': 60, 'differencing': False, 'ar_lags': 4,
                    'fit_model': 'OLS'}

pred_ols1, is_abs_ols1, is_pct_ols1, mod_prod_ols1, non_mod_prod_ols1 = pred.batch_prediction(
    prediction_dates=prediction_dates, model_settings=ols1_settings)

ols2_settings = {'prediction_window': 2, 'train_size': 60, 'differencing': False, 'ar_lags': 4,
                    'fit_model': 'OLS'}

pred_ols2, is_abs_ols2, is_pct_ols2, mod_prod_ols2, non_mod_prod_ols2 = pred.batch_prediction(
    prediction_dates=prediction_dates, model_settings=ols2_settings)


negbin1_settings = {'prediction_window': 1, 'train_size': 60, 'differencing': False, 'ar_lags': 4,
                    'fit_model': 'Negative-Binomial'}

pred_nb1, is_abs_nb1, is_pct_nb1, mod_prod_nb1, non_mod_prod_nb1 = pred.batch_prediction(
    prediction_dates=prediction_dates, model_settings=negbin1_settings)

negbin2_settings = {'prediction_window': 2, 'train_size': 60, 'differencing': False, 'ar_lags': 4,
                    'fit_model': 'Negative-Binomial'}

pred_nb2, is_abs_nb2, is_pct_nb2, mod_prod_nb2, non_mod_prod_nb2 = pred.batch_prediction(
    prediction_dates=prediction_dates,
    model_settings=negbin2_settings)

active_products_act = gf.import_temp_file(file_name=fm.ORDER_DATA_ACT, data_loc=fm.SAVE_LOC)
inactive_products_act = gf.import_temp_file(file_name=fm.ORDER_DATA_INACT, data_loc=fm.SAVE_LOC)
all_products_act = active_products_act.join(inactive_products_act, how='outer')

eval_ols1 = eval_pred.prediction_performance_evaluation(Y_true=all_products_act, Y_pred=pred_ols1,
                                                       Y_pred_mod=mod_prod_ols1, Y_pred_non_mod=non_mod_prod_ols1)

eval_ols2 = eval_pred.prediction_performance_evaluation(Y_true=all_products_act, Y_pred=pred_ols2,
                                                       Y_pred_mod=mod_prod_ols2, Y_pred_non_mod=non_mod_prod_ols2)

eval_nb1 = eval_pred.prediction_performance_evaluation(Y_true=all_products_act, Y_pred=pred_nb1,
                                                       Y_pred_mod=mod_prod_nb1, Y_pred_non_mod=non_mod_prod_nb1)

eval_nb2 = eval_pred.prediction_performance_evaluation(Y_true=all_products_act, Y_pred=pred_nb2,
                                                       Y_pred_mod=mod_prod_nb2, Y_pred_non_mod=non_mod_prod_nb2)


_gmod = eval_nb1['p_modelable_products_total'][['ym_pred_isum', 'ym_pred_psum', 'ym_true_sum', 'ym_error_isum', 'ym_error_psum']]
_gnmod = eval_nb1['p_nonmodelable_products_total'][['ynm_pred_isum', 'ynm_true_sum', 'ynm_error_isum']]

gf.save_to_csv(data=_gmod, file_name="oos_pred_nb1_mod", folder=fm.SAVE_LOC)
gf.save_to_csv(data=_gnmod, file_name="oos_pred_nb1_nmod", folder=fm.SAVE_LOC)

gf.save_to_pkl(data=eval_nb2, file_name="oos_pred_nb_2p_40pred", folder=fm.SAVE_LOC)

eval_is_nb1_time, eval_is_nb1_prod = eval_pred.in_sample_evaluation(pct_fits=is_pct_nb1)
eval_is_ols1_time, eval_is_ols1_prod = eval_pred.in_sample_evaluation(pct_fits=is_pct_ols1)
