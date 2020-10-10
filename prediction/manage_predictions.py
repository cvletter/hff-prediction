import prediction.make_prediction as pred
import prediction.prediction_evaluation as eval_pred
import prediction.column_names as cn
import prediction.file_management as fm
import prediction.general_purpose_functions as gf
import pandas as pd


prediction_dates = pd.DataFrame(pd.date_range(end='2020-10-05', periods=4, freq='W-MON').astype(str),
                                columns=[cn.FIRST_DOW])

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

eval_nb1 = eval_pred.prediction_performance_evaluation(Y_true=all_products_act, Y_pred=pred_nb1,
                                                        Y_pred_mod=mod_prod_nb1, Y_pred_non_mod=non_mod_prod_nb1)

eval_nb2 = eval_pred.prediction_performance_evaluation(Y_true=all_products_act, Y_pred=pred_nb2,
                                                        Y_pred_mod=mod_prod_nb2, Y_pred_non_mod=non_mod_prod_nb2)

pct_fits = is_pct_nb2

def in_sample_evaluation(pct_fits):
    eval_time = pd.DataFrame(index=pct_fits.index)
    eval_time['average'] = pct_fits.mean(axis=1)
    eval_time['5p'] = pct_fits.quantile(q=0.05, axis=1)
    eval_time['25p'] = pct_fits.quantile(q=0.25, axis=1)
    eval_time['median'] = pct_fits.quantile(q=0.5, axis=1)
    eval_time['75p'] = pct_fits.quantile(q=0.75, axis=1)
    eval_time['95p'] = pct_fits.quantile(q=0.95, axis=1)
    eval_time['count of products'] = pct_fits.count(axis=1, numeric_only=True)

    print("Overall median: {}; overall 5p: {}; overall 95p: {}".format(round(eval_time['median'].median(), 3),
                                                                       round(eval_time['5p'].median(), 3),
                                                                       round(eval_time['95p'].median(), 3)))

    eval_prod = pd.DataFrame(index=pct_fits.columns)
    eval_prod['average'] = pct_fits.mean(axis=0)
    eval_prod['5p'] = pct_fits.quantile(q=0.05, axis=0)
    eval_prod['25p'] = pct_fits.quantile(q=0.25, axis=0)
    eval_prod['median'] = pct_fits.quantile(q=0.5, axis=0)
    eval_prod['75p'] = pct_fits.quantile(q=0.75, axis=0)
    eval_prod['95p'] = pct_fits.quantile(q=0.95, axis=0)
    eval_time['count of fits'] = pct_fits.count(axis=0, numeric_only=True)

    print("Overall median: {}; overall 5p: {}; overall 95p: {}".format(round(eval_prod['median'].median(), 3),
                                                                       round(eval_prod['5p'].median(), 3),
                                                                       round(eval_prod['95p'].median(), 3)))


sns.relplot(data=is_pct_nb2[cn.MOD_PROD_SUM], kind="line")