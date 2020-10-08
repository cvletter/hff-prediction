import prediction.make_prediction as pred
import prediction.prediction_evaluation as eval_pred
import prediction.column_names as cn
import prediction.file_management as fm
import prediction.general_purpose_functions as gf
import pandas as pd
import numpy as np


prediction_dates = pd.DataFrame(pd.date_range('2020-06-01', periods=14, freq='W-MON').astype(str),
                                columns=[cn.FIRST_DOW])

negbin1_settings = {'prediction_window': 1, 'train_size': 70, 'differencing': False, 'ar_lags': 3,
                    'fit_model': 'Negative-Binomial'}

pred_nb1, is_abs_nb1, is_pct_nb1 = pred.batch_prediction(prediction_dates=prediction_dates,
                                                         model_settings=negbin1_settings)

negbin2_settings = {'prediction_window': 2, 'train_size': 70, 'differencing': False, 'ar_lags': 3,
                    'fit_model': 'Negative-Binomial'}

pred_nb2, is_abs_nb2, is_pct_nb2 = pred.batch_prediction(prediction_dates=prediction_dates,
                                                         model_settings=negbin2_settings)



active_products_act = gf.import_temp_file(file_name="actieve_halffabricaten_wk_2020926_1610.csv", data_loc=fm.SAVE_LOC)
inactive_products_act = gf.import_temp_file(file_name="inactieve_halffabricaten_wk_2020926_1610.csv", data_loc=fm.SAVE_LOC)
all_products_act = active_products_act.join(inactive_products_act, how='outer')


eval_pred.prediction_evaluation(product_name='Cervelaat Parmezaan 70g HF', Y_true=all_products_act, Y_pred=pred_nb2[pred_nb2.index > '2020-07-05'])

