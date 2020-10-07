import prediction.make_prediction as pred
import prediction.prediction_evaluation as eval_pred
import prediction.column_names as cn
import prediction.file_management as fm
import prediction.general_purpose_functions as gf
import pandas as pd
import numpy as np


prediction_dates = pd.DataFrame(pd.date_range('2020-08-01', periods=4, freq='W-MON').astype(str),
                                columns=[cn.FIRST_DOW])

negbin1_settings = {'prediction_window': 1, 'train_size': 70, 'differencing': False, 'ar_lags': 3,
                    'fit_model': 'Negative-Binomial'}

pred_nb1, is_abs_nb1, is_pct_nb1 = pred.batch_prediction(prediction_dates=prediction_dates,
                                                         model_settings=negbin1_settings)

active_products_act = gf.import_temp_file(file_name="actieve_halffabricaten_wk_2020926_1610.csv", data_loc=fm.SAVE_LOC)
inactive_products_act = gf.import_temp_file(file_name="inactieve_halffabricaten_wk_2020926_1610.csv", data_loc=fm.SAVE_LOC)
all_products_act = active_products_act.join(inactive_products_act, how='outer')

eval_pred.in_sample_plot(y_true=)

eval_pred.prediction_evaluation(product_name='total', Y_true=all_products_act, Y_pred=pred_nb2)

avg_pct_error =np.round((abs(results2['actual'] - results2['prediction']) / results2['actual']).mean(), 2)
graph_title = "Voorspelling 2 weken vooruit, totale productie, foutmarge: {}".format(avg_pct_error)
graph_fit = sns.relplot(data=results2, kind='line')
graph_fit.set(xlabel='week', ylabel='productie (CE)')
graph_fit.fig.suptitle(graph_title, fontsize=10)
plt.show()
