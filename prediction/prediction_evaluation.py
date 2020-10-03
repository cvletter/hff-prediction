import pandas as pd
import prediction.column_names as cn
import prediction.file_management as fm
import prediction.general_purpose_functions as gf
import seaborn as sns

active_products_act = gf.import_temp_file(file_name="actieve_halffabricaten_wk_2020926_1610.csv", data_loc=fm.SAVE_LOC)
inactive_products_act = gf.import_temp_file(file_name="inactieve_halffabricaten_wk_2020926_1610.csv", data_loc=fm.SAVE_LOC)
all_products_act = active_products_act.join(inactive_products_act, how='outer')

pred_1step = gf.import_temp_file(file_name="1step_predictions_2020103_1636.csv", data_loc=fm.SAVE_LOC)
pred_2step = gf.import_temp_file(file_name="2step_predictions_2020103_1636.csv", data_loc=fm.SAVE_LOC)

Y_true = all_products_act
Y_pred = pred_1step
product_name = 'total'


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
    # y_eval['prediction_error'] = y_eval['prediction'] - y_eval['actual']

    sns.relplot(data=y_eval, kind="line")
    return y_eval


test = prediction_evaluation(product_name='Copparol vijgroomk. 160g HF', Y_true=all_products_act, Y_pred=pred_1step)

