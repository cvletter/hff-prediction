import pandas as pd
import prediction.column_names as cn
import prediction.file_management as fm
import prediction.general_purpose_functions as gf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")


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


def in_sample_plot(y_true, y_fit, title, name=cn.MOD_PROD_SUM):
    y_comp = pd.concat([y_fit[name], y_true['y_true'][name]], axis=1)
    y_comp.columns = ['fitted', 'actual']
    y_comp['error'] = y_comp['fitted'] - y_comp['actual']
    y_comp['nullijn'] = 0

    avg_pct_error = np.round((abs(y_comp['error']) / y_comp['actual']).mean(), 3)

    graph_title = "{}, foutmarge: {}".format(title, avg_pct_error)
    graph_fit = sns.relplot(data=y_comp, kind='line')
    graph_fit.set(xlabel='week', ylabel='productie (CE)')
    graph_fit.fig.suptitle(graph_title, fontsize=10)
    plt.show()

    return y_comp


if __name__ == '__main__':
    active_products_act = gf.import_temp_file(file_name="actieve_halffabricaten_wk_2020926_1610.csv",
                                              data_loc=fm.SAVE_LOC)
    inactive_products_act = gf.import_temp_file(file_name="inactieve_halffabricaten_wk_2020926_1610.csv",
                                                data_loc=fm.SAVE_LOC)
    all_products_act = active_products_act.join(inactive_products_act, how='outer')


