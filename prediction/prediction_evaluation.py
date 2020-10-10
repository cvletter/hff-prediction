import pandas as pd
import prediction.column_names as cn
import prediction.file_management as fm
import prediction.general_purpose_functions as gf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sns.set_theme(style="darkgrid")


def prediction_performance_evaluation(Y_true, Y_pred, Y_pred_mod, Y_pred_non_mod):

    Y_mod_pred = pd.DataFrame(index=Y_pred.index)
    Y_nmod_pred = pd.DataFrame(index=Y_pred.index)

    Y_mod_true = pd.DataFrame(index=Y_pred.index)
    Y_nmod_true = pd.DataFrame(index=Y_pred.index)

    Y_mod_err = pd.DataFrame(index=Y_pred.index)
    Y_nmod_err = pd.DataFrame(index=Y_pred.index)

    Y_mod_pred_tot = pd.DataFrame(columns=['ym_pred_isum', 'ym_pred_psum', 'ym_true_sum', 'ym_error_psum',
                                           'ym_error_isum', 'ym_perror_isum', 'ym_perror_psum'], index=Y_pred.index)

    Y_nmod_pred_tot = pd.DataFrame(columns=['ynm_pred_isum', 'ynm_true_sum', 'ynm_error_isum',
                                            'ynm_perror_isum'], index=Y_pred.index)

    # dt = Y_pred.index[0]

    for dt in Y_pred.index:
        _dt = datetime.strftime(dt, '%Y-%m-%d')
        _ymod_col = Y_pred_mod[_dt]
        _ynmod_col = Y_pred_non_mod[_dt]

        _yp_mod = Y_pred.loc[dt, _ymod_col]
        _yp_mod_psum = int(_yp_mod[cn.MOD_PROD_SUM])
        _yp_mod.drop(cn.MOD_PROD_SUM, axis=0, inplace=True)

        _yp_nmod = Y_pred.loc[dt, _ynmod_col]

        _yt_mod = Y_true.loc[dt, _yp_mod.index]
        _yt_nmod = Y_true.loc[dt, _yp_nmod.index]

        # Fill totals
        Y_mod_pred_tot.loc[dt, 'ym_pred_isum'] = int(_yp_mod.sum())
        Y_mod_pred_tot.loc[dt, 'ym_pred_psum'] = _yp_mod_psum
        Y_mod_pred_tot.loc[dt, 'ym_true_sum'] = int(_yt_mod.sum())
        Y_mod_pred_tot.loc[dt, 'ym_error_isum'] = int(_yp_mod.sum()) - int(_yt_mod.sum())
        Y_mod_pred_tot.loc[dt, 'ym_perror_isum'] = int(int(_yp_mod.sum()) - int(_yt_mod.sum())) / int(_yt_mod.sum())
        Y_mod_pred_tot.loc[dt, 'ym_error_psum'] = _yp_mod_psum - int(_yt_mod.sum())
        Y_mod_pred_tot.loc[dt, 'ym_perror_psum'] = int(_yp_mod_psum - int(_yt_mod.sum())) / int(_yt_mod.sum())

        Y_nmod_pred_tot.loc[dt, 'ynm_pred_isum'] = int(_yp_nmod.sum())
        Y_nmod_pred_tot.loc[dt, 'ynm_true_sum'] = int(_yt_nmod.sum())
        Y_nmod_pred_tot.loc[dt, 'ynm_error_isum'] = int(int(_yp_nmod.sum()) - int(_yt_nmod.sum())) / int(_yt_nmod.sum())

        # Fill individual series
        Y_mod_pred.loc[dt, _yp_mod.index] = _yp_mod
        Y_nmod_pred.loc[dt, _yp_nmod.index] = _yp_nmod

        Y_mod_true.loc[dt, _yt_mod.index] = _yt_mod
        Y_nmod_true.loc[dt, _yt_nmod.index] = _yt_nmod

        Y_mod_err = Y_mod_pred.subtract(Y_mod_true, axis=1)
        Y_nmod_err = Y_nmod_pred.subtract(Y_nmod_true, axis=1)

    all_evaluations = {'p_modelable_products_total': Y_mod_pred_tot,
                       'p_nonmodelable_products_total': Y_nmod_pred_tot,
                       'p_modelable_products': Y_mod_pred,
                       'p_nonmodelable_products': Y_nmod_pred,
                       't_modelable_products': Y_mod_true,
                       't_nonmodelable_products': Y_nmod_true,
                       'e_modelable_products': Y_mod_err,
                       'e_nonmodelable_products': Y_nmod_err
                       }

    mod_isum_err = abs(all_evaluations['p_modelable_products_total']['ym_perror_isum']).mean()
    nmod_isum_err = abs(all_evaluations['p_nonmodelable_products_total']['ynm_error_isum']).mean()
    mod_psum_err = abs(all_evaluations['p_modelable_products_total']['ym_perror_psum']).mean()

    m_graph_title = "Foutmarges modelleerbaar: p-som: {}; i-som: {}".format(mod_psum_err, mod_isum_err)
    _gmod = all_evaluations['p_modelable_products_total'][['ym_pred_isum', 'ym_pred_psum', 'ym_true_sum', 'ym_error_isum', 'ym_error_psum']]
    _gmod.fillna(0, inplace=True)
    graph_fit = sns.relplot(data=_gmod, kind='line')
    graph_fit.set(xlabel='week', ylabel='productie (CE)')
    graph_fit.fig.suptitle(m_graph_title, fontsize=10)
    plt.show()

    nm_graph_title = "Foutmarges niet-modelleerbaar: i-som: {}".format(nmod_isum_err)
    _gnmod = all_evaluations['p_nonmodelable_products_total'][['ynm_pred_isum', 'ynm_true_sum', 'ynm_error_isum']]
    _gnmod.fillna(0, inplace=True)
    graph_fit = sns.relplot(data=_gnmod, kind='line')
    graph_fit.set(xlabel='week', ylabel='productie (CE)')
    graph_fit.fig.suptitle(nm_graph_title, fontsize=10)
    plt.show()


    return all_evaluations

# Oud
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
    eval_prod['count of fits'] = pct_fits.count(axis=0, numeric_only=True)

    print("Overall median: {}; overall 5p: {}; overall 95p: {}".format(round(eval_prod['median'].median(), 3),
                                                                       round(eval_prod['5p'].median(), 3),
                                                                       round(eval_prod['95p'].median(), 3)))

    return eval_time, eval_prod


if __name__ == '__main__':
    active_products_act = gf.import_temp_file(file_name="actieve_halffabricaten_wk_2020926_1610.csv",
                                              data_loc=fm.SAVE_LOC)
    inactive_products_act = gf.import_temp_file(file_name="inactieve_halffabricaten_wk_2020926_1610.csv",
                                                data_loc=fm.SAVE_LOC)
    all_products_act = active_products_act.join(inactive_products_act, how='outer')


