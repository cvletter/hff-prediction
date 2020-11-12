from prediction import general_purpose_functions as gf
from prediction import file_management as fm
from prediction import column_names as cn
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    results = gf.read_pkl(file_name='test_result_bs_2p_2l_70obs_20201112_1723.p',
                          data_loc=fm.SAVE_LOC)

    active_products_act = gf.import_temp_file(file_name=fm.ORDER_DATA_ACT,
                                              data_loc=fm.SAVE_LOC)
    inactive_products_act = gf.import_temp_file(file_name=fm.ORDER_DATA_INACT,
                                                data_loc=fm.SAVE_LOC)
    all_products_act = active_products_act.join(inactive_products_act, how='outer')

    a = "no"

    if a == "yes":
        from collections import Counter

        all_params = []
        for k in pars.keys():
            all_params += list(pars[k])

        Counter(all_params)


    def output_to_dict(result_file):
        for i in range(0, len(result_file)):
            if i == 0:
                all_dicts = result_file[i]
            else:
                all_dicts.update(result_file[i])

        return all_dicts

    all_dicts = output_to_dict(result_file=results)

    def get_predictions(result_dict):
        all_predictions = pd.DataFrame([])
        for k in result_dict.keys():
            _preds = result_dict[k][cn.PREDICTION_OS]
            all_predictions = pd.concat([all_predictions, _preds])

        return all_predictions

    def get_mod_products(result_dict):
        all_mod_products= {}
        all_nmod_products = {}

        for k in result_dict.keys():
            all_mod_products[k] = result_dict[k][cn.MOD_PROD]
            all_nmod_products[k] = result_dict[k][cn.NON_MOD_PROD]

        return all_mod_products, all_nmod_products


    def get_predictions(result_dict):
        all_predictions = pd.DataFrame([])
        for k in result_dict.keys():
            _preds = result_dict[k][cn.PREDICTION_OS]
            all_predictions = pd.concat([all_predictions, _preds])

        return all_predictions

    all_mod_prod, all_non_mod_prod = get_mod_products(result_dict=all_dicts)
    all_predictions = get_predictions(result_dict=all_dicts)
    all_true_values = all_products_act
    zero_pred=True

    def out_of_sample_performance(all_predictions, all_true_values, all_mod_prod, all_non_mod_prod, zero_pred=True):

        if zero_pred:
            all_predictions[all_predictions < 0] = 0

        predictions_total = all_predictions[all_predictions[cn.BOOTSTRAP_ITER] == 0]
        bootstraps_total = all_predictions[all_predictions[cn.BOOTSTRAP_ITER] != 0]

        all_prediction_dates = predictions_total.index.unique().sort_values(ascending=False)

        predictions_mod = pd.DataFrame(index=all_prediction_dates)
        predictions_nmod = pd.DataFrame(index=all_prediction_dates)
        true_values_mod = pd.DataFrame(index=all_prediction_dates)
        true_values_nmod = pd.DataFrame(index=all_prediction_dates)

        prediction_error_mod = pd.DataFrame(index=all_prediction_dates)
        prediction_error_nmod = pd.DataFrame(index=all_prediction_dates)
        prediction_perror_mod = pd.DataFrame(index=all_prediction_dates)
        prediction_perror_nmod = pd.DataFrame(index=all_prediction_dates)

        predictions_mod_total = pd.DataFrame(index=all_prediction_dates,
                                             columns=['prediction', 'q_prediction', 'true_value', 'pct_error', 'pct_qerror', 'lower_bound', 'upper_bound'])

        predictions_nmod_total = pd.DataFrame(index=all_prediction_dates,
                                             columns=['prediction', 'q_prediction', 'true_value', 'pct_error', 'pct_qerror', 'lower_bound', 'upper_bound'])

        predictions_tot = pd.DataFrame(index=all_prediction_dates,
                                             columns=['prediction', 'q_prediction', 'true_value', 'pct_error', 'pct_qerror', 'lower_bound', 'upper_bound'])

        for d in all_prediction_dates:
            _d = d.strftime("%Y-%m-%d")
            _raw_mod = all_mod_prod[_d].drop(cn.MOD_PROD_SUM)
            _raw_nmod = all_non_mod_prod[_d]

            _truev_mod = all_true_values.loc[d, _raw_mod]
            _truev_mod_sum = _truev_mod.sum()

            _truev_nmod = all_true_values.loc[d, _raw_nmod]
            _truev_nmod_sum = _truev_nmod.sum()

            _mod = _truev_mod.index[_truev_mod.notna()]
            _nmod = _truev_nmod.index[_truev_nmod.notna()]

            delm = len(_raw_mod) - len(_mod)
            delnm = len(_raw_nmod) - len(_nmod)

            print("Deleted {} mod prod, {} nonmod prod".format(delm, delnm))

            _truev_tot = _truev_mod_sum + _truev_nmod_sum

            _pred_int_mod_low = round(all_predictions.loc[_d, _mod].quantile(0.025, axis=0).sum(), 0)
            _pred_int_mod_high = round(all_predictions.loc[_d, _mod].quantile(0.975, axis=0).sum(), 0)

            _pred_int_nmod_low = round(all_predictions.loc[_d, _nmod].quantile(0.025, axis=0).sum(), 0)
            _pred_int_nmod_high = round(all_predictions.loc[_d, _nmod].quantile(0.975, axis=0).sum(), 0)

            _pred_int_tot_low = _pred_int_mod_low + _pred_int_nmod_low
            _pred_int_tot_high = _pred_int_mod_high + _pred_int_nmod_high

            _pred_mod_med = round(all_predictions.loc[_d, _mod].quantile(0.5, axis=0).sum(), 0)
            _pred_nmod_med = round(all_predictions.loc[_d, _nmod].quantile(0.5, axis=0).sum(), 0)
            _pred_tot_med = _pred_mod_med + _pred_nmod_med

            # Create
            _pred_mod = round(predictions_total.loc[_d, _mod], 0)
            _pred_mod_sum = _pred_mod.sum()

            _pred_nmod = round(predictions_total.loc[_d, _nmod], 0)
            _pred_nmod_sum = _pred_nmod.sum()

            _pred_tot = _pred_mod_sum + _pred_nmod_sum

            _pred_err_mod = abs(_pred_mod - _truev_mod)
            _pred_err_nmod = abs(_pred_nmod - _truev_nmod)

            _pred_perr_mod = round(abs(_pred_mod - _truev_mod) / _truev_mod, 3)
            _pred_perr_nmod = round(abs(_pred_nmod - _truev_nmod) / _truev_mod, 3)

            _pred_err_mod_sum = round(abs(_pred_mod_sum - _truev_mod_sum) / _truev_mod_sum, 3)
            _pred_err_nmod_sum = round(abs(_pred_nmod_sum - _truev_nmod_sum) / _truev_nmod_sum, 3)
            _pred_err_tot = round(abs(_pred_tot - _truev_tot) / _truev_tot, 3)

            _predq_err_mod = round(abs(_pred_mod_med - _truev_mod_sum) / _truev_mod_sum, 3)
            _predq_err_nmod = round(abs(_pred_nmod_med - _truev_nmod_sum) / _truev_nmod_sum, 3)
            _predq_err_tot = round(abs(_pred_tot_med - _truev_mod_sum) / _truev_tot, 3)

            # Collect
            predictions_mod.loc[d, _mod] = _pred_mod
            predictions_nmod.loc[d, _nmod] = _pred_nmod

            true_values_mod.loc[d, _mod] = _truev_mod
            true_values_nmod.loc[d, _nmod] = _truev_nmod

            prediction_error_mod.loc[d, _mod] = _pred_err_mod
            prediction_error_nmod.loc[d, _nmod] = _pred_err_nmod

            prediction_perror_mod.loc[d, _mod] = _pred_perr_mod
            prediction_perror_nmod.loc[d, _nmod] = _pred_perr_nmod

            predictions_mod_total.loc[d, 'prediction'] = _pred_mod_sum
            predictions_mod_total.loc[d, 'q_prediction'] = _pred_mod_med
            predictions_mod_total.loc[d, 'true_value'] = _truev_mod_sum
            predictions_mod_total.loc[d, 'pct_error'] = _pred_err_mod_sum
            predictions_mod_total.loc[d, 'pct_qerror'] = _predq_err_mod
            predictions_mod_total.loc[d, 'lower_bound'] = _pred_int_mod_low
            predictions_mod_total.loc[d, 'upper_bound'] = _pred_int_mod_high

            predictions_nmod_total.loc[d, 'prediction'] = _pred_nmod_sum
            predictions_nmod_total.loc[d, 'q_prediction'] = _pred_nmod_med
            predictions_nmod_total.loc[d, 'true_value'] = _truev_nmod_sum
            predictions_nmod_total.loc[d, 'pct_error'] = _pred_err_nmod_sum
            predictions_nmod_total.loc[d, 'pct_qerror'] = _predq_err_nmod
            predictions_nmod_total.loc[d, 'lower_bound'] = _pred_int_nmod_low
            predictions_nmod_total.loc[d, 'upper_bound'] = _pred_int_nmod_high

            predictions_tot.loc[d, 'prediction'] = _pred_tot
            predictions_tot.loc[d, 'q_prediction'] = _pred_tot_med
            predictions_tot.loc[d, 'true_value'] = _truev_tot
            predictions_tot.loc[d, 'pct_error'] = _pred_err_tot
            predictions_tot.loc[d, 'pct_qerror'] = _predq_err_tot
            predictions_tot.loc[d, 'lower_bound'] = _pred_int_tot_low
            predictions_tot.loc[d, 'upper_bound'] = _pred_int_tot_high

        return predictions_tot, predictions_mod_total, predictions_nmod_total, predictions_mod, true_values_mod


    all_mods, all_nmods = get_mod_products(result_dict=all_dicts)
    all_preds = get_predictions(result_dict=all_dicts)
    all_true_values = all_products_act

    pred_t, pred_m, pred_nm, r_predmod, r_truemod = out_of_sample_performance(all_predictions=all_preds,
                                                        all_true_values=all_true_values,
                                                        all_mod_prod=all_mods,
                                                        all_non_mod_prod=all_nmods)

    print("Average pct error, total: {}, modelable: {}, non-modelable: {}".format(
        round(pred_t['pct_error'].mean(), 2),
        round(pred_m['pct_error'].mean(), 2),
        round(pred_nm['pct_error'].mean(), 2)))

    pred_t_f = pred_t[(pred_t.index > '2020-06-01') | (pred_t.index < '2020-05-11')]
    pred_m_f = pred_m[(pred_m.index > '2020-06-01') | (pred_m.index < '2020-05-11')]
    pred_nm_f = pred_nm[(pred_nm.index > '2020-06-01') | (pred_nm.index < '2020-05-11')]

    print("Average pct error without peak-period, total: {}, modelable: {}, non-modelable: {}".format(
        round(pred_t_f['pct_error'].mean(), 2),
        round(pred_m_f['pct_error'].mean(), 2),
        round(pred_nm_f['pct_error'].mean(),2)))


    def plot_results(results):
        plot_data = results[['prediction', 'true_value']]
        pct_error = round((abs(plot_data['prediction'] - plot_data['true_value']) / plot_data['true_value']).mean(), 2)
        plot_data.fillna(0, inplace=True)
        graph_fit = sns.relplot(data=plot_data, kind='line')
        graph_fit.set(xlabel='week', ylabel='productie (CE)')
        title = "Predictions all production, average error: {}".format(pct_error)
        graph_fit.fig.suptitle(title, fontsize=10)
        plt.show()

    plot_results(results=pred_m)

    """
    gf.save_to_csv(data=r_predmod, file_name='raw_pred_2p_2p_mod', folder=fm.SAVE_LOC)
    gf.save_to_csv(data=r_truemod, file_name='raw_truev_2p_2p_mod', folder=fm.SAVE_LOC)
    gf.save_to_csv(data=pred_m, file_name='total_mod_predictions_2p_2l', folder=fm.SAVE_LOC)
    gf.save_to_csv(data=pred_t, file_name='total_predictions_2p_2l', folder=fm.SAVE_LOC)
    gf.save_to_csv(data=pred_m, file_name='total_mod_predictions_2p_2l', folder=fm.SAVE_LOC)
    gf.save_to_csv(data=pred_nm, file_name='total_nonmod_predictions_2p_2l', folder=fm.SAVE_LOC)
    """
