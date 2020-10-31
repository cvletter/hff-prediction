from prediction import general_purpose_functions as gf
from prediction import file_management as fm
from prediction import column_names as cn
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    results = gf.read_pkl(file_name='test_result_bs_20201031_1548.p',
                          data_loc=fm.SAVE_LOC)

    active_products_act = gf.import_temp_file(file_name=fm.ORDER_DATA_ACT,
                                              data_loc=fm.SAVE_LOC)
    inactive_products_act = gf.import_temp_file(file_name=fm.ORDER_DATA_INACT,
                                                data_loc=fm.SAVE_LOC)
    all_products_act = active_products_act.join(inactive_products_act, how='outer')

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

    all_mods, all_nmods = get_mod_products(result_dict=all_dicts)
    all_preds = get_predictions(result_dict=all_dicts)
    all_true_values = all_products_act


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
                                             columns=['prediction', 'true_value', 'pct_error', 'lower_bound', 'upper_bound'])

        predictions_nmod_total = pd.DataFrame(index=all_prediction_dates,
                                             columns=['prediction', 'true_value', 'pct_error', 'lower_bound', 'upper_bound'])

        predictions_tot = pd.DataFrame(index=all_prediction_dates,
                                             columns=['prediction', 'true_value', 'pct_error', 'lower_bound', 'upper_bound'])


        for d in all_prediction_dates:
            _d = d.strftime("%Y-%m-%d")
            _mod = all_mod_prod[_d].drop(cn.MOD_PROD_SUM)
            _nmod = all_non_mod_prod[_d]

            _bootstrap_mod = bootstraps_total.loc[_d, _mod].sum(axis=1)
            _pred_int_mod_low = _bootstrap_mod.quantile(0.20)
            _pred_int_mod_high = _bootstrap_mod.quantile(0.80)

            _bootstrap_nmod = bootstraps_total.loc[_d, _mod].sum(axis=1)
            _pred_int_nmod_low = _bootstrap_mod.quantile(0.20)
            _pred_int_nmod_high = _bootstrap_mod.quantile(0.80)

            _bootstrap_tot = _bootstrap_mod + _bootstrap_nmod
            _pred_int_tot_low = _bootstrap_mod.quantile(0.20)
            _pred_int_tot_high = _bootstrap_mod.quantile(0.80)

            # Create
            _pred_mod = round(predictions_total.loc[_d, _mod], 0)
            _pred_mod_sum = _pred_mod.sum()

            _pred_nmod = round(predictions_total.loc[_d, _nmod], 0)
            _pred_nmod_sum = _pred_nmod.sum()

            _pred_tot = _pred_mod_sum + _pred_nmod_sum

            _truev_mod = all_true_values.loc[d, _mod]
            _truev_mod_sum = _truev_mod.sum()

            _truev_nmod = all_true_values.loc[d, _nmod]
            _truev_nmod_sum = _truev_nmod.sum()

            _truev_tot = _truev_mod_sum + _truev_nmod_sum

            _pred_err_mod = abs(_pred_mod - _truev_mod)
            _pred_err_nmod = abs(_pred_nmod - _truev_nmod)

            _pred_perr_mod = round(abs(_pred_mod - _truev_mod) / _truev_mod, 3)
            _pred_perr_nmod = round(abs(_pred_nmod - _truev_nmod) / _truev_mod, 3)

            _pred_err_mod_sum = round(abs(_pred_mod_sum - _truev_mod_sum) / _truev_mod_sum, 3)
            _pred_err_nmod_sum = round(abs(_pred_nmod_sum - _truev_nmod_sum) / _truev_nmod_sum, 3)
            _pred_err_tot = round(abs(_pred_tot - _truev_tot) / _truev_tot, 3)

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
            predictions_mod_total.loc[d, 'true_value'] = _truev_mod_sum
            predictions_mod_total.loc[d, 'pct_error'] = _pred_err_mod_sum
            predictions_mod_total.loc[d, 'lower_bound'] = _pred_int_mod_low
            predictions_mod_total.loc[d, 'upper_bound'] = _pred_int_mod_high

            predictions_nmod_total.loc[d, 'prediction'] = _pred_nmod_sum
            predictions_nmod_total.loc[d, 'true_value'] = _truev_nmod_sum
            predictions_nmod_total.loc[d, 'pct_error'] = _pred_err_nmod_sum
            predictions_nmod_total.loc[d, 'lower_bound'] = _pred_int_nmod_low
            predictions_nmod_total.loc[d, 'upper_bound'] = _pred_int_nmod_high

            predictions_tot.loc[d, 'prediction'] = _pred_tot
            predictions_tot.loc[d, 'true_value'] = _truev_tot
            predictions_tot.loc[d, 'pct_error'] = _pred_err_tot
            predictions_tot.loc[d, 'lower_bound'] = _pred_int_tot_low
            predictions_tot.loc[d, 'upper_bound'] = _pred_int_tot_high

        return predictions_tot, predictions_mod_total, predictions_nmod_total


    all_mods, all_nmods = get_mod_products(result_dict=all_dicts)
    all_preds = get_predictions(result_dict=all_dicts)
    all_true_values = all_products_act

    pred_t, pred_m, pred_nm = out_of_sample_performance(all_predictions=all_preds,
                                                        all_true_values=all_true_values,
                                                        all_mod_prod=all_mods,
                                                        all_non_mod_prod=all_nmods)

    def plot_results(results):
        plot_data = results.drop('pct_error', axis=1, inplace=False)
        pct_error = round((abs(plot_data['prediction'] - plot_data['true_value']) / plot_data['true_value']).mean(), 2)
        plot_data.fillna(0, inplace=True)
        graph_fit = sns.relplot(data=plot_data, kind='line')
        graph_fit.set(xlabel='week', ylabel='productie (CE)')
        title = "Predictions all production, average error: {}".format(pct_error)
        graph_fit.fig.suptitle(title, fontsize=10)
        plt.show()

