import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
import hff_predictor.generic.dates as gf
import hff_predictor.generic.files

import numpy as np
import pandas as pd
from hff_predictor.generic.files import read_pkl, import_temp_file, save_to_csv


def output_to_dict(data_loc):
    result_file = read_pkl(data_loc=data_loc
                               )
    for i in range(0, len(result_file)):
        if i == 0:
            all_dicts = result_file[i]
        else:
            all_dicts.update(result_file[i])

    return all_dicts


def get_predictions(result_dict):
    all_predictions = pd.DataFrame([])
    for k in result_dict.keys():
        _preds = result_dict[k][cn.PREDICTION_OS]
        all_predictions = pd.concat([all_predictions, _preds])

    return all_predictions


def get_mod_products(result_dict):
    all_mod_products = {}
    all_nmod_products = {}

    for k in result_dict.keys():
        all_mod_products[k] = result_dict[k][cn.MOD_PROD]
        all_nmod_products[k] = result_dict[k][cn.NON_MOD_PROD]

    return all_mod_products, all_nmod_products


def get_benchmark(result_dict):
    all_benchmarks = pd.DataFrame([])
    for k in result_dict.keys():
        _preds = result_dict[k][cn.MA_BENCHMARK]
        all_benchmarks = pd.concat([all_benchmarks, _preds])

    return all_benchmarks


def performance_quality(predictions, benchmark, true_values,
                        modelable_prod, non_modelable_prod, grouping=None):

    # Remove negative predictions
    predictions[predictions <= 0] = 0.0

    # Keep only actual prediction
    predictions = predictions[predictions[cn.BOOTSTRAP_ITER] == 0]
    predictions.drop(cn.BOOTSTRAP_ITER, axis=1, inplace=True)

    # Collect all prediction dates in test set
    all_prediction_dates = predictions.index.unique().sort_values(
        ascending=False
    )

    # Prepare all data subsets
    predictions_mod = pd.DataFrame(index=all_prediction_dates)
    predictions_nmod = pd.DataFrame(index=all_prediction_dates)
    predictions_mod_rol = pd.DataFrame(index=all_prediction_dates)
    predictions_nmod_rol = pd.DataFrame(index=all_prediction_dates)

    benchmark_mod = pd.DataFrame(index=all_prediction_dates)
    benchmark_nmod = pd.DataFrame(index=all_prediction_dates)
    benchmark_mod_rol = pd.DataFrame(index=all_prediction_dates)
    benchmark_nmod_rol = pd.DataFrame(index=all_prediction_dates)

    true_values_mod = pd.DataFrame(index=all_prediction_dates)
    true_values_nmod = pd.DataFrame(index=all_prediction_dates)
    true_values_mod_rol = pd.DataFrame(index=all_prediction_dates)
    true_values_nmod_rol = pd.DataFrame(index=all_prediction_dates)

    predictions_mod_total = pd.DataFrame(
        index=all_prediction_dates,
        columns=[
            "prediction",
            "prediction_tot",
            "true_value",
            "pred_error_avg",
            "pred_error_sum",
            "pred_error_tot",
            "benchmark",
            "bmrk_error_avg",
            "bmrk_error_sum",
        ],
    )

    predictions_nmod_total = pd.DataFrame(
        index=all_prediction_dates,
        columns=[
            "prediction",
            "true_value",
            "pred_error_avg",
            "pred_error_sum",
            "benchmark",
            "bmrk_error_avg",
            "bmrk_error_sum",
        ],
    )

    predictions_tot = pd.DataFrame(
        index=all_prediction_dates,
        columns=[
            "prediction",
            "true_value",
            "pred_error_avg",
            "pred_error_sum",
            "benchmark",
            "bmrk_error_avg",
            "bmrk_error_sum",
        ],
    )

    # Filling in the predictions
    for d in all_prediction_dates:
        _d = d.strftime("%Y-%m-%d")
        _raw_mod = modelable_prod[_d].drop(cn.MOD_PROD_SUM)
        _raw_nmod = non_modelable_prod[_d]

        _truev_mod = true_values.loc[d, _raw_mod]

        _truev_nmod = true_values.loc[d, _raw_nmod]

        # Drop prediction if taken out of productions
        _mod = _truev_mod.index[_truev_mod.notna()]
        _nmod = _truev_nmod.index[_truev_nmod.notna()]

        _truev_mod = _truev_mod[_mod]
        _truev_nmod = _truev_nmod.loc[_nmod]

        # Create
        _truev_mod_sum = _truev_mod.sum()
        _truev_nmod_sum = _truev_nmod.sum()

        _truev_tot = _truev_mod_sum + _truev_nmod_sum

        _pred_mod = round(predictions.loc[_d, _mod].astype(float), 0)
        _pred_mod_sum = _pred_mod.sum()

        _pred_mod_tot = predictions.loc[_d, cn.MOD_PROD_SUM]

        _bmrk_mod = round(benchmark.loc[_d, _mod].astype(float), 0)
        _bmrk_mod_sum = _bmrk_mod.sum()

        _pred_nmod = round(predictions.loc[_d, _nmod].astype(float), 0)
        _pred_nmod_sum = _pred_nmod.sum()

        _bmrk_nmod = round(benchmark.loc[_d, _nmod].astype(float), 0)
        _bmrk_nmod_sum = _bmrk_nmod.sum()

        _pred_tot_sum = _pred_mod_sum + _pred_nmod_sum
        _bmrk_tot_sum = _bmrk_mod_sum + _bmrk_nmod_sum

        # Prediction error per product
        _pred_perr_mod = abs(_pred_mod - _truev_mod) / _truev_mod
        _pred_perr_nmod = abs(_pred_nmod - _truev_nmod) / _truev_nmod

        _bmrk_perr_mod = abs(_bmrk_mod - _truev_mod) / _truev_mod
        _bmrk_perr_nmod = abs(_bmrk_nmod - _truev_nmod) / _truev_nmod

        # Totals prediction (sum)
        _pred_err_mod_sum = abs(_pred_mod_sum - _truev_mod_sum) / _truev_mod_sum
        _pred_err_mod_tot = abs(_pred_mod_tot - _truev_mod_sum) / _truev_mod_sum

        _pred_err_nmod_sum = abs(_pred_nmod_sum - _truev_nmod_sum) / _truev_nmod_sum
        _pred_err_tot_sum = abs(_pred_tot_sum - _truev_tot) / _truev_tot

        _bmrk_err_mod_sum = abs(_bmrk_mod_sum - _truev_mod_sum) / _truev_mod_sum
        _bmrk_err_nmod_sum = abs(_bmrk_nmod_sum - _truev_nmod_sum) / _truev_nmod_sum
        _bmrk_err_tot_sum = abs(_bmrk_tot_sum - _truev_tot) / _truev_tot

        # Average prediction (avg)
        _pred_err_mod_avg = np.mean(list(abs(_pred_perr_mod)))
        _pred_err_nmod_avg = np.mean(list(abs(_pred_perr_nmod)))

        _pred_err_tot_avg = np.mean(list(abs(_pred_perr_mod)) + list(abs(_pred_perr_nmod)))

        _bmrk_err_mod_avg = np.mean(list(abs(_bmrk_perr_mod.astype(float))))
        _bmrk_err_nmod_avg = np.mean(list(abs(_bmrk_perr_nmod.astype(float))))
        _bmrk_err_tot_avg = np.mean(list(abs(_bmrk_perr_mod.astype(float))) + list(abs(_bmrk_perr_nmod.astype(float))))

        # Collect
        predictions_mod.loc[d, _mod] = _pred_mod
        predictions_nmod.loc[d, _nmod] = _pred_nmod

        benchmark_mod.loc[d, _mod] = _bmrk_mod
        benchmark_nmod.loc[d, _nmod] = _bmrk_nmod

        true_values_mod.loc[d, _mod] = _truev_mod
        true_values_nmod.loc[d, _nmod] = _truev_nmod

        predictions_mod_total.loc[d, "prediction"] = _pred_mod_sum
        predictions_mod_total.loc[d, "prediction_tot"] = _pred_mod_tot
        predictions_mod_total.loc[d, "true_value"] = _truev_mod_sum
        predictions_mod_total.loc[d, "pred_error_avg"] = _pred_err_mod_avg
        predictions_mod_total.loc[d, "pred_error_sum"] = _pred_err_mod_sum
        predictions_mod_total.loc[d, "pred_error_tot"] = _pred_err_mod_tot
        predictions_mod_total.loc[d, "benchmark"] = _bmrk_mod_sum
        predictions_mod_total.loc[d, "bmrk_error_avg"] = _bmrk_err_mod_avg
        predictions_mod_total.loc[d, "bmrk_error_sum"] = _bmrk_err_mod_sum

        predictions_nmod_total.loc[d, "prediction"] = _pred_nmod_sum
        predictions_nmod_total.loc[d, "true_value"] = _truev_nmod_sum
        predictions_nmod_total.loc[d, "pred_error_avg"] = _pred_err_nmod_avg
        predictions_nmod_total.loc[d, "pred_error_sum"] = _pred_err_nmod_sum
        predictions_nmod_total.loc[d, "benchmark"] = _bmrk_nmod_sum
        predictions_nmod_total.loc[d, "bmrk_error_avg"] = _bmrk_err_nmod_avg
        predictions_nmod_total.loc[d, "bmrk_error_sum"] = _bmrk_err_nmod_sum

        predictions_tot.loc[d, "prediction"] = _pred_tot_sum
        predictions_tot.loc[d, "true_value"] = _truev_tot
        predictions_tot.loc[d, "pred_error_avg"] = _pred_err_tot_avg
        predictions_tot.loc[d, "pred_error_sum"] = _pred_err_tot_sum
        predictions_tot.loc[d, "benchmark"] = _bmrk_tot_sum
        predictions_tot.loc[d, "bmrk_error_avg"] = _bmrk_err_tot_avg
        predictions_tot.loc[d, "bmrk_error_sum"] = _bmrk_err_tot_sum

    return (
        predictions_tot,
        predictions_mod_total,
        predictions_nmod_total,
        predictions_mod,
        true_values_mod,
    )


def performance_summary(prediction_table, subset="All", type="Test"):
    min_date = prediction_table.index.min().strftime("%Y-%m-%d")
    max_date = prediction_table.index.max().strftime("%Y-%m-%d")

    average_values = prediction_table.mean(axis=0)

    print("{} predictions, between {} and {}, {} total.".format(subset, min_date, max_date, prediction_table.shape[0]))

    if type == "Extensive":
        print("The sum prediction error: {}; the average error per product: {} ".format(
            round(average_values["pred_error_sum"], 2),
            round(average_values["pred_error_avg"], 2)))

        if subset == "Modelable":
            print("The total prediction error: {}".format(
                round(average_values["pred_error_tot"], 2)))

        print("The sum benchmark error: {}; the average benchmark error per product: {} ".format(
            round(average_values["bmrk_error_sum"], 2),
            round(average_values["bmrk_error_avg"], 2)))

    elif type == "Test":
        print("Sum prediction error: {}, benchmark error: {}".format(round(average_values["pred_error_sum"], 2),
                                                                     round(average_values["bmrk_error_sum"], 2)))
        if subset == "Modelable":
            print("Total prediction error: {}".format(round(average_values["pred_error_tot"], 2)))


def init_evaluate(summary):

    all_results = output_to_dict(data_loc=fm.TEST_RESULTS_FOLDER)




    def get_insample_error(result_dict):
        all_predictions = pd.DataFrame([])
        for k in result_dict.keys():
            _preds = result_dict[k][cn.FIT_ERROR_PCT]
            all_predictions = pd.concat([all_predictions, _preds])

        return all_predictions

    insample_errors = get_insample_error(result_dict=all_results)

    predictions = get_predictions(result_dict=all_results)
    benchmark = get_benchmark(result_dict=all_results)
    modelable_prod, non_modelable_prod = get_mod_products(result_dict=all_results)

    true_values = hff_predictor.generic.files.import_temp_file(
        data_loc=fm.ORDER_DATA_PR_FOLDER, set_index=True)

    product_cat = hff_predictor.generic.files.import_temp_file(
        data_loc=fm.ORDER_DATA_CG_PR_FOLDER, set_index=False)

    pred_tot, pred_mod_tot, pred_nmod_tot, pred_mod, true_valuesmod = performance_quality(predictions=predictions,
                                                                                          true_values=true_values,
                                                                                          benchmark=benchmark,
                                                                                          modelable_prod=modelable_prod,
                                                                                          non_modelable_prod=non_modelable_prod)

    save_to_csv(pred_mod_tot, file_name="test_voorspellingen_mod", folder=fm.TEST_PREDICTIONS_FOLDER)
    save_to_csv(pred_tot, file_name="test_voorspellingen_tot", folder=fm.TEST_PREDICTIONS_FOLDER)

    performance_summary(prediction_table=pred_tot, subset="All", type=summary)
    performance_summary(prediction_table=pred_mod_tot, subset="Modelable", type=summary)
    performance_summary(prediction_table=pred_nmod_tot, subset="Non-modelable", type=summary)

