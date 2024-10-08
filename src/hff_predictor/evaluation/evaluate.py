import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
import hff_predictor.generic.dates as gf
import hff_predictor.generic.files
import hff_predictor.data.transformations as dtr

import numpy as np
import pandas as pd
from hff_predictor.generic.files import read_pkl, import_temp_file, save_to_csv

import logging
LOGGER = logging.getLogger(__name__)


def output_to_dict(data_loc):
    result_file = read_pkl(data_loc=data_loc)

    for i in range(0, len(result_file)):
        if i == 0:
            all_dicts = result_file[i]
        else:
            all_dicts.update(result_file[i])

    return all_dicts


def get_insample_error(result_dict):
    all_predictions = pd.DataFrame([])
    for k in result_dict.keys():
        _preds = result_dict[k][cn.FIT_ERROR_PCT]
        all_predictions = pd.concat([all_predictions, _preds])

    return all_predictions


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
                        modelable_prod, non_modelable_prod):

    grouped_cols = [
        cn.MOD_PROD_SUM,
        cn.ALL_PROD_SUM,
        cn.ALL_ROL_SUM
    ]

    # Importeer consument groep nummers om groeperingen te kunnen maken
    consumentgroep_nr = import_temp_file(data_loc=fm.ORDER_DATA_CG_PR_FOLDER, set_index=False)
    consumentgroep_nr = consumentgroep_nr[[cn.INKOOP_RECEPT_NM, cn.CONSUMENT_GROEP_NR]]
    consumentgroep_nr.set_index(cn.INKOOP_RECEPT_NM, inplace=True)

    # Remove negative predictions
    predictions[predictions <= 0] = 0.0

    # Keep only actual prediction
    # predictions = predictions[predictions[cn.BOOTSTRAP_ITER] == 0]
    # predictions.drop(cn.BOOTSTRAP_ITER, axis=1, inplace=True, errors='ignore')

    # Collect all prediction dates in test set
    all_prediction_dates = predictions.index.unique().sort_values(
        ascending=False
    )

    # Prepare all data subsets
    predictions_mod = pd.DataFrame(index=all_prediction_dates)
    predictions_nmod = pd.DataFrame(index=all_prediction_dates)
    predictions_rol = pd.DataFrame(index=all_prediction_dates)

    benchmark_mod = pd.DataFrame(index=all_prediction_dates)
    benchmark_nmod = pd.DataFrame(index=all_prediction_dates)
    benchmark_rol = pd.DataFrame(index=all_prediction_dates)

    true_values_mod = pd.DataFrame(index=all_prediction_dates)
    true_values_nmod = pd.DataFrame(index=all_prediction_dates)
    true_values_rol = pd.DataFrame(index=all_prediction_dates)

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
            "prediction_80p",
            "true_value_80p",
            "pred_err80p",
            "pred_err80p_avg"
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
            "prediction_80p",
            "true_value_80p",
            "pred_err80p",
            "pred_err80p_avg"
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
            "prediction_80p",
            "true_value_80p",
            "pred_err80p",
            "pred_err80p_avg"
        ],
    )

    # Filling in the predictions
    for d in all_prediction_dates:
        _d = d.strftime("%Y-%m-%d")
        _raw_mod = modelable_prod[_d].drop(grouped_cols)
        _raw_nmod = non_modelable_prod[_d]

        _truev_mod = true_values.loc[d, _raw_mod]
        _truev_nmod = true_values.loc[d, _raw_nmod]

        # Drop prediction if taken out of productions
        _mod = _truev_mod.index[_truev_mod.notna()]
        _nmod = _truev_nmod.index[_truev_nmod.notna()]

        _rol_mod = dtr.find_rol_products(data=_mod, consumentgroep_nrs=consumentgroep_nr)
        _rol_nmod = dtr.find_rol_products(data=_nmod, consumentgroep_nrs=consumentgroep_nr)

        _truev_mod = _truev_mod[_mod]
        _truev_nmod = _truev_nmod.loc[_nmod]

        _truev_rol_mod = _truev_mod[_rol_mod]
        _truev_rol_nmod = _truev_nmod.loc[_rol_nmod]

        # Create
        _truev_mod_sum = _truev_mod.sum()
        _truev_nmod_sum = _truev_nmod.sum()

        _truev_tot = _truev_mod_sum + _truev_nmod_sum
        _truev_rol_tot = _truev_rol_mod.sum() + _truev_rol_nmod.sum()

        _pred_mod = round(predictions.loc[_d, _mod].astype(float), 0)
        _pred_mod_sum = _pred_mod.sum()

        # Totaal voorspellingen
        _pred_mod_tot = predictions.loc[_d, cn.MOD_PROD_SUM] # All modelable products
        _pred_tot = predictions.loc[_d, cn.ALL_PROD_SUM] # All products
        _pred_tot_rol = predictions.loc[_d, cn.ALL_ROL_SUM] # All rol products

        _bmrk_mod = round(benchmark.loc[_d, _mod].astype(float), 0)
        _bmrk_mod_rol = round(benchmark.loc[_d, _rol_mod].astype(float), 0)
        _bmrk_mod_sum = _bmrk_mod.sum()

        _pred_nmod = round(predictions.loc[_d, _nmod].astype(float), 0)
        _pred_nmod_sum = _pred_nmod.sum()

        _bmrk_nmod = round(benchmark.loc[_d, _nmod].astype(float), 0)
        _bmrk_nmod_rol = round(benchmark.loc[_d, _rol_nmod].astype(float), 0)
        _bmrk_nmod_sum = _bmrk_nmod.sum()

        _pred_tot_sum = _pred_mod_sum + _pred_nmod_sum
        _bmrk_tot_sum = _bmrk_mod_sum + _bmrk_nmod_sum
        _bmrk_tot_rol = _bmrk_mod_rol.sum() + _bmrk_nmod_rol.sum()

        # Prediction error per product
        _pred_perr_mod = abs(_pred_mod - _truev_mod) / _truev_mod
        _pred_perr_nmod = abs(_pred_nmod - _truev_nmod) / _truev_nmod

        _bmrk_perr_mod = abs(_bmrk_mod - _truev_mod) / _truev_mod
        _bmrk_perr_nmod = abs(_bmrk_nmod - _truev_nmod) / _truev_nmod

        # Totals prediction (sum)
        _pred_err_mod_sum = abs(_pred_mod_sum - _truev_mod_sum) / _truev_mod_sum
        _pred_err_mod_tot = abs(_pred_mod_tot - _truev_mod_sum) / _truev_mod_sum
        _pred_err_rol_tot = abs(_pred_tot_rol - _truev_rol_tot) / _truev_rol_tot

        _pred_err_nmod_sum = abs(_pred_nmod_sum - _truev_nmod_sum) / _truev_nmod_sum
        _pred_err_tot_sum = abs(_pred_tot_sum - _truev_tot) / _truev_tot
        _pred_err_tot = abs(_pred_tot - _truev_tot) / _truev_tot

        _bmrk_err_mod_sum = abs(_bmrk_mod_sum - _truev_mod_sum) / _truev_mod_sum
        _bmrk_err_nmod_sum = abs(_bmrk_nmod_sum - _truev_nmod_sum) / _truev_nmod_sum
        _bmrk_err_tot_sum = abs(_bmrk_tot_sum - _truev_tot) / _truev_tot
        _bmrk_err_tot_rol = abs(_bmrk_tot_rol - _truev_rol_tot) / _truev_rol_tot

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

        # Top 80%
        top_pct = 0.8
        _pred_mod_top = _pred_mod[_pred_mod.sort_values(ascending=False).cumsum() / _pred_mod.sum() <= top_pct]
        _truev_mod_top = _truev_mod[_pred_mod.sort_values(ascending=False).cumsum() / _pred_mod.sum() <= top_pct]
        _bmrk_mod_top = _bmrk_mod[_pred_mod.sort_values(ascending=False).cumsum() / _pred_mod.sum() <= top_pct]
        _perr_mod_top = np.round(abs(_pred_mod_top.sum() - _truev_mod_top.sum()) / _truev_mod_top.sum(), 3)
        _perr_mod_top_avg = np.round((abs(_pred_mod_top - _truev_mod_top) / _truev_mod_top).mean(), 3)
        _bmrk_mod_top_avg = np.round((abs(_bmrk_mod_top - _truev_mod_top) / _truev_mod_top).mean(), 3)

        _pred_nmod_top = _pred_nmod[_pred_nmod.sort_values(ascending=False).cumsum() / _pred_nmod.sum() <= top_pct]
        _truev_nmod_top = _truev_nmod[_pred_nmod.sort_values(ascending=False).cumsum() / _pred_nmod.sum() <= top_pct]
        _bmrk_nmod_top = _bmrk_nmod[_pred_nmod.sort_values(ascending=False).cumsum() / _pred_nmod.sum() <= top_pct]
        _perr_nmod_top = np.round(abs(_pred_nmod_top.sum() - _truev_nmod_top.sum()) / _truev_nmod_top.sum(), 3)
        _perr_nmod_top_avg = np.round((abs(_pred_nmod_top - _truev_nmod_top) / _truev_nmod_top).mean(), 3)
        _bmrk_nmod_top_avg = np.round((abs(_bmrk_nmod_top - _truev_nmod_top) / _truev_nmod_top).mean(), 3)

        _pred_tot_top = _pred_mod_top.sum() + _pred_nmod_top.sum()
        _bmrk_tot_top = _bmrk_mod_top.sum() + _bmrk_nmod_top.sum()
        _truev_tot_top = _truev_mod_top.sum() + _truev_nmod_top.sum()
        _perr_tot_top = np.round(abs(_pred_tot_top - _truev_tot_top) / _truev_tot_top, 3)
        _perr_tot_top_avg = np.round(np.mean(list(abs(_pred_mod_top - _truev_mod_top) / _truev_mod_top) +
                                             list(abs(_pred_nmod_top - _truev_nmod_top) / _truev_nmod_top)), 3)

        _bmrk_tot_top_avg = np.round(np.mean(list(abs(_bmrk_mod_top - _truev_mod_top) / _truev_mod_top) +
                                             list(abs(_bmrk_nmod_top - _truev_nmod_top) / _truev_nmod_top)), 3)

        # Collect
        predictions_mod_total.loc[d, "prediction"] = _pred_mod_sum
        predictions_mod_total.loc[d, "prediction_tot"] = _pred_mod_tot
        predictions_mod_total.loc[d, "true_value"] = _truev_mod_sum
        predictions_mod_total.loc[d, "pred_error_avg"] = _pred_err_mod_avg
        predictions_mod_total.loc[d, "pred_error_sum"] = _pred_err_mod_sum
        predictions_mod_total.loc[d, "pred_error_tot"] = _pred_err_mod_tot
        predictions_mod_total.loc[d, "benchmark"] = _bmrk_mod_sum
        predictions_mod_total.loc[d, "bmrk_error_avg"] = _bmrk_err_mod_avg
        predictions_mod_total.loc[d, "bmrk_error_sum"] = _bmrk_err_mod_sum
        predictions_mod_total.loc[d, "prediction_80p"] = _pred_mod_top.sum()
        predictions_mod_total.loc[d, "true_value_80p"] = _truev_mod_top.sum()
        predictions_mod_total.loc[d, "pred_err80p"] = _perr_mod_top
        predictions_mod_total.loc[d, "pred_err80p_avg"] = _perr_mod_top_avg
        predictions_mod_total.loc[d, "bmrk_80p"] = _bmrk_mod_top.sum()
        predictions_mod_total.loc[d, "bmrk_err80p_avg"] = _bmrk_mod_top_avg

        predictions_nmod_total.loc[d, "prediction"] = _pred_nmod_sum
        predictions_nmod_total.loc[d, "true_value"] = _truev_nmod_sum
        predictions_nmod_total.loc[d, "pred_error_avg"] = _pred_err_nmod_avg
        predictions_nmod_total.loc[d, "pred_error_sum"] = _pred_err_nmod_sum
        predictions_nmod_total.loc[d, "benchmark"] = _bmrk_nmod_sum
        predictions_nmod_total.loc[d, "bmrk_error_avg"] = _bmrk_err_nmod_avg
        predictions_nmod_total.loc[d, "bmrk_error_sum"] = _bmrk_err_nmod_sum
        predictions_nmod_total.loc[d, "prediction_80p"] = _pred_nmod_top.sum()
        predictions_nmod_total.loc[d, "true_value_80p"] = _truev_nmod_top.sum()
        predictions_nmod_total.loc[d, "pred_err80p"] = _perr_nmod_top
        predictions_nmod_total.loc[d, "pred_err80p_avg"] = _perr_nmod_top_avg
        predictions_nmod_total.loc[d, "bmrk_80p"] = _bmrk_nmod_top.sum()
        predictions_nmod_total.loc[d, "bmrk_err80p_avg"] = _bmrk_nmod_top_avg

        predictions_tot.loc[d, "prediction"] = _pred_tot_sum
        predictions_tot.loc[d, "true_value"] = _truev_tot
        predictions_tot.loc[d, "true_value_rol"] = _truev_rol_tot
        predictions_tot.loc[d, "prediction_tot"] = _pred_tot
        predictions_tot.loc[d, "predictions_rol_tot"] = _pred_tot_rol
        predictions_tot.loc[d, "pred_error_avg"] = _pred_err_tot_avg
        predictions_tot.loc[d, "pred_error_sum"] = _pred_err_tot_sum
        predictions_tot.loc[d, "pred_error_tot"] = _pred_err_tot
        predictions_tot.loc[d, "pred_error_rol_tot"] = _pred_err_rol_tot
        predictions_tot.loc[d, "benchmark"] = _bmrk_tot_sum
        predictions_tot.loc[d, "benchmark_rol"] = _bmrk_tot_rol
        predictions_tot.loc[d, "bmrk_error_avg"] = _bmrk_err_tot_avg
        predictions_tot.loc[d, "bmrk_error_sum"] = _bmrk_err_tot_sum
        predictions_tot.loc[d, "bmrk_error_rol"] = _bmrk_err_tot_rol
        predictions_tot.loc[d, "prediction_80p"] = _pred_tot_top
        predictions_tot.loc[d, "true_value_80p"] = _truev_tot_top
        predictions_tot.loc[d, "pred_err80p"] = _perr_tot_top
        predictions_tot.loc[d, "pred_err80p_avg"] = _perr_tot_top_avg
        predictions_tot.loc[d, "bmrk_80p"] = _bmrk_tot_top
        predictions_tot.loc[d, "bmrk_err80p_avg"] = _bmrk_tot_top_avg

    return (
        predictions_tot.astype(float),
        predictions_mod_total.astype(float),
        predictions_nmod_total.astype(float),
        predictions_mod.astype(float),
        true_values_mod.astype(float),
        predictions_nmod.astype(float),
        true_values_nmod.astype(float),
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
    predictions = get_predictions(result_dict=all_results)
    benchmark = get_benchmark(result_dict=all_results)
    modelable_prod, non_modelable_prod = get_mod_products(result_dict=all_results)

    true_values = hff_predictor.generic.files.import_temp_file(
        data_loc=fm.ORDER_DATA_PR_FOLDER, set_index=True)

    product_cat = hff_predictor.generic.files.import_temp_file(
        data_loc=fm.ORDER_DATA_CG_PR_FOLDER, set_index=False)

    pred_tot, pred_mod_tot, pred_nmod_tot, pred_mod, true_valuesmod, pred_nmod, true_valuesnmod = performance_quality(
        predictions=predictions,
        true_values=true_values,
        benchmark=benchmark,
        modelable_prod=modelable_prod,
        non_modelable_prod=non_modelable_prod)

    save_predictions = False
    if save_predictions:
        save_to_csv(pred_mod, file_name="raw_predictions_mod", folder=fm.TEST_PREDICTIONS_FOLDER)
        save_to_csv(predictions, file_name="raw_predictions_total", folder=fm.TEST_PREDICTIONS_FOLDER)
        save_to_csv(true_valuesmod, file_name="raw_actuals_mod", folder=fm.TEST_PREDICTIONS_FOLDER)
        save_to_csv(true_values, file_name="raw_actuals_total", folder=fm.TEST_PREDICTIONS_FOLDER)


    save_to_csv(pred_mod_tot, file_name="test_voorspellingen_mod", folder=fm.TEST_PREDICTIONS_FOLDER)
    save_to_csv(pred_tot, file_name="test_voorspellingen_tot", folder=fm.TEST_PREDICTIONS_FOLDER)

    performance_summary(prediction_table=pred_tot, subset="All", type=summary)
    performance_summary(prediction_table=pred_mod_tot, subset="Modelable", type=summary)
    performance_summary(prediction_table=pred_nmod_tot, subset="Non-modelable", type=summary)

