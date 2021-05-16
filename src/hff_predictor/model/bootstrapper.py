from hff_predictor.model.fit import fit_and_predict
import pandas as pd
import hff_predictor.config.column_names as cn

#TODO Bootstrap functie maken zodat het predictie intervallen geeft


def bootstrap(fit_dict,
              predict_dict,
              bootstrap=True,
              iterations=40,
              model_type="OLS",
              feature_threshold=None):

    all_predictions = pd.DataFrame([])

    for i in range(1, iterations):
        print("Running iteration {} of {}".format(i, iterations))
        fits, temp_os, pars = fit_and_predict(
            fit_dict=fit_dict,
            predict_dict=predict_dict,
            bootstrap=bootstrap,
            model_type=model_type,
            feature_threshold=feature_threshold,
        )
        temp_os[cn.BOOTSTRAP_ITER] = i

        all_predictions = pd.concat([all_predictions, temp_os])

        na_values = all_predictions.isna().sum().sum()
        logging.debug("In {} there are {} na_values".format(date_to_predict, na_values))