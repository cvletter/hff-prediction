from hff_predictor.model.fit import fit_and_predict
import pandas as pd
import hff_predictor.config.column_names as cn
import logging

LOGGER = logging.getLogger(__name__)


def bootstrap(prediction,
              fit_dict,
              predict_dict,
              bootstrap=True,
              iterations=40,
              model_type="OLS",
              feature_threshold=None):

    all_bootstraps = pd.DataFrame([])
    raw_prediction = prediction.copy(deep=True)

    for i in range(1, iterations+1):
        print(i)
        #LOGGER.debug("Running iteration {} of {}".format(i, iterations))

        fits, temp_os, pars = fit_and_predict(
            fit_dict=fit_dict,
            predict_dict=predict_dict,
            bootstrap=bootstrap,
            model_type=model_type,
            feature_threshold=feature_threshold,
        )
        temp_os[cn.BOOTSTRAP_ITER] = i

        all_bootstraps = pd.concat([all_bootstraps, temp_os])

    raw_prediction[cn.BOOTSTRAP_ITER] = 0

    all_predictions = pd.concat([raw_prediction, all_bootstraps])

    boundaries = pd.DataFrame([])
    boundaries["lower_boundary"] = all_predictions.quantile(q=0.05)
    boundaries["upper_boundary"] = all_predictions.quantile(q=0.95)
    boundaries.drop(cn.BOOTSTRAP_ITER, axis=0, inplace=True)

    return boundaries
