from hff_predictor.model.fit import fit_and_predict
import pandas as pd
import hff_predictor.config.column_names as cn
import logging

LOGGER = logging.getLogger(__name__)


def bootstrap(prediction: pd.DataFrame,
              fit_dict: dict,
              predict_dict: dict,
              bootstrap: bool = True,
              iterations: int = 2,
              model_type: str = "OLS",
              feature_threshold: list = None) -> pd.DataFrame:
    """
    Functie om bootstap voorspellingen te maken, t.b.v. onder en bovengrenzen van voorspelling
    :param prediction: Huidige voorspelling
    :param fit_dict: Fit dataset, gebruikt om modellen te schatten
    :param predict_dict: Predictie dataset, om voorspelling te maken
    :param bootstrap: Parameter om aan te geven of data moet worden ge-bootstrapped
    :param iterations: Aantal bootsrap iteraties
    :param model_type: Type voorspelmodel
    :param feature_threshold: Maximale features te selecteren
    :return: Bootstrapped onder en bovengrens
    """

    # Bereid dataframe voor om bootstraps op te vangen
    all_bootstraps = pd.DataFrame([])
    raw_prediction = prediction.copy(deep=True)

    # For-loop om bootstrap iteraties uit te voeren
    for i in range(1, iterations+1):
        LOGGER.debug("Bootstrap {} van totaal {}".format(i, iterations))

        fits, temp_os, pars = fit_and_predict(
            fit_dict=fit_dict,
            predict_dict=predict_dict,
            bootstrap=bootstrap,
            model_type=model_type,
            feature_threshold=feature_threshold,
        )

        # Voeg iteratienummer toe
        temp_os[cn.BOOTSTRAP_ITER] = i

        all_bootstraps = pd.concat([all_bootstraps, temp_os])

    # Geef de voorspelling bootstrap iteratie 0
    raw_prediction[cn.BOOTSTRAP_ITER] = 0

    # Voeg alles bij elkaar om bandbreedte te bepalen
    all_predictions = pd.concat([raw_prediction, all_bootstraps])

    # Bepaal bandbreedtes
    boundaries = pd.DataFrame([])
    #boundaries["lower_boundary"] = all_predictions.quantile(q=0.025)
    #boundaries["upper_boundary"] = all_predictions.quantile(q=0.975)
    boundaries["lower_boundary"] = all_predictions.min()
    boundaries["upper_boundary"] = all_predictions.max()

    boundaries.drop(cn.BOOTSTRAP_ITER, axis=0, inplace=True)

    return boundaries
