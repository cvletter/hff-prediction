from hff_predictor.model.model_types import regression_types, tree_based_types
import numpy as np
import pandas as pd
import logging
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)
LOGGER = logging.getLogger(__name__)


def fit_model(y: pd.Series, X: pd.DataFrame, model: str = "OLS"):
    """
    Fit wrapping functie om model te kiezen

    :param y: Afhankelijke variabele waar model op wordt gefit
    :param X: Set met verklarende variabelen
    :param model: Model type
    :return: Gefit model o.b.v model type
    """

    # Huidige set met mogelijke modellen
    regression_method = ["OLS", "Poisson", "Negative-Binomial"]
    tree_method = ["XGBoost", "LightGBM"]

    if model in regression_method:
        return regression_types.regression_model_fit(y=y, X=X)
    elif model in tree_method:
        return tree_based_types.tree_based_fit(y=y, X=X)


def predictor(Xpred: pd.DataFrame, fitted_model, weather_scenario: bool = False, model: str = "OLS"):
    """
    Voorspel wrapping funcite omwille van verschillende model types

    :param Xpred: Verklarende variabelen
    :param fitted_model: Geschat model
    :param model: Type model o.b.v. waarvan de input moet worden aangepast
    :return: De voorspelling
    """
    regression_method = ["OLS", "Poisson", "Negative-Binomial"]
    tree_method = ["XGBoost", "LightGBM"]

    if weather_scenario:
        pass


    # Model selectie
    if model in regression_method:
        prediction = fitted_model.predict(Xpred)
    elif model in tree_method:
        Xpred = np.ascontiguousarray(Xpred)
        prediction = fitted_model.predict(Xpred)
    else:
        logging.error("No correct model specified.")

    # Voorspelling afronden en negatieve waarden veranderen naar nul
    prediction = np.round(prediction, 0)
    prediction[prediction < 0] = 0

    return prediction

