import xgboost
import numpy as np
import pandas as pd
import warnings
import logging
import lightgbm

LOGGER = logging.getLogger(__name__)
warnings.filterwarnings(action='ignore', category=UserWarning)


def xgboost_regression_fit(y: pd.Series, X: pd.DataFrame):
    """
    XGBoost regressie functie, met fit

    :param y: Afhankelijke variabele
    :param X: Onafhankelijke variabelen
    :return: Gefit model
    """

    # alternative objective for count data: 'count:poisson'
    mdl = xgboost.XGBRegressor(objective='reg:squarederror', n_estimators=50)
    return mdl.fit(y=y, X=X)


def lightgbm_regression_fit(y: pd.Series, X: pd.DataFrame):
    """
    LightGBM model opzet

    :param y: Afhankelijke variabele
    :param X: Onafhankelijke variabelen
    :return: Gefit model
    """
    # alternative objective for count data: 'count:poisson'
    mdl = lightgbm.LGBMRegressor()
    return mdl.fit(y=y, X=X)


def tree_based_fit(y: pd.Series, X: pd.DataFrame, model_type="XGBoost"):
    """
    Wrap functie voor 'tree-based' modellen zoals XGBoost en LightGBM

    :param y: Afhankelijke variabele
    :param X: Onafhankelijke variabelen
    :param model_type: Type model wat is gebruikt
    :return: Geschat model

    """

    # Aanpassen van data types die beter aansluiten bij algoritme, werkt efficientie verhogend
    X = np.ascontiguousarray(X)
    y = np.ascontiguousarray(y)
    if model_type == "XGBoost":
        return xgboost_regression_fit(y=y, X=X)
    elif model_type == "LightGBM":
        return lightgbm_regression_fit(y=y, X=X)
    else:
        LOGGER.error("No correct model specified.")
