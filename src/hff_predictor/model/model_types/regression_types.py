import statsmodels.api as sm
import pandas as pd

import logging
LOGGER = logging.getLogger(__name__)


def ols_regression(y: pd.Series, X: pd.DataFrame):
    """
    OLS functie, nu nog zonder aanpassingen, kan worden uitgebreid

    :param y: Afhankelijke variabele
    :param X: Onafhankelijke variabelen
    :return: Geschat model
    """
    return sm.OLS(y, X, missing="drop")


def poisson_regression(y: pd.Series, X: pd.DataFrame):
    """
    Poisson regressie functie, kwan werken indien er veel 0-waarden zijn

    :param y: Afhankelijke variabele
    :param X: Onafhankelijke variabelen
    :return: Geschat model
    """
    return sm.GLM(y, X, family=sm.families.Poisson(), missing="drop")


def negative_binomial_regression(y, X):
    """
    Negatief-Binomiaal regressie model, generalisatie van Poisson model

    :param y: Afhankelijke variabele
    :param X: Onafhankelijke variabelen
    :return: Geschat model
    """

    # Auxiliaire features
    aux_reg_feat = pd.DataFrame(index=y.index)

    # Schat Possion model
    temp_mdl_poisson = sm.GLM(y, X, family=sm.families.Poisson(), missing="drop")
    temp_poisson_fit = temp_mdl_poisson.fit()

    # Auxiliaire regressie o.b.v. Poisson resultaten
    aux_reg_feat["lambda"] = temp_poisson_fit.mu
    aux_reg_feat["dep_var"] = ((y - temp_poisson_fit.mu) ** 2 - y) / temp_poisson_fit.mu
    aux_reg = sm.OLS(aux_reg_feat["dep_var"], aux_reg_feat["lambda"]).fit()

    # Bepaal alpha als input voor Negatief-Binomiaal model
    alpha_fit = aux_reg.params[0]

    return sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha_fit), missing="drop")


def regression_model_fit(y: pd.Series, X: pd.DataFrame, model_type: str = "OLS"):
    """
    Wrap functie voor de regressie modellen

    :param y: Afhankelijke variabele
    :param X: Onafhankelijke variabelen
    :param model_type: Gekozen regressie type
    :return: Geschat model

    """

    if model_type == "OLS":
        temp_mdl = ols_regression(y=y, X=X)

    elif model_type == "Poisson":
        temp_mdl = poisson_regression(y=y, X=X)

    elif model_type == "Negative-Binomial":
        temp_mdl = negative_binomial_regression(y=y, X=X)

    else:
        logging.error("No correct model specified.")

    return temp_mdl.fit()