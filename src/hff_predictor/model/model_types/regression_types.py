import statsmodels.api as sm
import pandas as pd

import logging
LOGGER = logging.getLogger(__name__)


def ols_regression(y, X):
    return sm.OLS(y, X, missing="drop")


def poisson_regression(y, X):
    return sm.GLM(y, X, family=sm.families.Poisson(), missing="drop")


def negative_binomial_regression(y, X):
    aux_reg_feat = pd.DataFrame(index=y.index)

    temp_mdl_poisson = sm.GLM(y, X, family=sm.families.Poisson(), missing="drop")

    temp_poisson_fit = temp_mdl_poisson.fit()

    aux_reg_feat["lambda"] = temp_poisson_fit.mu
    aux_reg_feat["dep_var"] = (
                                      (y - temp_poisson_fit.mu) ** 2 - y
                              ) / temp_poisson_fit.mu
    aux_reg = sm.OLS(aux_reg_feat["dep_var"], aux_reg_feat["lambda"]).fit()

    alpha_fit = aux_reg.params[0]

    return sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha_fit), missing="drop")


def regression_model_fit(y, X, model_type="OLS"):

    if model_type == "OLS":
        temp_mdl = ols_regression(y=y, X=X)

    elif model_type == "Poisson":
        temp_mdl = poisson_regression(y=y, X=X)

    elif model_type == "Negative-Binomial":
        temp_mdl = negative_binomial_regression(y=y, X=X)

    else:
        ValueError("No correct model specified.")

    return temp_mdl.fit()