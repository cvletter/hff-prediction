from hff_predictor.model.model_types import regression_types, tree_based_types
import numpy as np
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


def fit_model(y, X, model="OLS"):
    regression_method = ["OLS", "Poisson", "Negative-Binomial"]
    tree_method = ["XGBoost"]

    if model in regression_method:
        return regression_types.regression_model_fit(y=y, X=X)
    elif model in tree_method:
        return tree_based_types.tree_based_fit(y=y, X=X)


def predictor(Xpred, fitted_model, model="OLS"):
    regression_method = ["OLS", "Poisson", "Negative-Binomial"]
    tree_method = ["XGBoost"]

    if model in regression_method:
        prediction = fitted_model.predict(Xpred)
    elif model in tree_method:
        Xpred = np.ascontiguousarray(Xpred)
        prediction = fitted_model.predict(Xpred)
    else:
        ValueError("No correct model specified.")

    prediction = np.round(prediction, 0)
    prediction[prediction < 0] = 0

    return prediction

