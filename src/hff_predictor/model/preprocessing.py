from hff_predictor.model.model_types import regression_types, tree_based_types
import xgboost


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
        Xpred_xg = xgboost.DMatrix(Xpred)
        prediction = fitted_model.predict(Xpred_xg)
    else:
        ValueError("No correct model specified.")

    return prediction

