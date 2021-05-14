import xgboost
import numpy as np
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


def xgboost_regression_fit(y, X):
    # alternative objective for count data: 'count:poisson'
    mdl = xgboost.XGBRegressor(objective='reg:squarederror', n_estimators=50)
    return mdl.fit(y=y, X=X)


def tree_based_fit(y, X, model_type="XGBoost"):
    if model_type == "XGBoost":
        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)
        return xgboost_regression_fit(y=y, X=X)
    else:
        ValueError("No correct model specified.")
