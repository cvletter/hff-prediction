import xgboost
import numpy as np
import warnings
import logging
import lightgbm

LOGGER = logging.getLogger(__name__)
warnings.filterwarnings(action='ignore', category=UserWarning)


def xgboost_regression_fit(y, X):
    # alternative objective for count data: 'count:poisson'
    mdl = xgboost.XGBRegressor(objective='reg:squarederror', n_estimators=50)
    return mdl.fit(y=y, X=X)


def lightgbm_regression_fit(y, X):
    # alternative objective for count data: 'count:poisson'
    mdl = lightgbm.LGBMRegressor(n_jobs=1)
    return mdl.fit(y=y, X=X)


def tree_based_fit(y, X, model_type="XGBoost"):
    X = np.ascontiguousarray(X)
    y = np.ascontiguousarray(y)
    if model_type == "XGBoost":
        return xgboost_regression_fit(y=y, X=X)
    elif model_type == "LightGBM":
        return lightgbm_regression_fit(y=y, X=X)
    else:
        logging.error("No correct model specified.")
