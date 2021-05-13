import xgboost


def xgboost_regression_fit(y, X):
    # alternative objective for count data: 'count:poisson'
    mdl = xgboost.XGBRegressor(objective='reg:squarederror', n_estimators=50)
    return mdl.fit(y, X)


def tree_based_fit(y, X, model_type="XGBoost"):
    if model_type == "XGBoost":
        X_xg = xgboost.DMatrix(data=X)
        # y_xg = xgboost.DMatrix(data=y)
        return xgboost_regression_fit(y=y, X=X_xg)
    else:
        ValueError("No correct model specified.")
