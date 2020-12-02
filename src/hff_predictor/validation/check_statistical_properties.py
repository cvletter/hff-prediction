import datetime

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as acplots
from matplotlib import pyplot as plt
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller

import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
import hff_predictor.generic.dates as gf
from hff_predictor.features.create import prep_all_features
from hff_predictor.data.prepare import data_prep_wrapper
from hff_predictor.predict.setup import prediction_setup_wrapper


def prep_level_shifts():
    def str2date(date_str):
        return datetime.datetime.strptime(date_str, "%Y-%m-%d")

    level_shifts = pd.DataFrame(
        pd.date_range("2018-01-01", periods=1200, freq="D"), columns=["day"]
    )

    # this becomes the new constant
    # level_shifts['period_1'] = [1 if x <= str2date('2019-03-11') else 0 for x in level_shifts['day']]
    level_shifts["a_trans_period_1"] = [
        1 if (str2date("2019-03-18") <= x <= str2date("2019-04-08")) else 0
        for x in level_shifts["day"]
    ]
    level_shifts["b_period_2"] = [
        1 if str2date("2019-04-15") <= x <= str2date("2020-04-27") else 0
        for x in level_shifts["day"]
    ]
    level_shifts["c_trans_period_2"] = [
        1 if (str2date("2020-05-04") <= x <= str2date("2020-05-25")) else 0
        for x in level_shifts["day"]
    ]
    level_shifts["d_trans_period_2b"] = [
        1 if (str2date("2020-06-01") <= x <= str2date("2020-06-29")) else 0
        for x in level_shifts["day"]
    ]
    level_shifts["e_period_3"] = [
        1 if x >= str2date("2020-06-01") else 0 for x in level_shifts["day"]
    ]

    gf.add_week_year(data=level_shifts, date_name="day")
    gf.add_first_day_week(
        add_to=level_shifts, week_col_name=cn.WEEK_NUMBER, set_as_index=True
    )
    level_shifts.drop("day", axis=1, inplace=True)

    return level_shifts.groupby(cn.FIRST_DOW, as_index=True).max()


if __name__ == "main":

    date_to_predict = "2020-10-05"
    prediction_window = 2
    train_obs = cn.TRAIN_OBS
    difference = False
    lags = 5
    order_data = fm.RAW_DATA
    weather_data = fm.WEER_DATA
    product_data = fm.PRODUCT_STATUS
    model_type = "OLS"
    feature_threshold = [0.2, 25]

    # Import and prepare data
    (
        active_products,
        inactive_products,
        weather_data_processed,
        order_data_su,
        camp,
    ) = data_prep_wrapper(
        prediction_date=date_to_predict,
        prediction_window=prediction_window,
        reload_data=False,
        order_data_loc=order_data,
        weer_data_loc=weather_data,
        product_data_loc=product_data,
        agg_weekly=True,
        exclude_su=True,
        save_to_csv=False,
    )

    exogenous_features = prep_all_features(
        weather_data_processed=weather_data_processed,
        order_data_su=order_data_su,
        prediction_date=date_to_predict,
        train_obs=train_obs,
        save_to_csv=False,
    )

    fit_data, predict_data = prediction_setup_wrapper(
        prediction_date=date_to_predict,
        prediction_window=prediction_window,
        train_obs=train_obs,
        nlags=lags,
        difference=difference,
        act_products=active_products,
        exog_features=exogenous_features,
        save_to_pkl=False,
    )

    Y_true = fit_data[cn.Y_TRUE]
    y = Y_true[Y_true.columns[0]]

    def stat_properties(y):
        skewness = round(st.skew(y), 2)
        kurtosis = round(st.kurtosis(y), 2)
        avg = int(np.mean(y))
        med = int(np.median(y))
        adf = round(adfuller(y)[1], 3)

        return skewness, kurtosis, avg, med, adf

    def detrend(y):
        X = pd.DataFrame(index=y.index)
        X["constant"] = 1
        X["trend"] = sorted(np.arange(1, len(y) + 1), reverse=True)
        X["winter"] = [1 if x.month <= 3 else 0 for x in X.index]
        X["lente"] = [1 if 4 <= x.month <= 6 else 0 for x in X.index]
        X["zomer"] = [1 if 7 <= x.month <= 9 else 0 for x in X.index]

        X["lag2"] = y.shift(-2)
        X["lag3"] = y.shift(-3)

        X = X[:-3]
        y = y[:-3]

        mdl = sm.OLS(y, X, missing="drop")
        fit = mdl.fit()

        return y - fit.predict()

    plt.plot(acplots.acf(y))

    Y_true_dt = Y_true.apply(lambda x: detrend(x))
    Y_true_d = Y_true.apply(lambda x: x.diff(-1))[:-1]
    Y_true_ddt = Y_true_dt.apply(lambda x: x.diff(-1))[:-1]

    dst = Y_true.apply(lambda x: stat_properties(x)).T
    dst.columns = ["skew", "kurt", "med", "avg", "adf"]

    dst_dt = Y_true_dt.apply(lambda x: stat_properties(x)).T
    dst_dt.columns = ["skew", "kurt", "med", "avg", "adf"]

    dst_d = Y_true_d.apply(lambda x: stat_properties(x)).T
    dst_d.columns = ["skew", "kurt", "med", "avg", "adf"]

    dst_ddt = Y_true_ddt.apply(lambda x: stat_properties(x)).T
    dst_ddt.columns = ["skew", "kurt", "med", "avg", "adf"]

    level_features = prep_level_shifts()

    def ts_cleaner(y):

        # Determine stationarity
        adf_pval = adfuller(y)[1]

        if adf_pval < 0.05:
            print("Product {} is stationary by definition.".format(y.name))
        else:
            print("Product {} is not stationary by definition.".format(y.name))

        # Remove shocks
        level_features = prep_level_shifts().loc[y.index]
        level_features = level_features.loc[:, (level_features != 0).any(axis=0)]
        level_features = level_features[level_features.columns.sort_values()]

        if level_features.sum(axis=1).sum() == len(level_features):
            level_features = level_features.iloc[:, 1:]

        # Test shocks
        X = pd.DataFrame(index=y.index)
        X["constant"] = 1
        X = pd.concat([X, level_features], axis=1)

        fit_mdl = sm.OLS(y, X).fit()
        resid = fit_mdl.resid
        adf_resid = adfuller(resid)[1]

        if adf_resid < 0.05:
            print(
                "Product {} is stationary after correcting for shocks.".format(y.name)
            )
        else:
            print(
                "Product {} is not stationary after correcting for shocks.".format(
                    y.name
                )
            )

        if jarque_bera(resid)[1] < 0.05:
            print(
                "Product {} errors are normal after correcting for shocks.".format(
                    y.name
                )
            )
        else:
            print(
                "Product {} are not normal after correcting for shocks.".format(y.name)
            )

        # Test seasonality
        X["winter"] = [1 if x.month <= 3 else 0 for x in X.index]
        X["lente"] = [1 if 4 <= x.month <= 6 else 0 for x in X.index]
        X["zomer"] = [1 if 7 <= x.month <= 9 else 0 for x in X.index]

        fit_mdl = sm.OLS(y, X).fit()
        resid = fit_mdl.resid
        adf_resid = adfuller(resid)[1]

        if adf_resid < 0.05:
            print("Product {y.name} is stationary after correcting for seasons.")
        else:
            print(f"Product {y.name} is not stationary after correcting for seasons.")

        if jarque_bera(resid)[1] < 0.05:
            print(f"Product {y.name} errors are normal after correcting for seasons.")
        else:
            print(f"Product {y.name} are not normal after correcting for seasons.")

        # Test autocorrelation
        lag_index = [y.name in x for x in fit_data[cn.Y_AR].columns]
        y_ar = fit_data[cn.Y_AR].iloc[:, lag_index]

        X = pd.concat([y_ar, X], axis=1)

        y = y[:-3]
        X = X.sort_index(ascending=False)[:-3]

        fit_mdl = sm.OLS(y, X).fit()
        resid = fit_mdl.resid
        adf_resid = adfuller(resid)[1]

        if adf_resid < 0.05:
            print(
                f"Product {y.name} is stationary after correcting for autocorrelation."
            )
        else:
            print(
                f"Product {y.name} is not stationary after correcting for autocorrelation."
            )

        if jarque_bera(resid)[1] < 0.05:
            print(
                f"Product {y.name} errors are normal after correcting for autocorrelation."
            )
        else:
            print(
                f"Product {y.name} are not normal after correcting for autocorrelation."
            )
