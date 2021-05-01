import hff_predictor.generic.files
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime

import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm


def moving_average(active_products, prediction_date, periods=cn.MA_PERIOD, window=cn.PREDICTION_WINDOW):

    if type(prediction_date) == str:
        prediction_date = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")

    ma_start = prediction_date - datetime.timedelta(days=7*window)
    ma_end = prediction_date - datetime.timedelta(days=7*(window+periods))

    Y_org = active_products
    Y_ma_p = Y_org[(Y_org.index > ma_end) & (Y_org.index <= ma_start)]
    Y_ma_p.reset_index(inplace=True, drop=True)

    Y_ma_r = pd.DataFrame(index=[prediction_date], columns=Y_org.columns)
    Y_ma_r.loc[prediction_date, :] = Y_ma_p.mean(axis=0, skipna=True)

    return Y_ma_r
