import hff_predictor.generic.files
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime

import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm


def moving_average(fit_dict, periods=5, window=2):
    Y_org = fit_dict[cn.Y_TRUE]
    Y_ma = Y_org.shift(-2)[:periods]
    Y_ma_index = Y_org.index[:periods] + datetime.timedelta(days=7*window)
    Y_ma.set_index(Y_ma_index, inplace=True)

    return Y_ma.mean(axis=0, skip_na=True)
