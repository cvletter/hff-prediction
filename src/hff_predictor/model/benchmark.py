import pandas as pd
import datetime
import hff_predictor.config.column_names as cn
from hff_predictor.predict.setup import split_products
# FIT data toevoegen
# Mod prod en non mod prod
# dan de sommen verwijderen en opnieuw toevoegen


def moving_average(active_products, prediction_date, periods=cn.MA_PERIOD,
                   prediction_window=cn.PREDICTION_WINDOW, min_obs=cn.TRAIN_OBS):

    if type(prediction_date) == str:
        prediction_date = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")

    products_model, products_no_model = split_products(active_products=active_products,
                                                       min_obs=min_obs,
                                                       prediction_date=prediction_date,
                                                       prediction_window=prediction_window)

    ma_start = prediction_date - datetime.timedelta(days=7*prediction_window)
    ma_end = prediction_date - datetime.timedelta(days=7*(prediction_window+periods))

    Y_org = pd.concat([products_model, products_no_model], axis=1)
    Y_ma_p = Y_org[(Y_org.index > ma_end) & (Y_org.index <= ma_start)]
    Y_ma_p.reset_index(inplace=True, drop=True)

    Y_ma_r = pd.DataFrame(index=[prediction_date], columns=Y_org.columns)
    Y_ma_r.loc[prediction_date, :] = Y_ma_p.mean(axis=0, skipna=True)

    return Y_ma_r
