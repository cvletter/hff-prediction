import pandas as pd
import datetime
import hff_predictor.config.column_names as cn
from hff_predictor.predict.setup import split_products


def moving_average(active_products: pd.DataFrame, prediction_date: str, periods: int = cn.MA_PERIOD,
                   prediction_window: int = cn.PREDICTION_WINDOW, min_obs: int = cn.TRAIN_OBS) -> pd.DataFrame:
    """
    Maakt een voorspelling op basis van het gemiddelde van de afgelopen N weken (periods)
    :param active_products: Totaal dataframe met alle actieve producten
    :param prediction_date: Datum van voorspelling
    :param periods: Aantal weken waarover moet worden gemiddeld (nu 5 weken)
    :param prediction_window: Voorspelwindow
    :param min_obs: Aantal observaties in train set
    :return: Moving average dataset
    """

    # Vertaal datum als tekst naar datetime object
    if type(prediction_date) == str:
        prediction_date = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")

    # Splits producten op in modelleerbaar en niet modelleerbaar
    products_model, products_no_model = split_products(active_products=active_products,
                                                       min_obs=min_obs,
                                                       prediction_date=prediction_date,
                                                       prediction_window=prediction_window)

    # Bepaal window waarover gemiddelde moet worden genomen
    ma_start = prediction_date - datetime.timedelta(days=7*prediction_window)
    ma_end = prediction_date - datetime.timedelta(days=7*(prediction_window+periods))

    # Maak data selectie om gemiddelde te bepalen
    Y_org = pd.concat([products_model, products_no_model], axis=1)
    Y_ma_p = Y_org[(Y_org.index > ma_end) & (Y_org.index <= ma_start)]
    Y_ma_p.reset_index(inplace=True, drop=True)

    # Bepaalde moving average
    Y_ma_r = pd.DataFrame(index=[prediction_date], columns=Y_org.columns)
    Y_ma_r.loc[prediction_date, :] = Y_ma_p.mean(axis=0, skipna=True)

    return Y_ma_r
