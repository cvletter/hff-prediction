from hff_predictor.model.model_types import regression_types, tree_based_types
import hff_predictor.config.column_names as cn
import numpy as np
import pandas as pd
import logging
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)
LOGGER = logging.getLogger(__name__)


def fit_model(y: pd.Series, X: pd.DataFrame, model: str = "OLS"):
    """
    Fit wrapping functie om model te kiezen

    :param y: Afhankelijke variabele waar model op wordt gefit
    :param X: Set met verklarende variabelen
    :param model: Model type
    :return: Gefit model o.b.v model type
    """

    # Huidige set met mogelijke modellen
    regression_method = ["OLS", "Poisson", "Negative-Binomial"]
    tree_method = ["XGBoost", "LightGBM"]

    if model in regression_method:
        return regression_types.regression_model_fit(y=y, X=X)
    elif model in tree_method:
        return tree_based_types.tree_based_fit(y=y, X=X)


def predictor(Xpred: pd.DataFrame, fitted_model, weather_scenario = None, model: str = "OLS",
              prediction_window: int = 2):
    """
    Voorspel wrapping funcite omwille van verschillende model types

    :param Xpred: Verklarende variabelen
    :param fitted_model: Geschat model
    :param model: Type model o.b.v. waarvan de input moet worden aangepast
    :return: De voorspelling
    """

    # Model selectie
    def make_prediction(Xpred, model=model):
        regression_method = ["OLS", "Poisson", "Negative-Binomial"]
        tree_method = ["XGBoost", "LightGBM"]

        if model in regression_method:
            pred = fitted_model.predict(Xpred)
        elif model in tree_method:
            Xpred = np.ascontiguousarray(Xpred)
            pred = fitted_model.predict(Xpred)
        else:
            logging.error("No correct model specified.")

        return pred

    def replace_feature_values(X, factor, prediction_window=prediction_window):
        X_new = X.copy(deep=True)
        temperature = 'temperatuur_gem_next{}w'.format(prediction_window)
        weather_cols = [temperature]

        for c in weather_cols:
            if c == 'neerslag_mm_next2w':
                X_new.loc[:, c] = weather_scenario[c] * (1-factor)
            else:
                X_new.loc[:, c] = weather_scenario[c] * (1+factor)

        return X_new

    if weather_scenario is not None:

        better_weather_factor = 0.2
        worse_weather_factor = -0.2

        Xpred_bw = replace_feature_values(X=Xpred, factor=better_weather_factor)
        Xpred_ww = replace_feature_values(X=Xpred, factor=worse_weather_factor)


        prediction_bw = make_prediction(Xpred=Xpred_bw)
        prediction_bw = np.round(prediction_bw, 0)
        prediction_bw[prediction_bw < 0] = 0

        prediction_ww = make_prediction(Xpred=Xpred_ww)
        prediction_ww = np.round(prediction_ww, 0)
        prediction_ww[prediction_ww < 0] = 0

        return prediction_bw, prediction_ww

    else:
        prediction = make_prediction(Xpred=Xpred)

        # Voorspelling afronden en negatieve waarden veranderen naar nul
        prediction = np.round(prediction, 0)
        prediction[prediction < 0] = 0

        return prediction

