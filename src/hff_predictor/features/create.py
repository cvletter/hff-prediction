import pandas as pd
import hff_predictor.generic.files
import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
from hff_predictor.features.feature_types import weather, campaigns, covid, \
    seasonal, superunie, structural_breaks, sales

import logging
LOGGER = logging.getLogger(__name__)


def prep_all_features(
    weather_data_processed: pd.DataFrame,
    order_data_su: pd.DataFrame,
    campaign_data_su: pd.DataFrame,
    prediction_date: str,
    train_obs: int,
    index_col: str = cn.FIRST_DOW,
    import_file: bool = False,
    save_file: bool = False,
) -> pd.DataFrame:
    """
    Functie om alle factoren bij elkaar te brengen
    :param weather_data_processed: Ruwe weer data
    :param order_data_su: Order data
    :param campaign_data_su: Campagne data
    :param prediction_date: Voorspeldatum
    :param train_obs: Aantal observaties om model mee te trainen
    :param index_col: Kolom om te indexeren, vaak eerste dag van de week
    :param import_file: Mogelijkheid om data te importeren
    :param save_file: Mogelijkheid om data op te slaan
    :return: Totale set met voorspelfactoren
    """

    # Importeer weer data als deze al klaar staat
    if import_file:
        weather_data_processed = hff_predictor.generic.files.import_temp_file(
            file_name=weather_data_processed, data_loc=fm.SAVE_LOC, set_index=False
        )

    # Verzamel alle type features in een dict
    all_features = {}
    all_features['weather'] = weather.prep_weather_features(input_weer_data=weather_data_processed)
    all_features['holidays'] = seasonal.prep_holiday_features()
    all_features['seasons'] = seasonal.prep_seasonal_features()
    all_features['campaigns'] = campaigns.prep_campaign_features(campaign_data=campaign_data_su)
    all_features['covid'] = covid.prep_covid_features()
    all_features['breaks'] = structural_breaks.prep_level_shifts()
    all_features['plus_sales'] = sales.plus_sales()
    __, all_features['superunie_n'] = superunie.prep_su_features(
        input_order_data=order_data_su,
        prediction_date=prediction_date,
        train_obs=train_obs,
        index_col=index_col,
    )

    # Sla bestand op indien gewenst
    if save_file:
        hff_predictor.generic.files.save_to_pkl(
            data=all_features,
            file_name=fm.FEATURES_PROCESSED,
            folder=fm.FEATURES_PROCESSED_FOLDER,
        )

    return all_features


def init_create_features():
    # Import weer data
    order_data_su = hff_predictor.generic.files.import_temp_file(
        data_loc=fm.ORDER_DATA_SU_PR_FOLDER,
        set_index=True
    )

    campaign_data = hff_predictor.generic.files.import_temp_file(
        data_loc=fm.CAMPAIGN_DATA_PR_FOLDER,
        set_index=True,
    )

    weather_data = hff_predictor.generic.files.import_temp_file(
        data_loc=fm.WEATHER_DATA_PR_FOLDER,
        set_index=False
    )

    exog_features = prep_all_features(
        weather_data_processed=weather_data,
        order_data_su=order_data_su,
        campaign_data_su=campaign_data,
        prediction_date="2021-04-12",
        train_obs=cn.TRAIN_OBS,
        save_file=True,
        index_col=cn.FIRST_DOW,
    )

