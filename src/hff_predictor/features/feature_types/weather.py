import hff_predictor.config.column_names as cn
import pandas as pd


def prep_weather_features(input_weer_data: pd.DataFrame, index_col: str = cn.FIRST_DOW) -> pd.DataFrame:
    """
    Functie om ruwe weerdata te bewerken tot factoren
    :param input_weer_data: ruwe weer data
    :param index_col: Kolom op op de indexeren
    :return: Dataset met weerfactoren
    """

    # Zet index kolom goed indien nodig
    if not input_weer_data.index.name == index_col:
        input_weer_data.reset_index(inplace=True, drop=True)
        input_weer_data.set_index(index_col, inplace=True)

    input_weer_data.sort_index(ascending=False, inplace=True)

    # Selecteer weersfactoren
    cols = [cn.TEMP_GEM, cn.ZONUREN]
    weer_data_a = input_weer_data[cols]

    # Neem eeste verschillen van weer
    weer_data_d = weer_data_a.diff(periods=-1)
    weer_data_d.columns = ["d_temperatuur_gem", "d_zonuren"]

    return weer_data_a.join(weer_data_d, how="left").dropna(how="any")
