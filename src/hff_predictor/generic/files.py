import datetime
import pickle as pkl
import os
import glob
import pandas as pd
import hff_predictor.config.file_management as fm

import logging
LOGGER = logging.getLogger(__name__)


def read_latest_file(folder: str, file_extension: str):
    """
    Hulp functie om automatisch het laatst oppgeslagen bestand te selecteren in een map
    :param folder: Map waarin moet worden gezocht
    :param file_extension: Type bestand bijv. ".csv"
    :return: Het laatst opgeslagen bestand
    """
    search_in_folder = folder + file_extension
    return max(glob.iglob(pathname=search_in_folder), key=os.path.getctime)


def save_to_pkl(data: pd.DataFrame, file_name: str, folder: str):
    """
    Sla bestand op als Python pickle file
    :param data: Bestand wat moet worden opgelsagen
    :param file_name: Basis naam waar het mee moet worden opgeslagen
    :param folder: Plek waar het moet worden opgeslagen
    :return: Geeft niks terug
    """
    current_time = datetime.datetime.now()
    set_timestamp = "{}{:02d}{:02d}_{}{}".format(
        current_time.year,
        current_time.month,
        current_time.day,
        current_time.hour,
        current_time.minute,
    )


    save_as = "{}/{}_{}.p".format(folder, file_name, set_timestamp)
    search_in_folder = folder + "\*.p"
    current_latest_file = max(glob.iglob(pathname=search_in_folder), key=os.path.getctime)

    for i in glob.glob(search_in_folder):
        if i != current_latest_file:
            os.remove(i)

    with open(save_as, "wb") as f:
        pkl.dump(data, f)

    f.close()


def read_pkl(data_loc: str, file_name: str = None):
    """
    Functie om pickle bestand in te laden
    :param data_loc: Plek waar bestand is opgeslagen
    :param file_name: Naam van bestand
    :return:
    """

    # Als geen bestandsnaam wordt ingegeven, dan pakt het automatisch het laatst opgeslagen bestand (voorkeursoptie)
    if file_name is None:
        import_name = read_latest_file(folder=data_loc, file_extension="\*.p")
    else:
        import_name = "{}\{}".format(data_loc, file_name)

    LOGGER.debug(import_name)

    return pkl.load(open(import_name, "rb"))


def save_to_csv(data: pd.DataFrame, file_name: str, folder: str):
    """
    Schil om pandas to_csv functie om automatisch naam en opslagplek te bepalen
    :param data: Data die moet worden opgeslagen
    :param file_name: Basisnaam
    :param folder: Plek waar het moet worden opgeslagen
    :return: Geeft niks terug
    """
    current_time = datetime.datetime.now()
    set_timestamp = "{}{:02d}{:02d}_{}{}".format(
        current_time.year,
        current_time.month,
        current_time.day,
        current_time.hour,
        current_time.minute,
    )

    save_as = "{}/{}_{}.csv".format(folder, file_name, set_timestamp)
    search_in_folder = folder + "\*.csv"
    current_latest_file = max(glob.iglob(pathname=search_in_folder), key=os.path.getctime)


    if folder != fm.PREDICTIONS_FOLDER:
        for i in glob.glob(search_in_folder):
            if i != current_latest_file:
                os.remove(i)


    data.to_csv(save_as, sep=";", decimal=",")

    logging.debug("Data saved as {}".format(save_as))


def import_temp_file(data_loc: pd.DataFrame, file_name: str = None, set_index: bool = True):
    """
    Functie om snel tijdelijke bestanden in te laden
    :param data_loc: Plek waar data is opgeslagen
    :param file_name: Basisnaam
    :param set_index: Of er een index moet worden gezet, de index is eerste dag van week
    :return: Dataset
    """

    if file_name is None:
        import_name = read_latest_file(folder=data_loc, file_extension="\*.csv")
    else:
        import_name = "{}\{}".format(data_loc, file_name)

    LOGGER.debug("Used file named: {}".format(import_name))

    data = pd.read_csv(import_name, sep=";", decimal=",")

    if set_index:
        data["eerste_dag_week"] = pd.to_datetime(
            data["eerste_dag_week"], format="%Y-%m-%d"
        )
        data.set_index("eerste_dag_week", inplace=True)

    return data
