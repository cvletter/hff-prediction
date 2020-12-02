import datetime
import pickle as pkl

import pandas as pd


def save_to_pkl(data, file_name, folder):
    current_time = datetime.datetime.now()
    set_timestamp = "{}{}{}_{}{}".format(
        current_time.year,
        current_time.month,
        current_time.day,
        current_time.hour,
        current_time.minute,
    )

    save_as = "{}/{}_{}.p".format(folder, file_name, set_timestamp)

    with open(save_as, "wb") as f:
        pkl.dump(data, f)

    f.close()


def read_pkl(file_name, data_loc):
    import_name = "{}/{}".format(data_loc, file_name)
    return pkl.load(open(import_name, "rb"))


def save_to_csv(data, file_name, folder):
    current_time = datetime.datetime.now()
    set_timestamp = "{}{}{}_{}{}".format(
        current_time.year,
        current_time.month,
        current_time.day,
        current_time.hour,
        current_time.minute,
    )

    save_as = "{}/{}_{}.csv".format(folder, file_name, set_timestamp)
    data.to_csv(save_as, sep=";", decimal=",")

    print("Data saved as {}".format(save_as))


def import_temp_file(file_name, data_loc, set_index=True):
    import_name = "{}\{}".format(data_loc, file_name)
    data = pd.read_csv(import_name, sep=";", decimal=",")

    if set_index:
        data["eerste_dag_week"] = pd.to_datetime(
            data["eerste_dag_week"], format="%Y-%m-%d"
        )
        data.set_index("eerste_dag_week", inplace=True)

    return data
