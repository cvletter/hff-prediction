import datetime
import pandas as pd
import prediction.column_names as cn
import pickle as pkl


def save_to_pkl(data, file_name, folder):
    current_time = datetime.datetime.now()
    set_timestamp = "{}{}{}_{}{}".format(current_time.year,
                                         current_time.month,
                                         current_time.day,
                                         current_time.hour,
                                         current_time.minute)

    save_as = "{}/{}_{}.p".format(folder, file_name, set_timestamp)

    with open(save_as, 'wb') as f:
        pkl.dump(data, f)

    f.close()


def read_pkl(file_name, data_loc):
    import_name = '{}/{}'.format(data_loc, file_name)
    return pkl.load(open(import_name, "rb"))


def save_to_csv(data, file_name, folder):
    current_time = datetime.datetime.now()
    set_timestamp = "{}{}{}_{}{}".format(current_time.year,
                                         current_time.month,
                                         current_time.day,
                                         current_time.hour,
                                         current_time.minute)

    save_as = "{}/{}_{}.csv".format(folder, file_name, set_timestamp)
    data.to_csv(save_as, sep=";", decimal=",")

    print("Data saved as {}".format(save_as))


def import_temp_file(file_name, data_loc, set_index=True):
    import_name = '{}/{}'.format(data_loc, file_name)
    data = pd.read_csv(import_name, sep=";", decimal=",")

    if set_index:
        data['eerste_dag_week'] = pd.to_datetime(data['eerste_dag_week'], format='%Y-%m-%d')
        data.set_index('eerste_dag_week', inplace=True)

    return data


def add_week_year(data, date_name=cn.ORDER_DATE):
    set_date = False

    if data.index.name == date_name:
        set_date = True
        data.reset_index(inplace=True)

    week_num = data[date_name].apply(lambda x: x.isocalendar()[1])
    year_val = data[date_name].apply(lambda x: x.isocalendar()[0])
    data[cn.WEEK_NUMBER] = week_num.astype(str) + "-" + year_val.astype(str)

    if set_date:
        data.set_index(date_name, inplace=True)


def add_first_day_week(add_to, week_col_name=cn.WEEK_NUMBER, set_as_index=False):
    def gen_day_to_week_table():
        date_table = pd.DataFrame(pd.date_range('2018-01-01', periods=200, freq='W-MON'), columns=[cn.FIRST_DOW])

        week_num = date_table[cn.FIRST_DOW].apply(lambda x: x.isocalendar()[1])
        year_val = date_table[cn.FIRST_DOW].apply(lambda x: x.isocalendar()[0])
        date_table[cn.WEEK_NUMBER] = week_num.astype(str) + "-" + year_val.astype(str)

        return date_table

    source_table = gen_day_to_week_table()
    source_table.set_index(cn.WEEK_NUMBER, drop=True, inplace=True)

    if not add_to.index.name == week_col_name:
        add_to.reset_index(inplace=True, drop=True)
        add_to.set_index(week_col_name, inplace=True)

    add_to[cn.FIRST_DOW] = source_table[cn.FIRST_DOW]
    add_to.drop(add_to[add_to[cn.FIRST_DOW].isna()].index, inplace=True)

    if set_as_index:
        add_to.reset_index(inplace=True, drop=True)
        add_to.set_index(cn.FIRST_DOW, inplace=True)
