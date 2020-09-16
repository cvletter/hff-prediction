import datetime

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
