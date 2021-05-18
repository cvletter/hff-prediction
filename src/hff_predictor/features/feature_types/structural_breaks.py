import datetime
import pandas as pd
import hff_predictor.config.column_names as cn
import hff_predictor.generic.dates as gf


def prep_level_shifts() -> pd.DataFrame:
    """
    Functie om structurele verschuivingen in wekelijkse verkopen te kunnen modelleren
    :return: Dataset met geidentificeerde breuken
    """
    # Datum opgeslagen als tekst vertalen naar datetime object
    def str2date(date_str):
        return datetime.datetime.strptime(date_str, "%Y-%m-%d")

    # Verzamelpunt voor de level shift factoren
    level_shifts = pd.DataFrame(
        pd.date_range("2018-01-01", periods=cn.FEATURE_PERIOD_LENGTH, freq="D"), columns=["day"]
    )

    # Level shift periode 1 staat uit om conflict in model fitting te voorkomen
    # level_shifts['period_1'] = [1 if x <= str2date('2019-03-11') else 0 for x in level_shifts['day']]

    # Transitie periode van tijdelijke piek
    level_shifts["a_trans_period_1"] = [
        1 if (str2date("2019-03-18") <= x <= str2date("2019-04-08")) else 0
        for x in level_shifts["day"]
    ]

    # Tweede verschuiving
    level_shifts["b_period_2"] = [
        1 if str2date("2019-04-15") <= x <= str2date("2020-04-27") else 0
        for x in level_shifts["day"]
    ]

    # Transitieperiode met tijdelijke piek
    level_shifts["c_trans_period_2"] = [
        1 if (str2date("2020-05-04") <= x <= str2date("2020-05-25")) else 0
        for x in level_shifts["day"]
    ]

    # Breuk periode COVID
    level_shifts["d_trans_period_2b"] = [
        1 if (str2date("2020-06-01") <= x <= str2date("2020-06-29")) else 0
        for x in level_shifts["day"]
    ]

    # Periode post initiele COVID schok
    level_shifts["e_period_3"] = [
        1 if x >= str2date("2020-06-01") else 0 for x in level_shifts["day"]
    ]

    # Eerste dag van week toevoegen
    gf.add_week_year(data=level_shifts, date_name="day")
    gf.add_first_day_week(
        add_to=level_shifts, week_col_name=cn.WEEK_NUMBER, set_as_index=True
    )
    level_shifts.drop("day", axis=1, inplace=True)

    return level_shifts.groupby(cn.FIRST_DOW, as_index=True).max()
