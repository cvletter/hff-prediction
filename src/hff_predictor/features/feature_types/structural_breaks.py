import datetime
import pandas as pd
import hff_predictor.config.column_names as cn
import hff_predictor.generic.dates as gf
import hff_predictor.data.transformations as dtr

import hff_predictor.generic.files
import hff_predictor.config.file_management as fm
import hff_predictor.config.prediction_settings as ps
from hff_predictor.features.feature_types import weather, campaigns, covid, \
    seasonal, superunie, structural_breaks, sales

import logging
LOGGER = logging.getLogger(__name__)


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


def turning_points_analysis(prediction_date: str):

    # Vertaal voorspeldatum naar datetime object
    if type(prediction_date) == str:
        prediction_date = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")

    last_train_date = prediction_date - datetime.timedelta(weeks=(ps.TRAIN_OBS+ps.PREDICTION_WINDOW))

    order_data = hff_predictor.generic.files.import_temp_file(
        data_loc=fm.ORDER_DATA_ACT_PR_FOLDER,
        set_index=True
    )

    fitting_window = order_data.loc[prediction_date:last_train_date]

    d1 = dtr.change_col_names((fitting_window / fitting_window.shift(-1)) - 1, subscript='d1')
    d2 = dtr.change_col_names((fitting_window / fitting_window.shift(-2)) - 1, subscript='d2')
    d3 = dtr.change_col_names((fitting_window / fitting_window.shift(-3)) - 1, subscript='d3')

    d1_changes = dtr.change_col_names((d1 / abs(d1)), subscript='ch')
    d2_changes = dtr.change_col_names((d2 / abs(d2)), subscript='ch')
    d3_changes = dtr.change_col_names((d3 / abs(d3)), subscript='ch')

    d1_pos_change = dtr.change_col_names(d1_changes[d1 > 0.1].fillna(0), subscript='pos')
    d1_neg_change = dtr.change_col_names(d1_changes[d1 < -0.05].fillna(0), subscript='neg')

    turning_points = d1.join(
        d2).join(
        d3).join(
        d1_changes).join(
        d2_changes).join(
        d3_changes).join(
        d1_pos_change).join(
        d1_neg_change
    )

    return turning_points
