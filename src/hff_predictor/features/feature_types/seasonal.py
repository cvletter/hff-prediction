import datetime

import hff_predictor.generic.files
import pandas as pd

import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
import hff_predictor.generic.dates as gf


def prep_holiday_features():

    def add_holiday(holiday, output_df, dates):
        output_df[holiday] = [1 if x in dates else 0 for x in output_df["day"]]

    holiday_dates = pd.DataFrame(
        pd.date_range("2018-01-01", periods=cn.FEATURE_PERIOD_LENGTH, freq="D"), columns=["day"]
    )

    # Christmas
    pre_christmas_dt = pd.to_datetime(
        [
            "2018-12-10",
            "2018-12-17",
            "2019-12-09",
            "2019-12-16",
            "2020-12-14",
            "2020-12-21",
            "2021-12-13",
            "2021-12-20"
        ]
    )
    add_holiday(holiday="pre_christmas", output_df=holiday_dates, dates=pre_christmas_dt)


    post_christmas_dt = pd.to_datetime(
        ["2018-12-24", "2018-12-31", "2019-12-30", "2020-12-28", "2021-01-04", "2021-12-27", "2022-01-03"]
    )
    holiday_dates["post_christmas"] = [
        1 if x in post_christmas_dt else 0 for x in holiday_dates["day"]
    ]

    # Pasen, Pinksteren, Goede Vrijdag test

    # Koningsdag

    # Jaarwisseling

    gf.add_week_year(data=holiday_dates, date_name="day")
    gf.add_first_day_week(
        add_to=holiday_dates, week_col_name=cn.WEEK_NUMBER, set_as_index=True
    )
    holiday_dates.drop("day", axis=1, inplace=True)

    return holiday_dates.groupby(cn.FIRST_DOW, as_index=True).max()


def prep_seasonal_features():
    seasonal_dates = pd.DataFrame(
        pd.date_range("2018-01-01", periods=cn.FEATURE_PERIOD_LENGTH, freq="D"), columns=["day"]
    )

    seasonal_dates["winter"] = [
        1 if (x.month <= 2) or (x.month == 12) else 0 for x in seasonal_dates["day"]
    ]
    seasonal_dates["lente"] = [
        1 if 3 <= x.month <= 5 else 0 for x in seasonal_dates["day"]
    ]
    seasonal_dates["zomer"] = [
        1 if 6 <= x.month <= 8 else 0 for x in seasonal_dates["day"]
    ]

    # seasonal_dates["januari"] = [1 if x.month ==1 else 0 for x in seasonal_dates["day"]]
    seasonal_dates["februari"] = [1 if x.month == 2 else 0 for x in seasonal_dates["day"]]
    seasonal_dates["maart"] = [1 if x.month == 3 else 0 for x in seasonal_dates["day"]]
    seasonal_dates["maart"] = [1 if x.month == 3 else 0 for x in seasonal_dates["day"]]
    seasonal_dates["april"] = [1 if x.month == 4 else 0 for x in seasonal_dates["day"]]
    seasonal_dates["mei"] = [1 if x.month == 5 else 0 for x in seasonal_dates["day"]]
    seasonal_dates["juni"] = [1 if x.month == 6 else 0 for x in seasonal_dates["day"]]
    seasonal_dates["juli"] = [1 if x.month == 7 else 0 for x in seasonal_dates["day"]]
    seasonal_dates["augustus"] = [1 if x.month == 8 else 0 for x in seasonal_dates["day"]]
    seasonal_dates["september"] = [1 if x.month == 9 else 0 for x in seasonal_dates["day"]]
    seasonal_dates["oktober"] = [1 if x.month == 10 else 0 for x in seasonal_dates["day"]]
    seasonal_dates["november"] = [1 if x.month == 11 else 0 for x in seasonal_dates["day"]]
    seasonal_dates["december"] = [1 if x.month == 12 else 0 for x in seasonal_dates["day"]]

    gf.add_week_year(data=seasonal_dates, date_name="day")
    gf.add_first_day_week(
        add_to=seasonal_dates, week_col_name=cn.WEEK_NUMBER, set_as_index=True
    )
    seasonal_dates.drop("day", axis=1, inplace=True)

    seasonal_dates = seasonal_dates.groupby(cn.FIRST_DOW, as_index=True).max()
    # seasonal_dates['trend'] = np.arange(1, len(seasonal_dates)+1)

    return seasonal_dates
