import pandas as pd
import hff_predictor.config.column_names as cn
import hff_predictor.generic.dates as gf


def prep_holiday_features():

    def add_holiday(holiday, output_df, dates):
        output_df[holiday] = [1 if x in dates else 0 for x in output_df["day"]]

    holiday_dates = pd.DataFrame(
        pd.date_range("2018-01-01", periods=cn.FEATURE_PERIOD_LENGTH, freq="D"), columns=["day"]
    )

    # Christmas
    christmas_dt = pd.to_datetime(
        [
            "2018-12-24",
            "2019-12-23",
            "2020-12-21",
            "2021-12-20",
            "2022-12-19"
        ]
    )

    newyears_dt = pd.to_datetime(
        [
            "2018-12-31",
            "2019-12-30",
            "2020-12-28",
            "2021-12-27",
            "2022-12-26"
        ]
    )

    easter_dt = pd.to_datetime(
        [
            "2018-03-26",
            "2019-04-15",
            "2020-04-06",
            "2021-03-29",
            "2022-04-11"
        ]
    )

    pentecost_dt = pd.to_datetime(
        [
            "2018-05-14",
            "2019-06-03",
            "2020-05-25",
            "2021-05-17",
            "2022-05-30"
        ]
    )

    mothers_dt = pd.to_datetime(
        [
            "2018-05-07",
            "2019-05-06",
            "2020-05-24",
            "2021-05-03",
            "2022-05-02"
        ]
    )

    fathers_dt = pd.to_datetime(
        [
            "2018-06-11",
            "2019-06-10",
            "2020-06-15",
            "2021-06-14",
            "2022-06-13"
        ]
    )

    sinterklaas_dt = pd.to_datetime(
        [
            "2018-11-26",
            "2019-12-02",
            "2020-11-30",
            "2021-11-29",
            "2022-11-28"
        ]
    )

    add_holiday(holiday="christmas", output_df=holiday_dates, dates=christmas_dt)
    add_holiday(holiday="newyears", output_df=holiday_dates, dates=newyears_dt)
    add_holiday(holiday="easter", output_df=holiday_dates, dates=easter_dt)
    add_holiday(holiday="pentecost", output_df=holiday_dates, dates=pentecost_dt)
    add_holiday(holiday="christmas", output_df=holiday_dates, dates=christmas_dt)
    add_holiday(holiday="mothers_day", output_df=holiday_dates, dates=mothers_dt)
    add_holiday(holiday="fathers_day", output_df=holiday_dates, dates=fathers_dt)
    add_holiday(holiday="sinterklaas", output_df=holiday_dates, dates=sinterklaas_dt)

    gf.add_week_year(data=holiday_dates, date_name="day")
    gf.add_first_day_week(
        add_to=holiday_dates, week_col_name=cn.WEEK_NUMBER, set_as_index=True
    )
    holiday_dates.drop("day", axis=1, inplace=True)

    holiday_dates["all_holidays"] = holiday_dates.sum(axis=1)

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
