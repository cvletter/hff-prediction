import datetime

import hff_predictor.generic.files
import pandas as pd

import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
import hff_predictor.generic.dates as gf


def prep_covid_features():
    covid_dates = pd.DataFrame(
        pd.date_range("2018-01-01", periods=cn.FEATURE_PERIOD_LENGTH, freq="D"), columns=["day"]
    )

    persco_1 = pd.to_datetime(["2020-03-09"])
    persco_2_horeca_dicht = pd.to_datetime(["2020-03-16"])
    persco_3_lockdown = pd.to_datetime(["2020-03-23"])
    persco_4 = pd.to_datetime(["2020-04-02"])
    persco_5_scholen_open = pd.to_datetime(["2020-04-21"])
    persco_6_kappers_open = pd.to_datetime(["2020-05-06"])
    # kappers_open = pd.to_datetime(['2020-05-11'])
    persco_7_horeca_open = pd.to_datetime(["2020-05-19"])
    # horeca_open = pd.to_datetime(['2020-06-01'])
    persco_8 = pd.to_datetime(["2020-06-24"])
    # verdere_versoepelingen = pd.to_datetime(['2020-07-01'])
    persco_9 = pd.to_datetime(["2020-08-06"])
    persco_10 = pd.to_datetime(["2020-08-18"])
    persco_11 = pd.to_datetime(["2020-09-01"])
    persco_12_aanscherping1_horeca = pd.to_datetime(["2020-09-25"])
    persco_13_aanscherping2_horeca = pd.to_datetime(["2020-09-28"])
    persco_14 = pd.to_datetime(["2020-10-02"])
    persco_15_horeca_dicht = pd.to_datetime(["2020-10-13"])
    persco_16 = pd.to_datetime(["2020-11-03"])
    persco_17 = pd.to_datetime(["2020-11-17"])
    persco_18 = pd.to_datetime(["2020-12-14"])
    persco_19 = pd.to_datetime(["2021-01-12"])
    persco_20 = pd.to_datetime(["2021-01-20"])
    persco_21 = pd.to_datetime(["2021-02-02"])
    persco_22 = pd.to_datetime(["2021-02-23"])
    persco_23 = pd.to_datetime(["2021-03-09"])
    persco_24 = pd.to_datetime(["2021-03-23"])
    persco_25 = pd.to_datetime(["2021-04-13"])
    persco_26 = pd.to_datetime(["2021-04-20"])

    persconferentie = [
        persco_1,
        persco_2_horeca_dicht,
        persco_3_lockdown,
        persco_4,
        persco_5_scholen_open,
        persco_6_kappers_open,
        persco_7_horeca_open,
        persco_8,
        persco_9,
        persco_10,
        persco_11,
        persco_12_aanscherping1_horeca,
        persco_13_aanscherping2_horeca,
        persco_14,
        persco_15_horeca_dicht,
        persco_16,
        persco_17,
        persco_18,
        persco_19,
        persco_20,
        persco_21,
        persco_22,
        persco_23,
        persco_24,
        persco_25,
        persco_26,
    ]

    negatieve_persconferenties = [
        persco_1,
        persco_2_horeca_dicht,
        persco_3_lockdown,
        persco_10,
        persco_11,
        persco_12_aanscherping1_horeca,
        persco_13_aanscherping2_horeca,
        persco_14,
        persco_15_horeca_dicht,
        persco_16,
        persco_17,
        persco_18,
        persco_19,
        persco_20,
        persco_21,
        persco_22,
        persco_23,
        persco_24,
    ]

    positieve_persconferenties = [
        persco_4,
        persco_5_scholen_open,
        persco_6_kappers_open,
        persco_7_horeca_open,
        persco_8,
        persco_9,
        persco_25,
        persco_26,
    ]

    covid_dates["negatieve_persconf"] = [
        1 if x in negatieve_persconferenties else 0 for x in covid_dates["day"]
    ]
    covid_dates["positieve_persconf"] = [
        1 if x in positieve_persconferenties else 0 for x in covid_dates["day"]
    ]
    covid_dates["persconferentie"] = [
        1 if x in persconferentie else 0 for x in covid_dates["day"]
    ]
    covid_dates["horeca_dicht"] = [
        1
        if (
            (persco_2_horeca_dicht <= x <= persco_7_horeca_open)
            or (persco_15_horeca_dicht <= x <= persco_26)
        )
        else 0
        for x in covid_dates["day"]
    ]

    covid_dates["persco_1"] = [1 if x == persco_1 else 0 for x in covid_dates["day"]]
    covid_dates["persco_2"] = [
        1 if x == persco_2_horeca_dicht else 0 for x in covid_dates["day"]
    ]
    covid_dates["persco_3"] = [
        1 if x == persco_3_lockdown else 0 for x in covid_dates["day"]
    ]
    covid_dates["persco_4"] = [1 if x == persco_4 else 0 for x in covid_dates["day"]]
    covid_dates["persco_5"] = [
        1 if x == persco_5_scholen_open else 0 for x in covid_dates["day"]
    ]
    covid_dates["persco_6"] = [
        1 if x == persco_6_kappers_open else 0 for x in covid_dates["day"]
    ]
    covid_dates["persco_7"] = [
        1 if x == persco_7_horeca_open else 0 for x in covid_dates["day"]
    ]
    covid_dates["persco_8"] = [1 if x == persco_8 else 0 for x in covid_dates["day"]]
    covid_dates["persco_9"] = [1 if x == persco_9 else 0 for x in covid_dates["day"]]
    covid_dates["persco_10"] = [1 if x == persco_10 else 0 for x in covid_dates["day"]]
    covid_dates["persco_11"] = [1 if x == persco_11 else 0 for x in covid_dates["day"]]
    covid_dates["persco_12"] = [
        1 if x == persco_12_aanscherping1_horeca else 0 for x in covid_dates["day"]
    ]
    covid_dates["persco_13"] = [
        1 if x == persco_13_aanscherping2_horeca else 0 for x in covid_dates["day"]
    ]
    covid_dates["persco_14"] = [1 if x == persco_14 else 0 for x in covid_dates["day"]]
    covid_dates["persco_15"] = [
        1 if x == persco_15_horeca_dicht else 0 for x in covid_dates["day"]
    ]

    gf.add_week_year(data=covid_dates, date_name="day")
    gf.add_first_day_week(
        add_to=covid_dates, week_col_name=cn.WEEK_NUMBER, set_as_index=True
    )
    covid_dates.drop("day", axis=1, inplace=True)

    return covid_dates.groupby(cn.FIRST_DOW, as_index=True).max()
