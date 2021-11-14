import pandas as pd
import hff_predictor.config.column_names as cn
import hff_predictor.generic.dates as gf


def prep_covid_features() -> pd.DataFrame:
    """
    Houdt bij wanneer persconferenties zijn geweest rondom Corona.
    :return: Dataset met indicatoren per persconferentie
    """

    # Dataframe om Covid features aan toe te voegen
    covid_dates = pd.DataFrame(
        pd.date_range("2018-01-01", periods=cn.FEATURE_PERIOD_LENGTH, freq="D"), columns=["day"]
    )

    # Datums van persconferenties
    persco_1 = pd.to_datetime(['2020-03-06'])
    persco_2 = pd.to_datetime(['2020-03-09'])
    persco_3 = pd.to_datetime(['2020-03-12'])
    persco_4 = pd.to_datetime(['2020-03-17'])
    persco_5 = pd.to_datetime(['2020-03-19'])
    persco_6 = pd.to_datetime(['2020-03-20'])
    persco_7 = pd.to_datetime(['2020-03-23'])
    persco_8 = pd.to_datetime(['2020-03-25'])
    persco_9 = pd.to_datetime(['2020-03-27'])
    persco_10 = pd.to_datetime(['2020-03-31'])
    persco_11 = pd.to_datetime(['2020-04-02'])
    persco_12 = pd.to_datetime(['2020-04-07'])
    persco_13 = pd.to_datetime(['2020-04-15'])
    persco_14 = pd.to_datetime(['2020-04-21'])
    persco_15 = pd.to_datetime(['2020-04-29'])
    persco_16 = pd.to_datetime(['2020-05-13'])
    persco_17 = pd.to_datetime(['2020-05-15'])
    persco_18 = pd.to_datetime(['2020-05-19'])
    persco_19 = pd.to_datetime(['2020-05-20'])
    persco_20 = pd.to_datetime(['2020-05-27'])
    persco_21 = pd.to_datetime(['2020-06-3'])
    persco_22 = pd.to_datetime(['2020-06-24'])
    persco_23 = pd.to_datetime(['2020-07-22'])
    persco_24 = pd.to_datetime(['2020-08-6'])
    persco_25 = pd.to_datetime(['2020-08-18'])
    persco_26 = pd.to_datetime(['2020-09-01'])
    persco_27 = pd.to_datetime(['2020-09-18'])
    persco_28 = pd.to_datetime(['2020-09-28'])
    persco_29 = pd.to_datetime(['2020-10-13'])
    persco_30 = pd.to_datetime(['2020-10-27'])
    persco_31 = pd.to_datetime(['2020-11-03'])
    persco_32 = pd.to_datetime(['2020-11-17'])
    persco_33 = pd.to_datetime(['2020-12-08'])
    persco_34 = pd.to_datetime(['2021-01-12'])
    persco_35 = pd.to_datetime(['2021-01-20'])
    persco_36 = pd.to_datetime(['2021-02-02'])
    persco_37 = pd.to_datetime(['2021-02-23'])
    persco_38 = pd.to_datetime(['2021-03-08'])
    persco_39 = pd.to_datetime(['2021-03-23'])
    persco_40 = pd.to_datetime(['2021-04-13'])
    persco_41 = pd.to_datetime(['2021-04-20'])
    persco_42 = pd.to_datetime(['2021-05-11'])
    persco_43 = pd.to_datetime(['2021-05-28'])
    persco_44 = pd.to_datetime(['2021-06-18'])
    persco_45 = pd.to_datetime(['2021-07-09'])
    persco_46 = pd.to_datetime(['2021-07-12'])
    persco_47 = pd.to_datetime(['2021-08-13'])
    persco_48 = pd.to_datetime(['2021-09-14'])
    persco_49 = pd.to_datetime(['2021-11-02'])
    persco_50 = pd.to_datetime(['2021-11-12'])

    # Verzameling van alle persconferenties
    persconferentie = [
        persco_1,
        persco_2,
        persco_3,
        persco_4,
        persco_5,
        persco_6,
        persco_7,
        persco_8,
        persco_9,
        persco_10,
        persco_11,
        persco_12,
        persco_13,
        persco_14,
        persco_15,
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
        persco_27,
        persco_28,
        persco_29,
        persco_30,
        persco_31,
        persco_32,
        persco_33,
        persco_34,
        persco_35,
        persco_36,
        persco_37,
        persco_38,
        persco_39,
        persco_40,
        persco_41,
        persco_42,
        persco_43,
        persco_44,
        persco_45,
        persco_46,
        persco_47,
        persco_48,
        persco_49,
        persco_50
    ]

    # Persconferenties die negatief waren met beperkingen
    negatieve_persconferenties = [
        persco_1,
        persco_2,
        persco_3,
        persco_4,
        persco_5,
        persco_6,
        persco_7,
        persco_8,
        persco_9,
        persco_10,
        persco_11,
        persco_12,
        persco_13,
        persco_23,
        persco_24,
        persco_25,
        persco_26,
        persco_27,
        persco_28,
        persco_29,
        persco_30,
        persco_31,
        persco_32,
        persco_33,
        persco_34,
        persco_35,
        persco_36,
        persco_37,
        persco_38,
        persco_45,
        persco_46,
        persco_49,
        persco_50
    ]

    # Meer hoopvolle persconferenties
    positieve_persconferenties = [
        persco_14,
        persco_15,
        persco_16,
        persco_17,
        persco_18,
        persco_19,
        persco_20,
        persco_21,
        persco_22,
        persco_39,
        persco_40,
        persco_41,
        persco_42,
        persco_43,
        persco_44,
        persco_47,
        persco_48
    ]

    # Toevoegen van de persconferentie groeperingen
    for i in range(1, len(persconferentie) + 1):
        _name = "persco_{}".format(i)
        covid_dates[_name] = [1 if x == persconferentie[i-1] else 0 for x in covid_dates["day"]]

    covid_dates["horeca_dicht"] = [
        1
        if (
            (persco_4 <= x <= persco_18)
            or (persco_29 <= x <= persco_41)
        )
        else 0
        for x in covid_dates["day"]
    ]

    covid_dates["negatieve_persconf"] = [
    1 if x in negatieve_persconferenties else 0
    for x in covid_dates["day"]
        ]
    covid_dates["positieve_persconf"] = [
    1 if x in positieve_persconferenties else 0
    for x in covid_dates["day"]
        ]
    covid_dates["persconferentie"] = [
    1 if x in persconferentie else 0
    for x in covid_dates["day"]]

    # Voeg eerste dag van week toe ter voorbereiding op groepering
    gf.add_week_year(data=covid_dates, date_name="day")
    gf.add_first_day_week(
        add_to=covid_dates, week_col_name=cn.WEEK_NUMBER, set_as_index=True
    )
    # Verwijder dag, niet nodig
    covid_dates.drop("day", axis=1, inplace=True)

    return covid_dates.groupby(cn.FIRST_DOW, as_index=True).max()
