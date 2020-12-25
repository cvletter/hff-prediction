import datetime
from typing import Union

import hff_predictor.generic.files
import pandas as pd
import numpy as np
from knmy import knmy

import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
import hff_predictor.generic.dates as gf
from hff_predictor.generic.files import read_latest_file


def process_order_data() -> pd.DataFrame:
    """
    :param order_data:
    :return:
    """

    order_data = read_latest_file(folder=fm.ORDER_DATA_FOLDER, file_extension="\*.xlsx")
    raw_data = pd.read_excel(
        order_data,
        dtype={
            "Consumentgroep": str,
            "Inkooprecept": str,
            "VerkString": str,
            "SU": str,
            "Organisatie": str,
            "Weekjaar": str,
            "Week": str,
            "Datum": str,
            "Besteld #CE": int,
        },
    )

    raw_data.rename(
        columns={
            "ConsumentGroep": cn.CONSUMENT_GROEP,
            "InkoopRecept": cn.INKOOP_RECEPT,
            "VerkString": cn.VERKOOP_ART,
            "SU": cn.SELECT_ORG,
            "Organisatie": cn.ORGANISATIE,
            "Weekjaar": cn.WEEK_NUMBER,
            "Week": cn.WEEK,
            "Datum": cn.ORDER_DATE,
            "Besteld #CE": cn.CE_BESTELD,
        },
        errors="raise",
        inplace=True,
    )

    raw_data[cn.ORDER_DATE] = pd.to_datetime(raw_data[cn.ORDER_DATE], format="%Y-%m-%d")

    # Splitsen van consumentgroepnummers, verkoopartikelen en inkoopreceptnummers
    raw_data[cn.CONSUMENT_GROEP_NR] = (
        raw_data[cn.CONSUMENT_GROEP].str.split("-", expand=True, n=1)[0].astype(int)
    )
    raw_data[[cn.VERKOOP_ART_NR, cn.VERKOOP_ART_NM]] = raw_data[
        cn.VERKOOP_ART
    ].str.split(" - ", expand=True, n=1)
    raw_data[[cn.INKOOP_RECEPT_NR, cn.INKOOP_RECEPT_NM]] = raw_data[
        cn.INKOOP_RECEPT
    ].str.split(" - ", expand=True, n=1)

    raw_data[cn.VERKOOP_ART_NR] = raw_data[cn.VERKOOP_ART_NR].astype(int)
    raw_data[cn.INKOOP_RECEPT_NR] = raw_data[cn.INKOOP_RECEPT_NR].astype(int)

    # Voeg hier het weeknummer en jaar toe o.b.v. de besteldatum
    gf.add_week_year(data=raw_data, date_name=cn.ORDER_DATE)

    return raw_data[
        [
            cn.ORDER_DATE,
            cn.WEEK_NUMBER,
            cn.INKOOP_RECEPT_NR,
            cn.INKOOP_RECEPT_NM,
            cn.VERKOOP_ART_NR,
            cn.VERKOOP_ART_NM,
            cn.CE_BESTELD,
            cn.SELECT_ORG,
            cn.ORGANISATIE,
            cn.CONSUMENT_GROEP_NR,
        ]
    ]


def process_weather_data(weekly=True) -> pd.DataFrame:

    today = int(datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d"))
    _, _, _, data_temp_sun = knmy.get_knmi_data(type='daily',
                                                stations=[260],
                                                start=20180801, end=today,
                                                variables=['TEMP', 'SUNR'],
                                                parse=True)

    raw_weer_data = data_temp_sun.loc[1:, :][['YYYYMMDD', 'TG', 'TN', 'TX', 'SQ']]
    raw_weer_data[cn.W_DATE] = pd.to_datetime(raw_weer_data['YYYYMMDD'], format="%Y%m%d")
    raw_weer_data.set_index(cn.W_DATE, inplace=True)
    raw_weer_data.drop('YYYYMMDD', axis=1, inplace=True)

    raw_weer_data.columns = [cn.TEMP_GEM, cn.TEMP_MIN, cn.TEMP_MAX, cn.ZONUREN]
    # Deel alle cijfers door 10 om tot normale waarden voor temp, uren en mm te komen

    raw_weer_data = np.round(raw_weer_data.astype(int) / 10, 1)
    gf.add_week_year(data=raw_weer_data, date_name=cn.W_DATE)

    # Indien data moet worden geaggregeerd naar week
    if weekly:
        raw_weer_data.reset_index(inplace=True)
        raw_weer_data = raw_weer_data.groupby(cn.WEEK_NUMBER, as_index=False).agg(
            {
                cn.TEMP_GEM: "mean",
                cn.TEMP_MIN: "min",
                cn.TEMP_MAX: "max",
                cn.ZONUREN: "sum",
                #cn.NEERSLAG_DUUR: "sum",
                #cn.NEERSLAG_MM: "sum",
            }
        )

        raw_weer_data.columns = [
            cn.WEEK_NUMBER,
            cn.TEMP_GEM,
            cn.TEMP_MIN,
            cn.TEMP_MAX,
            cn.ZONUREN,
            #cn.NEERSLAG_DUUR,
            #cn.NEERSLAG_MM,
        ]

    return raw_weer_data


def process_campaigns() -> pd.DataFrame:

    campaign_data = read_latest_file(folder=fm.CAMPAIGN_DATA_FOLDER, file_extension="\*.csv")
    raw_campaign_data = pd.read_csv(campaign_data, sep=";")
    raw_campaign_data["Datum"] = pd.to_datetime(
        raw_campaign_data["Datum"], format="%d-%m-%Y"
    )
    raw_campaign_data.set_index("Datum", inplace=True)
    raw_campaign_data.index.name = cn.FIRST_DOW

    raw_campaign_data.columns = [
        "Camp_Boni",
        "Camp_Boon",
        "Camp_Coop",
        "Camp_Deen",
        "Camp_Detailresult",
        "Camp_Hoogvliet",
        "Camp_Linders",
        "Camp_Nettorama",
        "Camp_Plus",
        "Camp_Poiesz",
        "Camp_Sligro",
        "Camp_Spar",
        "Camp_Picnic",
        "Camp_Vomar",
        "Camp_Emte",
        "Camp_Total",
    ]

    return raw_campaign_data


# TODO Afmaken column_names
def process_product_status() -> pd.DataFrame:

    product_status_data = read_latest_file(folder=fm.PRODUCT_STATUS_FOLDER, file_extension="\*.xlsx")
    raw_product_status = pd.read_excel(
        product_status_data,
        sheet_name="Blad2",
        dtype={"Nummer": str, "Omschrijving": str, "Geblokkeerd": str},
    ).dropna(how="all")

    raw_product_status.rename(
        columns={
            "Nummer": "inkooprecept_nr",
            "Omschrijving": "inkooprecept_naam",
            "Geblokkeerd": "geblokkeerd",
        },
        errors="raise",
        inplace=True,
    )

    raw_product_status["inkooprecept_nr"] = raw_product_status[
        "inkooprecept_nr"
    ].astype(int)

    raw_product_status.set_index("inkooprecept_nr", inplace=True)

    return raw_product_status


# Toevoegen van de status van een product, 'geblokkeerd' of nog 'actief'
def add_product_status(
    order_data_processed: pd.DataFrame,
    product_status_processed: pd.DataFrame,
    join_col="inkooprecept_nr",
):

    order_data_processed.reset_index(inplace=True)
    order_data_processed.set_index(join_col, inplace=True)

    reset_product_index = False
    product_index = product_status_processed.index.name

    if not product_index == join_col:
        reset_product_index = True
        product_status_processed.reset_index(inplace=True)
        product_status_processed.set_index(join_col, inplace=True)

    order_data_processed["inkooprecept_geblokkeerd"] = product_status_processed[
        "geblokkeerd"
    ]

    if reset_product_index:
        product_status_processed.set_index(product_index, inplace=True)

    order_data_processed.reset_index(inplace=True)


def data_filtering(unfiltered_data: pd.DataFrame, su_filter=True) -> pd.DataFrame:

    print("Unfiltered data: {} lines".format(len(unfiltered_data)))

    # Enkel bulk, rol en aankoopproducten
    filter_1 = unfiltered_data[
        (unfiltered_data["consumentgroep_nr"].between(14, 16, inclusive=True))
    ]
    print("Bul, rol, aankoop data: {} lines".format(len(filter_1)))

    # Enkel bestellingen van leden van de SuperUnie
    if su_filter:
        filter_2 = filter_1[(filter_1[cn.SELECT_ORG] == "Superunie")]
        print("Bestellingen leden: {} lines".format(len(filter_2)))
    else:
        filter_2 = filter_1

    # Bestellingen na 1 augustus 2018, vanaf dat moment bestellingen betrouwbaar
    filter_3 = filter_2[
        filter_2["besteldatum"] >= pd.Timestamp(year=2018, month=8, day=1)
    ]
    print("Bestellingen na 01/08/2018: {} lines".format(len(filter_3)))

    # Enkel actieve producten
    filter_4 = filter_3[filter_3["inkooprecept_geblokkeerd"] == "Nee"]
    print("Actieve producten: {} lines".format(len(filter_4)))

    return filter_4


def data_aggregation(
    filtered_data: pd.DataFrame, weekly=True, exclude_su=True
) -> pd.DataFrame:
    time_agg = "week_jaar" if weekly else "besteldatum"
    product_agg = "ce_besteld"

    group_cols = [time_agg, "inkooprecept_naam", "inkooprecept_nr"]

    if not exclude_su:
        group_cols += ["organisatie"]

    selected_cols = [product_agg] + group_cols

    ungrouped_data = filtered_data[selected_cols]
    aggregated_data = ungrouped_data.groupby(group_cols, as_index=False).agg(
        {product_agg: "sum"}
    )

    if not weekly:
        gf.add_week_year(data=aggregated_data)

    return aggregated_data


def make_pivot(aggregated_data: pd.DataFrame, weekly=True) -> pd.DataFrame:

    date_granularity = cn.FIRST_DOW if weekly else cn.ORDER_DATE

    if aggregated_data.index.name == date_granularity:
        aggregated_data.reset_index(inplace=True, drop=False)

    pivoted_data = pd.DataFrame(
        aggregated_data.pivot(
            index=date_granularity, columns=cn.INKOOP_RECEPT_NM, values=cn.CE_BESTELD
        )
    )

    return pivoted_data.sort_index(ascending=False, inplace=False)


# Selecteert hierdoor alleen producten die zijn verkocht in de week dat de voorspelling wordt gemaakt
def find_active_products(
    raw_product_ts: pd.DataFrame, eval_week=cn.LAST_TRAIN_DATE
) -> [pd.DataFrame, pd.DataFrame]:
    eval_data = raw_product_ts.loc[eval_week].T
    eval_data.drop("week_jaar", inplace=True, errors="ignore")
    all_active_products = eval_data.index
    active_sold_products = eval_data.dropna(how="all").index
    active_not_sold_products = list(
        set(all_active_products) - set(active_sold_products)
    )

    return (
        raw_product_ts[active_sold_products],
        raw_product_ts[active_not_sold_products],
    )


def process_data(
    agg_weekly=True,
    exclude_su=True,
    save_to_csv=False,
) -> [pd.DataFrame, pd.DataFrame]:

    # Importeren van order data
    order_data = process_order_data()

    # Importeren van weer data, op wekelijks niveau
    weer_data = process_weather_data(weekly=agg_weekly)

    gf.add_first_day_week(
        add_to=weer_data, week_col_name=cn.WEEK_NUMBER, set_as_index=True
    )
    weer_data.sort_index(ascending=False, inplace=True)

    # Importeren van product status data
    product_status = process_product_status()

    # Toevoegen van product status
    add_product_status(
        order_data_processed=order_data, product_status_processed=product_status
    )

    # Filteren van besteldata
    order_data_filtered = data_filtering(order_data)

    # Aggregeren van data naar wekelijks niveau en halffabrikaat
    order_data_wk = data_aggregation(
        filtered_data=order_data_filtered, weekly=agg_weekly, exclude_su=exclude_su
    )
    gf.add_first_day_week(
        add_to=order_data_wk, week_col_name=cn.WEEK_NUMBER, set_as_index=True
    )

    order_data_wk_su = data_aggregation(
        filtered_data=order_data_filtered, weekly=agg_weekly, exclude_su=False
    )

    gf.add_first_day_week(
        add_to=order_data_wk_su, week_col_name=cn.WEEK_NUMBER, set_as_index=True
    )

    # Pivoteren van data

    #  Shape: 110 producten, 112 datapunten
    order_data_pivot_wk = make_pivot(aggregated_data=order_data_wk, weekly=True)

    campaign_data = process_campaigns()

    # Bestanden worden hier tussentijds opgeslagen omdat hier nog geen restricties op zijn gelegd m.b.t. een datum
    # Dit maakt het mogelijk om de data later in te laden en de laatste bewerkingen te doen, over een veelvoud aan ver-
    # schillende datums. Elke keer de data opnieuw inladen kost veel tijd (zeker tijdens testen).

    if save_to_csv:
        hff_predictor.generic.files.save_to_csv(
            data=weer_data, file_name=fm.WEATHER_DATA_PREPROCESSED, folder=fm.WEATHER_DATA_PPR_FOLDER
        )
        hff_predictor.generic.files.save_to_csv(
            data=order_data_pivot_wk,
            file_name=fm.ORDER_DATA_PROCESSED,
            folder=fm.ORDER_DATA_PR_FOLDER,
        )
        hff_predictor.generic.files.save_to_csv(
            data=order_data_wk_su, file_name=fm.ORDER_DATA_SU_PREPROCESSED, folder=fm.ORDER_DATA_SU_PPR_FOLDER
        )
        hff_predictor.generic.files.save_to_csv(
            data=campaign_data, file_name=fm.CAMPAIGN_DATA_PROCESSED, folder=fm.CAMPAIGN_DATA_PR_FOLDER
        )

    return order_data_pivot_wk, weer_data, order_data_wk_su, campaign_data


# Wrapping function to do entire data preparation
def data_prep_wrapper(
    prediction_date: datetime.datetime,
    prediction_window: int,
    reload_data=False,
    agg_weekly=True,
    exclude_su=True,
    save_to_csv=False,
) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if type(prediction_date) == str:
        prediction_date = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")

    last_train_date = prediction_date - datetime.timedelta(weeks=prediction_window)

    if reload_data:
        (
            order_data_pivot_wk,
            weather_data,
            order_data_wk_su,
            campaign_data,
        ) = process_data(
            agg_weekly=agg_weekly,
            exclude_su=exclude_su,
            save_to_csv=False
        )

    else:
        order_data_pivot_wk = hff_predictor.generic.files.import_temp_file(
            data_loc=fm.ORDER_DATA_PR_FOLDER, set_index=True
        )

        weather_data = hff_predictor.generic.files.import_temp_file(
            data_loc=fm.WEATHER_DATA_PPR_FOLDER, set_index=True
        )

        order_data_wk_su = hff_predictor.generic.files.import_temp_file(
            data_loc=fm.ORDER_DATA_SU_PPR_FOLDER,
            set_index=True
        )

        campaign_data = hff_predictor.generic.files.import_temp_file(
            data_loc=fm.CAMPAIGN_DATA_PR_FOLDER, set_index=True
        )

    # Actieve producten selecteren: 66 actief; 45 inactief
    order_data_wk_a, order_data_wk_ia = find_active_products(
        raw_product_ts=order_data_pivot_wk, eval_week=last_train_date
    )

    order_data_wk_su_a = order_data_wk_su[
        order_data_wk_su["inkooprecept_naam"].isin(order_data_wk_a.columns)
    ]

    if save_to_csv:
        hff_predictor.generic.files.save_to_csv(
            data=weather_data, file_name=fm.WEATHER_DATA_PROCESSED, folder=fm.WEATHER_DATA_PR_FOLDER
        )
        hff_predictor.generic.files.save_to_csv(
            data=order_data_wk_a,
            file_name=fm.ORDER_DATA_ACT_PROCESSED,
            folder=fm.ORDER_DATA_ACT_PR_FOLDER,
        )
        hff_predictor.generic.files.save_to_csv(
            data=order_data_wk_ia,
            file_name=fm.ORDER_DATA_INACT_PROCESSED,
            folder=fm.ORDER_DATA_INACT_PR_FOLDER,
        )
        hff_predictor.generic.files.save_to_csv(
            data=order_data_wk_su_a,
            file_name=fm.ORDER_DATA_SU_PROCESSED,
            folder=fm.ORDER_DATA_SU_PR_FOLDER,
        )

    return (
        order_data_wk_a,
        order_data_wk_ia,
        weather_data,
        order_data_wk_su_a,
        campaign_data,
    )


def init_prepare_data():
    """
    order_data, weer_data, order_data_su, campaigns = process_data(
        agg_weekly=True,
        exclude_su=True,
        save_to_csv=True,
    )
    """

    order_data_wk_a,  order_data_wk_ia, weather_data, order_data_wk_su_a, campaign_data = data_prep_wrapper(
        prediction_date="2020-10-05",
        prediction_window=2,
        reload_data=True,
        agg_weekly=True,
        exclude_su=True,
        save_to_csv=True
      )

