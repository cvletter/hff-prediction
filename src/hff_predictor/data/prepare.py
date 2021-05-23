import datetime
import hff_predictor.generic.files
import pandas as pd
import numpy as np
import requests as re
import zipfile as zf
import os
import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
import hff_predictor.generic.dates as gf
from hff_predictor.generic.files import read_latest_file

import logging

LOGGER = logging.getLogger(__name__)


def process_order_data() -> pd.DataFrame:
    """Importeert automatisch laatst beschikbaar gemaakte Excel bestand met orderdata, maakt gebruik van data export uit
    Qlikview.
    :return: Verwerking van ruwe data, voegt datum toe, hernoemt variabelen en maakt ze geschikt voor verder gebruik.
    Hier wordt nog geen data verwijderd.
    """

    order_data = read_latest_file(folder=fm.ORDER_DATA_FOLDER, file_extension="\*.xlsx")

    raw_data = pd.read_excel(
        order_data,
        dtype={
            "Artikelgroep": int,
            "ConsGrp Naam": str,
            "Organisatie": str,
            "Superunie": str,
            "InkoopRecept": int,
            "InkoopRecept Omschrijving": str,
            "Weekjaar": str,
            "Periode": str,
            "Week": int,
            "Leverdatum": str,
            "Order": str,
            "Artikelen": str,
            "Artikelomschrijving": str,
            "Besteld #CE": int,
        },
    )

    raw_data.rename(
        columns={
            "Artikelgroep": cn.CONSUMENT_GROEP_NR,
            "ConsGrp Naam": cn.CONSUMENT_GROEP,
            "InkoopRecept": cn.INKOOP_RECEPT_NR,
            "InkoopRecept Omschrijving": cn.INKOOP_RECEPT_NM,
            "Artikelen": cn.VERKOOP_ART_NR,
            "Artikelomschrijving": cn.VERKOOP_ART_NM,
            "Superunie": cn.SELECT_ORG,
            "Organisatie": cn.ORGANISATIE,
            "Weekjaar": cn.WEEK_NUMBER,
            "Week": cn.WEEK,
            "Leverdatum": cn.ORDER_DATE,
            "Besteld #CE": cn.CE_BESTELD,
        },
        errors="raise",
        inplace=True,
    )

    raw_data[cn.ORDER_DATE] = pd.to_datetime(raw_data[cn.ORDER_DATE], format="%Y-%m-%d")

    raw_data[cn.VERKOOP_ART_NR] = raw_data[cn.VERKOOP_ART_NR].astype(int)
    raw_data[cn.INKOOP_RECEPT_NR] = raw_data[cn.INKOOP_RECEPT_NR].astype(int)

    # Voeg hier het weeknummer en jaar toe o.b.v. de besteldatum
    gf.add_week_year(data=raw_data, date_name=cn.ORDER_DATE)

    logging.debug("Loaded data with orders between {} and {}".format(raw_data[cn.ORDER_DATE].min(),
                                                                     raw_data[cn.ORDER_DATE].max()))

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
    """
    Automatische data import van weerdata van het KNMI. Maakt gebruik van module 'knmy' om weerinformatie op te halen.
    De data wordt opgehaald over een periode van 1 aug. 2018 tot de dag waarop dit script wordt uigevoerd.
    Merk op dat neerslagcijfers achterwege zijn gelaten, deze worden met een te grote vertraging beschikbaar gemaakt.
    :param weekly: Optie om dagelijkse cijfers te aggregeren naar wekelijkse cijfers.
    :return: Verwerkte data m.b.t. het weer zoals temperatuur en zonuren.
    """

    # Bepalen van datum 'vandaag'
    today = int(datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d"))

    # Tijdelijke oplossing van ophalen weerdata
    temp_weather_name = 'current_weather_data.zip'

    url = "https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/daggegevens/etmgeg_260.zip"
    r = re.get(url, allow_redirects=True)
    zfile = open(temp_weather_name, 'wb')
    zfile.write(r.content)
    zfile.close()

    weer_data_zip = zf.ZipFile(temp_weather_name)  # having First.csv zipped file.
    data_download = pd.read_csv(weer_data_zip.open('etmgeg_260.txt'), skiprows=47, delimiter=",", low_memory=False)
    weer_data_zip.close()

    data_download[cn.W_DATE] = pd.to_datetime(data_download['YYYYMMDD'], format='%Y%m%d')
    data_download = data_download[data_download[cn.W_DATE] >= "2018-08-01"]

    logging.debug("Downloaded weather data between {} and {}".format(data_download[cn.W_DATE].min(),
                                                                    data_download[cn.W_DATE].max()))

    data_download.rename(columns=lambda x: x.strip(), inplace=True)
    raw_weer_data = data_download.loc[1:, :][[cn.W_DATE, 'TG', 'TN', 'TX', 'SQ', 'RH']]

    os.remove(temp_weather_name)

    raw_weer_data.set_index(cn.W_DATE, inplace=True)
    raw_weer_data.columns = [cn.TEMP_GEM, cn.TEMP_MIN, cn.TEMP_MAX, cn.ZONUREN, cn.NEERSLAG_MM]

    for c in raw_weer_data.columns:
        if raw_weer_data[c].dtype == 'O':
            raw_weer_data[c] = raw_weer_data[c].str.strip().replace('', np.NaN)
            raw_weer_data[c] = raw_weer_data[c].fillna(method='backfill')
            raw_weer_data[c] = raw_weer_data[c].astype(int)

    # Deel alle cijfers door 10 om tot normale waarden voor temp, uren en mm te komen
    raw_weer_data = np.round(raw_weer_data.astype(int) / 10, 1)

    # Voeg weeknumemr toe
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
                cn.NEERSLAG_MM: "sum",
            }
        )

        raw_weer_data.columns = [
            cn.WEEK_NUMBER,
            cn.TEMP_GEM,
            cn.TEMP_MIN,
            cn.TEMP_MAX,
            cn.ZONUREN,
            cn.NEERSLAG_MM,
        ]

    return raw_weer_data


def process_campaigns() -> pd.DataFrame:
    """
    Importeert en verwerkt automatisch de (verwerkte) data m.b.t. de weken wanneer tapas in de actie zijn bij een
    lid van de superunie. Deze data vereist voorverwerking en kan niet ruw worden ingeladen. Zie desbestreffende map
    voor een voorbeeld. Selecteert wel automatisch het laatst beschikbare bestand.
    :return: Verwerkte campagne datums, per supermarkt.
    """

    # Importeren van ruwe data, datum leesbaar maken
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


def process_product_status() -> pd.DataFrame:
    """
    Importeert en verwerkt automatisch de (verwerkte) data m.b.t. welke producten actief zijn. Deze data vereist
    voorverwerking en kan niet ruw worden ingeladen. Zie desbestreffende map voor een voorbeeld. Selecteert wel
    automatisch het laatst beschikbare bestand.

    :return: Verwerkte productstatus data, klaar om gekoppeld te worden aan order data.
    """

    # Import van ruwe data
    product_status_data = read_latest_file(folder=fm.PRODUCT_STATUS_FOLDER, file_extension="\*.xlsx")
    raw_product_status = pd.read_excel(
        product_status_data,
        sheet_name="Inkoopartikelen",
        dtype={"Artikel": str, "Block ID": str, "Blocktekst": str},
    ).dropna(how="all")

    raw_product_status.rename(
        columns={
            "Artikel": "inkooprecept_nr",
            "Block ID": "geblokkeerd",
            "Blocktekst": "blocktekst",
        },
        errors="raise",
        inplace=True,
    )

    raw_product_status.drop("blocktekst", axis=1, inplace=True)

    raw_product_status.sort_values(by=["inkooprecept_nr", "geblokkeerd"], ascending=True, inplace=True)
    raw_product_status.drop_duplicates(subset="inkooprecept_nr", keep="first", inplace=True)
    raw_product_status["inkooprecept_nr"] = raw_product_status[
        "inkooprecept_nr"
    ].astype(int)

    raw_product_status["geblokkeerd"] = raw_product_status[
        "geblokkeerd"
    ].astype(int)

    raw_product_status.set_index("inkooprecept_nr", inplace=True)
    raw_product_status["geblokkeerd"].replace({-1: "Ja", 0: "Nee"}, inplace=True)

    return raw_product_status


def add_product_status(
        order_data_processed: pd.DataFrame,
        product_status_processed: pd.DataFrame,
        join_col="inkooprecept_nr",
):
    """
    Functie om product status toe te voegen aan order data, m.a.w. is het product nog actief of niet.
    :param order_data_processed: Verwerkte order data
    :param product_status_processed: Verwerkte order status data
    :param join_col: Kolom waarop beide bronnen met elkaar kunnen worden gevoegd
    :return: Geeft niets terug, voegt kolom toe aan 'order_data_processed'
    """

    # Tijdelijk de 'join_col' instellen als index om makkelijk bronnen bij elkaar te voegen
    order_data_processed.reset_index(inplace=True)
    order_data_processed.set_index(join_col, inplace=True)

    reset_product_index = False
    product_index = product_status_processed.index.name

    if not product_index == join_col:
        reset_product_index = True
        product_status_processed.reset_index(inplace=True)
        product_status_processed.set_index(join_col, inplace=True)

    # Hier wordt de productstatus toegevoegd
    order_data_processed["inkooprecept_geblokkeerd"] = product_status_processed[
        "geblokkeerd"
    ]

    if reset_product_index:
        product_status_processed.set_index(product_index, inplace=True)

    order_data_processed.reset_index(inplace=True)


def data_filtering(unfiltered_data: pd.DataFrame, su_filter=True) -> pd.DataFrame:
    """
    Filteren en verwijderen van data om tot gewenste data selectie te komen
    :param unfiltered_data: Verwerkte, maar ongefilterde data
    :param su_filter: Als deze filter op 'true' staat worden orderregels die niet van een SU lid komen weggelaten
    :return: Gefilterde data
    """

    logging.debug("Unfiltered data: {} lines".format(len(unfiltered_data)))

    # Selecteert enkel de bulk, rol en aankoopproducten, corresponderen met nummers 14-16
    filter_1 = unfiltered_data[
        (unfiltered_data["consumentgroep_nr"].between(14, 16, inclusive=True))
    ]
    logging.debug("Bul, rol, aankoop data: {} lines".format(len(filter_1)))

    # Enkel bestellingen van leden van de SuperUnie
    if su_filter:
        filter_2 = filter_1[(filter_1[cn.SELECT_ORG] == "Superunie")]
        logging.debug("Bestellingen leden: {} lines".format(len(filter_2)))
    else:
        filter_2 = filter_1

    # Bestellingen na 1 augustus 2018, vanaf dat moment bestellingen betrouwbaar
    filter_3 = filter_2[
        filter_2["besteldatum"] >= pd.Timestamp(year=2018, month=8, day=1)
        ]
    logging.debug("Bestellingen na 01/08/2018: {} lines".format(len(filter_3)))

    # Enkel actieve producten
    #filter_4 = filter_3[filter_3["inkooprecept_geblokkeerd"] == "Nee"]
    #print("Actieve producten: {} lines".format(len(filter_4)))

    return filter_3


def data_aggregation(
        filtered_data: pd.DataFrame, weekly=True, exclude_su=True
) -> pd.DataFrame:
    """
    Order data wordt in principe per bestelling aangeleverd, deze functie wordt gebruikt om deze orders tea aggregeren
    tot weekniveau
    :param filtered_data: Gefilterde, niet geaggregeerde data
    :param weekly: Data aggregeren naar wekelijks niveau
    :param exclude_su: Als deze op true staat betekent dat data alle orders per week worden geaggregeerd,
    hoeveel een lid van de SU heeft besteld is niet meer herleidbaar
    :return: Geaggregeerde data
    """

    # Te gebruiken kolom om data op gewenste niveau te aggregeren
    time_agg = cn.WEEK_NUMBER if weekly else cn.ORDER_DATE
    product_agg = cn.CE_BESTELD

    group_cols = [time_agg, cn.INKOOP_RECEPT_NM, cn.INKOOP_RECEPT_NR]

    if not exclude_su:
        group_cols += [cn.ORGANISATIE]

    # Aggregeren gebeurt om een zo klein mogelijke selectie van de data
    selected_cols = [product_agg] + group_cols

    ungrouped_data = filtered_data[selected_cols]

    # Aggregatie van de data, dit leidt tot totale verkopen per halffabrikaat per week
    aggregated_data = ungrouped_data.groupby(group_cols, as_index=False).agg(
        {product_agg: "sum"}
    )

    if not weekly:
        gf.add_week_year(data=aggregated_data)

    return aggregated_data


def make_pivot(aggregated_data: pd.DataFrame, weekly=True) -> pd.DataFrame:
    """
    De geaggregeerde data is nu nog gestapeld, de gewenste structuur is dat elk product een eigen kolom heeft en
    de rijen de bestellingen per week laten zien (dit is pivoteren)
    :param aggregated_data: Geaggregeerde data
    :param weekly: Helpt bepalen welke kolom moet worden gebruikt om te pivoteren
    :return:
    """
    date_granularity = cn.FIRST_DOW if weekly else cn.ORDER_DATE

    if aggregated_data.index.name == date_granularity:
        aggregated_data.reset_index(inplace=True, drop=False)

    # Hier wordt de pivot uitgevoerd
    pivoted_data = pd.DataFrame(
        aggregated_data.pivot(
            index=date_granularity, columns=cn.INKOOP_RECEPT_NM, values=cn.CE_BESTELD
        )
    )

    # Data wordt gesorteerd teruggestuurd
    return pivoted_data.sort_index(ascending=False, inplace=False)


# Selecteert hierdoor alleen producten die zijn verkocht in de week dat de voorspelling wordt gemaakt
def find_active_products(
        raw_product_ts: pd.DataFrame, eval_week=cn.LAST_TRAIN_DATE
) -> [pd.DataFrame, pd.DataFrame]:
    """
    Deze functie bepaalt welke producten afgelopen week nog zijn verkocht en waar dus een voorspelling
    van kan worden gemaakt
    :param raw_product_ts: Vewerkte en gepivoteerde data
    :param eval_week: De laatste week die kan worden gebruikt om een voorspelling te maken
    :return: Twee bestanden, een met 'actieve' producten en een met 'inactieve' producten

    """

    # Isoleert de rij met data van producten in de week waar een order moet zijn gemaakt
    eval_data = raw_product_ts.loc[eval_week].T
    eval_data.drop("week_jaar", inplace=True, errors="ignore")

    # Evalueert hier welke producten een waarde beschikbaar hebben, m.a.w. bestellingen
    all_active_products = eval_data.index
    active_sold_products = eval_data.dropna(how="all").index

    # Inactieve producten worden bepaald door het verschil te nemen van totaal en actieve producten
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
    """
    Dit is de functie waar alles bij elkaar wordt gebracht tot en met de pivot. Dit betekent in feite dat alle functies
    worden uitgevoerd om de data te verwerken, waar nog geen restricties worden gelegd op periode waairn data
    beschikbaar moet zijn
    :param agg_weekly: Wekelijks aggregeren
    :param exclude_su: Aggrgeren over SU of niet
    :param save_to_csv: Opslaan van de resultaten
    :return:
    """

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

    # Sla op tot welke categorie elk product behoort: Bulk, rol of aankoop

    product_consumentgroep = order_data[['inkooprecept_naam', 'inkooprecept_nr', 'consumentgroep_nr']].drop_duplicates(
        keep='first')

    hff_predictor.generic.files.save_to_csv(
        data=product_consumentgroep, file_name=fm.PRODUCT_CONSUMENTGROEP_NR, folder=fm.ORDER_DATA_CG_PR_FOLDER
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

    # Pivoteren van data; bevat ongeveer (jan '21): 110 producten, 112 datapunten
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
        prediction_date:str,
        prediction_window: int,
        reload_data=False,
        agg_weekly=True,
        exclude_su=True,
        save_to_csv=False,
) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """

    :param prediction_date:
    :param prediction_window:
    :param reload_data:
    :param agg_weekly:
    :param exclude_su:
    :param save_to_csv:
    :return:
    """
    if type(prediction_date) == str:
        prediction_date = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")

    last_train_date = prediction_date - datetime.timedelta(weeks=prediction_window)

    LOGGER.debug("The value for reload_data in data prep wrapper is: {}.".format(reload_data))

    if reload_data:
        (
            order_data_pivot_wk,
            weather_data,
            order_data_wk_su,
            campaign_data,
        ) = process_data(
            agg_weekly=agg_weekly,
            exclude_su=exclude_su,
            save_to_csv=True
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

    """order_data, weer_data, order_data_su, campaigns = process_data(
        agg_weekly=True,
        exclude_su=True,
        save_to_csv=False,
    )"""


    order_data_wk_a, order_data_wk_ia, weather_data, order_data_wk_su_a, campaign_data = data_prep_wrapper(
        prediction_date="2021-04-12",
        prediction_window=2,
        reload_data=True,
        agg_weekly=True,
        exclude_su=True,
        save_to_csv=True
    )
