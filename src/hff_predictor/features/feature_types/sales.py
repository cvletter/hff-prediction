import pandas as pd
import hff_predictor.config.column_names as cn
import hff_predictor.generic.dates as gf

import logging
LOGGER = logging.getLogger(__name__)


def plus_sales():

    file_loc = "U:\Productie Voorspelmodel\Input\Bestellingen Hollander"
    raw_data = "hollander_bestellingen.xlsx"
    ean_hp = "hollander_ean.xlsx"
    ean_1b = "1bite_ean.xlsx"
    artikel_overzicht = "artikelen_overzicht.csv"

    order_data = pd.read_excel(io=file_loc + "\\" + raw_data, sheet_name='661B100')
    ean_data_hp = pd.read_excel(file_loc + "\\" + ean_hp)
    ean_data_1b = pd.read_excel(file_loc + "\\" + ean_1b)
    artikelen = pd.read_csv(file_loc + "\\" + artikel_overzicht, sep=";")

    order_data[cn.WEEK_NUMBER] = order_data["WeekQV"].astype(str) + "-" + order_data["Jaar"].astype(str)
    gf.add_first_day_week(add_to=order_data)

    ean_1b_join = pd.merge(ean_data_1b, artikelen, how="left", left_on="ArtikelNummer", right_on="Artikelen")

    ean_1b_hp = pd.merge(ean_1b_join, ean_data_hp, how="inner", left_on="Artikel EAN CE", right_on="CEAN")

    # TODO Juiste kolommen toevoegen voor vertaling naar inkoop recept naam
    # Vertaling maken van HE naar CE
    # Voorraad uitrekenen
    # Koppelen op productniveau
    order_data_tot = order_data[[cn.FIRST_DOW, 'TOT', 'Artikel omschrijving']]

    order_data_agg = order_data_tot.groupby(['Artikel omschrijving', cn.FIRST_DOW], as_index=False).agg({'TOT': "sum"})

    # Hier wordt de pivot uitgevoerd
    pivoted_data = pd.DataFrame(
        order_data_agg.pivot(
            index=cn.FIRST_DOW, columns='Artikel omschrijving', values='TOT'
        )
    )

    LOGGER.critical("The last available week of data for sales data is {}.".format(pivoted_data.index.max()))

    pivoted_data['total'] = pivoted_data.sum(axis=1)
    pivoted_data.sort_index(ascending=False, inplace=True)

    final_data = pivoted_data.fillna(value=0)

    return final_data

