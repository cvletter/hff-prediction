import pandas as pd
import hff_predictor.config.column_names as cn
import hff_predictor.generic.dates as gf
import hff_predictor.generic.files
import hff_predictor.config.file_management as fm

import logging
LOGGER = logging.getLogger(__name__)


def plus_sales():

    # Location of all Plus data sales
    file_loc = "U:\Productie Voorspelmodel\Input\Bestellingen Hollander"

    raw_data = "hollander_bestellingen.xlsx"  # Orders from the DC to local stores
    raw_sales_data = "hollander_verkopen.xlsx"
    ean_hp = "hollander_ean.xlsx"  # Overview of products and product numbers from Plus
    ean_1b = "1bite_ean.xlsx"  # Overview of products and product numbers from 1BITE
    artikel_overzicht = "artikelen_overzicht.csv" # Combination of both Plus and 1BITE product numbers

    # Import data
    order_data = pd.read_excel(io=file_loc + "\\" + raw_data, sheet_name='661B100')
    ean_data_hp = pd.read_excel(file_loc + "\\" + ean_hp)
    ean_data_1b = pd.read_excel(file_loc + "\\" + ean_1b)
    artikelen = pd.read_csv(file_loc + "\\" + artikel_overzicht, sep=";")

    # Process all sales data into one DataFrame
    sub_files = ["2020 - 1-26", "2020 - 27-53", "2021 - 1-26", "2021 - 27-52"]
    total_sales_data = pd.DataFrame([])

    for sf in sub_files:
        sf_year = sf[:4]
        sales_data = pd.read_excel(io=file_loc + "\\" + raw_sales_data, sheet_name=sf)
        sales_data.drop('Artikelomschrijving', axis=1, inplace=True)

        sales_data.set_index('ArtnrCE', inplace=True)
        sales_data_stacked = pd.DataFrame(sales_data.stack(dropna=False))
        sales_data_stacked.reset_index(inplace=True)
        sales_data_stacked.columns = ['ArtnrCE', cn.WEEK_NUMBER, "plus_sales"]

        sales_data_stacked[cn.WEEK_NUMBER] = sales_data_stacked[cn.WEEK_NUMBER].astype(str) + "-" + sf_year
        gf.add_first_day_week(add_to=sales_data_stacked)
        total_sales_data = total_sales_data.append(sales_data_stacked)

    total_sales_data["ArtnrCE"] += 1

    order_data[cn.WEEK_NUMBER] = order_data["WeekQV"].astype(str) + "-" + order_data["Jaar"].astype(str)
    gf.add_first_day_week(add_to=order_data)

    ean_1b_join = pd.merge(ean_data_1b, artikelen, how="left", left_on="ArtikelNummer", right_on="Artikelen")

    ean_1b_hp = pd.merge(ean_1b_join, ean_data_hp, how="inner", left_on="Artikel EAN CE", right_on="CEAN")
    ean_1b_hp = ean_1b_hp[ean_1b_hp["Plant"] == "Katwijk"]

    selected_columns = ['ArtikelNummer', 'Artikel Naam', 'aantal_pp', 'ArtikelnrPlus',
                        'InkoopRecept', 'InkoopRecept Omschrijving', 'Artikel code omschrijving']

    selected_data = ean_1b_hp[selected_columns]
    selected_data['Artikel code'] = selected_data['Artikel code omschrijving'].str.split(" ").str[0].astype(int)
    join_table = selected_data[['InkoopRecept', 'InkoopRecept Omschrijving',
                                'ArtikelnrPlus', 'aantal_pp', 'Artikel code']]

    join_table.drop_duplicates(inplace=True, keep='first')
    sales_data_join = pd.merge(total_sales_data, join_table, how="inner", left_on="ArtnrCE", right_on="ArtikelnrPlus")
    order_data_join = pd.merge(order_data, join_table, how="inner", left_on="Artikel code", right_on="Artikel code")

    # TODO Juiste kolommen toevoegen voor vertaling naar inkoop recept naam
    # Vertaling maken van HE naar CE
    # Voorraad uitrekenen
    # Koppelen op productniveau
    order_data_tot = order_data_join[[cn.FIRST_DOW, 'TOT', 'Artikel omschrijving', 'InkoopRecept', 'InkoopRecept Omschrijving', 'aantal_pp']]
    order_data_tot["sales_ce"] = order_data_tot["TOT"] * order_data_tot["aantal_pp"]

    order_data_agg = order_data_tot.groupby(['InkoopRecept Omschrijving', cn.FIRST_DOW], as_index=False).agg({'sales_ce': "sum"})

    # Hier wordt de pivot uitgevoerd
    pivoted_data = pd.DataFrame(
        order_data_agg.pivot(
            index=cn.FIRST_DOW, columns='InkoopRecept Omschrijving', values='sales_ce'
        )
    )

    LOGGER.critical("The last available week of data for sales data is {}.".format(pivoted_data.index.max()))

    pivoted_data['total'] = pivoted_data.sum(axis=1)
    pivoted_data.sort_index(ascending=False, inplace=True)

    final_data = pivoted_data.fillna(value=0)

    new_col_names = []
    for i in final_data.columns:
        col_name = "{}_sales".format(i)
        new_col_names.append(col_name)

    final_data.columns = new_col_names

    return final_data


"""order_data = hff_predictor.generic.files.import_temp_file(
    data_loc=fm.ORDER_DATA_ACT_PR_FOLDER,
    set_index=True
)

shared_cols = list(set(final_data.columns).intersection(order_data.columns))

final_data_act = final_data[shared_cols]
order_data_act = order_data[shared_cols]"""