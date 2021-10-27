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

        sales_data.set_index(['ArtnrCE', 'Artikelomschrijving'], inplace=True)
        sales_data_stacked = pd.DataFrame(sales_data.stack(dropna=False))

        sales_data_stacked.reset_index(inplace=True)

        sales_data_stacked.columns = ['ArtnrCE', 'Artikelomschrijving', cn.WEEK_NUMBER, "plus_sales"]

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

    sales_pre_join = pd.merge(total_sales_data, join_table["ArtikelnrPlus"], how="left", left_on="ArtnrCE", right_on="ArtikelnrPlus")
    art_nrs = pd.DataFrame(sales_pre_join.groupby('Artikelomschrijving').agg({'ArtikelnrPlus': 'max'})).dropna(how='any')
    art_nrs.reset_index(drop=False, inplace=True)
    art_nrs.rename(columns={'ArtikelnrPlus': 'ArtikelnrPlusMatch'}, inplace=True)

    total_sales_data = pd.merge(total_sales_data, art_nrs, how="left", left_on="Artikelomschrijving",
                                right_on="Artikelomschrijving")

    sales_data_join = pd.merge(total_sales_data, join_table, how="left", left_on="ArtikelnrPlusMatch", right_on="ArtikelnrPlus")
    order_data_join = pd.merge(order_data, join_table, how="left", left_on="Artikel code", right_on="Artikel code")

    def prepare_data(input_data, orders=True):
        if orders:
            prep_data = input_data[[cn.FIRST_DOW, 'TOT', 'InkoopRecept', 'InkoopRecept Omschrijving', 'aantal_pp']]
            prep_data["sales_ce"] = prep_data["TOT"] * prep_data["aantal_pp"]
        else:
            prep_data = input_data[[cn.FIRST_DOW, 'plus_sales', 'InkoopRecept', 'InkoopRecept Omschrijving', 'aantal_pp']]
            prep_data["sales_ce"] = prep_data["plus_sales"]

        order_data_agg = prep_data.groupby(['InkoopRecept Omschrijving', cn.FIRST_DOW], as_index=False).agg({'sales_ce': "sum"})

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
        subscript = "sales" if orders else "sales_cons"

        for i in final_data.columns:
            col_name = "{}_{}".format(i, subscript)
            new_col_names.append(col_name)

        final_data.columns = new_col_names
        # print(final_data.head())

        return final_data

    # sales_data = prepare_data(input_data=order_data_join, orders=True)
    sales_cons_data = prepare_data(input_data=sales_data_join, orders=False)

    # Drop if too many missing values
    sales_cons_data.sort_index(ascending=False, inplace=True)

    def zero_filter(data, days, limit_missing):
        total_cols = data.shape[1]
        selection = data.iloc[:days, :]
        filter1 = selection[selection > 0].isna().sum()
        filter1 = filter1[filter1 <= limit_missing].index
        dropped = total_cols - len(filter1)
        LOGGER.debug("Dropped {} columns of total {} columns".format(dropped, total_cols))
        return data[filter1]

    sales_cons_f1 = zero_filter(data=sales_cons_data, days=5, limit_missing=0)
    sales_cons_f2 = zero_filter(data=sales_cons_f1, days=70, limit_missing=5)

    LOGGER.debug("Added Plus sales data to total feature set.")

    return sales_cons_f2
