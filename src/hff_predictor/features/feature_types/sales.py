import pandas as pd
import datetime
import numpy as np
import hff_predictor.config.column_names as cn
import hff_predictor.generic.dates as gf
import hff_predictor.generic.files
import hff_predictor.config.file_management as fm
import hff_predictor.config.prediction_settings as ps

import logging
LOGGER = logging.getLogger(__name__)


def plus_sales(prediction_date: str):

    if type(prediction_date) == str:
        prediction_date = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")

    # Location of all Plus data sales
    file_loc = "U:\Productie Voorspelmodel\Input\Bestellingen Hollander"
    raw_bestel_data = "bestellingen_totaal.csv"

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

    # Stel lijst samen met unieke artikelen besteld vanuit Plus
    bestellingen = pd.read_csv(file_loc + "\\" + raw_bestel_data, sep=",")

    bestellingen_plus = bestellingen[(bestellingen["Organisatie"] == "Hollander Plus")]
    bestellingen_plus = bestellingen_plus[bestellingen_plus["Weekjaar"] >= 2020]

    plus_artikelen = bestellingen_plus[['InkoopRecept', 'InkoopRecept Omschrijving',
                                   'Artikelen', 'Artikelomschrijving', 'Besteld #CE']].drop_duplicates(keep='first')

    plus_art = pd.DataFrame(plus_artikelen.groupby(
        ['InkoopRecept', 'InkoopRecept Omschrijving', 'Artikelen', 'Artikelomschrijving']).agg(
        {'Besteld #CE': 'sum'})).reset_index()

    bestellingen_plus[cn.WEEK_NUMBER] = bestellingen_plus["Week"].astype(str) + "-" + bestellingen_plus["Weekjaar"].astype(str)
    gf.add_first_day_week(add_to=bestellingen_plus)

    plus_bestellingen = pd.DataFrame(bestellingen_plus.groupby(
        ['InkoopRecept Omschrijving', cn.FIRST_DOW]).agg(
        {'Besteld #CE': 'sum'})).reset_index()

    plus_bestellingen = pd.DataFrame(
        plus_bestellingen.pivot(
            index=cn.FIRST_DOW, columns='InkoopRecept Omschrijving', values="Besteld #CE"
        )
    )
    plus_bestellingen[cn.MOD_PROD_SUM] = plus_bestellingen.sum(axis=1)

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

    # Article numbers with EAN
    plus_art_ean = pd.merge(plus_art, ean_data_1b, how="left", left_on="Artikelen", right_on="ArtikelNummer")

    # Find and match outdated product numbers and replace them with most recent
    art_nrs = pd.DataFrame(plus_art_ean.groupby('InkoopRecept Omschrijving').agg({'Artikel EAN CE': 'max'}))
    art_nrs.reset_index(drop=False, inplace=True)
    art_nrs.rename(columns={'Artikel EAN CE': 'Artikel EAN CE_match'}, inplace=True)

    plus_art_ean_2 = pd.merge(plus_art_ean, art_nrs, how="left", left_on="InkoopRecept Omschrijving",
                                right_on="InkoopRecept Omschrijving")

    ean_1b_hp = pd.merge(plus_art_ean_2, ean_data_hp, how="left", left_on="Artikel EAN CE_match", right_on="CEAN")
    ean_1b_hp.drop_duplicates(keep='first', inplace=True)

    selected_columns = ['ArtikelNummer', 'Artikel Naam', 'aantal_pp', 'ArtikelnrPlus',
                        'InkoopRecept', 'InkoopRecept Omschrijving', 'Artikel code omschrijving']

    selected_data = ean_1b_hp[selected_columns]
    selected_data.drop_duplicates(keep='first', inplace=True)
    selected_data.dropna(subset=['Artikel code omschrijving'], inplace=True)

    selected_data['Artikel code'] = selected_data['Artikel code omschrijving'].str.split(" ").str[0].astype(int)
    join_table = selected_data[['InkoopRecept', 'InkoopRecept Omschrijving',
                                'ArtikelnrPlus', 'Artikel code']]

    join_table.drop_duplicates(inplace=True, keep='first')

    sales_data_join = pd.merge(total_sales_data, join_table, how="left", left_on="ArtnrCE", right_on="ArtikelnrPlus")
    order_data_join = pd.merge(order_data, join_table, how="left", left_on="Artikel code", right_on="Artikel code")

    matched = sales_data_join[~sales_data_join["InkoopRecept Omschrijving"].isnull()]
    matched = matched[["Artikelomschrijving", "InkoopRecept", "InkoopRecept Omschrijving"]].drop_duplicates(keep='first')
    sales_data_join.rename(columns={"InkoopRecept Omschrijving": "InkoopRecept Omschrijving_oud",
                                    "InkoopRecept": "InkoopRecept_oud"}, inplace=True)

    sales_data_enriched = pd.merge(sales_data_join,
                                   matched[["Artikelomschrijving", "InkoopRecept", "InkoopRecept Omschrijving"]],
                                   how="left", left_on="Artikelomschrijving",
                                   right_on="Artikelomschrijving")

    def prepare_data(input_data, orders=True):
        if orders:
            prep_data = input_data[[cn.FIRST_DOW, 'TOT', 'InkoopRecept', 'InkoopRecept Omschrijving']]
        else:
            prep_data = input_data[[cn.FIRST_DOW, 'plus_sales', 'InkoopRecept', 'InkoopRecept Omschrijving']]
            prep_data["sales_ce"] = prep_data["plus_sales"]

        order_data_agg = prep_data.groupby(['InkoopRecept Omschrijving', cn.FIRST_DOW], as_index=False).agg({'sales_ce': "sum"})

        # Hier wordt de pivot uitgevoerd
        pivoted_data = pd.DataFrame(
            order_data_agg.pivot(
                index=cn.FIRST_DOW, columns='InkoopRecept Omschrijving', values='sales_ce'
            )
        )

        LOGGER.critical("The last available week of data for sales data is {}.".format(pivoted_data.index.max()))

        pivoted_data[cn.MOD_PROD_SUM] = pivoted_data.sum(axis=1)
        pivoted_data.sort_index(ascending=False, inplace=True)

        final_data = pivoted_data.fillna(value=0)

        new_col_names = []
        subscript = "orders" if orders else "sales"

        for i in final_data.columns:
            col_name = "{}_{}".format(i, subscript)
            new_col_names.append(col_name)

        final_data.columns = new_col_names
        # print(final_data.head())

        return final_data

    # sales_data = prepare_data(input_data=order_data_join, orders=True)
    sales_cons_data_temp = prepare_data(input_data=sales_data_enriched, orders=False)
    # sales_cons_data.to_csv("sales_cons.csv", sep=";", decimal=".")
    # Drop if too many missing values
    sales_cons_data_temp.sort_index(ascending=False, inplace=True)

    first_train_date = prediction_date - datetime.timedelta(weeks=ps.TRAIN_OBS + ps.N_LAGS)
    sales_cons_data = sales_cons_data_temp.loc[prediction_date:first_train_date]

    print("first {}; last {}".format(sales_cons_data.index.min(), sales_cons_data.index.max()))

    def zero_filter(data, days, limit_missing):
        total_cols = data.shape[1]
        selection = data.iloc[:days, :]
        filter1 = selection[selection > 0].isna().sum()
        filter1 = filter1[filter1 <= limit_missing].index
        dropped = total_cols - len(filter1)
        LOGGER.debug("Dropped {} columns of total {} columns".format(dropped, total_cols))
        return data[filter1]

    sales_cons_f1 = zero_filter(data=sales_cons_data, days=5, limit_missing=0)
    sales_cons_f2 = zero_filter(data=sales_cons_f1, days=ps.TRAIN_OBS, limit_missing=5)

    # Additional feature types

    sales_plus_all = {}
    sales_plus_all["plus_sales"] = sales_cons_f2

    def change_col_names(input_data, subscript):
        new_col_names = []
        for i in input_data.columns:
            col_name = "{}_{}".format(i, subscript)
            new_col_names.append(col_name)
        input_data.columns = new_col_names
        return input_data

    sales_plus = sales_cons_f2.sort_index(ascending=True)
    sales_plus_d1 = (sales_plus.diff(1) / sales_plus.shift(1)).sort_index(ascending=False, inplace=False)
    sales_plus_d2 = (sales_plus.diff(2) / sales_plus.shift(2)).sort_index(ascending=False, inplace=False)
    sales_plus_2w = (sales_plus.rolling(2).sum()).sort_index(ascending=False, inplace=False)
    sales_plus_3w = (sales_plus.rolling(3).sum()).sort_index(ascending=False, inplace=False)
    sales_plus_5w = (sales_plus.rolling(5).sum()).sort_index(ascending=False, inplace=False)


    sales_columns = list(set([x[:-6] for x in sales_plus.columns]))
    sales_columns.sort()
    plus_bestellingen_match = plus_bestellingen.loc[sales_plus.index, sales_columns]
    plus_bestellingen_match.columns = sales_plus.columns

    plus_bestellingen_3w = (plus_bestellingen_match.rolling(3).sum()).sort_index(ascending=False, inplace=False)
    plus_bestellingen_2w = (plus_bestellingen_match.rolling(2).sum()).sort_index(ascending=False, inplace=False)
    plus_diff_2w = plus_bestellingen_2w.subtract(sales_plus_2w)
    plus_diff_3w = plus_bestellingen_3w.subtract(sales_plus_3w)


    sales_plus_all['plus_sales_1d'] = change_col_names(sales_plus_d1, subscript="d1")
    sales_plus_all['plus_sales_2d'] = change_col_names(sales_plus_d2, subscript="d2")
    sales_plus_all['plus_sales_3ma'] = change_col_names(sales_plus_3w, subscript="ma3")
    sales_plus_all['plus_sales_5ma'] = change_col_names(sales_plus_5w, subscript="ma5")
    sales_plus_all['plus_sales_2diff'] = change_col_names(plus_diff_2w, subscript="diff2")
    sales_plus_all['plus_sales_3diff'] = change_col_names(plus_diff_3w, subscript="diff3")

    sales_plus_final = sales_plus_all["plus_sales"].join(
        sales_plus_all['plus_sales_1d']).join(
        sales_plus_all['plus_sales_2d']).join(
        sales_plus_all['plus_sales_3ma']).join(
        sales_plus_all['plus_sales_5ma']).join(
        sales_plus_all['plus_sales_2diff']).join(
        sales_plus_all['plus_sales_3diff'].join(
        )
    )

    sales_plus_final = sales_plus_final.iloc[:-4, :]
    sales_plus_final.replace([np.inf, -np.inf], np.nan, inplace=True)

    LOGGER.debug("Added Plus sales data to total feature set.")

    return sales_plus_final
