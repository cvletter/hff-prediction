import pandas as pd
import hff_predictor.config.column_names as cn
import hff_predictor.generic.dates as gf


def plus_sales():

    file_loc = "U:\Productie Voorspelmodel\Input\Bestellingen Hollander"
    raw_data = "hollander_bestellingen.xlsx"
    ean_data = "hollander_ean.xlsx"

    order_data = pd.read_excel(file_loc + "\\" + raw_data)
    ean_data = pd.read_excel(file_loc + "\\" + ean_data)

    order_data[cn.WEEK_NUMBER] = order_data["WeekQV"].astype(str) + "-" + order_data["Jaar"].astype(str)
    gf.add_first_day_week(add_to=order_data)

    order_data_tot = order_data[[cn.FIRST_DOW, 'TOT', 'Artikel omschrijving']]

    order_data_agg = order_data_tot.groupby(['Artikel omschrijving', cn.FIRST_DOW], as_index=False).agg({'TOT': "sum"})

    # Hier wordt de pivot uitgevoerd
    pivoted_data = pd.DataFrame(
        order_data_agg.pivot(
            index=cn.FIRST_DOW, columns='Artikel omschrijving', values='TOT'
        )
    )

    pivoted_data['total'] = pivoted_data.sum(axis=1)
    pivoted_data.sort_index(ascending=False, inplace=True)

    final_data = pivoted_data.fillna(value=0)

    return final_data

