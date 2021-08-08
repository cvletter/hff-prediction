import pandas as pd
import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
import hff_predictor.generic.dates as gf
import numpy as np


file_name = "order_data_unagg_filtered.csv"

data = pd.read_csv (file_name, sep=";", decimal=",")
data['besteldatum'] = pd.to_datetime(data['besteldatum'], format="%Y-%m-%d")
data.drop(data.columns[0], axis=1, inplace=True)
data = data[data['inkooprecept_naam'].notna()]
data = data[data[cn.INKOOP_RECEPT_NM] != '1BITE SAMPLE ETIKET']

gf.add_first_day_week(
    add_to=data, week_col_name=cn.WEEK_NUMBER, set_as_index=True
)

aggregated_data = data.groupby([cn.ORDER_DATE,cn.INKOOP_RECEPT_NM], as_index=False).agg(
    {cn.CE_BESTELD: "sum"}
)



pivoted_data = pd.DataFrame(
    aggregated_data.pivot(
        index=cn.ORDER_DATE, columns=cn.INKOOP_RECEPT_NM, values=cn.CE_BESTELD
    )
)

pivoted_data['Total'] = pivoted_data.sum(axis=1)
pivoted_data.reset_index(inplace=True, drop=False)
pivoted_data['weekday_num'] = pivoted_data['besteldatum'].dt.weekday

days_cond = [
    pivoted_data['weekday_num'] == 0,
    pivoted_data['weekday_num'] == 1,
    pivoted_data['weekday_num'] == 2,
    pivoted_data['weekday_num'] == 3,
    pivoted_data['weekday_num'] == 4,
    pivoted_data['weekday_num'] == 5,
    pivoted_data['weekday_num'] == 6
    ]

days_labels = ['Maandag', 'Dinsdag', 'Woensdag', 'Donderdag', 'Vrijdag', 'Zaterdag', 'Zondag']

pivoted_data['weekday_name'] = np.select(days_cond, days_labels)

weeks_cond = [
pivoted_data['weekday_num'] < 3,
(pivoted_data['weekday_num'] >= 3) & (pivoted_data['weekday_num'] < 5),
pivoted_data['weekday_num'] >= 5
]

week_labels = ['madiwo', 'dovr', 'zazo']

pivoted_data['subweek_name'] = np.select(weeks_cond, week_labels)


]