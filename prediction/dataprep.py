import pandas as pd
import datetime


def rawdata_processing(data_loc):
    raw_data = pd.read_excel(data_loc,
                             dtype={'Consumentgroep': str,
                                    'Inkooprecept':str,
                                    'VerkString': str,
                                    'SU': str,
                                    'Organisatie': str,
                                    'Weekjaar': str,
                                    'Week': str,
                                    'Datum' : str,
                                    'Besteld #CE': int})

    raw_data.rename(columns={'ConsumentGroep' : 'consumentgroep',
                             'InkoopRecept' : 'inkooprecept',
                             'VerkString' : 'verkoopartikel',
                             'SU' : 'superunielid',
                             'Organisatie' : 'organisatie',
                             'Weekjaar' : 'weekjaar',
                             'Week' : 'week',
                             'Datum' : 'besteldatum',
                             'Besteld #CE' : 'ce_besteld'},
                    errors="raise",
                    inplace=True)

    raw_data['besteldatum'] = pd.to_datetime(raw_data['besteldatum'], format='%Y-%m-%d')

    raw_data['consumentgroep_nr'] = raw_data['consumentgroep'].str.split("-", expand=True, n=1)[0].astype(int)
    raw_data[['verkoopartikel_nr', 'verkoopartikel_naam']] = raw_data['verkoopartikel'].str.split(" - ", expand=True, n=1)
    raw_data[['inkooprecept_nr', 'inkooprecept_naam']] = raw_data['inkooprecept'].str.split(" - ", expand=True, n=1)

    raw_data['verkoopartikel_nr'] = raw_data['verkoopartikel_nr'].astype(int)
    raw_data['inkooprecept_nr'] = raw_data['inkooprecept_nr'].astype(int)

    raw_data['weeknummer'] = raw_data['besteldatum'].apply(lambda x: x.isocalendar()[1])
    raw_data['jaar'] = raw_data['besteldatum'].apply(lambda x: x.isocalendar()[0])
    raw_data['week_jaar'] = raw_data['weeknummer'].astype(str) + "-" + raw_data['jaar'].astype(str)

    return raw_data[['besteldatum',
                     'week_jaar',
                     'inkooprecept_nr',
                     'inkooprecept_naam',
                     'verkoopartikel_nr',
                     'verkoopartikel_naam',
                     'ce_besteld',
                     'superunielid',
                     'organisatie',
                     'consumentgroep_nr']]

def create_datetable(raw_data_processed):
    date_cols = raw_data_processed[['week_jaar', 'besteldatum']]
    date_table =  pd.DataFrame(date_cols.groupby(['week_jaar'],
                                               as_index=False).agg({'besteldatum': 'min'})).set_index('week_jaar')
    date_table.columns = ['eerste_dag_week']

    return date_table


def product_status_processing(data_loc):
    raw_product_status = pd.read_excel(data_loc,
                                       sheet_name='Blad2',
                                       dtype={'Nummer': str,
                                              'Omschrijving': str,
                                              'Geblokkeerd': str}).dropna(how='all')

    raw_product_status.rename(columns={'Nummer' : 'inkooprecept_nr',
                                              'Omschrijving' : 'inkooprecept_naam',
                                              'Geblokkeerd' : 'geblokkeerd'},
                                     errors="raise",
                                     inplace=True)

    raw_product_status['inkooprecept_nr'] = raw_product_status['inkooprecept_nr'].astype(int)

    return raw_product_status


def add_product_status(sales_data, product_status):

    sales_tmp = sales_data.set_index('inkooprecept_nr', inplace=False)
    product_tmp = product_status.set_index('inkooprecept_nr', inplace=False)
    sales_tmp['inkooprecept_geblokkeerd'] = product_tmp['geblokkeerd']

    return sales_tmp.reset_index(inplace=False)


def data_filtering(unfiltered_data):

    print("Unfiltered data: {} lines".format(len(unfiltered_data)))

    filter_1 = unfiltered_data[(unfiltered_data['consumentgroep_nr'].between(14, 16, inclusive=True))]
    print("Bul, rol, aankoop data: {} lines".format(len(filter_1)))

    filter_2 = filter_1[(filter_1['superunielid'] == 'Superunie')]
    print("Bestellingen Superunie leden: {} lines".format(len(filter_2)))

    filter_3 = filter_2[filter_2['besteldatum'] >= pd.Timestamp(year=2018, month=8, day=1)]
    print("Bestellingen na 01/08/2018: {} lines".format(len(filter_3)))

    filter_4 = filter_3[filter_3['inkooprecept_geblokkeerd'] == 'Nee']
    print("Actieve producten: {} lines".format(len(filter_4)))

    return filter_4


def data_aggregation(unaggregated_data, weekly=True):

    agg_col = 'week_jaar' if weekly else 'besteldatum'
    data_cols = [agg_col, 'inkooprecept_naam', 'inkooprecept_nr', 'ce_besteld']

    ungrouped_data = unaggregated_data[data_cols]

    return pd.DataFrame(ungrouped_data.groupby([agg_col, 'inkooprecept_naam', 'inkooprecept_nr'],
                                               as_index=False).agg({'ce_besteld': 'sum'}))


def make_pivot(aggregated_data, weekly=True, date_table=[]):

    data_granularity = 'week_jaar' if weekly else 'besteldatum'

    pivoted_data = pd.DataFrame(aggregated_data.pivot(index=data_granularity,
                                       columns='inkooprecept_naam',
                                       values='ce_besteld'))
    if weekly:
        pivoted_data['eerste_dag_week'] = date_table['eerste_dag_week']

    pivoted_data.reset_index(inplace=True)
    return pivoted_data.set_index('eerste_dag_week', inplace=False)


def find_active_products(raw_product_ts, eval_week='2020-08-31'):
    eval_data = raw_product_ts.loc[eval_week].T
    eval_data.drop('week_jaar', inplace=True)
    all_active_products = eval_data.index
    active_sold_products = eval_data.dropna(how='all').index
    active_not_sold_products = list(set(all_active_products) - set(active_sold_products))

    return raw_product_ts[active_sold_products], raw_product_ts[active_not_sold_products]


def select_products_to_predict(active_sold_products, min_obs=70, eval_week='2020-08-31'):
    eval_date = datetime.datetime.strptime(eval_week, "%Y-%m-%d")
    end_date = eval_date - datetime.timedelta(weeks=min_obs)
    fitting_window = active_sold_products.loc[end_date:eval_date]
    obs_count = pd.DataFrame(fitting_window.count())
    obs_count.columns = ['count']

    series_to_model = obs_count[obs_count['count'] >= 70].index
    series_not_to_model = obs_count[obs_count['count'] < 70].index

    return active_sold_products[series_to_model], active_sold_products[series_not_to_model]


# Run functions
RAW_DATA = '/Users/cornelisvletter/Google Drive/HFF/Data/Betellingen met HF-artikel.xlsx'
PRODUCT_STATUS = '/Users/cornelisvletter/Google Drive/HFF/Data/productstatus.xlsx'

raw_data_proc = rawdata_processing(data_loc=RAW_DATA)
date_table = create_datetable(raw_data_proc)
raw_product_status = product_status_processing(data_loc=PRODUCT_STATUS)
raw_data_app = add_product_status(sales_data = raw_data_proc, product_status=raw_product_status)
raw_data_filtered = data_filtering(unfiltered_data=raw_data_proc)
data_aggregated_weekly = data_aggregation(raw_data_filtered, weekly=True)
data_pivot_weekly = make_pivot(data_aggregated_weekly, weekly=True, date_table=date_table)

data_sold_products, data_not_sold_products = find_active_products(raw_product_ts=data_pivot_weekly)
data_to_model, data_not_to_model = select_products_to_predict(data_sold_products)