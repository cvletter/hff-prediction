import pandas as pd
import datetime
import prediction.general_purpose_functions as gf


def add_week_year(data, date_name='besteldatum'):
    set_date = False

    if data.index.name == date_name:
        set_date = True
        data.reset_index(inplace=True)

    week_num = data[date_name].apply(lambda x: x.isocalendar()[1])
    year_val = data[date_name].apply(lambda x: x.isocalendar()[0])
    data['week_jaar'] = week_num.astype(str) + "-" + year_val.astype(str)

    if set_date:
        data.set_index(date_name, inplace=True)


def order_data_processing(order_data_loc):
    raw_data = pd.read_excel(order_data_loc,
                             dtype={'Consumentgroep': str,
                                    'Inkooprecept': str,
                                    'VerkString': str,
                                    'Gebruiken': str,
                                    'Org_ano': str,
                                    'Weekjaar': str,
                                    'Week': str,
                                    'Datum': str,
                                    'Besteld #CE': int})

    raw_data.rename(columns={'ConsumentGroep': 'consumentgroep',
                             'InkoopRecept': 'inkooprecept',
                             'VerkString': 'verkoopartikel',
                             'Gebruiken': 'gebruiken',
                             'Org_ano': 'organisatie',
                             'Weekjaar': 'weekjaar',
                             'Week': 'week',
                             'Datum': 'besteldatum',
                             'Besteld #CE': 'ce_besteld'},
                    errors="raise",
                    inplace=True)

    raw_data['besteldatum'] = pd.to_datetime(raw_data['besteldatum'], format='%Y-%m-%d')

    raw_data['consumentgroep_nr'] = raw_data['consumentgroep'].str.split("-", expand=True, n=1)[0].astype(int)
    raw_data[['verkoopartikel_nr', 'verkoopartikel_naam']] = raw_data['verkoopartikel'].str.split(" - ", expand=True, n=1)
    raw_data[['inkooprecept_nr', 'inkooprecept_naam']] = raw_data['inkooprecept'].str.split(" - ", expand=True, n=1)

    raw_data['verkoopartikel_nr'] = raw_data['verkoopartikel_nr'].astype(int)
    raw_data['inkooprecept_nr'] = raw_data['inkooprecept_nr'].astype(int)

    add_week_year(data=raw_data, date_name='besteldatum')

    return raw_data[['besteldatum',
                     'week_jaar',
                     'inkooprecept_nr',
                     'inkooprecept_naam',
                     'verkoopartikel_nr',
                     'verkoopartikel_naam',
                     'ce_besteld',
                     'gebruiken',
                     'organisatie',
                     'consumentgroep_nr']]


def weer_data_processing(weer_data_loc, weekly=True):
    raw_weer_data = pd.read_csv(weer_data_loc, sep=";")

    raw_weer_data.columns = ['date', 'temperatuur_gem', 'temperatuur_min',
                             'temperatuur_max', 'zonuren', 'neerslag_duur', 'neerslag_mm']

    raw_weer_data['date'] = pd.to_datetime(raw_weer_data['date'], format='%Y%m%d')
    raw_weer_data.set_index('date', inplace=True)

    raw_weer_data = raw_weer_data / 10
    add_week_year(data=raw_weer_data, date_name='date')

    if weekly:
        raw_weer_data.reset_index(inplace=True)
        raw_weer_data = raw_weer_data.groupby('week_jaar', as_index=False).agg({
            'temperatuur_gem': 'mean',
            'temperatuur_min': 'min',
            'temperatuur_max': 'max',
            'zonuren': 'mean',
            'neerslag_duur': 'sum',
            'neerslag_mm': 'sum',
            })

        raw_weer_data.columns = ['date', 'temperatuur_gem', 'temperatuur_min',
                                 'temperatuur_max', 'zonuren', 'neerslag_duur', 'neerslag_mm']

    return raw_weer_data


def first_day_week_table(processed_order_data):

    date_cols = processed_order_data[['week_jaar', 'besteldatum']]
    day_to_week_table = pd.DataFrame(date_cols.groupby(['week_jaar'],
                                               as_index=False).agg({'besteldatum': 'min'})).set_index('week_jaar')
    day_to_week_table.columns = ['eerste_dag_week']

    return day_to_week_table


def add_first_day_week(add_to, source_table, week_col_name='week_jaar'):

    if not add_to.index.name == week_col_name:
        add_to.reset_index(inplace=True, errors='ignore')
        add_to.set_index(week_col_name, inplace=True)

    add_to['eerste_dag_week'] = source_table['eerste_dag_week']


def product_status_processing(product_data_loc):
    raw_product_status = pd.read_excel(product_data_loc,
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

    raw_product_status.set_index('inkooprecept_nr', inplace=True)

    return raw_product_status


def add_product_status(order_data_processed, product_status_processed, join_col='inkooprecept_nr'):

    order_data_processed.reset_index(inplace=True)
    order_data_processed.set_index(join_col, inplace=True)

    reset_product_index = False
    product_index = product_status_processed.index.name

    if not product_index == join_col:
        reset_product_index = True
        product_status_processed.reset_index(inplace=True)
        product_status_processed.set_index(join_col, inplace=True)

    order_data_processed['inkooprecept_geblokkeerd'] = product_status_processed['geblokkeerd']

    if reset_product_index:
        product_status_processed.set_index(product_index, inplace=True)

    order_data_processed.reset_index(inplace=True)


def data_filtering(unfiltered_data, su_filter=True):

    print("Unfiltered data: {} lines".format(len(unfiltered_data)))

    filter_1 = unfiltered_data[(unfiltered_data['consumentgroep_nr'].between(14, 16, inclusive=True))]
    print("Bul, rol, aankoop data: {} lines".format(len(filter_1)))

    if su_filter:
        filter_2 = filter_1[(filter_1['superunielid'] == 'Superunie')]
        print("Bestellingen Superunie leden: {} lines".format(len(filter_2)))

    filter_3 = filter_2[filter_2['besteldatum'] >= pd.Timestamp(year=2018, month=8, day=1)]
    print("Bestellingen na 01/08/2018: {} lines".format(len(filter_3)))

    filter_4 = filter_3[filter_3['inkooprecept_geblokkeerd'] == 'Nee']
    print("Actieve producten: {} lines".format(len(filter_4)))

    return filter_4


def data_aggregation(filtered_data, weekly=True, su=False):
    time_agg = 'week_jaar' if weekly else 'besteldatum'
    product_agg = 'ce_besteld'

    group_cols = [time_agg, 'inkooprecept_naam', 'inkooprecept_nr']

    if su:
        group_cols += ['organisatie']

    selected_cols = [product_agg] + group_cols

    ungrouped_data = filtered_data[selected_cols]
    aggregated_data = ungrouped_data.groupby(group_cols, as_index=False).agg({product_agg: 'sum'})

    if not weekly:
        add_week_year(data=aggregated_data)

    return aggregated_data


def make_pivot(aggregated_data, day_to_week_table, weekly=True):

    date_granularity = 'week_jaar' if weekly else 'besteldatum'

    pivoted_data = pd.DataFrame(aggregated_data.pivot(index=date_granularity,
                                                      columns='inkooprecept_naam',
                                                      values='ce_besteld'))

    if weekly:
        add_first_day_week(add_to=pivoted_data, source_table=day_to_week_table)
        pivoted_data.reset_index(inplace=True)
        pivoted_data.set_index('eerste_dag_week', inplace=True)
        pivoted_data.sort_index()
    else:
        add_week_year(data=pivoted_data, date_name=date_granularity)

    return pivoted_data.sort_index(ascending=False, inplace=False)


def find_active_products(raw_product_ts, eval_week='2020-08-24'):
    eval_data = raw_product_ts.loc[eval_week].T
    eval_data.drop('week_jaar', inplace=True, errors='ignore')
    all_active_products = eval_data.index
    active_sold_products = eval_data.dropna(how='all').index
    active_not_sold_products = list(set(all_active_products) - set(active_sold_products))

    return raw_product_ts[active_sold_products], raw_product_ts[active_not_sold_products]


def select_products_to_predict(active_sold_products, min_obs=70, eval_week='2020-08-24'):
    eval_date = datetime.datetime.strptime(eval_week, "%Y-%m-%d")
    end_date = eval_date - datetime.timedelta(weeks=min_obs)
    fitting_window = active_sold_products.loc[end_date:eval_date]
    obs_count = pd.DataFrame(fitting_window.count())
    obs_count.columns = ['count']

    series_to_model = obs_count[obs_count['count'] >= min_obs].index
    series_not_to_model = obs_count[obs_count['count'] < min_obs].index

    return active_sold_products[series_to_model], active_sold_products[series_not_to_model]


if __name__ == '__main__':
    # Run functions
    RAW_DATA = '/Users/cornelisvletter/Google Drive/HFF/Data/Betellingen met HF-artikel.xlsx'
    PRODUCT_STATUS = '/Users/cornelisvletter/Google Drive/HFF/Data/productstatus.xlsx'
    WEER_DATA = '/Users/cornelisvletter/Google Drive/HFF/Data/knmi_200913_debilt.csv'
    SAVE_LOC = '/Users/cornelisvletter/Google Drive/HFF/Data/Prepared'



    # Importeren van order data
    order_data = order_data_processing(order_data_loc=RAW_DATA)

    # Tabel maken met eerste dag van de week
    first_dow_table = first_day_week_table(processed_order_data=order_data)

    # Importeren van weer data, op wekelijks niveau
    weer_data = weer_data_processing(weer_data_loc=WEER_DATA, weekly=True)

    # Importeren van product status data
    product_status = product_status_processing(product_data_loc=PRODUCT_STATUS)

    # Toevoegen van product status
    add_product_status(order_data_processed=order_data, product_status_processed=product_status)

    # Filteren van besteldata
    order_data_filtered = data_filtering(order_data)

    # Aggregeren van data naar wekelijks niveau en halffabrikaat
    order_data_wk = data_aggregation(filtered_data=order_data_filtered, weekly=True, su=False)

    # Aggregeren van data naar besteldatum niveau en halffabrikaat
    order_data_dg = data_aggregation(filtered_data=order_data_filtered, weekly=False, su=False)

    # Pivoteren van data

    #  Shape: 111 producten, 112 datapunten
    order_data_pivot_wk = make_pivot(aggregated_data=order_data_wk,
                                         day_to_week_table=first_dow_table,
                                         weekly=True)

    #  Shape: 111 producten, 570 datapunten
    order_data_pivot_dg = make_pivot(aggregated_data=order_data_dg,
                                        day_to_week_table=first_dow_table,
                                        weekly=False)

    # Actieve producten selecteren: 66 actief; 45 inactief
    order_data_wk_a, order_data_wk_ia = find_active_products(
        raw_product_ts=order_data_pivot_wk,
        eval_week='2020-08-24')

    gf.save_to_csv(data=order_data_wk_a, file_name='actieve_halffabricaten_wk', folder=SAVE_LOC)
    gf.save_to_csv(data=order_data_wk_ia, file_name='inactieve_halffabricaten_wk', folder=SAVE_LOC)
