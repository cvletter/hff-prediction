import pandas as pd
import numpy as np

RAW_DATA = '/Users/cornelisvletter/Google Drive/HFF/Data/Betellingen met HF-artikel.xlsx'
PRODUCT_STATUS = '/Users/cornelisvletter/Google Drive/HFF/Data/productstatus.xlsx'

raw_data = pd.read_excel(RAW_DATA)

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

raw_data_proc = rawdata_processing(data_loc=RAW_DATA)

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

raw_product_status = product_status_processing(data_loc=PRODUCT_STATUS)
raw_data_app = add_product_status(sales_data = raw_data_proc, product_status=raw_product_status)

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

raw_data_filtered = data_filtering(unfiltered_data=raw_data_proc)


def data_aggregation(unaggregated_data, weekly=True):

    agg_col = ['week_jaar' if weekly else 'besteldatum']


        unaggregated_data_week = unaggregated_da



product_state = pd.read_excel(PRODUCT_STATE,sheet_name='Blad2')

