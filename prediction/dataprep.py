import pandas as pd
import numpy as np

RAW_DATA = '/Users/cornelisvletter/Google Drive/HFF/Data/Betellingen met HF-artikel.xlsx'

raw_data = pd.read_excel(RAW_DATA)

def rawdata_processing(data_loc):
    raw_data = pd.read_excel(data_loc)
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


def data_filtering(unfiltered_data):

    print("Unfiltered data: {} lines".format(len(unfiltered_data)))

    filter_1 = unfiltered_data[(unfiltered_data['consumentgroep_nr'].between(14, 16, inclusive=True))]
    print("Bul, rol, aankoop data: {} lines".format(len(filter_1)))

    filter_2 = filter_1[(filter_1['superunielid'] == 'Superunie')]
    print("Bestellingen Superunie leden: {} lines".format(len(filter_2)))

    filter_3 = filter_2[filter_2['besteldatum'] >= pd.Timestamp(year=2018, month=8, day=1)]
    print("Bestellingen na 01/08/2018: {} lines".format(len(filter_3)))

    return filter_3

raw_data_filter = data_filtering(unfiltered_data=raw_data_proc)


def data_aggregation(unaggregated_data, weekly=True):

    agg_col = 'besteldatum'

    if weekly:
        agg_col = 'week_jaar'


        unaggregated_data_week = unaggregated_data[[]]

