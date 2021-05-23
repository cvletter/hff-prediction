import pandas as pd
import hff_predictor.config.column_names as cn
import hff_predictor.generic.files as fl
import hff_predictor.config.file_management as fm


def first_difference_data(undifferenced_data: pd.DataFrame, delta: int = 1, scale: bool = True) -> pd.DataFrame:
    """
    Transformeert data naar wekelijkse of meerweeklijkse verschillen, in plaats van absolute waarden
    :param undifferenced_data: Ruwe data met absolute waarden
    :param delta: Aantal weken waarover verschil moet worden genomen
    :param scale: Of het verschil geschaald moet worden tot percentage of niet
    :return: Data die bestaat uit verschillen ipv absolute waarden
    """

    # Data in de juiste aflopende volgorde sorteren
    undifferenced_data.sort_index(ascending=True, inplace=True)
    differenced_data = undifferenced_data.diff(periods=delta)
    differenced_data.sort_index(ascending=False, inplace=True)
    undifferenced_data.sort_index(ascending=False, inplace=True)

    # Schalen van data indien geselecteerd
    if scale:
        differenced_data = differenced_data / undifferenced_data.shift(-1)

    return differenced_data[:-delta]


def fill_missing_values(data):
    """
    Vervangt missende (N/A) waarden voor nul
    :param data: Ruwe data met missende waarden
    :return: Geen output, maakt de veranderingen in de ingevoerde data
    """
    data.fillna(value=0, inplace=True)


def create_lags(data: pd.DataFrame, lag_range: int) -> pd.DataFrame:
    """
    Maakt een nieuwe dataset met vertraagde waarde van ingegeven data
    :param data: Onvertraagde, ruwe data
    :param lag_range: Aantal periodes dat data moet worden vertraagd, kan ook een list zijn met meerdere opties
    :return: Dataset met lags
    """

    # Verzekeren dat data in juiste volgorde staat gesorteerd
    data_temp = data.sort_index(ascending=False, inplace=False)
    data_lags = pd.DataFrame(index=data_temp.index)

    # Vertaal de ingegeven lag tot een range
    if type(lag_range) is int:
        lag_range = list(reversed(range(-lag_range, 1)))

    # Haalt de kolomnamen op waarover de lags moeten worden bepaald
    data_columns = data.columns

    # Per kolom / feature worden hier de lags bepaald
    for i in data_columns:
        for l in lag_range:
            if l <= 0:
                _temp_name = "{}_last{}w".format(i, abs(l)) # lag in het verleden
            else:
                _temp_name = "{}_next{}w".format(i, abs(l)) # waarde in de toekomst

            data_lags[_temp_name] = data_temp[i].shift(l)

    return data_lags


def find_rol_products(data: pd.DataFrame, consumentgroep_nrs: pd.DataFrame) -> list:
    """
    Hulpfunctie om per product te bepalen of het een bulk, rol of aankoopproduct is
    :param data: Data met halffabricaatnamen als kolommen
    :param consumentgroep_nrs: Data met halffabricaatnamen en typering
    :return: Lijst met rol producten
    """

    # Haal de kolommen uit een DataFrame als ingegeven data een dataset is en geen list
    if type(data) == pd.DataFrame:
        data_cols = data.columns
    else:
        data_cols = data

    # Bepalen welke van de actieve producten rol producten zijn
    rol_products = consumentgroep_nrs[consumentgroep_nrs[cn.CONSUMENT_GROEP_NR] == 16].index
    return list(set.intersection(set(rol_products), set(data_cols)))


def add_product_number(data: pd.DataFrame) -> pd.DataFrame:
    product_nrs = fl.import_temp_file(data_loc=fm.ORDER_DATA_CG_PR_FOLDER, set_index=False)
    product_nrs = product_nrs[[cn.INKOOP_RECEPT_NM, cn.INKOOP_RECEPT_NR]]
    product_nrs = product_nrs.drop_duplicates(cn.INKOOP_RECEPT_NM, keep='first')
    product_nrs.set_index(cn.INKOOP_RECEPT_NM, inplace=True)

    data_with_prod_nr = data.join(product_nrs, how='left')
    data_with_prod_nr.reset_index(inplace=True)
    data_with_prod_nr = data_with_prod_nr.set_index(cn.INKOOP_RECEPT_NR, inplace=False)
    data_with_prod_nr.rename(columns={'index': cn.INKOOP_RECEPT_NM}, inplace=True)

    return data_with_prod_nr.sort_index(ascending=True)
