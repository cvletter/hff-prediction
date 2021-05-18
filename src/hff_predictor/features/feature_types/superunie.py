import datetime
import pandas as pd
import hff_predictor.config.column_names as cn


def prep_su_features(input_order_data: pd.DataFrame, prediction_date: str, train_obs: int, index_col: str):
    """
    Functie om factoren af te leiden per lid Superunie op basis van orderdata
    :param input_order_data: Ruwe order data
    :param prediction_date: Moment van voorspellen
    :param train_obs: Aantal observaties waar model mee wordt getraind, standaard 70
    :param index_col: Kolom waar data mee wordt geindexeerd
    :return: Dataset met Superunie factoren
    """

    # Vertaal voorspeldatum naar datetime object
    if type(prediction_date) == str:
        prediction_date = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")

    # Hulpfunctie om kolommen te hernoemen met een andere suffix
    def rename_cols(input_data, suffix="SU_pct"):
        new_cols = ["{}_{}".format(x, suffix) for x in input_data.columns]
        input_data.columns = new_cols

    # Zet index kolom goed als dat niet het geval is
    if not input_order_data.index.name == index_col:
        input_order_data.reset_index(inplace=True, drop=True)
        input_order_data.set_index(index_col, inplace=True)

    # Sorteer aflopend (datums)
    input_order_data.sort_index(ascending=False, inplace=True)

    # Bepaal de window waarbinnen de features moeten worden bepaald
    first_train_date = prediction_date - datetime.timedelta(weeks=train_obs)
    fitting_window = input_order_data.loc[prediction_date:first_train_date]

    fitting_window.reset_index(inplace=True, drop=False)

    # Bepaal bestellingen per SU lid, zowel de som als het aantal unique producten
    products_p_su = fitting_window.groupby(
        [cn.FIRST_DOW, cn.ORGANISATIE], as_index=False
    ).agg({cn.CE_BESTELD: "sum", cn.INKOOP_RECEPT_NM: "nunique"})

    # Maak een dataframe met SU leden als de kolommen en aantal unique producten besteld per week als waarden
    products_p_su_t = pd.DataFrame(
        products_p_su.pivot(
            index=cn.FIRST_DOW, columns=cn.ORGANISATIE, values=cn.INKOOP_RECEPT_NM
        )
    )

    # Maak een dataframe met SU leden als de kolommen en aantal besteld per week als waarden
    products_p_su_ce = pd.DataFrame(
        products_p_su.pivot(
            index=cn.FIRST_DOW, columns=cn.ORGANISATIE, values=cn.CE_BESTELD
        )
    )

    # Groepeer per Superunie lid, in deze periode, het totaal aantal bestelde producten en unique producten
    su_totals = fitting_window.groupby([cn.ORGANISATIE], as_index=False).agg(
        {cn.CE_BESTELD: "sum", cn.INKOOP_RECEPT_NM: "nunique"}
    )

    # Zet lid Superunie als index
    su_totals.set_index(cn.ORGANISATIE, inplace=True)

    # Bepaal per lid Superunie het percentage bestellingen van totaal
    su_totals["pct_total"] = round(
        su_totals[cn.CE_BESTELD] / su_totals[cn.CE_BESTELD].sum(), 3
    )
    # Bepaal per lid Superunie het gemiddeld aantal bestellingen per product
    su_totals["ce_pp"] = round(
        su_totals[cn.CE_BESTELD] / su_totals[cn.INKOOP_RECEPT_NM], 3
    )

    # Bepaal hoe groot dit percentage is per lid Superunie, ten opzichte van andere leden
    su_totals["ce_pp_pct"] = round(su_totals["ce_pp"] / su_totals["ce_pp"].sum(), 3)

    su_totals.reset_index(inplace=True, drop=False)

    # Breng DataFrames samen
    su_totals_grouped = pd.merge(
        fitting_window,
        su_totals[[cn.ORGANISATIE, "ce_pp_pct"]],
        how="left",
        left_on=cn.ORGANISATIE,
        right_on=cn.ORGANISATIE,
    )

    su_totals_wk = su_totals_grouped.groupby(
        [cn.FIRST_DOW, cn.INKOOP_RECEPT_NM], as_index=False
    ).agg({cn.CE_BESTELD: "sum", cn.ORGANISATIE: "nunique", "ce_pp_pct": "sum"})

    su_pct = pd.DataFrame(
        su_totals_wk.pivot(
            index=cn.FIRST_DOW, columns=cn.INKOOP_RECEPT_NM, values="ce_pp_pct"
        )
    )

    su_n = pd.DataFrame(
        su_totals_wk.pivot(
            index=cn.FIRST_DOW, columns=cn.INKOOP_RECEPT_NM, values=cn.ORGANISATIE
        )
    )

    rename_cols(input_data=su_pct, suffix="SU_pct")
    rename_cols(input_data=su_n, suffix="SU_count")

    return su_pct.sort_index(ascending=False, inplace=False), su_n.sort_index(
        ascending=False, inplace=False
    )