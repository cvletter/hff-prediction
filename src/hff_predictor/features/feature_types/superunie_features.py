import datetime

import hff_predictor.generic.files
import pandas as pd

import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
import hff_predictor.generic.dates as gf


def prep_su_features(input_order_data, prediction_date, train_obs, index_col):

    if type(prediction_date) == str:
        prediction_date = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")

    def rename_cols(input_data, suffix="SU_pct"):
        new_cols = ["{}_{}".format(x, suffix) for x in input_data.columns]
        input_data.columns = new_cols

    if not input_order_data.index.name == index_col:
        input_order_data.reset_index(inplace=True, drop=True)
        input_order_data.set_index(index_col, inplace=True)

    input_order_data.sort_index(ascending=False, inplace=True)

    first_train_date = prediction_date - datetime.timedelta(weeks=train_obs)
    fitting_window = input_order_data.loc[prediction_date:first_train_date]

    fitting_window.reset_index(inplace=True, drop=False)

    products_p_su = fitting_window.groupby(
        [cn.FIRST_DOW, cn.ORGANISATIE], as_index=False
    ).agg({cn.CE_BESTELD: "sum", cn.INKOOP_RECEPT_NM: "nunique"})

    products_p_su_t = pd.DataFrame(
        products_p_su.pivot(
            index=cn.FIRST_DOW, columns=cn.ORGANISATIE, values=cn.INKOOP_RECEPT_NM
        )
    )
    products_p_su_ce = pd.DataFrame(
        products_p_su.pivot(
            index=cn.FIRST_DOW, columns=cn.ORGANISATIE, values=cn.CE_BESTELD
        )
    )

    su_totals = fitting_window.groupby([cn.ORGANISATIE], as_index=False).agg(
        {cn.CE_BESTELD: "sum", cn.INKOOP_RECEPT_NM: "nunique"}
    )

    su_totals.set_index(cn.ORGANISATIE, inplace=True)

    su_totals["pct_total"] = round(
        su_totals[cn.CE_BESTELD] / su_totals[cn.CE_BESTELD].sum(), 3
    )
    su_totals["ce_pp"] = round(
        su_totals[cn.CE_BESTELD] / su_totals[cn.INKOOP_RECEPT_NM], 3
    )
    su_totals["ce_pp_pct"] = round(su_totals["ce_pp"] / su_totals["ce_pp"].sum(), 3)

    su_totals.reset_index(inplace=True, drop=False)

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