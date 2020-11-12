import pandas as pd
import numpy as np
import prediction.column_names as cn
import prediction.file_management as fm
import prediction.general_purpose_functions as gf
import datetime


def prep_su_features(input_order_data, prediction_date, train_obs, index_col):

    if type(prediction_date) == str:
        prediction_date = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")

    def rename_cols(input_data, suffix='SU_pct'):
        new_cols = ["{}_{}".format(x, suffix) for x in input_data.columns]
        input_data.columns = new_cols

    if not input_order_data.index.name == index_col:
        input_order_data.reset_index(inplace=True, drop=True)
        input_order_data.set_index(index_col, inplace=True)

    input_order_data.sort_index(ascending=False, inplace=True)

    first_train_date = prediction_date - datetime.timedelta(weeks=train_obs)
    fitting_window = input_order_data.loc[prediction_date:first_train_date]

    fitting_window.reset_index(inplace=True, drop=False)

    su_totals = fitting_window.groupby([cn.ORGANISATIE], as_index=False).agg(
        {cn.CE_BESTELD: 'sum', cn.INKOOP_RECEPT_NM: 'nunique'})

    su_totals.set_index(cn.ORGANISATIE, inplace=True)

    su_totals['pct_total'] = round(su_totals[cn.CE_BESTELD] / su_totals[cn.CE_BESTELD].sum(), 3)
    su_totals['ce_pp'] = round(su_totals[cn.CE_BESTELD] / su_totals[cn.INKOOP_RECEPT_NM], 3)
    su_totals['ce_pp_pct'] = round(su_totals['ce_pp'] / su_totals['ce_pp'].sum(), 3)

    su_totals.reset_index(inplace=True, drop=False)

    su_totals_grouped = pd.merge(fitting_window, su_totals[[cn.ORGANISATIE, 'ce_pp_pct']], how='left',
                                 left_on=cn.ORGANISATIE, right_on=cn.ORGANISATIE)

    su_totals_wk = su_totals_grouped.groupby([cn.FIRST_DOW, cn.INKOOP_RECEPT_NM], as_index=False).agg(
        {cn.CE_BESTELD: 'sum', cn.ORGANISATIE: 'nunique', 'ce_pp_pct': 'sum'})

    su_pct = pd.DataFrame(su_totals_wk.pivot(index=cn.FIRST_DOW,
                                             columns=cn.INKOOP_RECEPT_NM,
                                             values='ce_pp_pct'))

    su_n = pd.DataFrame(su_totals_wk.pivot(index=cn.FIRST_DOW,
                                           columns=cn.INKOOP_RECEPT_NM,
                                           values=cn.ORGANISATIE))

    rename_cols(input_data=su_pct, suffix='SU_pct')
    rename_cols(input_data=su_n, suffix='SU_count')

    return su_pct.sort_index(ascending=False, inplace=False), su_n.sort_index(ascending=False, inplace=False)


def prep_weather_features(input_weer_data, index_col=cn.FIRST_DOW):
    if not input_weer_data.index.name == index_col:
        input_weer_data.reset_index(inplace=True, drop=True)
        input_weer_data.set_index(index_col, inplace=True)

    input_weer_data.sort_index(ascending=False, inplace=True)

    cols = [cn.TEMP_GEM, cn.NEERSLAG_MM, cn.ZONUREN]

    weer_data_a = input_weer_data[cols]
    weer_data_d = weer_data_a.diff(periods=-1)
    weer_data_d.columns = ['d_temperatuur_gem', 'd_neerslag_mm', 'd_zonuren']

    return weer_data_a.join(weer_data_d, how='left').dropna(how='any')


def prep_campaign_features():
    campaign_data = gf.import_temp_file()
    pass


def prep_seasonal_features():
    seasonal_dates = pd.DataFrame(pd.date_range('2018-01-01', periods=1200, freq='D'), columns=['day'])

    for i in range(1, 12):
        name = "month_{}".format(i)
        seasonal_dates[name] = [1 if x.month == i else 0 for x in seasonal_dates['day']]

    gf.add_week_year(data=seasonal_dates, date_name='day')
    gf.add_first_day_week(add_to=seasonal_dates, week_col_name=cn.WEEK_NUMBER, set_as_index=True)
    seasonal_dates.drop('day', axis=1, inplace=True)

    seasonal_dates = seasonal_dates.groupby(cn.FIRST_DOW, as_index=True).max()
    seasonal_dates['trend'] = np.arange(1, len(seasonal_dates)+1)

    return seasonal_dates


def prep_holiday_features():
    holiday_dates = pd.DataFrame(pd.date_range('2018-01-01', periods=1200, freq='D'), columns=['day'])

    pre_christmas_dt = pd.to_datetime(['2018-12-10', '2018-12-17', '2019-12-09',
                                       '2019-12-16', '2020-12-14', '2020-12-21'])

    holiday_dates['pre_christmas'] = [1 if x in pre_christmas_dt else 0 for x in holiday_dates['day']]

    post_christmas_dt = pd.to_datetime(['2018-12-24', '2018-12-31', '2019-12-30', '2020-12-28', '2021-01-04'])
    holiday_dates['post_christmas'] = [1 if x in post_christmas_dt else 0 for x in holiday_dates['day']]

    gf.add_week_year(data=holiday_dates, date_name='day')
    gf.add_first_day_week(add_to=holiday_dates, week_col_name=cn.WEEK_NUMBER, set_as_index=True)
    holiday_dates.drop('day', axis=1, inplace=True)

    return holiday_dates.groupby(cn.FIRST_DOW, as_index=True).max()


def prep_level_shifts():
    def str2date(date_str):
        return datetime.datetime.strptime(date_str, "%Y-%m-%d")

    level_shifts = pd.DataFrame(pd.date_range('2018-01-01', periods=1200, freq='D'), columns=['day'])

    # this becomes the new constant
    # level_shifts['period_1'] = [1 if x <= str2date('2019-03-11') else 0 for x in level_shifts['day']]
    level_shifts['period_2'] = [1 if str2date('2019-04-15') <= x <= str2date('2020-04-27') else 0 for x in
                                level_shifts['day']]

    level_shifts['period_3'] = [1 if x >= str2date('2020-06-01') else 0 for x in level_shifts['day']]
    level_shifts['trans_period_1'] = [1 if (str2date('2019-03-18') <= x <= str2date('2019-04-08')) else 0 for x in
                                      level_shifts['day']]
    level_shifts['trans_period_2'] = [1 if (str2date('2020-05-04') <= x <= str2date('2020-05-25')) else 0 for x in
                                      level_shifts['day']]

    gf.add_week_year(data=level_shifts, date_name='day')
    gf.add_first_day_week(add_to=level_shifts, week_col_name=cn.WEEK_NUMBER, set_as_index=True)
    level_shifts.drop('day', axis=1, inplace=True)

    return level_shifts.groupby(cn.FIRST_DOW, as_index=True).max()


def prep_covid_features():
    covid_dates = pd.DataFrame(pd.date_range('2018-01-01', periods=1200, freq='D'), columns=['day'])

    persco_1 = pd.to_datetime(['2020-03-09'])
    persco_2_horeca_dicht = pd.to_datetime(['2020-03-16'])
    persco_3_lockdown = pd.to_datetime(['2020-03-23'])
    persco_4 = pd.to_datetime(['2020-04-02'])
    persco_5_scholen_open = pd.to_datetime(['2020-04-21'])
    persco_6_kappers_open = pd.to_datetime(['2020-05-06'])
    # kappers_open = pd.to_datetime(['2020-05-11'])
    persco_7_horeca_open = pd.to_datetime(['2020-05-19'])
    # horeca_open = pd.to_datetime(['2020-06-01'])
    persco_8 = pd.to_datetime(['2020-06-24'])
    # verdere_versoepelingen = pd.to_datetime(['2020-07-01'])
    persco_9 = pd.to_datetime(['2020-08-06'])
    persco_10 = pd.to_datetime(['2020-08-18'])
    persco_11 = pd.to_datetime(['2020-09-01'])
    persco_12_aanscherping1_horeca = pd.to_datetime(['2020-09-25'])
    persco_13_aanscherping2_horeca = pd.to_datetime(['2020-09-28'])
    persco_14 = pd.to_datetime(['2020-10-02'])
    persco_15_horeca_dicht = pd.to_datetime(['2020-10-13'])

    persconferentie = [persco_1, persco_2_horeca_dicht, persco_3_lockdown, persco_4, persco_5_scholen_open,
                       persco_6_kappers_open, persco_7_horeca_open, persco_8, persco_9, persco_10,
                       persco_11, persco_12_aanscherping1_horeca,
                       persco_13_aanscherping2_horeca, persco_14, persco_15_horeca_dicht]

    negatieve_persconferenties = [persco_1, persco_2_horeca_dicht, persco_3_lockdown, persco_10,
                                  persco_11, persco_12_aanscherping1_horeca,
                                  persco_13_aanscherping2_horeca, persco_14, persco_15_horeca_dicht]

    positieve_persconferenties = [persco_4, persco_5_scholen_open, persco_6_kappers_open, persco_7_horeca_open,
                                  persco_8, persco_9]

    covid_dates['negatieve_persconf'] = [1 if x in negatieve_persconferenties else 0 for x in covid_dates['day']]
    covid_dates['positieve_persconf'] = [1 if x in positieve_persconferenties else 0 for x in covid_dates['day']]
    covid_dates['persconferentie'] = [1 if x in persconferentie else 0 for x in covid_dates['day']]
    covid_dates['horeca_dicht'] = [
        1 if ((persco_2_horeca_dicht <= x <= persco_7_horeca_open) or (x >= persco_15_horeca_dicht))
        else 0 for x in covid_dates['day']]

    covid_dates['persco_1'] = [1 if x == persco_1 else 0 for x in covid_dates['day']]
    covid_dates['persco_2'] = [1 if x == persco_2_horeca_dicht else 0 for x in covid_dates['day']]
    covid_dates['persco_3'] = [1 if x == persco_3_lockdown else 0 for x in covid_dates['day']]
    covid_dates['persco_4'] = [1 if x == persco_4 else 0 for x in covid_dates['day']]
    covid_dates['persco_5'] = [1 if x == persco_5_scholen_open else 0 for x in covid_dates['day']]
    covid_dates['persco_6'] = [1 if x == persco_6_kappers_open else 0 for x in covid_dates['day']]
    covid_dates['persco_7'] = [1 if x == persco_7_horeca_open else 0 for x in covid_dates['day']]
    covid_dates['persco_8'] = [1 if x == persco_8 else 0 for x in covid_dates['day']]
    covid_dates['persco_9'] = [1 if x == persco_9 else 0 for x in covid_dates['day']]
    covid_dates['persco_10'] = [1 if x == persco_10 else 0 for x in covid_dates['day']]
    covid_dates['persco_11'] = [1 if x == persco_11 else 0 for x in covid_dates['day']]
    covid_dates['persco_12'] = [1 if x == persco_12_aanscherping1_horeca else 0 for x in covid_dates['day']]
    covid_dates['persco_13'] = [1 if x == persco_13_aanscherping2_horeca else 0 for x in covid_dates['day']]
    covid_dates['persco_14'] = [1 if x == persco_14 else 0 for x in covid_dates['day']]
    covid_dates['persco_15'] = [1 if x == persco_15_horeca_dicht else 0 for x in covid_dates['day']]

    gf.add_week_year(data=covid_dates, date_name='day')
    gf.add_first_day_week(add_to=covid_dates, week_col_name=cn.WEEK_NUMBER, set_as_index=True)
    covid_dates.drop('day', axis=1, inplace=True)

    return covid_dates.groupby(cn.FIRST_DOW, as_index=True).max()


def prep_all_features(weather_data_processed, order_data_su,
                      prediction_date, train_obs,
                      index_col=cn.FIRST_DOW, import_file=False, save_to_csv=False):
    if import_file:
        weather_data_processed = gf.import_temp_file(file_name=weather_data_processed,
                                                     data_loc=fm.SAVE_LOC, set_index=False)

    weather_f = prep_weather_features(input_weer_data=weather_data_processed)
    holiday_f = prep_holiday_features()
    covid_f = prep_covid_features()
    level_f = prep_level_shifts()
    season_f = prep_seasonal_features()

    su_pct, su_n = prep_su_features(input_order_data=order_data_su, prediction_date=prediction_date,
                                    train_obs=train_obs, index_col=index_col)

    def create_lagged_features(data, lag_range=None):

        if lag_range is None:
            lag_range = [1, -1, -2]

        data_columns = data.columns

        for i in data_columns:
            for l in lag_range:
                if l < 0:
                    _temp_name = "{}_last{}w".format(i, abs(l))
                else:
                    _temp_name = "{}_next{}w".format(i, abs(l))

                data[_temp_name] = data[i].shift(l)

        return data

    all_shift_features = weather_f.join(
        holiday_f, how='left').join(
        covid_f, how='left')

    all_shift_features.sort_index(ascending=False, inplace=True)

    all_su_features = su_pct.join(su_n, how='left')
    all_su_features.sort_index(ascending=False, inplace=True)
    all_su_features_lags = create_lagged_features(data=all_su_features, lag_range=[2, 1, -1, -2])

    all_shift_features_lags = create_lagged_features(data=all_shift_features)

    all_exog_features = all_shift_features_lags.join(all_su_features_lags, how='left').join(season_f)
    #.join(level_f, how='left') excluded level features

    eval_cols = all_exog_features.loc[prediction_date].T
    cols_include = eval_cols.dropna(how='any', axis=0)

    all_exog_features_non_zero = all_exog_features[cols_include.index]

    if save_to_csv:
        gf.save_to_csv(data=all_exog_features_non_zero, file_name='exogenous_features', folder=fm.SAVE_LOC)

    return all_exog_features_non_zero


if __name__ == '__main__':
    # Import weer data
    order_data_su = gf.import_temp_file(file_name=fm.ORDER_DATA_ACT_SU, data_loc=fm.SAVE_LOC,
                                        set_index=True)

    weather_data = gf.import_temp_file(file_name=fm.WEER_DATA_PREP, data_loc=fm.SAVE_LOC, set_index=False)
    weather_features = prep_weather_features(input_weer_data=weather_data)
    holiday_features = prep_holiday_features()
    covid_features = prep_covid_features()

    exog_features = prep_all_features(weather_data_processed=weather_data, order_data_su=order_data_su,
                                      prediction_date='2020-10-05', train_obs=cn.TRAIN_OBS, save_to_csv=False, index_col=cn.FIRST_DOW)


    gf.save_to_csv(data=exog_features, file_name='exogenous_features', folder=fm.SAVE_LOC)
