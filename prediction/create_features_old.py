import pandas as pd
import prediction.column_names as cn
import prediction.file_management as fm
import prediction.general_purpose_functions as gf
import datetime


def prep_weather_features(input_weer_data, min_max=False, index_col=cn.FIRST_DOW,
                          shift=True, prediction_window=cn.PREDICTION_WINDOW):

    # TODO check of ik niet beter gem. temp overdag kan nemen
    if not input_weer_data.index.name == index_col:
        input_weer_data.reset_index(inplace=True, drop=True)
        input_weer_data.set_index(index_col, inplace=True)

    input_weer_data.sort_index(ascending=False, inplace=True)

    cols = [cn.TEMP_GEM, cn.NEERSLAG_MM, cn.ZONUREN]

    if min_max:
        # cols = [cn.TEMP_GEM, cn.NEERSLAG_MM, cn.ZONUREN, cn.TEMP_MIN, cn.TEMP_MAX]
        pass

    weather_current_week = input_weer_data[cols]

    weather_min_1w = weather_current_week.shift(-1)
    weather_min_1w.columns = [cn.TEMP_GEM_L1W, cn.NEERSLAG_MM_L1W, cn.ZONUREN_L1W]

    weather_min_2w = weather_current_week.shift(-2)
    weather_min_2w.columns = [cn.TEMP_GEM_L2W, cn.NEERSLAG_MM_L2W, cn.ZONUREN_L2W]

    weather_combined = weather_current_week.join(
        weather_min_1w, how='left').join(
        weather_min_2w, how='left')

    if shift:
        return weather_combined.shift(-prediction_window + 1).dropna(how='any')

    return weather_combined


def prep_holiday_features(weekly=False, shift=True, prediction_window=cn.PREDICTION_WINDOW):
    holiday_dates = pd.DataFrame(pd.date_range('2018-01-01', periods=1200, freq='D'), columns=['day'])

    pre_christmas_dt = pd.to_datetime(['2018-12-10', '2018-12-17', '2019-12-09',
                                       '2019-12-16', '2020-12-14', '2020-12-21'])

    holiday_dates['pre_christmas'] = [1 if x in pre_christmas_dt else 0 for x in holiday_dates['day']]

    post_christmas_dt = pd.to_datetime(['2018-12-24', '2018-12-31', '2019-12-30', '2020-12-28', '2021-01-04'])
    holiday_dates['post_christmas'] = [1 if x in post_christmas_dt else 0 for x in holiday_dates['day']]

    """
    sinterklaas_dt = pd.to_datetime(['2018-12-05', '2019-12-05', '2020-12-05'])
    holiday_dates['sinterklaas'] = [1 if x in sinterklaas_dt else 0 for x in holiday_dates['day']]

    newyears_dt = pd.to_datetime(['2018-12-31', '2019-12-31', '2020-12-31'])
    holiday_dates['newyears'] = [1 if x in newyears_dt else 0 for x in holiday_dates['day']]

    easter_dt = pd.to_datetime(['2018-04-01', '2019-04-21', '2020-04-12'])
    holiday_dates['easter'] = [1 if x in easter_dt else 0 for x in holiday_dates['day']]

    pentecost_dt = pd.to_datetime(['2018-05-20', '2019-06-09', '2020-05-31'])
    holiday_dates['pentecost'] = [1 if x in pentecost_dt else 0 for x in holiday_dates['day']]

    mothers_day_dt = pd.to_datetime(['2018-05-13', '2019-05-12', '2020-05-10'])
    holiday_dates['mothers_day'] = [1 if x in mothers_day_dt else 0 for x in holiday_dates['day']]

    fathers_day_dt = pd.to_datetime(['2018-06-17', '2019-06-16', '2020-06-21'])
    holiday_dates['fathers_day'] = [1 if x in fathers_day_dt else 0 for x in holiday_dates['day']]

    kings_day_dt = pd.to_datetime(['2018-04-27', '2019-04-27', '2020-04-27'])
    holiday_dates['kings_day'] = [1 if x in kings_day_dt else 0 for x in holiday_dates['day']]

    carnaval_dt = pd.to_datetime(['2018-02-11', '2019-03-05', '2020-02-23'])
    holiday_dates['carnaval'] = [1 if x in carnaval_dt else 0 for x in holiday_dates['day']]
    """

    gf.add_week_year(data=holiday_dates, date_name='day')
    gf.add_first_day_week(add_to=holiday_dates, week_col_name=cn.WEEK_NUMBER, set_as_index=True)
    holiday_dates.drop('day', axis=1, inplace=True)

    if weekly:
        holiday_weeks = holiday_dates.groupby(cn.FIRST_DOW, as_index=True).max()
        if shift:
            holiday_weeks = holiday_weeks.shift(-prediction_window)

        return holiday_weeks.dropna(how='any', inplace=False)

    return holiday_dates


def prep_level_shifts():

    def str2date(date_str):
        return datetime.datetime.strptime(date_str, "%Y-%m-%d")

    level_shifts = pd.DataFrame(pd.date_range('2018-01-01', periods=1200, freq='D'), columns=['day'])

    # this becomes the new constant
    # level_shifts['period_1'] = [1 if x <= str2date('2019-03-11') else 0 for x in level_shifts['day']]
    level_shifts['period_2'] = [1 if str2date('2019-04-15') <= x <= str2date('2020-04-27') else 0 for x in level_shifts['day']]

    level_shifts['period_3'] = [1 if x >= str2date('2020-06-01') else 0 for x in level_shifts['day']]
    level_shifts['trans_period_1'] = [1 if (str2date('2019-03-18') <= x <= str2date('2019-04-08')) else 0 for x in level_shifts['day']]
    level_shifts['trans_period_2'] = [1 if (str2date('2020-05-04') <= x <= str2date('2020-05-25')) else 0 for x in level_shifts['day']]

    gf.add_week_year(data=level_shifts, date_name='day')
    gf.add_first_day_week(add_to=level_shifts, week_col_name=cn.WEEK_NUMBER, set_as_index=True)
    level_shifts.drop('day', axis=1, inplace=True)

    return level_shifts.groupby(cn.FIRST_DOW, as_index=True).max()


def prep_covid_features(weekly=False):

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

    positieve_persconferenties = [persco_4, persco_5_scholen_open, persco_6_kappers_open, persco_7_horeca_open, persco_8, persco_9]

    covid_dates['negatieve_persconf'] = [1 if x in negatieve_persconferenties else 0 for x in covid_dates['day']]
    covid_dates['positieve_persconf'] = [1 if x in positieve_persconferenties else 0 for x in covid_dates['day']]
    covid_dates['persconferentie'] = [1 if x in persconferentie else 0 for x in covid_dates['day']]
    covid_dates['horeca_dicht'] = [1 if ((persco_2_horeca_dicht <= x <= persco_7_horeca_open) or (x >= persco_15_horeca_dicht))
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

    if weekly:
        return covid_dates.groupby(cn.FIRST_DOW, as_index=True).max()

    return covid_dates


def prep_exogenous_features(weather_data_processed, prediction_window, import_file=False, save_to_csv=False, shift=True):

    if import_file:
        weather_data_processed = gf.import_temp_file(file_name=weather_data_processed,
                                                     data_loc=fm.SAVE_LOC, set_index=False)

    weather_f = prep_weather_features(input_weer_data=weather_data_processed, prediction_window=prediction_window,
                                      shift=shift)
    holiday_f = prep_holiday_features(weekly=True, prediction_window=prediction_window, shift=shift)
    covid_f = prep_covid_features(weekly=True)
    level_f = prep_level_shifts()

    exog_features = weather_f.join(holiday_f, how='left').join(covid_f, how='left').join(level_f, how='left')

    if save_to_csv:
        gf.save_to_csv(data=exog_features, file_name='exogenous_features', folder=fm.SAVE_LOC)

    return exog_features


if __name__ == '__main__':
    # Import weer data
    weather_data = gf.import_temp_file(file_name=fm.WEER_DATA_PREP, data_loc=fm.SAVE_LOC, set_index=False)
    weather_features = prep_weather_features(input_weer_data=weather_data)
    holiday_features = prep_holiday_features(weekly=True)
    covid_features = prep_covid_features(weekly=True)

    exog_features = prep_exogenous_features(weather_data_processed=weather_data, prediction_window=1,
                                            save_to_csv=False, shift=False)

    gf.save_to_csv(data=exog_features, file_name='exogenous_features', folder=fm.SAVE_LOC)

    fit_data = gf.read_pkl(file_name=fm.FIT_DATA, data_loc=fm.SAVE_LOC)
    y_true = fit_data['y_true']
    X_exog = fit_data['x_exog']

    y_sum = y_true['model_products_sum']
    yd_true = y_true.diff(periods=-1)


    def get_correlations(dep_var, indep_var):
        corrs = pd.DataFrame([])
        for j in dep_var.columns:
            for i in indep_var.columns:
                _corr = round(dep_var[j].corr(indep_var[i].shift(0)), 3)
                corrs.loc[i, j] = _corr

                _corr_p2 = round(dep_var[j].corr(indep_var[i].shift(2)), 3)
                _temp_name = "{}_p2".format(i)
                corrs.loc[_temp_name, j] = _corr_p2

                _corr_p1 = round(dep_var[j].corr(indep_var[i].shift(1)), 3)
                _temp_name = "{}_p1".format(i)
                corrs.loc[_temp_name, j] = _corr_p1

                _corr_m1 = round(dep_var[j].corr(indep_var[i].shift(-1)), 3)
                _temp_name = "{}_m1".format(i)
                corrs.loc[_temp_name, j] = _corr_m1

                _corr_m2 = round(dep_var[j].corr(indep_var[i].shift(-2)), 3)
                _temp_name = "{}_m2".format(i)
                corrs.loc[_temp_name, j] = _corr_m2

        corr_stat = pd.DataFrame(index=corrs.index, columns=['avg', '25p', 'med', '75p'])
        corr_stat['avg'] = round(corrs.mean(axis=1), 2)
        corr_stat['25p'] = round(corrs.quantile(q=0.25, axis=1), 2)
        corr_stat['med'] = round(corrs.quantile(q=0.5, axis=1), 2)
        corr_stat['75p'] = round(corrs.quantile(q=0.75, axis=1), 2)

        return corr_stat

    yd2_true = y_true.diff(periods=-2)

    corr = get_correlations(dep_var=y_true, indep_var=exog_features)
    corr_d = get_correlations(dep_var=yd_true, indep_var=exog_features)
    corr_d2 = get_correlations(dep_var=yd2_true, indep_var=exog_features)

    corr_d2.to_csv("correlations_diff2.csv", sep=";", decimal=",")
