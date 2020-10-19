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

    christmas_dt = pd.to_datetime(['2018-12-25', '2019-12-25', '2020-12-25'])
    holiday_dates['christmas'] = [1 if x in christmas_dt else 0 for x in holiday_dates['day']]

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

    covid_start_dt = pd.to_datetime(['2018-03-13'])
    covid_end_dt = pd.to_datetime(['2018-06-02'])
    covid_dates['covid_period'] = [1 if (covid_start_dt <= x <= covid_end_dt) else 0 for x in covid_dates['day']]

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

