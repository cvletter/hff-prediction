import pandas as pd
import prediction.column_names as cn
import prediction.file_management as fm
import prediction.general_purpose_functions as gf


def prep_weather_features(input_weer_data, min_max=False, index_col=cn.FIRST_DOW):

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
    weather_plus_1w = weather_current_week.shift(1)
    weather_plus_1w.columns = [cn.TEMP_GEM_P1W, cn.NEERSLAG_MM_P1W, cn.ZONUREN_P1W]

    weather_min_1w = weather_current_week.shift(-1)
    weather_min_1w.columns = [cn.TEMP_GEM_L1W, cn.NEERSLAG_MM_L1W, cn.ZONUREN_L1W]

    weather_min_2w = weather_current_week.shift(-2)
    weather_min_2w.columns = [cn.TEMP_GEM_L2W, cn.NEERSLAG_MM_L2W, cn.ZONUREN_L2W]

    return weather_current_week.join(
        weather_min_1w, how='left').join(
        weather_min_2w, how='left').join(
        weather_plus_1w).dropna(how='any')


def prep_holiday_features(weekly=False):
    holiday_dates = pd.DataFrame(pd.date_range('2018-01-01', periods=1200, freq='D'), columns=['day'])

    christmas_dt = pd.to_datetime(['2018-12-25', '2019-12-25', '2020-12-25'])
    holiday_dates['christmas'] = [1 if x in christmas_dt else 0 for x in holiday_dates['day']]

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
        return holiday_dates.groupby(cn.FIRST_DOW, as_index=True).max()

    return holiday_dates


def prep_covid_features(weekly=False):

    covid_dates = pd.DataFrame(pd.date_range('2018-01-01', periods=1200, freq='D'), columns=['day'])

    covid_start_dt = pd.to_datetime(['2018-03-13'])
    covid_dates['covid_start'] = [1 if x >= covid_start_dt else 0 for x in covid_dates['day']]

    gf.add_week_year(data=covid_dates, date_name='day')
    gf.add_first_day_week(add_to=covid_dates, week_col_name=cn.WEEK_NUMBER, set_as_index=True)
    covid_dates.drop('day', axis=1, inplace=True)

    if weekly:
        return covid_dates.groupby(cn.FIRST_DOW, as_index=True).max()

    return covid_dates


def prep_exogenous_features(weather_f, holiday_f, covid_f):

    exog_features = weather_f.join(holiday_f, how='left').join(covid_f, how='left')

    return exog_features


if __name__ == '__main__':
    # Import weer data
    weather_data = pd.read_csv(fm.WEER_DATA_PREP, decimal=",", sep=';')
    weather_features = prep_weather_features(input_weer_data=weather_data)
    holiday_features = prep_holiday_features(weekly=True)
    covid_features = prep_covid_features(weekly=True)

    exog_features = prep_exogenous_features(weather_f=weather_features,
                                            holiday_f=holiday_features,
                                            covid_f=covid_features)

    gf.save_to_csv(data=exog_features, file_name='exogenous_features', folder=fm.SAVE_LOC)