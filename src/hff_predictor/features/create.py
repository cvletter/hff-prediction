import datetime

import hff_predictor.generic.files
import pandas as pd

import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
import hff_predictor.generic.dates as gf

from hff_predictor.features.feature_types import weather_features, campaign_features, covid_features, \
    seasonal_features, superunie_features


def prep_level_shifts():
    def str2date(date_str):
        return datetime.datetime.strptime(date_str, "%Y-%m-%d")

    level_shifts = pd.DataFrame(
        pd.date_range("2018-01-01", periods=1200, freq="D"), columns=["day"]
    )

    # this becomes the new constant
    # level_shifts['period_1'] = [1 if x <= str2date('2019-03-11') else 0 for x in level_shifts['day']]
    level_shifts["a_trans_period_1"] = [
        1 if (str2date("2019-03-18") <= x <= str2date("2019-04-08")) else 0
        for x in level_shifts["day"]
    ]

    level_shifts["b_period_2"] = [
        1 if str2date("2019-04-15") <= x <= str2date("2020-04-27") else 0
        for x in level_shifts["day"]
    ]

    level_shifts["c_trans_period_2"] = [
        1 if (str2date("2020-05-04") <= x <= str2date("2020-05-25")) else 0
        for x in level_shifts["day"]
    ]

    level_shifts["d_trans_period_2b"] = [
        1 if (str2date("2020-06-01") <= x <= str2date("2020-06-29")) else 0
        for x in level_shifts["day"]
    ]

    level_shifts["e_period_3"] = [
        1 if x >= str2date("2020-06-01") else 0 for x in level_shifts["day"]
    ]

    gf.add_week_year(data=level_shifts, date_name="day")
    gf.add_first_day_week(
        add_to=level_shifts, week_col_name=cn.WEEK_NUMBER, set_as_index=True
    )
    level_shifts.drop("day", axis=1, inplace=True)

    return level_shifts.groupby(cn.FIRST_DOW, as_index=True).max()


def prep_all_features(
    weather_data_processed,
    order_data_su,
    campaign_data_su,
    prediction_date,
    train_obs,
    index_col=cn.FIRST_DOW,
    import_file=False,
    save_to_csv=False,
):
    if import_file:
        weather_data_processed = hff_predictor.generic.files.import_temp_file(
            file_name=weather_data_processed, data_loc=fm.SAVE_LOC, set_index=False
        )

    weather_f = weather_features.prep_weather_features(input_weer_data=weather_data_processed)
    holiday_f = seasonal_features.prep_holiday_features()
    season_f = seasonal_features.prep_seasonal_features()
    campaign_f = campaign_features.prep_campaign_features(campaign_data=campaign_data_su)
    covid_f = covid_features.prep_covid_features()

    level_f = prep_level_shifts()

    su_pct, su_n = superunie_features.prep_su_features(
        input_order_data=order_data_su,
        prediction_date=prediction_date,
        train_obs=train_obs,
        index_col=index_col,
    )

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

    all_shift_features = weather_f.join(holiday_f, how="left").join(covid_f, how="left")

    all_shift_features.sort_index(ascending=False, inplace=True)

    all_su_features = su_pct.join(su_n, how="left").join(campaign_f, how="left")
    all_su_features.sort_index(ascending=False, inplace=True)
    all_su_features_lags = create_lagged_features(
        data=all_su_features, lag_range=[2, 1, -1, -2]
    )

    all_shift_features_lags = create_lagged_features(data=all_shift_features)

    all_exog_features = (
        all_shift_features_lags.join(all_su_features_lags, how="left")
        .join(season_f)
        .join(level_f, how="left")
    )

    eval_cols = all_exog_features.loc[prediction_date].T
    cols_include = eval_cols.dropna(how="any", axis=0)

    all_exog_features_non_zero = all_exog_features[cols_include.index]

    if save_to_csv:
        hff_predictor.generic.files.save_to_csv(
            data=all_exog_features_non_zero,
            file_name="exogenous_features",
            folder=fm.SAVE_LOC,
        )

    return all_exog_features_non_zero


def init_create_features():
    # Import weer data
    order_data_su = hff_predictor.generic.files.import_temp_file(
        file_name=fm.ORDER_DATA_ACT_SU, data_loc=fm.SAVE_LOC, set_index=True
    )

    campaign_data = hff_predictor.generic.files.import_temp_file(
        file_name="campaign_data_processed_20201114_1222.csv",
        data_loc=fm.SAVE_LOC,
        set_index=True,
    )

    weather_data = hff_predictor.generic.files.import_temp_file(
        file_name=fm.WEER_DATA_PREP, data_loc=fm.SAVE_LOC, set_index=False
    )
    weather_features = prep_weather_features(input_weer_data=weather_data)
    holiday_features = prep_holiday_features()
    covid_features = prep_covid_features()

    exog_features = prep_all_features(
        weather_data_processed=weather_data,
        order_data_su=order_data_su,
        campaign_data_su=campaign_data,
        prediction_date="2020-10-05",
        train_obs=cn.TRAIN_OBS,
        save_to_csv=False,
        index_col=cn.FIRST_DOW,
    )

    hff_predictor.generic.files.save_to_csv(
        data=exog_features, file_name="exogenous_features", folder=fm.SAVE_LOC
    )
