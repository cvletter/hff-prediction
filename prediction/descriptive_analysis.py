import pandas as pd
from prediction import file_management as fm
from prediction import general_purpose_functions as gf
from prediction import column_names as cn
import datetime


fit_data = gf.read_pkl(file_name=fm.FIT_DATA, data_loc=fm.SAVE_LOC)
y_true = fit_data['y_true']
X_exog = fit_data['x_exog']


pre_corona = datetime.datetime.strptime("2020-04-20", "%Y-%m-%d")
post_corona = datetime.datetime.strptime("2020-06-02", "%Y-%m-%d")

dependent_variable = y_true
exog_features = X_exog
y_pre_corona = y_true[y_true.index <= pre_corona]
y_post_corona = y_true[y_true.index >= post_corona]


def weather_correl(dependent_variable, exog_features):
    weather_cols = [cn.TEMP_GEM, cn.TEMP_GEM_L1W, cn.TEMP_GEM_L2W, cn.ZONUREN, cn.ZONUREN_L1W,
                    cn.ZONUREN_L2W, cn.NEERSLAG_MM, cn.NEERSLAG_MM_L1W, cn.NEERSLAG_MM_L2W]

    X_weather = exog_features[weather_cols]

    X_weather[cn.TEMP_GEM_P1W] = X_weather[cn.TEMP_GEM].shift(1)
    X_weather[cn.NEERSLAG_MM_P1W] = X_weather[cn.NEERSLAG_MM].shift(1)
    X_weather[cn.ZONUREN_P1W] = X_weather[cn.ZONUREN].shift(1)
    X_weather.dropna(how='any', inplace=True)

    Y_weer_correl = pd.DataFrame(index=dependent_variable.columns, columns=X_weather.columns)

    for c in dependent_variable.columns:
        _ytemp = dependent_variable[c]
        for w in X_weather.columns:
            _wtemp = X_weather[w]
            Y_weer_correl.loc[c, w] = round(_ytemp.corr(_wtemp), 2)

    return Y_weer_correl


def holiday_correl(dependent_variable, exog_features):
    holiday_cols = ['christmas', 'sinterklaas', 'newyears', 'easter', 'pentecost',
                    'mothers_day', 'fathers_day', 'kings_day', 'carnaval']

    X_holiday = exog_features[holiday_cols]

    Y_holiday_correl = pd.DataFrame(index=dependent_variable.columns, columns=X_holiday.columns)

    for c in dependent_variable.columns:
        _ytemp = dependent_variable[c]
        for w in X_holiday.columns:
            _wtemp = X_holiday[w]
            Y_holiday_correl.loc[c, w] = round(_ytemp.corr(_wtemp), 2)

    return Y_holiday_correl


def lag_holiday_correl(dependent_variable, exog_features, holiday='christmas'):

    X_holiday_lags = pd.DataFrame(exog_features[holiday])

    for l in range(-3, 4):
        if l > 0:
            _name = "{}_p{}".format(holiday, l)
        else:
            _name = "{}_l{}".format(holiday, l)

        X_holiday_lags[_name] = X_holiday_lags[holiday].shift(l)

    X_holiday_lags.dropna(how='any', inplace=True)

    Y_holiday_correl = pd.DataFrame(index=dependent_variable.columns, columns=X_holiday_lags.columns)

    for c in dependent_variable.columns:
        _ytemp = dependent_variable[c]
        for w in X_holiday_lags.columns:
            _wtemp = X_holiday_lags[w]
            Y_holiday_correl.loc[c, w] = round(_ytemp.corr(_wtemp), 2)

    return Y_holiday_correl


def autocorrelation_check(dependent_variable):
    Y_correls = pd.DataFrame(index=dependent_variable.columns, columns=['lag1', 'lag2', 'lag3',
                                                                        'lag4', 'lag5', 'lag6', 'lag7'])

    for i in dependent_variable.columns:
        _ytemp = dependent_variable[i]
        _ylags = pd.concat([_ytemp.shift(-1), _ytemp.shift(-2), _ytemp.shift(-3), _ytemp.shift(-4),
                            _ytemp.shift(-5), _ytemp.shift(-6), _ytemp.shift(-7)], axis=1)
        _ylags.columns = ['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'lag6', 'lag7']
        for l in _ylags.columns:
            Y_correls.loc[i, l] = round(_ytemp.corr(_ylags[l]), 2)

    return Y_correls


def level_check(dependent_variable, pre_corona='2020-04-20', post_corona='2020-06-02'):

    pre_corona = datetime.datetime.strptime(pre_corona, "%Y-%m-%d")
    post_corona = datetime.datetime.strptime(post_corona, "%Y-%m-%d")

    Y_level_check = pd.DataFrame(index=dependent_variable.columns, columns=['total', 'pre', 'post', 'peak'])
    y_pre = dependent_variable[dependent_variable.index <= pre_corona]
    y_post = dependent_variable[dependent_variable.index >= post_corona]
    y_peak = dependent_variable[(dependent_variable.index > pre_corona) & (dependent_variable.index < post_corona)]

    Y_level_check['total'] = round(dependent_variable.mean(), 1)
    Y_level_check['pre'] = round(y_pre.mean(), 1)
    Y_level_check['post'] = round(y_post.mean(), 1)
    Y_level_check['peak'] = round(y_peak.mean(), 1)
    Y_level_check['post/pre'] = round((y_post.mean() / y_pre.mean()) - 1, 2)
    Y_level_check['post/total'] = round((y_post.mean() / dependent_variable.mean()) - 1, 2)

    return Y_level_check



# Level check
Y_level = level_check(dependent_variable=y_true)

# Autocorrelatie check
Y_corr_total = autocorrelation_check(dependent_variable=y_true)
Y_corr_pre_cor = autocorrelation_check(dependent_variable=y_pre_corona)
Y_corr_post_cor = autocorrelation_check(dependent_variable=y_post_corona)

# Weer check
Y_weer = weather_correl(dependent_variable=y_true, exog_features=X_exog)

# Holiday
Y_holiday = holiday_correl(dependent_variable=y_true, exog_features=X_exog)
Y_christmas = lag_holiday_correl(dependent_variable=y_true, exog_features=X_exog, holiday='christmas')