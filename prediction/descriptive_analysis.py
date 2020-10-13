import pandas as pd
from prediction import file_management as fm
from prediction import general_purpose_functions as gf

fit_data = gf.read_pkl(file_name=fm.FIT_DATA, data_loc=fm.SAVE_LOC)
y_true = fit_data['y_true']


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


y_pre_corona = y_true[y_true.index <= '20-4-2020']
y_post_corona = y_true[y_true.index >= '2-6-2020']

Y_corr_total = autocorrelation_check(dependent_variable=y_true)
Y_corr_pre_cor = autocorrelation_check(dependent_variable=y_pre_corona)
Y_corr_post_cor = autocorrelation_check(dependent_variable=y_post_corona)