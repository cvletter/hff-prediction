import prediction.general_purpose_functions as gpf
import statsmodels.api as sm
import pandas as pd
import prediction.general_purpose_functions as gf
import prediction.file_management as fm


def batch_fit_ar_model(Y, X_ar, add_constant=True):
    Y_pred = pd.DataFrame(index=Y.index)

    for product in Y.columns:
        y_name = product
        y = Y[y_name]

        lag_index = [y_name in x for x in X_ar.columns]
        x_ar = X_ar.iloc[:, lag_index]

        if add_constant:
            x_ar.insert(0, 'constant', 1)

        temp_mdl = sm.OLS(y, x_ar, missing='drop')
        temp_fit = temp_mdl.fit()
        Y_pred[y_name] = temp_fit.predict()

    return Y_pred


if __name__ == '__main__':
    fit_data = gf.read_pkl(file_name=fm.FIT_DATA, data_loc=fm.SAVE_LOC)


