import prediction.general_purpose_functions as gpf
import statsmodels.api as sm
import pandas as pd

DATA_LOC = '/Users/cornelisvletter/Google Drive/HFF/Data/Prepared'
FILE_NAME_AR = 'producten_pred_ar_diff_2020916_178.csv'
FILE_NAME_Y = 'producten_pred_diff_2020916_178.csv'

Y = gpf.import_temp_file(file_name=FILE_NAME_Y, data_loc=DATA_LOC, set_index=True)
ar_comp = gpf.import_temp_file(file_name=FILE_NAME_AR, data_loc=DATA_LOC, set_index=True)


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


predictions = batch_fit_ar_model(Y, ar_comp)
errors = Y - predictions








y = pd.read_csv(import_name, sep=";", decimal=",")
order_data['eerste_dag_week'] = pd.to_datetime(order_data['eerste_dag_week'], format='%Y-%m-%d')
order_data.set_index('eerste_dag_week', inplace=True)