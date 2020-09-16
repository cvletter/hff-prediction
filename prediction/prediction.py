import prediction.general_purpose_functions as gpf

DATA_LOC = '/Users/cornelisvletter/Google Drive/HFF/Data/Prepared'
FILE_NAME_AR = 'producten_pred_ar_diff_2020916-1529.csv'
FILE_NAME_Y = 'producten_pred_diff_2020916-1529.csv'

y = gpf.import_temp_file(file_name=FILE_NAME_Y, data_loc=DATA_LOC, set_index=True)





y = pd.read_csv(import_name, sep=";", decimal=",")
order_data['eerste_dag_week'] = pd.to_datetime(order_data['eerste_dag_week'], format='%Y-%m-%d')
order_data.set_index('eerste_dag_week', inplace=True)