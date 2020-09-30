from prediction.dataprep import data_prep_wrapper
from prediction.create_features import prep_exogenous_features
import prediction.file_management as fm

# Parameters

PREDICTION_DATE = '2020-06-22'
PREDICTION_WINDOW = 2
ORDER_DATA = fm.RAW_DATA
WEER_DATA = fm.WEER_DATA
PRODUCT_DATA = fm.PRODUCT_STATUS


# Import and prepare data
active_products, inactive_products, weather_data_processed = data_prep_wrapper(
    prediction_date=PREDICTION_DATE,
    prediction_window=PREDICTION_WINDOW,
    order_data_loc=ORDER_DATA,
    weer_data_loc=WEER_DATA,
    product_data_loc=PRODUCT_DATA,
    agg_weekly=True, exclude_su=True,
    save_to_csv=False)

exogenous_features = prep_exogenous_features(weather_data_processed=fm.WEER_DATA_PREP, save_to_csv=False)