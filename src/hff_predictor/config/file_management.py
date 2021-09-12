# Key folders
BASE_FOLDER = "E:\Productie Voorspelmodel\Data"
# BASE_FOLDER = "U:\Productie Voorspelmodel"

# Input folders
ORDER_DATA_FOLDER = BASE_FOLDER + "\Input\Bestellingen"
PRODUCT_STATUS_FOLDER = BASE_FOLDER + "\Input\Productstatus"
CAMPAIGN_DATA_FOLDER = BASE_FOLDER + "\Input\Campagnes"

# Input SU leden
ORDER_DATA_FOLDER_HP = BASE_FOLDER + "\Input\Bestellingen Hollander"

# Processed folders
WEATHER_DATA_PPR_FOLDER = BASE_FOLDER + "\Processed\Weer\Preprocessed"
WEATHER_DATA_PR_FOLDER = BASE_FOLDER + "\Processed\Weer\Processed"
ORDER_DATA_PR_FOLDER = BASE_FOLDER + "\Processed\Bestellingen\Standaard"
ORDER_DATA_SU_PPR_FOLDER = BASE_FOLDER + "\Processed\Bestellingen\Superunie eigenschappen\Preprocessed"
CAMPAIGN_DATA_PR_FOLDER = BASE_FOLDER + "\Processed\Campagnes"

ORDER_DATA_ACT_PR_FOLDER = BASE_FOLDER + "\Processed\Bestellingen\Actief"
ORDER_DATA_INACT_PR_FOLDER = BASE_FOLDER + "\Processed\Bestellingen\Inactief"
ORDER_DATA_SU_PR_FOLDER = BASE_FOLDER + "\Processed\Bestellingen\Superunie eigenschappen\Processed"

ORDER_DATA_CG_PR_FOLDER = BASE_FOLDER + "\Processed\Bestellingen\Consumentgroepnummer"
FEATURES_PROCESSED_FOLDER = BASE_FOLDER + "\Processed\Features"


# Processed files
WEATHER_DATA_PREPROCESSED = "weather_data_preprocessed"
WEATHER_DATA_PROCESSED = "weather_data_processed"
ORDER_DATA_PROCESSED = "order_data_pivot_week_processed"
ORDER_DATA_ACT_PROCESSED = "actieve_halffabricaten_week"
ORDER_DATA_INACT_PROCESSED = "inactieve_halffabricaten_week"
ORDER_DATA_SU_PREPROCESSED = "order_data_week_superunie_preprocessed"
ORDER_DATA_SU_PROCESSED = "order_data_week_superunie_processed"
CAMPAIGN_DATA_PROCESSED = "campaign_data_processed"
PRODUCT_CONSUMENTGROEP_NR = "product_consumentgroep_nr"
FEATURES_PROCESSED = "exogenous_features_processed"

# Output
PREDICTIONS_FOLDER = BASE_FOLDER + "\Output\Voorspellingen"
INTERMEDIARY_RESULTS_FOLDER = BASE_FOLDER + "\Output\Tussenresultaten"
TEST_RESULTS_FOLDER = BASE_FOLDER + "\Output\Testvoorspellingen"
TEST_PREDICTIONS_FOLDER = BASE_FOLDER + "\Output\Tussenresultaten\Performance"

# Feature groups
WEATHER = "weather"
SEASONAL = "seasonal"
HOLIDAYS = "holidays"
COVID = "covid"
STRUC_BREAKS = "breaks"
CAMPAIGNS = "campaigns"
SUPERUNIE_PCT = "superunie_pct"
SUPERUNIE_N = "superunie_n"