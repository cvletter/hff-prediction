
import datetime

# ORDER DATA
ORDER_DATE = 'besteldatum'
CONSUMENT_GROEP = 'consumentgroep'
INKOOP_RECEPT = 'inkooprecept'
VERKOOP_ART = 'verkoopartikel'
SELECT_ORG = 'gebruiken'
ORGANISATIE = 'organisatie'
CE_BESTELD = 'ce_besteld'
WEEK = 'week'

# WEER DATA
W_DATE = 'date'
TEMP_GEM = 'temperatuur_gem'
TEMP_MIN = 'temperatuur_min'
TEMP_MAX = 'temperatuur_max'
ZONUREN = 'zonuren'
NEERSLAG_DUUR = 'neerslag_duur'
NEERSLAG_MM = 'neerslag_mm'

TEMP_GEM_L1W = 'temperatuur_gem_l1w'
TEMP_GEM_L2W = 'temperatuur_gem_l2w'
TEMP_GEM_P1W = 'temperatuur_gem_p1w'

ZONUREN_L1W = 'zonuren_l1w'
ZONUREN_L2W = 'zonuren_l2w'
ZONUREN_P1W = 'zonuren_p1w'

NEERSLAG_MM_L1W = 'neerslag_duur_l1w'
NEERSLAG_MM_L2W = 'neerslag_duur_l2w'
NEERSLAG_MM_P1W = 'neerslag_duur_p1w'

# GENERATED
WEEK_NUMBER = 'week_jaar'
MOD_PROD_SUM = 'model_products_sum'
CONSUMENT_GROEP_NR = 'consumentgroep_nr'
VERKOOP_ART_NR = 'verkoopartikel_nr'
VERKOOP_ART_NM ='verkoopartikel_naam'
INKOOP_RECEPT_NR = 'inkooprecept_nr'
INKOOP_RECEPT_NM = 'inkooprecept_naam'
FIRST_DOW = 'eerste_dag_week'

# MODEL NAMES
Y_TRUE = 'y_true'
Y_AR = 'y_ar'
X_EXOG = 'x_exog'

Y_AR_M = 'y_ar_m'
Y_AR_NM = 'y_ar_nm'


# INPUT VARIABLES


FIRST_AVAILABLE_DATE = datetime.datetime.strptime('2018-08-01', "%Y-%m-%d")
PREDICTION_DATE = datetime.datetime.strptime('2020-08-31', "%Y-%m-%d")
PREDICTION_WINDOW = 2
LAST_TRAIN_DATE = PREDICTION_DATE - datetime.timedelta(weeks=PREDICTION_WINDOW)

TRAIN_OBS = 70

HOLIDAY_FORWARD = 2
N_LAGS = 2

