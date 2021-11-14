# Settings waar voorspelling mee wordt gemaakt, zowel voor test als make
TRAIN_OBS = 70                   # Aantal observasties waar modellen mee worden getraind
PREDICTION_WINDOW = 2            # Voorspelwindow
HOLIDAY_FORWARD = 2
N_LAGS = 4                       # Aantal weken dat wordt teruggekeken
MA_PERIOD = 5                    # Aantal weken waar benchmark over middelt
DIFFERENCING = False             # Eerste verschillen nemen voor voorspelling
BOOTSTRAP_ITER = 40              # Aantal bootstraps voor grenswaarden voorspelling
MODEL_TYPE = "OLS"               # Voorspelalgoritme: OLS, Poisson, XGBoost, etc
MAX_FEATURES = 20                # Maximaal te selecteren features in optimalisatie
MIN_CORRELATION = 0.3            # Minimum correlatie die een feature moet hebben
FEATURE_OPT = [MIN_CORRELATION,  # Combinatie van bovenstaande parameters
               MAX_FEATURES]
FEATURE_PERIOD_LENGTH = 2500     # Number of days to generate for features
WEATHER_FORECAST = False         # Weersfactoren worden ook als voorspelling meegenomen
ADD_PLUS_SALES = False           # Option to add Plus sales data to feature set (data until 13 Sep 2021)
TOP_DOWN = False                 # Perform top down prediction, default is bottom up (prediction per product)
STANDARDIZE = False              # Standardizes all features between IID(0,1) for more optimal performance
SU_MEMBER = None                 # Only make a prediction for a SU Member, should be combined with a '--reload'
