# Settings waar voorspelling mee wordt gemaakt, zowel voor test als make
TRAIN_OBS = 70                   # Aantal observasties waar modellen mee worden getraind
PREDICTION_WINDOW = 2            # Voorspelwindow
HOLIDAY_FORWARD = 2
N_LAGS = 3                       # Aantal weken dat wordt teruggekeken
MA_PERIOD = 5                    # Aantal weken waar benchmark over middelt
DIFFERENCING = False             # Eerste verschillen nemen voor voorspelling
BOOTSTRAP_ITER = 1              # Aantal bootstraps voor grenswaarden voorspelling
MODEL_TYPE = "OLS"               # Voorspelalgoritme: OLS, Poisson, XGBoost, etc
MAX_FEATURES = 30                # Maximaal te selecteren features in optimalisatie
MIN_CORRELATION = 0.2            # Minimum correlatie die een feature moet hebben
FEATURE_OPT = [MIN_CORRELATION,  # Combinatie van bovenstaande parameters
               MAX_FEATURES]
FEATURE_PERIOD_LENGTH = 2500     # Number of days to generate for features
WEATHER_FORECAST = False          # Weersfactoren worden ook als voorspelling meegenomen
