import hff_predictor.generic.files
import pandas as pd
import numpy as np
import hff_predictor.config.column_names as cn
import hff_predictor.config.file_management as fm
from hff_predictor.model.preprocessing import fit_model, predictor

import logging
LOGGER = logging.getLogger(__name__)


def get_top_correlations(y: pd.DataFrame, y_lags: pd.DataFrame, top_correl: int = 5) -> tuple:
    """
    Bepaalt de variabelen die het meest correleren met geselecteerde target

    :param y: De target variabele, in dit geval vaak de bestellingen per halffabricaat
    :param y_lags: De verklarende variabelen
    :param top_correl: Top x variabelen die moeten worden geselecteerd
    :return: Top x correlerende variabelen
    """

    # Onderstaande operaties bepalen zoveel mogelijk o.b.v. lineaire algebra de correlaties voor heel veel variabelen

    # Correlatie formule deel 1
    A_mA = y - y.mean()
    B_mB = y_lags - y_lags.mean()

    # Correlatie formule deel 2
    ssA = (A_mA ** 2).sum()
    ssB = (B_mB ** 2).sum()

    # Correlatie formule deel 3
    numerator = np.dot(A_mA.T, B_mB)
    denominator = np.sqrt(np.dot(pd.DataFrame(ssA), pd.DataFrame(ssB).T))

    # De correlaties
    correls = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )

    # Correlaties opgeslagen als dataframe, met goede kolommen en index
    corrs = pd.DataFrame(correls, index=y.columns, columns=y_lags.columns)

    # Negeert correlaties met zichzelf, die worden apart behandeld in AR optimalisatie
    for i in corrs.index:
        for j in corrs.columns:
            if i == j[:-7]:
                corrs.loc[i, j] = -1e9

    # Bepaal top correlaties
    top_correlations = {}

    # Als alleen de meest correlerende feature eruit moet worden gehaald
    if len(y.columns) == 1 & top_correl == 1:
        top_name = corrs.T.idxmax()[0]
        top_value = round(corrs[top_name].values[0], 3)
        return top_name, top_value

    # Als een grote set met top p correlerend variabelende moet worden bepaald
    else:
        for p in corrs.index:
            top_correlations[p] = (
                corrs.loc[p].sort_values(ascending=False)[:top_correl].index
            )

        # Finally get corr coeff
        return top_correlations, corrs


def optimize_ar_model(y: pd.Series, y_ar: pd.DataFrame, X_exog: pd.DataFrame, weather_forecast: bool,
                      constant: bool = True, model: str = "OLS"):
    """
    Optimaliseren van autoregressieve componenten in model

    :param y: Target variabele
    :param y_ar: Vertraagde autoregressieve componenten
    :param X_exog: Exogene variabelen
    :param constant: Constante toevoegen of niet
    :param model: Model type
    :return: Geoptimaliseerde AR componenten
    """

    sorted_lags = y_ar.columns.sort_values(ascending=True)
    # Optie om weersvoorspelling aan benchmark toe te voegen

    # Zet vooraf een startpunt voor optimale lags
    optimal_lags = 1
    min_fit_val = 1e9

    # Loop over verschillende lag opties heen
    for lag in range(1, len(sorted_lags) + 1):

        _y_ar = y_ar.iloc[:, :lag]
        X_ar = _y_ar

        if constant:
            X_ar.insert(0, "constant", 1)

        # Fit model met huidige lags
        _fit = fit_model(y=y, X=X_ar, model=model)

        # Bepaal in-sample fout
        _fit_value = round((abs(y - _fit.predict(X_ar)) / y).median(), 5)

        # Update optimale lags, als de fitwaarde beter is
        if _fit_value < min_fit_val:
            min_fit_val = _fit_value
            optimal_lags = lag

    # Optimale lag waarden
    lag_values = y_ar.iloc[:, :optimal_lags]

    X_exog_rf = X_exog
     #.drop(columns=drop_cols, inplace=False, errors="ignore")
    # lag_values.join(use_baseline_features, how="left")
    return lag_values, X_exog_rf


def batch_fit_model(Y: pd.DataFrame, Y_ar: pd.DataFrame, X_exog: pd.DataFrame, weather_forecast: bool,
                    add_constant: bool = True, model: str = "OLS", feature_threshold: list = None,
                    prediction_window: int = 2):
    """
    Hier worden de modellen gefit in batch vorm, of wel voor alle producten

    :param Y: Alle producten, werkelijke waarden
    :param Y_ar: Vertraagde (AR) componenten
    :param X_exog: Externe factoren
    :param add_constant: Voeg een constante toe indien nodig
    :param model: Type model
    :param feature_threshold: Grenswaarden optimalisatie features
    :param prediction_window:
    :return: Geschatte modellen
    """

    # Standaard settings voor feature optimalisatie
    if feature_threshold is None:
        feature_threshold = [0.2, 15]

    # Prepareer objecten om resultaten in te verzamelen
    Y_pred = pd.DataFrame(index=Y.index)
    fitted_models = {}
    all_params = {}
    optimized_ar_features = {}
    optimized_exog_features = {}

    # X_exog_nw = X_exog.drop(cn.WEATHER_PRED_COLS, inplace=False, axis=1, errors='ignore')

    if prediction_window == 2:
        X_weather_baseline = X_exog[cn.TEMP_GEM_N2W]
    else:
        X_weather_baseline = X_exog[cn.TEMP_GEM_N1W]

    all_weather_cols = cn.WEATHER_PRED_COLS
    all_weather_cols.extend([cn.TEMP_GEM_N1W, cn.TEMP_GEM_N2W])
    X_exog_nw = X_exog.drop(all_weather_cols, inplace=False, axis=1, errors='ignore')

    # Schat en optimaliseer model per product
    for product in Y.columns:
        y_name = product
        y = Y[y_name]

        # Haal de lags op voor dat product
        lag_index = [y_name in x for x in Y_ar.columns]
        y_ar = Y_ar.iloc[:, lag_index]

        # Haal de andere lags op, die mogelijk voorspelfactoren kunnen zijn
        lag_index_other = [y_name not in x for x in Y_ar.columns]
        y_ar_other = Y_ar.iloc[:, lag_index_other]

        # Optimaliseer eerst autoregressieve componenten
        ar_baseline, X_exog_rf = optimize_ar_model(y=y, y_ar=y_ar, X_exog=X_exog_nw,
                                                   constant=add_constant, model=model,
                                                   weather_forecast=weather_forecast)

        if weather_forecast:
            ar_baseline = ar_baseline.join(X_weather_baseline, how='left')

        # Schat het baseline model, nog zonder exogene factoren
        if cn.ADD_PLUS:
            sales_name = "{}_sales_cons_last".format(y_name)

            cols_check = [sales_name in x for x in X_exog_rf.columns]
            sales_cols = X_exog_rf.iloc[:, cols_check].columns

            if len(sales_cols):
                sales_cols.sort_values()
                sales_cols_select = sales_cols[:2]
                ar_baseline[sales_cols_select] = X_exog_rf[sales_cols_select]
                X_exog_rf.drop(sales_cols_select, axis=1, inplace=True)
                # LOGGER.debug("Found Plus sales column for {}, added to baseline.".format(y_name))

        baseline_fit = fit_model(y=y, X=ar_baseline, model=model)

        # Collectie van alle mogelijke factoren
        all_possible_features = y_ar_other.join(X_exog_rf, how="left")

        # Overgebleven residuen als verschil tussen werkelijke waarde en geschatte baseline model
        resid = y - baseline_fit.predict(ar_baseline)
        correlation_val = 1 # startwaarde correlatie
        selected_features = ar_baseline.copy(deep=True) # startset met geselecteerde features

        if add_constant:
            selected_features.insert(0, "constant", 1)

        # Deze functie blijft lopen totdat het geselecteerde aantal features zijn toegevoegd met hoogste correlaties
        while (correlation_val > feature_threshold[0]
               and selected_features.shape[1] < feature_threshold[1]):

            # Bepaalde top correlaties met huidgie residuen
            corr_name, correlation_val = get_top_correlations(y=pd.DataFrame(resid),
                                                              y_lags=all_possible_features, top_correl=1)

            # Voeg hoogst correlerende feature toe aan set geselecteerde features
            selected_features = selected_features.join(
                all_possible_features[corr_name], how="left"
            )

            # Verwijder deze feature dan uit de mogelijk te selecteren variabelen
            all_possible_features.drop(corr_name, axis=1, inplace=True)

            # Schat model met huidige features en bewaar nieuwe residuen
            mdl_fit = fit_model(y=y, X=selected_features, model=model)
            resid = y - mdl_fit.predict(selected_features)

        # Verzamel de AR componenten
        ar_name = "{}_last".format(y_name)
        ar_cols = [ar_name in x for x in selected_features.columns]

        # Selecteer de exogenene factoren en scheid ze van de AR factoren
        exog_cols = [not x for x in ar_cols]
        ar_features = selected_features.iloc[:, ar_cols]
        exog_features = selected_features.iloc[:, exog_cols]

        # Schat het finale model en bewaar de fit, parameters en optimale AR factoren
        Y_pred[y_name] = mdl_fit.predict(selected_features)
        fitted_models[y_name] = mdl_fit
        all_params[y_name] = selected_features.columns
        optimized_ar_features[y_name] = ar_features.columns
        optimized_exog_features[y_name] = exog_features.columns

    return Y_pred, fitted_models, all_params, optimized_ar_features, optimized_exog_features,


def batch_make_prediction(Yp_ar_m: pd.DataFrame, Yp_ar_nm: pd.DataFrame, Xp_exog: pd.DataFrame,
                          fitted_models: list, Yf_ar_opt: pd.DataFrame, Yf_exog_opt: pd.DataFrame,
                          weather_forecast: bool, add_constant: bool = True, prep_input: bool = True,
                          model_type: str = "OLS", prediction_window: int = 2, weather_values: dict = None,
                          find_comparable_model: bool = True):
    """
    Maakt hier de  voorspellingen in batch vorm

    :param Yp_ar_m: AR componenten van modelleerbare producten
    :param Yp_ar_nm: AR componenten vna niet modelleerbare producten
    :param Xp_exog: Exogene factoren
    :param fitted_models: Geschatte modellen
    :param Yf_ar_opt: Geoptimaliseerde AR componenten
    :param Yf_exog_opt: Geoptimaliseerde exogene factoren
    :param add_constant: Voeg een constante toe indien nodig
    :param prep_input: Bereid input voor
    :param model_type: Type model
    :param prediction_window:
    :param find_comparable_model: Vind een vergelijkbaar model voor niet voorspelbare modellen
    :return: Voorspelling per producten
    """

    # Transformeer tot DataFrame
    def series_to_dataframe(pd_series):
        return pd.DataFrame(pd_series).transpose()

    if prep_input:
        Yp_ar_m = series_to_dataframe(Yp_ar_m)
        Yp_ar_nm = series_to_dataframe(Yp_ar_nm)
        Xp_exog = series_to_dataframe(Xp_exog)

    Y_pred = pd.DataFrame(index=Yp_ar_m.index)

    fit_params_wf = {}
    fit_params_reg = {}

    if weather_values is not None:
        Y_pred_bw = pd.DataFrame(index=Yp_ar_m.index)
        Y_pred_ww = pd.DataFrame(index=Yp_ar_m.index)

    # Verwijder de lag underscript ('_last1w'), is exact 7 tekens
    Ym_products = list(set([x[:-7] for x in Yp_ar_m.columns]))  # Remove 'lag' tag

    # Maak een voorspelling per product
    for y_name_m in Ym_products:

        # Haal vertraagde (lag) kolommen op, om juiste featureset samen te stellen o.b.v model fit
        lag_index = [y_name_m in x for x in Yp_ar_m.columns]
        Xp_ar_m = Yp_ar_m.iloc[:, lag_index]

        Xf_ar_m = Yf_ar_opt[y_name_m]
        Xp_ar_m = Xp_ar_m.iloc[:, :Xf_ar_m.shape[0]]

        # Voeg de vertraagde kolommen toe aan de feature set
        Xp_all_features = Yp_ar_m.join(Xp_exog, how="left")

        Xf_exog_m = Yf_exog_opt[y_name_m].drop("constant")

        Xp_arx_m = Xp_all_features[Xf_exog_m]

        if add_constant:
            Xp_ar_m.insert(0, "constant", 1)

        # Totale set met features
        Xp_tot = Xp_ar_m.join(Xp_arx_m, how="left")


        # Maak voorspelling aan de hand van predictor functie
        if weather_values is not None:
            fit_params_wf[y_name_m] = fitted_models[y_name_m].params

            Y_pred_bw[y_name_m], Y_pred_ww[y_name_m] = predictor(Xpred=Xp_tot, fitted_model=fitted_models[y_name_m],
                                                                 model=model_type, weather_scenario=weather_values,
                                                                 prediction_window=prediction_window)
        else:


            fit_params_reg[y_name_m] = fitted_models[y_name_m].params
            Y_pred[y_name_m] = predictor(Xpred=Xp_tot, fitted_model=fitted_models[y_name_m],
                                         model=model_type, weather_scenario=weather_values,
                                         prediction_window=prediction_window)



    # pd.DataFrame(fit_params_reg).to_csv("params_reg.csv", sep=";", decimal=",")

    # Selecteer hier de producten die niet-modelleerbaar zijn
    Ynm_products = list(set([x[:-7] for x in Yp_ar_nm.columns]))
    for y_name_nm in Ynm_products:

        lag_index = [y_name_nm in x for x in Yp_ar_nm.columns]
        Xp_ar_nm = Yp_ar_nm.iloc[:, lag_index]

        # Zoek naar een vergelijkbaar product om toch een voorspelling te maken
        if find_comparable_model:
            # Find product which has similar magnitude absolute sales
            lag_val = "_last0w"

            # Haal de laatst beschikbare waarde op van te voorspellen pnroduct
            _y_nm_val = Yp_ar_nm["{}{}".format(y_name_nm, lag_val)][0]

            # Haal de waarden op over dezelfde periode van modelleerbare producten
            lag1_index = [lag_val in x for x in Yp_ar_m.columns]
            _Y_m_vals = Yp_ar_m.iloc[:, lag1_index]

            # Selecteer meest vergelijkbare product als product die er abosluut gezien het dichtst bij zit
            _closest_prod = (abs(_Y_m_vals - _y_nm_val) / _y_nm_val).T
            closest_product_name = _closest_prod.idxmin()[0][:-7]

        else:
            # Alternatief model is simpelweg de som van modelleerbare producten
            closest_product_name = cn.MOD_PROD_SUM

        # Vanaf hier worden dezelfde stappen utigevoerd als voor modelleerbare producten

        Xf_ar_cp = Yf_ar_opt[closest_product_name]

        Xp_ar_nm = Xp_ar_nm.iloc[:, : Xf_ar_cp.shape[0]]

        Xp_all_features = Yp_ar_m.join(Xp_exog, how="left")

        Xf_exog_cp = Yf_exog_opt[closest_product_name].drop("constant")

        Xp_arx_cp = Xp_all_features[Xf_exog_cp]

        if add_constant:
            Xp_ar_nm.insert(0, "constant", 1)

        Xp_tot = Xp_ar_nm.join(Xp_arx_cp, how="left")

        if weather_values is not None:
            Y_pred_bw[y_name_nm], Y_pred_ww[y_name_nm] = predictor(Xpred=Xp_tot,
                                                                   fitted_model=fitted_models[closest_product_name],
                                                                   model=model_type, weather_scenario=weather_values,
                                                                   prediction_window=prediction_window)

        else:
            Y_pred[y_name_nm] = predictor(Xpred=Xp_tot, fitted_model=fitted_models[closest_product_name],
                                          model=model_type, weather_scenario=weather_values,
                                          prediction_window=prediction_window)

    if weather_values is not None:
        Y_pred_bw = Y_pred_bw.T
        Y_pred_bw.columns = ['beter_weer']

        Y_pred_ww = Y_pred_ww.T
        Y_pred_ww.columns = ['slechter_weer']

        Y_pred = Y_pred_bw.join(Y_pred_ww)

    return Y_pred


def fit_and_predict(fit_dict: dict, predict_dict: dict, weather_forecast: bool, model_type: str = "OLS",
                    bootstrap: bool = False, feature_threshold: list = None, prediction_window: int = 2) -> tuple:
    """
    Combinatie functie van schatten van het model en het maken van de voorspelling

    :param fit_dict: Verzamelobject om model te fitten
    :param predict_dict: Verzamel object om model te schatten
    :param model_type: Type voorspelmodel
    :param bootstrap: Bootstrap functie
    :param feature_threshold: Optimalisatie parameters
    :param prediction_window
    :return: Voorspellingen
    """

    def reset_index(data: pd.DataFrame) -> pd.DataFrame:
        """
        Hulpfunctie om indices te resetten

        :param data: Data met verkeerde index
        :return: Data met nieuwe index, tbv bootstrapping
        """
        data_new = data.reset_index(drop=True, inplace=False)
        data_new["bootstrap_index"] = np.arange(data.shape[0])
        return data_new.set_index("bootstrap_index", inplace=False, drop=True)

    if feature_threshold is None:
        feature_threshold = [0.2, 15]

    # Input variabelen, om eventueel bootstrap samples van te maken
    Y_org = fit_dict[cn.Y_TRUE]
    Yar_org = fit_dict[cn.Y_AR]
    X_org = fit_dict[cn.X_EXOG]

    # Data wordt tot een bootstrap sample gemaakt, wat in feite een random sample is van het origineel
    if bootstrap:
        Y_fit = Y_org.sample(n=Y_org.shape[0], replace=True)
        Yar_fit = Yar_org.loc[Y_fit.index, :]
        X_fit = X_org.loc[Y_fit.index, :]

        Y_fit = reset_index(data=Y_fit)
        Yar_fit = reset_index(data=Yar_fit)
        X_fit = reset_index(data=X_fit)

    else:
        Y_fit = Y_org
        Yar_fit = Yar_org
        X_fit = X_org


    # Fit standaard model
    Yis_fit, model_fits, all_pars, Yar_opt, X_opt = batch_fit_model(
        Y=Y_fit,
        Y_ar=Yar_fit,
        add_constant=True,
        X_exog=X_fit,
        model=model_type,
        feature_threshold=[feature_threshold[0], feature_threshold[1]],
        weather_forecast=weather_forecast,
        prediction_window=prediction_window
    )

    # Maak standaard voorspellingen
    Yos_pred= batch_make_prediction(
        Yp_ar_m=predict_dict[cn.Y_AR_M],
        Yp_ar_nm=predict_dict[cn.Y_AR_NM],
        Xp_exog=predict_dict[cn.X_EXOG],
        fitted_models=model_fits,
        Yf_ar_opt=Yar_opt,
        Yf_exog_opt=X_opt,
        add_constant=True,
        model_type=model_type,
        find_comparable_model=True,
        prediction_window=prediction_window,
        weather_forecast=weather_forecast
    )
    # Fit weersvoorspelling model

    if bootstrap:
        return Yis_fit, Yos_pred, all_pars

    if not bootstrap:
        wYis_fit, wmodel_fits, wall_pars, wYar_opt, wX_opt = batch_fit_model(
            Y=Y_fit,
            Y_ar=Yar_fit,
            add_constant=True,
            X_exog=X_fit,
            model=model_type,
            prediction_window=prediction_window,
            feature_threshold=[feature_threshold[0], feature_threshold[1]],
            weather_forecast=True
        )

        temperature = 'temperatuur_gem_next{}w'.format(prediction_window)
        sun_hours = 'zonuren_next{}w'.format(prediction_window)
        rain = 'neerslag_mm_next{}w'.format(prediction_window)

        weather_values = {temperature: np.mean(X_fit[temperature][0:1]),
                          sun_hours: np.mean(X_fit[sun_hours][0:1]),
                          rain: np.mean(X_fit[rain][0:1])}

        wYos_pred = batch_make_prediction(
            Yp_ar_m=predict_dict[cn.Y_AR_M],
            Yp_ar_nm=predict_dict[cn.Y_AR_NM],
            Xp_exog=predict_dict[cn.X_EXOG],
            fitted_models=wmodel_fits,
            Yf_ar_opt=wYar_opt,
            Yf_exog_opt=wX_opt,
            add_constant=True,
            model_type=model_type,
            find_comparable_model=True,
            prediction_window=prediction_window,
            weather_forecast=True,
            weather_values=weather_values
        )

        return Yis_fit, Yos_pred, all_pars, wYos_pred




def init_train():
    fit_dict = hff_predictor.generic.files.read_pkl(
        file_name=fm.FIT_DATA, data_loc=fm.SAVE_LOC
    )
    predict_dict = hff_predictor.generic.files.read_pkl(
        file_name=fm.PREDICT_DATA, data_loc=fm.SAVE_LOC
    )

    Yis_fit, model_fits, all_pars, ar_f, exog_f = batch_fit_model(
        Y=fit_dict[cn.Y_TRUE],
        Y_ar=fit_dict[cn.Y_AR],
        X_exog=fit_dict[cn.X_EXOG],
        model="OLS",
        feature_threshold=[0.2, 25],
    )

    Yis_fit, Yos_pred, all_pars = batch_make_prediction(
        Yp_ar_m=predict_dict[cn.Y_AR_M],
        Yp_ar_nm=predict_dict[cn.Y_AR_NM],
        Xp_exog=predict_dict[cn.X_EXOG],
        Yf_ar_opt=ar_f,
        Yf_exog_opt=exog_f,
        fitted_models=model_fits,
        find_comparable_model=True,
    )

    yisfit, yosfit, pars = fit_and_predict(
        fit_dict=fit_dict, predict_dict=predict_dict, feature_threshold=[0.2, 25]
    )
