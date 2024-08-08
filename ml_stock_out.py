###############################################################Import APIs#####################################################
import os
import numpy as np
import datetime
from sqlalchemy import create_engine, BigInteger, Integer, String, Date
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import  Dense, LSTM, GRU
from keras.callbacks import EarlyStopping
import psycopg2
from psycopg2 import sql

##########################################Read Data###############################################################
def select_fact_inventaire(user, pwd, host, port, dbname, schema, table) :
    engine = create_engine(f'postgresql://{user}:{pwd}@{host}:{port}/{dbname}')
    connexion = engine.connect()
    df_fact_inventaire = pd.read_sql_table(table_name=table, schema=schema, con=engine)
    connexion.close()
    return df_fact_inventaire

##############################################################################################Prevision des ventes#####################################################################################################
##########################################Data Preprocessing###############################################################
def get_prep_sales(df_fact_inventaire) :
    df_prep_vente = df_fact_inventaire[['DATE_MVT', 'ID_ARTICLE', 'NOM_ARTICLE', 'SOMME_QTE_VENTE']]
    #df_prep_vente = df_prep_vente.set_index('DATE_MVT')
    return df_prep_vente

def sales_per_month(df_prep_vente):
    df_prep_vente_copy = df_prep_vente.copy()
    df_prep_vente_copy['MOIS'] = df_prep_vente_copy['DATE_MVT'].dt.strftime('%Y-%m')
    # Grouper les données par mois et par produit, puis calculer la somme de QTE_SORT pour chaque groupe
    df_vente_month = df_prep_vente_copy.groupby(['ID_ARTICLE', 'NOM_ARTICLE', 'MOIS'])['SOMME_QTE_VENTE'].sum().reset_index()
    return df_vente_month

##########################################Data Analysis###############################################################
##Courbe de distribution de vente par jour
def visualize_sales_per_article_day(df_prep_vente) :
    # Créer un graphique pour le NOM_ARTICLE actuel
    plt.figure(figsize=(14, 9))
    plt.bar(df_prep_vente['DATE_MVT'], df_prep_vente['SOMME_QTE_VENTE'], color='lightblue')
    plt.title(f"Quantité vendue par jour pour {df_prep_vente['NOM_ARTICLE']}")
    plt.xlabel("Date")
    plt.ylabel("Quantité de vente")
    plt.legend()
    plt.grid(True)
    plt.show()
##Courbe de distribution de vente par mois
def visualize_sales_per_article_month(df_vente_month) :
    # Créer un graphique pour le NOM_ARTICLE actuel
    plt.figure(figsize=(14, 9))
    plt.bar(df_vente_month['MOIS'], df_vente_month['SOMME_QTE_VENTE'], color='lightblue')
    plt.title(f"Quantité vendue par mois pour {df_vente_month['NOM_ARTICLE']}")
    plt.xlabel("Mois")
    plt.ylabel("Quantité de vente")
    plt.legend()
    plt.grid(True)
    plt.show()


#Tester la stationnarité de la série tomporelle
def test_serie_stationnaire_sales(df_prep_vente) :
    # Appliquer le test ADF à la colonne QTE_SORT de la DataFrame monthly_sales
    resultat_adf = adfuller(df_prep_vente['SOMME_QTE_VENTE'])
    # Afficher les résultats
    print('Statistique ADF : %f' % resultat_adf[0])
    print('p-value : %f' % resultat_adf[1])
    print('Valeurs Critiques :')
    for cle, valeur in resultat_adf[4].items():
        print('\t%s: %.3f' % (cle, valeur))
        # Interprétation
    if resultat_adf[1] < 0.05:
        print("La série de vente journalière est stationnaire.")
        return True
    else:
        print("La série de vente journalière n'est pas stationnaire.")
        return False
    
############################################################################################Prédiction avec ARIMA############################################################################################################
#Récupérer les paramètres d'ARIMA
def get_arima_parameters_sales(df_prep_vente) :
    arima_params = {}
    arima_params_article = {}
    article = df_prep_vente['NOM_ARTICLE'].iloc[0]
    # Utiliser auto_arima pour trouver les meilleurs paramètres ARIMA pour l'article actuel
    model_arima = auto_arima(df_prep_vente['SOMME_QTE_VENTE'], start_p=1, start_q=1,
                         max_p=3, max_q=3,
                         seasonal=False,
                         trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)
    # Stocker les paramètres ARIMA dans le dictionnaire
    arima_params[article] = model_arima.get_params()
    # Modifier ici pour stocker directement les paramètres du meilleur modèle
    arima_params_article[article] = model_arima.order
    # Afficher les paramètres ARIMA trouvés pour chaque article
    print(f"Paramètres ARIMA pour {article}: {arima_params_article[article]}")
    return arima_params_article

#Lancer le modèle et Calculer les métriques de performance de modèle ARIMA
def evaluate_model_arima_sales(df_prep_vente,arima_params_article) :    
    # netoyage des données
    df_prep_vente.dropna(inplace=True)
    # Définir la liste des dates de test
    dates_test = df_prep_vente.index[int(len(df_prep_vente) * 0.65):]
    # Créer un DataFrame pour les prédictions
    df_predictions = pd.DataFrame(index=dates_test)
    performance = []
    # Boucle sur chaque article
    for article, order in arima_params_article.items():
        print("Processing article:", article)
    # Division des données en ensembles d'entraînement et de test
        train_size = int(len(df_prep_vente) * 0.65)  # Utilisation de 80% des données pour l'entraînement
        train, test = df_prep_vente.iloc[:train_size], df_prep_vente.iloc[train_size:]
        # Ajustement du modèle ARIMA aux données d'entraînement
        model = ARIMA(train['SOMME_QTE_VENTE'], order=order)
        model_fit = model.fit()
        # Évaluation de la performance du modèle
        predictions = model_fit.predict(start=len(train),end= len(train)+len(test)-1,typ='levles')
        df_predictions['SOMME_QTE_VENTE'] = predictions
        # Calculer les mesures de performance
        rmse = mean_squared_error(test['SOMME_QTE_VENTE'], predictions, squared=False)
        mae = mean_absolute_error(test['SOMME_QTE_VENTE'], predictions)
        r2 = r2_score(test['SOMME_QTE_VENTE'], predictions)
        # Stocker les performances dans la liste
        performance.append({'Model':'ARIMA','Article': article, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
        # Convertir la liste en DataFrame pour l'affichage
        df_performance_arima = pd.DataFrame(performance)
        # Tracer les Prédictions pour l'année 2023
        plt.figure(figsize=(10, 6))
        plt.plot(df_prep_vente.index, df_prep_vente['SOMME_QTE_VENTE'], label='Valeurs réelles')
        plt.plot(test.index, predictions, color='red', label='Prédictions')
        plt.title(f'Prédiction des ventes pour {article} en 2023 avec ARIMA')
        plt.xlabel('Date de mouvement')
        plt.ylabel('Quantité de vente')
        plt.legend()
        plt.grid(True)
        plt.show()
    return df_performance_arima

#Forecast avec le modèle ARIMA
def forecast_sales_arima(df_prep_vente, arima_params_article):
    # Fit ARIMA model (p,d,q)
    nom_article = df_prep_vente['NOM_ARTICLE'].iloc[0]
    model = ARIMA(df_prep_vente['SOMME_QTE_VENTE'], order=arima_params_article[nom_article])
    model_arima_fit = model.fit()
    # Forecast for each business day of the year 2024
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    forecast_index = pd.bdate_range(start=start_date, end=end_date)
    num_weekdays = len(forecast_index)
    forecast_arima = model_arima_fit.forecast(steps=num_weekdays)
    # Convertir les Prédictions en DataFrame et définir l'index
    forecast_vente_df = pd.DataFrame(forecast_arima.values, index=forecast_index, columns=['PREV_SOMME_QTE_VENTE'])
    forecast_vente_df['PREV_SOMME_QTE_VENTE'] = forecast_vente_df['PREV_SOMME_QTE_VENTE'].round().astype(int)
    # Récupérer dynamiquement les valeurs pour 'ID_ARTICLE' et 'NOM_ARTICLE'
    id_article = df_prep_vente['ID_ARTICLE'].iloc[0]
    # Ajouter à forecast_stock_df
    forecast_vente_df['ID_ARTICLE'] = id_article
    forecast_vente_df['NOM_ARTICLE'] = nom_article
    # Tracer les ventes réelles et les Prédictions de ventes pour les jours ouvrés de l'année 2024
    plt.figure(figsize=(14, 7))
    plt.plot(df_prep_vente.index, df_prep_vente['SOMME_QTE_VENTE'], label='Historique des ventes')
    plt.plot(forecast_vente_df.index, forecast_vente_df['PREV_SOMME_QTE_VENTE'], label='Prédiction des ventes', linestyle='--')
    plt.title(f'Prédiction des ventes pour {nom_article} sur les jours ouvrés de 2024 avec ARIMA')
    plt.xlabel('Date de mouvement')
    plt.ylabel('Quantité de vente')
    plt.legend()
    plt.grid(True)
    plt.show()
    return forecast_vente_df

############################################################################################Prédiction avec SARIMA############################################################################################################
#Récupérer les paramètres de SARIMA
def get_sarima_parameters_sales(df_prep_vente) :
    article = df_prep_vente['NOM_ARTICLE'].iloc[0]
    # Initialiser un dictionnaire pour stocker les paramètres ARIMA de chaque article
    sarima_params_article = {}
    # Utiliser auto_arima pour trouver les meilleurs paramètres ARIMA pour l'article actuel
    model_sarima = auto_arima(df_prep_vente['SOMME_QTE_VENTE'], start_p=1, start_q=1,
                   max_p=3, max_q=3, m=12,
                   start_P=0, seasonal=True,
                   trace=True,
                   error_action='ignore',  
                   suppress_warnings=True, 
                   stepwise=True)
    # Extraire les paramètres ARIMA et saisonniers
    order = model_sarima.get_params()['order']
    seasonal_order = model_sarima.get_params()['seasonal_order']
    # Stocker les paramètres SARIMA dans le dictionnaire
    sarima_params_article[article] = {'order': order, 'seasonal_order': seasonal_order}
    # Afficher les paramètres SARIMA trouvés pour chaque article
    print("Paramètres SARIMA pour chaque article:")
    for article, params in sarima_params_article.items():
        print(f"{article}: {params}")
    return sarima_params_article

#Lancer le modèle et Calculer les métriques de performance de modèle SARIMA
def evaluate_model_sarima_sales(df_prep_vente,sarima_params_article) :    
    # Définir la liste des dates de test
    dates_test = df_prep_vente.index[int(len(df_prep_vente) * 0.65 ):]
    # Créer un DataFrame pour les prédictions
    df_predictions = pd.DataFrame(index=dates_test)
    performance = []
    # Boucle sur chaque article
    for article, params in sarima_params_article.items():
        print("Processing article:", article)
        # Division des données en ensembles d'entraînement et de test
        train_size = int(len(df_prep_vente) * 0.65 )  # Utilisation de 80% des données pour l'entraînement
        train, test = df_prep_vente.iloc[:train_size], df_prep_vente.iloc[train_size:]
        order = params['order']
        seasonal_order = params['seasonal_order']
        model = SARIMAX(train['SOMME_QTE_VENTE'], order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        # Évaluation de la performance du modèle
        predictions = model_fit.predict(start=len(train),end= len(train)+len(test)-1,typ='levels')
        predictions.index=dates_test
        df_predictions['PREV_SOMME_QTE_VENTE'] = predictions
        # Calculer les mesures de performance
        rmse = mean_squared_error(test['SOMME_QTE_VENTE'], predictions, squared=False)
        mae = mean_absolute_error(test['SOMME_QTE_VENTE'], predictions)
        r2 = r2_score(test['SOMME_QTE_VENTE'], predictions)
        # Stocker les performances dans la liste
        performance.append({'Article': article, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
    # Convertir la liste en DataFrame pour l'affichage
    df_performance = pd.DataFrame(performance)
    # Tracer les Prédictions pour chaque article
    for article in df_predictions.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df_predictions.index, df_predictions['PREV_SOMME_QTE_VENTE'], label='Prédiction des ventes',color='blue')
        plt.plot(df_prep_vente.index, df_prep_vente['SOMME_QTE_VENTE'], label='Historique des ventes', color='red')
        plt.title(f'Prédiction des ventes pour {article} en 2023 avec SARIMA')
        plt.xlabel('Date de mouvement')
        plt.ylabel('Quantité de vente')
        plt.legend()
        plt.grid(True)
        plt.show()
    return df_performance

#Forecast avec le modèle SARIMA
def forecast_sales_sarima(df_prep_vente, sarima_params_article):
    nom_article = df_prep_vente['NOM_ARTICLE'].iloc[0]
    model_sarima = SARIMAX(df_prep_vente['SOMME_QTE_VENTE'],
                           order=sarima_params_article[nom_article]['order'],
                           seasonal_order=sarima_params_article[nom_article]['seasonal_order'])
    model_sarima_fit = model_sarima.fit()
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    forecast_index = pd.bdate_range(start=start_date, end=end_date)
    num_weekdays = len(forecast_index)
    forecast_sarima = model_sarima_fit.forecast(steps=num_weekdays)
    forecast_stock_df = pd.DataFrame(forecast_sarima, index=forecast_index, columns=['PREV_SOMME_QTE_VENTE'])
    # Remplacer NA par 0 et les valeurs infinies par la valeur maximale des données non infinies
    forecast_stock_df['PREV_SOMME_QTE_VENTE'] = forecast_stock_df['PREV_SOMME_QTE_VENTE'].replace([np.inf, -np.inf], np.nan).fillna(0)
    # Conversion en entiers est maintenant possible
    forecast_stock_df['PREV_SOMME_QTE_VENTE'] = forecast_stock_df['PREV_SOMME_QTE_VENTE'].astype(int)
    id_article = df_prep_vente['ID_ARTICLE'].iloc[0]
    forecast_stock_df['ID_ARTICLE'] = id_article
    forecast_stock_df['NOM_ARTICLE'] = nom_article
    plt.figure(figsize=(14, 7))
    plt.plot(df_prep_vente.index, df_prep_vente['SOMME_QTE_VENTE'], label='Historique des ventes')
    plt.plot(forecast_stock_df.index, forecast_stock_df['PREV_SOMME_QTE_VENTE'], label='Prédiction des ventes', linestyle='--')
    plt.title(f'Prédiction des ventes pour {nom_article} sur les jours ouvrés de 2024 avec SARIMA')
    plt.xlabel('Date de mouvement')
    plt.ylabel('Quantité de vente')
    plt.legend()
    plt.grid(True)
    plt.show()
    return forecast_stock_df

############################################################################################Prédiction avec LSTM############################################################################################################
#Fonction pour créer des séquences pour l'entraînement LSTM, modifiée pour travailler avec les données indexées par date
def create_dataset_lstm_sales(df_prep_vente,dataset, look_back):
    dataX, dataY = [], []
    dates = []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
        dates.append(df_prep_vente.index[i + look_back])
    return np.array(dataX), np.array(dataY), np.array(dates)
#Lancer le modèle et Calculer les métriques de performance de modèle LSTM
def evaluate_lstm_model_sales(df_prep_vente) :
    article = df_prep_vente['NOM_ARTICLE'].iloc[0]
    performance = []
    # Normalisation des données
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_prep_vente['SOMME_QTE_VENTE'].values.reshape(-1,1))
    look_back = 3
    # Division des données en ensembles d'entraînement et de test, avec les dates
    train_size = int(len(scaled_data) * 0.65)
    test_size = len(scaled_data) - train_size
    train, test = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]
    trainX, trainY, train_dates = create_dataset_lstm_sales(df_prep_vente,train, look_back)
    testX, testY, test_dates = create_dataset_lstm_sales(df_prep_vente,test, look_back)
    # Redimensionnement pour LSTM
    trainX = np.reshape(trainX, (trainX.shape[0], look_back, 1))  
    testX = np.reshape(testX, (testX.shape[0], look_back, 1))
    # Définition et entraînement du modèle LSTM
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, validation_data=(testX, testY), callbacks=[early_stop])
    # Prédictions et inversement de la normalisation
    testPredict = model.predict(testX)
    testPredict = scaler.inverse_transform(testPredict)
    testY_inverse = scaler.inverse_transform([testY])
    # Calcul des métriques de performance
    testScore_RMSE = np.sqrt(mean_squared_error(testY_inverse[0], testPredict[:,0]))
    testScore_MAE = mean_absolute_error(testY_inverse[0], testPredict[:,0])
    testScore_R2 = r2_score(testY_inverse[0], testPredict[:,0])
    # Stocker les performances dans la liste
    performance.append({'Model' : 'LSTM','Article': article,  'RMSE': testScore_RMSE, 'MAE': testScore_MAE, 'R2': testScore_R2})
    df_performance = pd.DataFrame(performance)
    # DataFrame pour les valeurs prédites avec dates
    predictions_df = pd.DataFrame({'DATE_MVT': test_dates, 'SOMME_QTE_VENTE': testY_inverse[0], 'PREV_SOMME_QTE_VENTE': testPredict[:,0]})
    predictions_df.set_index('DATE_MVT', inplace=True)
    # Affichage des courbes des valeurs réelles et prédites par date
    plt.figure(figsize=(10,6))
    plt.plot(predictions_df.index, predictions_df['SOMME_QTE_VENTE'], label='Historique des ventes')
    plt.plot(predictions_df.index, predictions_df['PREV_SOMME_QTE_VENTE'], label='Prédiction des ventes')
    plt.title(f'Prédiction des ventes pour {article} en 2023 avec LSTM')
    plt.xlabel('Date de mouvement')
    plt.ylabel('Quantité de vente')
    plt.legend()
    plt.grid(True)
    plt.show()
    return df_performance, model,scaled_data,scaler

#Forecast avec le modèle LSTM
def forecast_sales_lstm(df_prep_vente,model,scaled_data,scaler):
    article = df_prep_vente['NOM_ARTICLE'].iloc[0]
    predictions_2024 = []
    look_back = 3
    # Parcourir chaque mois et chaque jour pour 2024
    for month in range(1, 13):
        for day in range(1, 32):
            try:
                # Créer une date pour 2024 et vérifier si c'est un weekend
                date_2024 = pd.to_datetime(f'2024-{month}-{day}')
                # Si c'est un weekend (5 pour samedi, 6 pour dimanche), continuer à la prochaine itération
                if date_2024.weekday() in [5, 6]:
                    continue
                # Trouver la date correspondante en 2023
                date_2023 = date_2024.replace(year=2023)
                # Assurer que la date est valide et présente dans df_fact_vente
                if date_2023 not in df_prep_vente.index:
                    continue
                # Trouver l'index de cette date dans vos données normalisées
                idx = np.where(df_prep_vente.index == date_2023)[0][0]
                # Assurer qu'il y a assez de données pour le look_back
                if idx < look_back:
                    continue
                # Préparer l'entrée pour le modèle
                input_data = scaled_data[idx-look_back+1:idx+1].reshape(1, look_back, 1)
                # Faire la prédiction
                prediction = model.predict(input_data)
                # Inverser la normalisation
                prediction_inversed = scaler.inverse_transform(prediction)
                # Ajouter la prédiction à la liste
                predictions_2024.append((date_2024, prediction_inversed[0][0]))
            except Exception as e:
                print(e)
                pass
    # Conversion des prédictions en DataFrame
    predictions_df_lstm_2024 = pd.DataFrame(predictions_2024, columns=['DATE_MVT', 'PREV_SOMME_QTE_VENTE'])
    predictions_df_lstm_2024.set_index('DATE_MVT', inplace=True)
    predictions_df_lstm_2024['PREV_SOMME_QTE_VENTE'] = predictions_df_lstm_2024['PREV_SOMME_QTE_VENTE'].round()
    predictions_df_lstm_2024['PREV_SOMME_QTE_VENTE'] = predictions_df_lstm_2024['PREV_SOMME_QTE_VENTE'].astype(int)
    id_article = df_prep_vente['ID_ARTICLE'].iloc[0]
    nom_article = df_prep_vente['NOM_ARTICLE'].iloc[0]
    # Ajouter à forecast_stock_df
    predictions_df_lstm_2024['ID_ARTICLE'] = id_article
    predictions_df_lstm_2024['NOM_ARTICLE'] = nom_article
    # Afficher les prédictions
    plt.figure(figsize=(15,6))
    plt.plot(df_prep_vente.index, df_prep_vente['SOMME_QTE_VENTE'], label='Historique des ventes')
    plt.plot(predictions_df_lstm_2024.index, predictions_df_lstm_2024['PREV_SOMME_QTE_VENTE'], label='Prédiction des ventes')
    plt.title(f'Prédiction des ventes pour {article} en 2024 avec LSTM')
    plt.xlabel('Date de mouvement')
    plt.ylabel('Quantité de vente')
    plt.legend()
    plt.grid(True)
    plt.show()
    return predictions_df_lstm_2024
############################################################################################Prédiction avec GRU############################################################################################################
#Fonction pour créer des séquences pour l'entraînement GRU
def create_dataset_gru_sales(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

#Lancer le modèle et Calculer les métriques de performance de modèle GRU
def evaluate_gru_model_sales(df_prep_vente) :
    article = df_prep_vente['NOM_ARTICLE'].iloc[0]
    performance = []
    # Normalisation des données
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_prep_vente['SOMME_QTE_VENTE'].values.reshape(-1,1))
    # Division des données en ensembles d'entraînement
    train_size = int(len(scaled_data) * 0.65)
    test_size = len(scaled_data) - train_size
    train, test = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]
    look_back = 1
    trainX, trainY = create_dataset_gru_sales(train, look_back)
    testX, testY = create_dataset_gru_sales(test, look_back)
    # Redimensionnement pour GRU
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # Construction du modèle avec GRU
    model = Sequential()
    model.add(GRU(50, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # Entraînement du modèle
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2, validation_data=(testX, testY), callbacks=[early_stop])
    # Prédictions
    testPredict = model.predict(testX)
    # Inverser la normalisation pour les valeurs prédites et les valeurs réelles pour évaluation
    testPredict = scaler.inverse_transform(testPredict)
    testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))
    # Calcul des métriques de performance
    testScore_RMSE = np.sqrt(mean_squared_error(testY_inverse, testPredict))
    testScore_MAE = mean_absolute_error(testY_inverse, testPredict)
    testScore_R2 = r2_score(testY_inverse, testPredict)
    performance.append({'Model' : 'GRU','Article': article,  'RMSE': testScore_RMSE, 'MAE': testScore_MAE, 'R2': testScore_R2})
    df_performance = pd.DataFrame(performance)
    # Affichage des résultats
    dates = df_prep_vente.index[-len(testPredict):]
    plt.figure(figsize=(10,6))
    plt.plot(dates, testY_inverse.flatten(), 'b-', label='Historique des ventes')
    plt.plot(dates, testPredict.flatten(), 'r--', label='Prédiction des ventes')
    plt.title(f'Prédiction des ventes pour {article} en 2023 avec GRU')
    plt.xlabel('Date de mouvement')
    plt.ylabel('Quantité de vente')
    plt.legend()
    plt.grid(True)
    plt.show()
    return df_performance,model,scaled_data,scaler

#Forecast avec le modèle GRU
def forecast_sales_gru(df_prep_vente,model,scaled_data,scaler):
    article = df_prep_vente['NOM_ARTICLE'].iloc[0]
    look_back = 1
    predictions_2024 = []
    # Parcourir chaque mois et chaque jour pour 2024
    for month in range(1, 13):
        for day in range(1, 32):
            try:
                # Créer une date pour 2024 et vérifier si c'est un weekend
                date_2024 = pd.to_datetime(f'2024-{month}-{day}')
                # Si c'est un weekend (5 pour samedi, 6 pour dimanche), continuer à la prochaine itération
                if date_2024.weekday() in [5, 6]:
                    continue
                # Trouver la date correspondante en 2023
                date_2023 = date_2024.replace(year=2023)
                # Assurer que la date est valide et présente dans df_fact_vente
                if date_2023 not in df_prep_vente.index:
                    continue
                # Trouver l'index de cette date dans vos données normalisées
                idx = np.where(df_prep_vente.index == date_2023)[0][0]
                # Assurer qu'il y a assez de données pour le look_back
                if idx < look_back:
                    continue
                # Préparer l'entrée pour le modèle
                input_data = scaled_data[idx-look_back+1:idx+1].reshape(1, look_back, 1)
                # Faire la prédiction
                prediction = model.predict(input_data)
                # Inverser la normalisation
                prediction_inversed = scaler.inverse_transform(prediction)
                # Ajouter la prédiction à la liste
                predictions_2024.append((date_2024, prediction_inversed[0][0]))
            except Exception as e:
                print(e)
                pass
    # Conversion des prédictions en DataFrame
    predictions_df_gru_2024 = pd.DataFrame(predictions_2024, columns=['DATE_MVT', 'PREV_SOMME_QTE_VENTE'])
    predictions_df_gru_2024.set_index('DATE_MVT', inplace=True)
    predictions_df_gru_2024['PREV_SOMME_QTE_VENTE'] = predictions_df_gru_2024['PREV_SOMME_QTE_VENTE'].round()
    predictions_df_gru_2024['PREV_SOMME_QTE_VENTE'] = predictions_df_gru_2024['PREV_SOMME_QTE_VENTE'].astype(int)
    id_article = df_prep_vente['ID_ARTICLE'].iloc[-1]
    nom_article = df_prep_vente['NOM_ARTICLE'].iloc[-1]
    # Ajouter à forecast_stock_df
    predictions_df_gru_2024['ID_ARTICLE'] = id_article
    predictions_df_gru_2024['NOM_ARTICLE'] = nom_article
    # Afficher les prédictions
    plt.figure(figsize=(15,6))
    plt.plot(df_prep_vente.index, df_prep_vente['SOMME_QTE_VENTE'], label='Historique des ventes')
    plt.plot(predictions_df_gru_2024.index, predictions_df_gru_2024['PREV_SOMME_QTE_VENTE'], label='Prédiction des ventes')
    plt.title(f'Prédiction des ventes pour {article} en 2024 avec GRU')
    plt.xlabel('Date de mouvement')
    plt.ylabel('Quantité de vente')
    plt.legend()
    plt.grid(True)
    plt.show()
    return predictions_df_gru_2024

##############################################################################################Prevision de quantité de stock#####################################################################################################
##########################################Data Preprocessing###############################################################
def get_prep_stock(df_fact_inventaire) :
    df_prep_stock = df_fact_inventaire[['DATE_MVT', 'ID_ARTICLE', 'NOM_ARTICLE','QTE_STOCK_INITIAL','QTE_ENT']]
    return df_prep_stock

def calculate_somme_stock_month(df_prep_stock):
    df_prep_stock = df_prep_stock[~((df_prep_stock["DATE_MVT"].dt.month == 12) & (df_prep_stock["DATE_MVT"].dt.day == 31))]
    # Grouper par année et mois, et calculer la somme de QTE_STOCK_INITIAL et QTE_ENT
    df_stock_month = df_prep_stock.groupby(['ID_ARTICLE', 'NOM_ARTICLE', df_prep_stock['DATE_MVT'].dt.to_period("M")]).agg({"QTE_STOCK_INITIAL": "sum", "QTE_ENT": "sum"}).reset_index()
    # Calculer la quantité totale
    df_stock_month["SOMME_QTE_STOCK"] = df_stock_month["QTE_STOCK_INITIAL"] + df_stock_month["QTE_ENT"]
    # Ajuster le format de DATE_MVT pour correspondre exactement à la demande
    df_stock_month["DATE_MVT"] = df_stock_month["DATE_MVT"].dt.to_timestamp().dt.strftime("%Y-%m-%d")
    df_stock_month = df_stock_month.set_index('DATE_MVT')
    return df_stock_month
##########################################Data Analysis###################################################################
def visualize_stock_per_article_month(df_stock_month) : 
    # Créer un graphique pour le NOM_ARTICLE actuel
    plt.figure(figsize=(10, 6))
    plt.plot(df_stock_month.index, df_stock_month['SOMME_QTE_STOCK'], marker='o', linestyle='-', color='lightblue')
    plt.title(f"Quantité du stock par mois pour {df_stock_month['NOM_ARTICLE']}")
    plt.xlabel("Mois")
    plt.ylabel("Quantité du Stock")
    plt.legend()
    plt.grid(True)
    plt.show()

#Tester la stationnarité de la série tomporelle
def test_serie_stationnaire_stock(df_stock_month) :
    # Appliquer le test ADF à la colonne QTE_SORT de la DataFrame monthly_sales
    resultat_adf = adfuller(df_stock_month['SOMME_QTE_STOCK'])
    # Afficher les résultats
    print('Statistique ADF : %f' % resultat_adf[0])
    print('p-value : %f' % resultat_adf[1])
    print('Valeurs Critiques :')
    for cle, valeur in resultat_adf[4].items():
        print('\t%s: %.3f' % (cle, valeur))
        # Interprétation
    if resultat_adf[1] < 0.05:
        print("La série de vente journalière est stationnaire.")
        return True
    else:
        print("La série de vente journalière n'est pas stationnaire.")
        return False
############################################################################################Prédiction avec ARIMA############################################################################################################
#Récupérer les paramètres d'ARIMA
def get_arima_parameters_stock(df_stock_month) : 
    # Définir le nom de l'article
    article = df_stock_month['NOM_ARTICLE'].iloc[0]
    # Initialiser un dictionnaire pour stocker les paramètres ARIMA de chaque article
    arima_params = {}
    arima_params_article = {}
    # Utiliser auto_arima pour trouver les meilleurs paramètres ARIMA pour l'article actuel
    model_arima = auto_arima(df_stock_month['SOMME_QTE_STOCK'], start_p=1, start_q=1,
                         max_p=3, max_q=3,
                         seasonal=False,
                         trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)
    # Stocker les paramètres ARIMA dans le dictionnaire
    arima_params[article] = model_arima.get_params()
    # Modifier ici pour stocker directement les paramètres du meilleur modèle
    arima_params_article[article] = model_arima.order
    # Afficher les paramètres ARIMA trouvés pour chaque article
    print(f"Paramètres ARIMA pour {article}: {arima_params_article[article]}")
    return arima_params_article

#Lancer le modèle et Calculer les métriques de performance de modèle ARIMA
def evaluate_model_arima_stock(df_stock_month,arima_params_article) :    
    # Définir la liste des dates de test
    dates_test = df_stock_month.index[int(len(df_stock_month) * 0.65):]
    # Créer un DataFrame pour les prédictions
    df_predictions = pd.DataFrame(index=dates_test)
    performance = []
    # Boucle sur chaque article
    for article, order in arima_params_article.items():
        print("Processing article:", article)
        # Division des données en ensembles d'entraînement et de test
        train_size = int(len(df_stock_month) * 0.65)  # Utilisation de 80% des données pour l'entraînement
        train, test = df_stock_month.iloc[:train_size], df_stock_month.iloc[train_size:]
        # Ajustement du modèle ARIMA aux données d'entraînement
        model = ARIMA(train['SOMME_QTE_STOCK'], order=order)
        #model = ARIMA(train['AMOXICILLINE'], order=(1, 1, 2))
        model_fit = model.fit()
        # Évaluation de la performance du modèle
        predictions = model_fit.predict(start=len(train),end= len(train)+len(test)-1,typ='levles')
        df_predictions['SOMME_QTE_STOCK'] = predictions
        # Calculer les mesures de performance
        rmse = mean_squared_error(test['SOMME_QTE_STOCK'], predictions, squared=False)
        mae = mean_absolute_error(test['SOMME_QTE_STOCK'], predictions)
        r2 = r2_score(test['SOMME_QTE_STOCK'], predictions)
        # Stocker les performances dans la liste
        performance.append({'Model':'ARIMA','Article': article, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
        # Convertir la liste en DataFrame pour l'affichage
        df_performance = pd.DataFrame(performance)
        # Tracer les Prédictions
        plt.figure(figsize=(10, 6))
        plt.plot(df_stock_month.index, df_stock_month['SOMME_QTE_STOCK'], label='Historique des stocks')
        plt.plot(test.index, predictions, color='red', label='Prédiction des stocks')
        plt.title(f'Prédiction des stocks pour {article} en 2023 avec ARIMA')
        plt.xlabel('Date de mouvement')
        plt.ylabel('Quantité du stock')
        plt.legend()
        plt.grid(True)
        plt.show()
    return df_performance

#Forecast avec le modèle ARIMA
def forecast_stock_arima(df_stock_month, arima_params_article):
    article = df_stock_month['NOM_ARTICLE'].iloc[0]
    # Fit ARIMA model (p,d,q)
    model = ARIMA(df_stock_month['SOMME_QTE_STOCK'], order=arima_params_article[article])
    model_arima_fit = model.fit()
    # Forecast for the next 12 months (2024)
    forecast_arima = model_arima_fit.forecast(steps=12)
    # Génération de l'index des Prédictions qui suit la dernière date de vos données historiques
    last_historical_date = df_stock_month.index[-1]
    forecast_index = pd.date_range(start='2024-01-01', periods=12, freq='MS')
    # Convertir les Prédictions en DataFrame et définir l'index
    forecast_stock_df = pd.DataFrame(forecast_arima.values, index=forecast_index, columns=['PREV_SOMME_QTE_STOCK'])
    forecast_stock_df['PREV_SOMME_QTE_STOCK'] = forecast_stock_df['PREV_SOMME_QTE_STOCK'].round().astype(int)
    # Récupérer dynamiquement les valeurs pour 'ID_ARTICLE' et 'NOM_ARTICLE'
    id_article = df_stock_month['ID_ARTICLE'].iloc[-1]
    nom_article = df_stock_month['NOM_ARTICLE'].iloc[-1]
    # Ajouter à forecast_stock_df
    forecast_stock_df['ID_ARTICLE'] = id_article
    forecast_stock_df['NOM_ARTICLE'] = nom_article
    # Tracer les données historiques et des Prédictions de stock
    plt.figure(figsize=(14, 7))
    plt.plot(df_stock_month.index, df_stock_month['SOMME_QTE_STOCK'], label='Historique des stocks')
    plt.plot(forecast_stock_df.index, forecast_stock_df['PREV_SOMME_QTE_STOCK'], label='Prédiction des stocks', linestyle='--')
    plt.title(f'Prédiction des stocks pour {article} en 2024 avec ARIMA')
    plt.xlabel('Date de mouvement')
    plt.ylabel('Quantité du stock')
    plt.legend()
    plt.grid(True)
    plt.show()
    return forecast_stock_df
############################################################################################Prédiction avec SARIMA############################################################################################################
#Récupérer les paramètres de SARIMA
def get_sarima_parameters_stock(df_stock_month) :
    article = df_stock_month['NOM_ARTICLE'].iloc[0]
    # Initialiser un dictionnaire pour stocker les paramètres ARIMA de chaque article
    sarima_params_article = {}
    # Utiliser auto_arima pour trouver les meilleurs paramètres ARIMA pour l'article actuel
    model_sarima = auto_arima(df_stock_month['SOMME_QTE_STOCK'], start_p=1, start_q=1,
                       max_p=3, max_q=3, m=12,
                       start_P=0, seasonal=True,
                       trace=True,
                       error_action='ignore',  
                       suppress_warnings=True, 
                       stepwise=True)
    # Extraire les paramètres ARIMA et saisonniers
    order = model_sarima.get_params()['order']
    seasonal_order = model_sarima.get_params()['seasonal_order']
    # Stocker les paramètres SARIMA dans le dictionnaire
    sarima_params_article[article] = {'order': order, 'seasonal_order': seasonal_order}
    # Afficher les paramètres SARIMA trouvés pour chaque article
    print("Paramètres SARIMA pour chaque article:")
    for article, params in sarima_params_article.items():
        print(f"{article}: {params}")
    # Afficher le dictionnaire complet
    return sarima_params_article

#Lancer le modèle et Calculer les métriques de performance de modèle SARIMA
def evaluate_model_sarima_stock(df_stock_month,sarima_params_article):
    dates_test = df_stock_month.index[int(len(df_stock_month) * 0.65 ):]
    # Créer un DataFrame pour les prédictions
    df_predictions = pd.DataFrame(index=dates_test)
    performance = []
    # Boucle sur chaque article
    for article, params in sarima_params_article.items():
        print("Processing article:", article)
        # Division des données en ensembles d'entraînement et de test
        train_size = int(len(df_stock_month) * 0.65 )  
        train, test = df_stock_month.iloc[:train_size], df_stock_month.iloc[train_size:]
        print(test)
        order = params['order']
        seasonal_order = params['seasonal_order']
        model = SARIMAX(train['SOMME_QTE_STOCK'], order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        # Évaluation de la performance du modèle
        predictions = model_fit.predict(start=len(train),end= len(train)+len(test)-1,typ='levels')
        print(predictions)
        predictions.index=dates_test
        df_predictions['SOMME_QTE_STOCK'] = predictions
        # Calculer les mesures de performance
        rmse = mean_squared_error(test['SOMME_QTE_STOCK'], predictions, squared=False)
        mae = mean_absolute_error(test['SOMME_QTE_STOCK'], predictions)
        r2 = r2_score(test['SOMME_QTE_STOCK'], predictions)
        # Stocker les performances dans la liste
        performance.append({'Model':'SARIMA','Article': article, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
    # Convertir la liste en DataFrame pour l'affichage
    df_performance = pd.DataFrame(performance)
    # Tracer les Prédictions pour chaque article
    for article in df_predictions.columns:
        plt.figure(figsize=(14, 9))
        plt.plot(df_predictions.index, df_predictions['SOMME_QTE_STOCK'], label='Prediction des stocks',color='red')
        plt.plot(test.index, test['SOMME_QTE_STOCK'], label='Historique des stocks', color='blue')
        plt.title(f'Prédiction des stocks pour {article} en 2023 avec SARIMA')
        plt.xlabel('Date de mouvement')
        plt.ylabel('Quantité du stock')
        plt.legend()
        plt.grid(True)
        plt.show()
    return df_performance

#Forecast avec le modèle SARIMA
def forecast_stock_sarima(df_stock_month, sarima_params_article):
    article = df_stock_month['NOM_ARTICLE'].iloc[0]
    model_sarima = SARIMAX(df_stock_month['SOMME_QTE_STOCK'], order=sarima_params_article[article]['order'], seasonal_order=sarima_params_article[article]['seasonal_order'])
    model_sarima_fit = model_sarima.fit()
    # Forecast for the next 12 months (2024)
    forecast_sarima = model_sarima_fit.forecast(steps=12)
    # Génération de l'index des Prédictions qui suit la dernière date de vos données historiques
    last_historical_date = df_stock_month.index[-1]
    forecast_index = pd.date_range(start=last_historical_date + pd.DateOffset(months=1), periods=12, freq='MS')
    # Convertir les Prédictions en DataFrame et définir l'index
    forecast_stock_df = pd.DataFrame(forecast_sarima.values, index=forecast_index, columns=['PREV_SOMME_QTE_STOCK'])
    forecast_stock_df['PREV_SOMME_QTE_STOCK'] = forecast_stock_df['PREV_SOMME_QTE_STOCK'].round().astype(int)
    # Récupérer dynamiquement les valeurs pour 'ID_ARTICLE' et 'NOM_ARTICLE'
    id_article = df_stock_month['ID_ARTICLE'].iloc[-1]
    nom_article = df_stock_month['NOM_ARTICLE'].iloc[-1]
    forecast_stock_df['ID_ARTICLE'] = id_article
    forecast_stock_df['NOM_ARTICLE'] = nom_article
    # Tracer les données historiques et prédictions
    plt.figure(figsize=(14, 7))
    plt.plot(df_stock_month.index, df_stock_month['SOMME_QTE_STOCK'], label='Historique des stocks')
    plt.plot(forecast_stock_df.index, forecast_stock_df['PREV_SOMME_QTE_STOCK'], label='Prédiction des stocks', linestyle='--')
    plt.title(f'Prédiction des stocks pour {article} en 2024 avec SARIMA')
    plt.xlabel('Date de mouvement')
    plt.ylabel('Quantité du stock')
    plt.legend()
    plt.grid(True)
    plt.show()
    return forecast_stock_df

############################################################################################Prédiction avec LSTM############################################################################################################
# Préparation des séquences
def create_dataset_lstm_stock(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

#Lancer le modèle et Calculer les métriques de performance de modèle LSTM
def evaluate_lstm_model_stock(df_stock_month) :
    article = df_stock_month['NOM_ARTICLE'].iloc[0]
    performance = []
    # Division des données en ensembles d'entraînement et de test
    train_size = int(len(df_stock_month) * 0.65)
    train_df = df_stock_month.iloc[:train_size]
    test_df = df_stock_month.iloc[train_size:]
    # Normalisation des données
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled  = scaler.fit_transform(train_df['SOMME_QTE_STOCK'].values.reshape(-1,1))
    look_back = 1  
    X_train, y_train = create_dataset_lstm_stock(train_scaled, look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    # Construction du modèle LSTM
    model = Sequential([LSTM(4, input_shape=(1, look_back)),Dense(1)])
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Entraînement du modèle
    model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)
    # Préparer les données de test pour la prédiction
    test_scaled = scaler.transform(test_df["SOMME_QTE_STOCK"].values.reshape(-1,1))
    X_test, y_test = create_dataset_lstm_stock(test_scaled, look_back)
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    # Prédiction
    predictions_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled)
    # Calcul des métriques de performance
    y_true = test_df['SOMME_QTE_STOCK'].values[look_back:]  
    # Ajuster y_true pour qu'il corresponde à la longueur des prédictions
    y_true_adjusted = y_true[:len(predictions)]  # Assurez-vous que y_true et predictions ont la même longueur
    # Maintenant, calculez les métriques avec les tableaux ajustés
    mse = mean_squared_error(y_true_adjusted, predictions.flatten())
    mae = mean_absolute_error(y_true_adjusted, predictions.flatten())
    r2 = r2_score(y_true_adjusted, predictions.flatten())
    # Stocker les performances dans la liste
    performance.append({'Model' : 'LSTM','Article': article,  'RMSE': mse, 'MAE': mae, 'R2': r2})
    df_performance = pd.DataFrame(performance)
    index_for_plot = test_df.index[look_back:len(predictions) + look_back]
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, test_df['SOMME_QTE_STOCK'], label='Historique des stocks', marker='o', color='blue')
    plt.plot(index_for_plot, predictions, label='Prédiction des stocks', marker='x', color='red')
    plt.title('Prédiction des stocks pour {article} en 2023 avec LSTM')
    plt.xlabel('Date')
    plt.ylabel('SOMME_QTE_STOCK')
    plt.legend()
    plt.show()
    return df_performance,model,scaler

#Forecast avec le modèle LSTM
def forecast_stock_lstm(df_stock_month,model,scaler):
    article = df_stock_month['NOM_ARTICLE'].iloc[0]
    df_stock_month.index = pd.to_datetime(df_stock_month.index)
    # Normalisation de la colonne 'SOMME_QTE_STOCK' pour l'année 2023
    values_2023 = df_stock_month.loc['2023', 'SOMME_QTE_STOCK'].values.reshape(-1, 1)
    values_2023_scaled = scaler.transform(values_2023)
    # Préparation pour les prédictions de 2024
    predictions_2024 = []

    # Génération des prédictions pour chaque mois de 2024
    for month in range(1, 13):
        # Création de la date pour 2024
        date_2024 = pd.Timestamp(f'2024-{month:02d}-01')
        prediction_scaled = model.predict(values_2023_scaled[month - 1].reshape(1, 1, 1))
        prediction = scaler.inverse_transform(prediction_scaled)
        # Ajout de la prédiction à la liste
        predictions_2024.append((date_2024, prediction[0][0]))
    # Conversion des prédictions en DataFrame
    predictions_df_lstm_2024 = pd.DataFrame(predictions_2024, columns=['DATE_MVT', 'PREV_SOMME_QTE_STOCK'])
    predictions_df_lstm_2024.set_index('DATE_MVT', inplace=True)
    predictions_df_lstm_2024['PREV_SOMME_QTE_STOCK'] = predictions_df_lstm_2024['PREV_SOMME_QTE_STOCK'].round()
    predictions_df_lstm_2024['PREV_SOMME_QTE_STOCK'] = predictions_df_lstm_2024['PREV_SOMME_QTE_STOCK'].astype(int)
    id_article = df_stock_month['ID_ARTICLE'].iloc[0]
    nom_article = df_stock_month['NOM_ARTICLE'].iloc[0]
    # Ajouter à forecast_stock_df
    predictions_df_lstm_2024['ID_ARTICLE'] = id_article
    predictions_df_lstm_2024['NOM_ARTICLE'] = nom_article
    plt.figure(figsize=(15, 6))
    plt.plot(df_stock_month.index, df_stock_month['SOMME_QTE_STOCK'], label='Historique des stocks', marker='o', color='blue', linestyle='-')
    plt.plot(predictions_df_lstm_2024.index, predictions_df_lstm_2024['PREV_SOMME_QTE_STOCK'], label='Prédictions pour 2024', marker='x', color='red', linestyle='--')
    plt.title('Prédiction des stocks pour {article} en 2024 avec LSTM')
    plt.xlabel('Date')
    plt.ylabel('SOMME_QTE_STOCK')
    plt.legend()
    plt.grid(True)
    plt.show()
    return predictions_df_lstm_2024

############################################################################################Prédiction avec GRU############################################################################################################
# Préparation des séquences
def create_dataset_gru_stock(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

#Lancer le modèle et Calculer les métriques de performance de modèle GRU
def evaluate_gru_model_stock(df_stock_month) :
    article = df_stock_month['NOM_ARTICLE'].iloc[0]
    performance = []
    # Division des données en ensembles d'entraînement et de test
    train_size = int(len(df_stock_month) * 0.65)
    train_df = df_stock_month.iloc[:train_size]
    test_df = df_stock_month.iloc[train_size:]
    # Normalisation des données
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled  = scaler.fit_transform(train_df['SOMME_QTE_STOCK'].values.reshape(-1,1))
    look_back = 1  
    X_train, y_train = create_dataset_gru_stock(train_scaled, look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    # Construction du modèle LSTM
    model = Sequential([
        GRU(4, input_shape=(1, look_back)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Entraînement du modèle
    model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)
    # Préparer les données de test pour la prédiction
    test_scaled = scaler.transform(test_df["SOMME_QTE_STOCK"].values.reshape(-1,1))
    X_test, y_test = create_dataset_gru_stock(test_scaled, look_back)
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    # Prédiction
    predictions_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled)
    # Calcul des métriques de performance
    y_true = test_df['SOMME_QTE_STOCK'].values[look_back:]  
    # Ajuster y_true pour qu'il corresponde à la longueur des prédictions
    y_true_adjusted = y_true[:len(predictions)]  # Assurez-vous que y_true et predictions ont la même longueur
    # Maintenant, calculez les métriques avec les tableaux ajustés
    mse = mean_squared_error(y_true_adjusted, predictions.flatten())
    mae = mean_absolute_error(y_true_adjusted, predictions.flatten())
    r2 = r2_score(y_true_adjusted, predictions.flatten())
    # Stocker les performances dans la liste
    performance.append({'Model' : 'GRU','Article': article,  'RMSE': mse, 'MAE': mae, 'R2': r2})
    df_performance = pd.DataFrame(performance)
    index_for_plot = test_df.index[look_back:len(predictions) + look_back]
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, test_df['SOMME_QTE_STOCK'], label='Historique des stocks', marker='o', color='blue')
    plt.plot(index_for_plot, predictions, label='Prédiction des stocks', marker='x', color='red')
    plt.title(f'Prédiction des stocks pour {article} en 2023 avec GRU')
    plt.xlabel('Date')
    plt.ylabel('SOMME_QTE_STOCK')
    plt.legend()
    plt.show()
    return df_performance,model,scaler

#Forecast avec le modèle GRU
def forecast_stock_gru(df_stock_month,model,scaler):
    article = df_stock_month['NOM_ARTICLE'].iloc[0]
    df_stock_month.index = pd.to_datetime(df_stock_month.index)
    # Normalisation de la colonne 'SOMME_QTE_STOCK' pour l'année 2023
    values_2023 = df_stock_month.loc['2023', 'SOMME_QTE_STOCK'].values.reshape(-1, 1)
    values_2023_scaled = scaler.transform(values_2023)
    # Préparation pour les prédictions de 2024
    predictions_2024 = []

    # Génération des prédictions pour chaque mois de 2024
    for month in range(1, 13):
        # Création de la date pour 2024
        date_2024 = pd.Timestamp(f'2024-{month:02d}-01')
        prediction_scaled = model.predict(values_2023_scaled[month - 1].reshape(1, 1, 1))
        prediction = scaler.inverse_transform(prediction_scaled)
        # Ajout de la prédiction à la liste
        predictions_2024.append((date_2024, prediction[0][0]))
    # Conversion des prédictions en DataFrame
    predictions_df_gru_2024 = pd.DataFrame(predictions_2024, columns=['DATE_MVT', 'PREV_SOMME_QTE_STOCK'])
    predictions_df_gru_2024.set_index('DATE_MVT', inplace=True)
    predictions_df_gru_2024['PREV_SOMME_QTE_STOCK'] = predictions_df_gru_2024['PREV_SOMME_QTE_STOCK'].round()
    predictions_df_gru_2024['PREV_SOMME_QTE_STOCK'] = predictions_df_gru_2024['PREV_SOMME_QTE_STOCK'].astype(int)
    id_article = df_stock_month['ID_ARTICLE'].iloc[0]
    nom_article = df_stock_month['NOM_ARTICLE'].iloc[0]
    # Ajouter à forecast_stock_df
    predictions_df_gru_2024['ID_ARTICLE'] = id_article
    predictions_df_gru_2024['NOM_ARTICLE'] = nom_article
    plt.figure(figsize=(15, 6))
    plt.plot(df_stock_month.index, df_stock_month['SOMME_QTE_STOCK'], label='Historique des stocks', marker='o', color='blue', linestyle='-')
    plt.plot(predictions_df_gru_2024.index, predictions_df_gru_2024['PREV_SOMME_QTE_STOCK'], label='Prédictions pour 2024', marker='x', color='red', linestyle='--')
    plt.title('Prédiction des stocks pour {article} en 2024 avec LSTM')
    plt.xlabel('Date')
    plt.ylabel('SOMME_QTE_STOCK')
    plt.legend()
    plt.grid(True)
    plt.show()
    return predictions_df_gru_2024

############################################################################################Calcul d'optimisation d'inventaire############################################################################################################
#create and calculate prédiction d'inventaire dataframe
def calculate_fact_prev_inventaire(forecast_stock_df,forecast_sales_df) :
    forecast_stock_df['PREV_SOMME_QTE_STOCK_CUMUL'] = forecast_stock_df['PREV_SOMME_QTE_STOCK'].cumsum()
    forecast_stock_df.reset_index(inplace=True)
    expanded_rows = []
    for i, row in forecast_stock_df.iterrows():
        start_date = row['DATE_MVT']
        end_date = start_date + pd.offsets.MonthEnd(1)
        month_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        # Répéter les valeurs pour chaque jour ouvrable
        for date in month_dates:
            expanded_rows.append({
                'DATE_MVT': date,
                'ID_ARTICLE': row['ID_ARTICLE'],
                'NOM_ARTICLE': row['NOM_ARTICLE'],
                'PREV_SOMME_QTE_STOCK': row['PREV_SOMME_QTE_STOCK'],
                'PREV_SOMME_QTE_STOCK_CUMUL': row['PREV_SOMME_QTE_STOCK_CUMUL']
            })
    # Créer un nouveau dataframe à partir des lignes générées
    expanded_forecast_stock_df = pd.DataFrame(expanded_rows)
    fact_pred_inventaire_df = forecast_sales_df.merge(expanded_forecast_stock_df, on='DATE_MVT', how='left')
    fact_pred_inventaire_df.sort_values('DATE_MVT', inplace=True)
    fact_pred_inventaire_df.rename(columns={'ID_ARTICLE_x': 'ID_ARTICLE', 'NOM_ARTICLE_x': 'NOM_ARTICLE'}, inplace=True)
    fact_pred_inventaire_df.drop('ID_ARTICLE_y', axis=1, inplace=True)
    fact_pred_inventaire_df.drop('NOM_ARTICLE_y', axis=1, inplace=True)
    fact_pred_inventaire_df['PREV_SOMME_QTE_VENTE_CUMUL'] = fact_pred_inventaire_df.groupby(['ID_ARTICLE'])['PREV_SOMME_QTE_VENTE'].cumsum()
    fact_pred_inventaire_df['PREV_SOMME_QTE_STOCK'] = fact_pred_inventaire_df.groupby(['ID_ARTICLE', fact_pred_inventaire_df['DATE_MVT'].dt.month])['PREV_SOMME_QTE_STOCK'].transform('first')
    fact_pred_inventaire_df['PREV_QTE_RESTE_STOCK_REEL'] = fact_pred_inventaire_df['PREV_SOMME_QTE_STOCK_CUMUL'] - fact_pred_inventaire_df['PREV_SOMME_QTE_VENTE_CUMUL']
    initial_qte = fact_pred_inventaire_df['PREV_SOMME_QTE_STOCK'].iloc[0] 
    fact_pred_inventaire_df['PREV_QTE_STOCK_INITIAL'] = initial_qte
    demande_annuelle = fact_pred_inventaire_df.groupby([fact_pred_inventaire_df['DATE_MVT'].dt.year,'ID_ARTICLE'])['PREV_SOMME_QTE_VENTE'].sum()
    stock_de_securite = 0.1 * demande_annuelle
    stock_de_securite_dict = (stock_de_securite.round().astype(int)).to_dict() 
    fact_pred_inventaire_df['YEAR'] = fact_pred_inventaire_df['DATE_MVT'].dt.year
    #calcul de quantité de cumule de vente annuelle
    fact_pred_inventaire_df['PREV_QTE_VENTE_CUMULE_ANNUELLE'] = fact_pred_inventaire_df.groupby(['ID_ARTICLE', 'YEAR'])['PREV_SOMME_QTE_VENTE'].cumsum()
    fact_pred_inventaire_df['PREV_QTE_STOCK_SECURITE'] = fact_pred_inventaire_df.apply(lambda row: stock_de_securite_dict.get((row['YEAR'], row['ID_ARTICLE'])), axis=1)
    fact_pred_inventaire_df.drop('YEAR', axis=1, inplace=True)
    #calcul de stock de securité initial optimal
    fact_pred_inventaire_df['PREV_QTE_STOCK_INITIAL_OPTIMAL'] = fact_pred_inventaire_df['PREV_QTE_STOCK_INITIAL'].iloc[0] + fact_pred_inventaire_df['PREV_QTE_STOCK_SECURITE']
    #calcul de quantité de reste de stock optimal
    fact_pred_inventaire_df['PREV_QTE_RESTE_STOCK_OPTIMAL'] = fact_pred_inventaire_df['PREV_QTE_STOCK_INITIAL_OPTIMAL'] - fact_pred_inventaire_df['PREV_QTE_VENTE_CUMULE_ANNUELLE']
    return fact_pred_inventaire_df

#Calculer le reaprovisionnement
def calcul_pred_reaprovisionnement_necessaire(fact_pred_inventaire_df):
    fact_pred_inventaire_df['PREV_REAPROVISIONNEMENT'] = 0
   # fact_pred_inventaire_df.drop('YEAR', axis=1, inplace=True)
    fact_pred_inventaire_df.sort_values(by=['ID_ARTICLE', 'DATE_MVT'], inplace=True)
    fact_pred_inventaire_df = fact_pred_inventaire_df.set_index('DATE_MVT')
    for i in fact_pred_inventaire_df.index:
        if fact_pred_inventaire_df.at[i, 'PREV_QTE_RESTE_STOCK_OPTIMAL'] < fact_pred_inventaire_df.at[i, 'PREV_QTE_STOCK_SECURITE']:
            fact_pred_inventaire_df.at[i, 'PREV_REAPROVISIONNEMENT'] = fact_pred_inventaire_df.at[i, 'PREV_QTE_STOCK_INITIAL_OPTIMAL'] - fact_pred_inventaire_df.at[i, 'PREV_QTE_RESTE_STOCK_OPTIMAL']
            # Mise à jour de QTE_RESTE_STOCK_OPTIMAL pour les enregistrements futurs
            fact_pred_inventaire_df.loc[i:, 'PREV_QTE_RESTE_STOCK_OPTIMAL'] += fact_pred_inventaire_df.at[i, 'PREV_REAPROVISIONNEMENT']
    return fact_pred_inventaire_df

#calculer maximum entre 4 variables 
def found_maximum(valeur1, valeur2, valeur3, valeur4):
    max_value = max(valeur1, valeur2)
    if valeur3 > max_value:
        max_value = valeur3
    if valeur4 > max_value:
        max_value = valeur4
    return max_value

############################################################################################Load de la table d'inventaire############################################################################################################
#Load de prévsion de prédiction de fact_inventaire
def load_predict_fact_inventaire(user,pwd,host,port,dbname,schema,table,df):
    engine = create_engine('postgresql://'+user+':'+pwd+'@'+host+':'+port+'/'+dbname+'')
    connexion = engine.connect()
    df['ID_FACT_PRED_INVENTAIRE'] = np.arange(len(df)) + 1
    df.to_sql(table, engine, schema= schema, if_exists='replace', index=False, dtype={
    'ID_FACT_PRED_INVENTAIRE': BigInteger,
    'DATE_MVT': Date,
    'ID_ARTICLE': Integer,
    'NOM_ARTICLE': String,
    'PREV_SOMME_QTE_VENTE': Integer,
    'PREV_SOMME_QTE_STOCK': Integer,
    'PREV_SOMME_QTE_STOCK_CUMUL': Integer,
    'PREV_SOMME_QTE_VENTE_CUMUL': Integer,
    'PREV_QTE_RESTE_STOCK_REEL': Integer,
    'PREV_QTE_STOCK_INITIAL': Integer,
    'PREV_QTE_STOCK_SECURITE': Integer,
    'PREV_QTE_STOCK_INITIAL_OPTIMAL': Integer,
    'PREV_QTE_RESTE_STOCK_OPTIMAL' : Integer,
    'PREV_REAPROVISIONNEMENT' : Integer
    })
    #with engine.connect() as con:
    #    con.execute(f'ALTER TABLE {schema}."Fact_Pred_Inventaire" ADD PRIMARY KEY ("ID_FACT_PRED_INVENTAIRE")')
    #with engine.connect() as con:
    #    con.execute(f'ALTER TABLE {schema}."Fact_Pred_Inventaire" ADD FOREIGN KEY ("ID_ARTICLE") REFERENCES {schema}."Dim_Produit"("ID_ARTICLE")')
    print('chargement de données avec succèes')
    connexion.close()
def predict_fact_inventaire_keys(user,pwd,host,port,dbname):
    conn_string = f"dbname='{dbname}' user='{user}' host='{host}' password='{pwd}' port='{port}'"
    # Connexion à la base de données
    conn = psycopg2.connect(conn_string)
    try:
        # Création d'un cursor pour exécuter des requêtes
        cur = conn.cursor()
    # Votre requête SQL
        query_key_Pred_Inventaire=sql.SQL("""ALTER TABLE public."Fact_Pred_Inventaire" ADD PRIMARY KEY ("ID_FACT_PRED_INVENTAIRE");""")
        cur.execute(query_key_Pred_Inventaire)
        foreign_query_key_Pred_Inventaire= sql.SQL("""ALTER TABLE public."Fact_Pred_Inventaire" ADD FOREIGN KEY ("ID_ARTICLE") REFERENCES public."Dim_Produit"("ID_ARTICLE");""")
        cur.execute(foreign_query_key_Pred_Inventaire)
        conn.commit()

        print("Requêtes exécutées avec succès")

    except Exception as e:
        print("Une erreur est survenue:", e)
        conn.rollback()

    finally:
        # Fermeture de la connexion
        cur.close()
        conn.close()
def drop_table_pred_inventaire(user,pwd,host,port,dbname):
    conn_string = f"dbname='{dbname}' user='{user}' host='{host}' password='{pwd}' port='{port}'"
    # Connexion à la base de données
    conn = psycopg2.connect(conn_string)
    try:
        # Création d'un cursor pour exécuter des requêtes
        cur = conn.cursor()
    # Votre requête SQL
        query_drop_pred_fact_inventaire=sql.SQL("""DROP TABLE IF EXISTS public."Fact_Pred_Inventaire";""")
        cur.execute(query_drop_pred_fact_inventaire)
        # Commit des modifications
        conn.commit()
        print("Requêtes exécutées avec succès")

    except Exception as e:
        print("Une erreur est survenue:", e)
        conn.rollback()

    finally:
        # Fermeture de la connexion
        cur.close()
        conn.close()
############################################################################################Main#####################################################################################################################
def main():
    # Paramètres de connexion à la base de données
    user = 'postgres'
    pwd = 'zied1990'
    host = 'localhost'
    port = '5432'
    dbname = 'postgres'
    schema = 'public'
    table = 'Fact_Inventaire'
    df_fact_inventaire = select_fact_inventaire(user, pwd, host, port, dbname, schema, table)
    fact_pred_inventaire_df = pd.DataFrame()
    for nom_article in df_fact_inventaire['NOM_ARTICLE'].unique():
        df_fact_inventaire_per_article = df_fact_inventaire[df_fact_inventaire['NOM_ARTICLE'] == nom_article]
        #prevision de vente 
        df_prep_vente = get_prep_sales(df_fact_inventaire_per_article)
        df_prep_vente
        visualize_sales_per_article_day(df_prep_vente)
        df_vente_month = sales_per_month(df_prep_vente)
        visualize_sales_per_article_month(df_vente_month)
        df_prep_vente = df_prep_vente.set_index('DATE_MVT')
        #test_serie = test_serie_stationnaire_sales(df_prep_vente)
        #if test_serie == True :
        arima_params_article = get_arima_parameters_sales(df_prep_vente)
        df_performance_arima = evaluate_model_arima_sales(df_prep_vente,arima_params_article)
        sarima_params_article = get_sarima_parameters_sales(df_prep_vente)
        df_performance_sarima = evaluate_model_sarima_sales(df_prep_vente,sarima_params_article)
        df_performance_lstm, model_lstm,scaled_data_lstm,scaler_lstm = evaluate_lstm_model_sales(df_prep_vente)
        df_performance_gru,model_gru,scaled_data_gru,scaler_gru = evaluate_gru_model_sales(df_prep_vente)
        r2_arima = df_performance_arima['R2'].iloc[0]
        r2_sarima = df_performance_sarima['R2'].iloc[0]
        r2_lstm = df_performance_lstm['R2'].iloc[0]
        r2_gru = df_performance_gru['R2'].iloc[0]
        max_performance=found_maximum(r2_arima, r2_sarima, r2_lstm, r2_gru)
        if max_performance==r2_arima:
            df_pred_vente = forecast_sales_arima(df_prep_vente, arima_params_article)
        elif max_performance==r2_sarima:
            df_pred_vente = forecast_sales_sarima(df_prep_vente, sarima_params_article)
        elif max_performance==r2_lstm:
            df_pred_vente = forecast_sales_lstm(df_prep_vente,model_lstm,scaled_data_lstm,scaler_lstm)
        else:
            df_pred_vente = forecast_sales_gru(df_prep_vente,model_gru,scaled_data_gru,scaler_gru)
        #else :
            #df_performance_lstm, model_lstm,scaled_data_lstm,scaler_lstm = evaluate_lstm_model_sales(df_prep_vente)
            #df_performance_gru,model_gru,scaled_data_gru,scaler_gru = evaluate_gru_model_sales(df_prep_vente)
            #r2_lstm = df_performance_lstm['R2'].iloc[0]
            #r2_gru = df_performance_gru['R2'].iloc[0]
            #if r2_lstm > r2_gru :
               # df_pred_vente = forecast_sales_lstm(df_prep_vente,model_lstm,scaled_data_lstm,scaler_lstm)
            #else :
               # df_pred_vente = forecast_sales_gru(df_prep_vente,model_gru,scaled_data_gru,scaler_gru)
        #prevision de  stock
        df_prep_stock = get_prep_stock(df_fact_inventaire_per_article)
        df_stock_month = calculate_somme_stock_month(df_prep_stock)
        df_stock_month.index = pd.to_datetime(df_stock_month.index)
        visualize_stock_per_article_month(df_stock_month)
       # test_serie = test_serie_stationnaire_stock(df_stock_month)
       # if test_serie == True :
        arima_params_article = get_arima_parameters_stock(df_stock_month)
        df_performance_arima = evaluate_model_arima_stock(df_stock_month,arima_params_article)
        sarima_params_article = get_sarima_parameters_stock(df_stock_month)
        df_performance_sarima = evaluate_model_sarima_stock(df_stock_month,sarima_params_article)
        df_performance_lstm, model_lstm,scaler_lstm = evaluate_lstm_model_stock(df_stock_month)
        df_performance_gru,model_gru,scaler_gru = evaluate_gru_model_stock(df_stock_month)
        r2_arima = df_performance_arima['R2'].iloc[0]
        r2_sarima = df_performance_sarima['R2'].iloc[0]
        r2_lstm = df_performance_lstm['R2'].iloc[0]
        r2_gru = df_performance_gru['R2'].iloc[0]
        max_performance=found_maximum(r2_arima, r2_sarima, r2_lstm, r2_gru)
        if max_performance==r2_arima :
            df_pred_stock = forecast_stock_arima(df_stock_month, arima_params_article)
        elif max_performance== r2_sarima:
            df_pred_stock = forecast_stock_sarima(df_stock_month, sarima_params_article)
        elif max_performance== r2_lstm:
            df_pred_stock = forecast_stock_lstm(df_stock_month,model_lstm,scaler_lstm)
        else:
            df_pred_stock = forecast_stock_gru(df_stock_month,model_gru,scaler_gru)
                
        #else :
            #df_performance_lstm, model_lstm,scaler_lstm = evaluate_lstm_model_stock(df_stock_month)
            #df_performance_gru,model_gru,scaler_gru = evaluate_gru_model_stock(df_stock_month)
            #r2_lstm = df_performance_lstm['R2'].iloc[0]
            #r2_gru = df_performance_gru['R2'].iloc[0]
            #if r2_lstm > r2_gru :
               # df_pred_stock = forecast_stock_lstm(df_stock_month,model_lstm,scaler_lstm)
            #else :
             #   df_pred_stock = forecast_stock_gru(df_stock_month,model_gru,scaler_gru)
        # calculer prediction d'ineventaire 
        # Reset the index, which moves the current index to a column
        df_pred_stock = df_pred_stock.reset_index()
        # Rename the new column to 'DATE_MVT'
        df_pred_stock = df_pred_stock.rename(columns={'index': 'DATE_MVT'})
        fact_pred_inventaire_article_df = calculate_fact_prev_inventaire(df_pred_stock,df_pred_vente)
        fact_pred_inventaire_article_df = calcul_pred_reaprovisionnement_necessaire(fact_pred_inventaire_article_df)
        fact_pred_inventaire_df = pd.concat([fact_pred_inventaire_df, fact_pred_inventaire_article_df])
    fact_pred_inventaire_df=fact_pred_inventaire_df.reset_index()
    drop_table_pred_inventaire(user,pwd,host,port,dbname)
    load_predict_fact_inventaire(user,pwd,host,port,dbname,schema,'Fact_Pred_Inventaire',fact_pred_inventaire_df)
    predict_fact_inventaire_keys(user,pwd,host,port,dbname)

if __name__ == "__main__":
    main()