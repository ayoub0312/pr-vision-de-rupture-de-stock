import datetime
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import subprocess
import os
import time
import seaborn as sns


st.title("Historique des ventes")

# Requête vers l'API Django
response = requests.get("http://localhost:8000/api/fact_inventaire/")
data = response.json()

# Créer un DataFrame
df = pd.DataFrame(data)

# Convertir 'date_mvt' en format date
df['date_mvt'] = pd.to_datetime(df['date_mvt'])

# Extraire l'année
df['year'] = df['date_mvt'].dt.year

# Créer une liste de noms d'articles uniques
nom_articles = df['nom_article'].unique()
years = df['year'].unique()

# Sélectionner les filtres
selected_nom_article = st.selectbox("Sélectionnez un article:", nom_articles)
selected_year = st.selectbox("Sélectionnez une année:", years)

filtered_df = df[(df['nom_article'] == selected_nom_article) & (df['year'] == selected_year)]

# Ajout du graphique de distribution pour somme_qte_vente par date_mvt
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(filtered_df['date_mvt'], filtered_df['somme_qte_vente'], marker='o')
ax.set_xlabel("Date MVT")
ax.set_ylabel("Quantité")
ax.set_title(f"Distribution de Somme de quantité de Vente par Date MVT pour {selected_nom_article} en {selected_year}")
st.pyplot(fig)

st.title("Optimisation d'Inventaire")
# Sélectionner une date
dates = filtered_df['date_mvt'].dt.strftime('%Y-%m-%d')
selected_date = st.selectbox("Sélectionnez une date:", dates)
selected_date = datetime.datetime.strptime(selected_date, '%Y-%m-%d')
# Filtrer les données pour la date sélectionnée
selected_date_df = filtered_df[filtered_df['date_mvt'] == selected_date]
# Récupérer la deuxième ligne
if len(selected_date_df) < 2:
    selected_date_df = selected_date_df.iloc[[0]]
else :
    selected_date_df = selected_date_df.iloc[[1]]
# Déterminer la couleur pour QTE RESTE STOCK REEL
qte_reste_stock_reel = selected_date_df['qte_reste_stock_reel'].iloc[0]
qte_reste_stock_optimal = selected_date_df['qte_reste_stock_optimal'].iloc[0]
qte_stock_securite = selected_date_df['qte_stock_securite'].iloc[0]
qte_stock_initial_optimal = selected_date_df['qte_stock_initial_optimal'].iloc[0]

# Calcul des seuils
seuil_inf = 0.1 * qte_stock_securite + qte_stock_securite
seuil_sup = qte_stock_initial_optimal - 0.1 * qte_stock_initial_optimal

if qte_reste_stock_reel < qte_stock_securite or qte_reste_stock_reel > qte_stock_initial_optimal:
    delta = -1
    delta_color = 'inverse'
elif seuil_sup < qte_reste_stock_reel < seuil_inf:
    delta = 0
    delta_color = 'off'
else:
    delta = 1
    delta_color = 'normal'

if qte_reste_stock_optimal < qte_stock_securite or qte_reste_stock_optimal > qte_stock_initial_optimal:
    delta1 = -1
    delta_color1 = 'inverse'
elif seuil_sup < qte_reste_stock_optimal < seuil_inf:
    delta1 = 0
    delta_color1 = 'off'
else:
    delta1 = 1
    delta_color1 = 'normal'




col1, col2, col3 = st.columns(3)

# Affichage des valeurs dans chaque colonne
with col1:
    st.metric("QTE VENTE", selected_date_df['somme_qte_vente'].iloc[0])
    st.metric("REAPROVISIONNEMENT", selected_date_df['reaprovisionnement'].iloc[0])

with col2:
    st.metric("QTE STOCK INITIAL OPTIMAL", selected_date_df['qte_stock_initial_optimal'].iloc[0])
    st.metric("QTE RESTE STOCK REEL", f"{qte_reste_stock_reel:}", delta, delta_color=delta_color)
    

with col3:
    st.metric("QTE STOCK SECURITE", selected_date_df['qte_stock_securite'].iloc[0])
    st.metric("QTE RESTE STOCK OPTIMAL", f"{qte_reste_stock_optimal:}", delta1, delta_color=delta_color1)


  

# Plot pour les données d'inventaire
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(filtered_df['date_mvt'], filtered_df['qte_stock_initial_optimal'], label='Stock Initial Optimal', marker='o')
ax.plot(filtered_df['date_mvt'], filtered_df['qte_stock_securite'], label='Stock Sécurité', marker='o')
ax.plot(filtered_df['date_mvt'], filtered_df['reaprovisionnement'], label='Réapprovisionnement', marker='o')
ax.plot(filtered_df['date_mvt'], filtered_df['qte_reste_stock_reel'], label='Reste Stock Réel', marker='o')
ax.plot(filtered_df['date_mvt'], filtered_df['qte_reste_stock_optimal'], label='Reste Stock Optimal', marker='o')

ax.set_xlabel("Date MVT")
ax.set_ylabel("Quantité")
ax.set_title(f"Inventaire pour {selected_nom_article} en {selected_year}")
ax.legend(loc='upper left')
st.pyplot(fig)



st.title("Prévision de vente")
# Ajouter un bouton pour exécuter le script de prévision
if st.button('Exécuter la prévision'):
    script_path = os.path.join(os.path.dirname(__file__), 'base','scripts', 'ml_stock_out.py')
    if os.path.exists(script_path):
        # Créer une barre de progression
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)  # Simuler le temps de traitement
            progress_bar.progress(percent_complete + 1)
        result = subprocess.run(['python3', script_path], capture_output=True, text=True)
        if result.returncode == 0:
            st.success('Prévision exécutée avec succès!')
            st.text(result.stdout)
        else:
            st.error('Erreur lors de l\'exécution de la prévision.')
            st.text(result.stderr)
    else:
        st.error('Le script de prévision est introuvable.')


# Requête vers l'API Django pour récupérer les données de pred_fact_inventaire
response = requests.get("http://localhost:8000/api/pred_fact_inventaire/")
pred_data = response.json()

# Créer un DataFrame pour les données de prévision
pred_df = pd.DataFrame(pred_data)

# Convertir 'date_mvt' en format date
pred_df['date_mvt'] = pd.to_datetime(pred_df['date_mvt'])
# Extraire l'année
pred_df['year'] = pred_df['date_mvt'].dt.year
# Créer une liste de noms d'articles uniques pour les prévisions
pred_nom_articles = pred_df['nom_article'].unique()
pred_years = pred_df['year'].unique()
# Sélectionner les filtres pour les données de prévision
selected_pred_nom_article = st.selectbox("Sélectionnez un article (Prévision):", pred_nom_articles)
selected_pred_year = st.selectbox("Sélectionnez une année (Prévision):", pred_years)
# Filtrer les données pour l'année et l'article sélectionnés
filtered_pred_df = pred_df[(pred_df['nom_article'] == selected_pred_nom_article) & (pred_df['year'] == selected_pred_year)]

# Ajout du graphique de distribution pour somme_qte_vente par date_mvt
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(filtered_pred_df['date_mvt'], filtered_pred_df['prev_somme_qte_vente'], marker='o')
ax.set_xlabel("Date MVT")
ax.set_ylabel("Quantité")
ax.set_title(f"Distribution de prévision de quantité de Vente par Date MVT pour {selected_pred_nom_article} en {selected_pred_year}")
st.pyplot(fig)

st.title("Prévision d'Optimisation d'Inventaire")
# Sélectionner une date pour les données de prévision
pred_dates = filtered_pred_df['date_mvt'].dt.strftime('%Y-%m-%d')
selected_pred_date = st.selectbox("Sélectionnez une date (Prévision):", pred_dates)
selected_pred_date = pd.to_datetime(selected_pred_date)
# Filtrer les données pour la date sélectionnée pour les prévisions
selected_pred_date_df = filtered_pred_df[filtered_pred_df['date_mvt'] == selected_pred_date]
# Récupérer la deuxième ligne pour les prévisions, sinon la première si une seule ligne est présente
if len(selected_pred_date_df) > 1:
    selected_pred_date_df = selected_pred_date_df.iloc[[1]]
else:
    selected_pred_date_df = selected_pred_date_df.iloc[[0]]
# Déterminer la couleur pour PREV QTE RESTE STOCK REEL
prev_qte_reste_stock_reel = selected_pred_date_df['prev_qte_reste_stock_reel'].iloc[0]
prev_qte_reste_stock_optimal = selected_pred_date_df['prev_qte_reste_stock_optimal'].iloc[0]
prev_qte_stock_securite = selected_pred_date_df['prev_qte_stock_securite'].iloc[0]
prev_qte_stock_initial_optimal = selected_pred_date_df['prev_qte_stock_initial_optimal'].iloc[0]
# Calcul des seuils pour les prévisions
prev_seuil_inf = 0.1 * prev_qte_stock_securite + prev_qte_stock_securite
prev_seuil_sup = prev_qte_stock_initial_optimal - 0.1 * prev_qte_stock_initial_optimal
# Définir les deltas et les couleurs basés sur les seuils
prev_delta, prev_delta_color = (-1, 'inverse') if prev_qte_reste_stock_reel < prev_qte_stock_securite or prev_qte_reste_stock_reel > prev_qte_stock_initial_optimal else (0, 'off') if prev_seuil_sup < prev_qte_reste_stock_reel < prev_seuil_inf else (1, 'normal')
prev_delta1, prev_delta_color1 = (-1, 'inverse') if prev_qte_reste_stock_optimal < prev_qte_stock_securite or prev_qte_reste_stock_optimal > prev_qte_stock_initial_optimal else (0, 'off') if prev_seuil_sup < prev_qte_reste_stock_optimal < prev_seuil_inf else (1, 'normal')
col1, col2, col3 = st.columns(3)
# Affichage des valeurs de prévision dans chaque colonne
with col1:
    st.metric("PREV QTE VENTE", selected_pred_date_df['prev_somme_qte_vente'].iloc[0])
    st.metric("PREV REAPPROVISIONNEMENT", selected_pred_date_df['prev_reapprovisionnement'].iloc[0])
with col2:
    st.metric("PREV QTE STOCK INITIAL OPTIMAL", selected_pred_date_df['prev_qte_stock_initial_optimal'].iloc[0])
    st.metric("PREV QTE RESTE STOCK REEL", f"{prev_qte_reste_stock_reel:}", prev_delta, delta_color=prev_delta_color)
with col3:
    st.metric("PREV QTE STOCK SECURITE", selected_pred_date_df['prev_qte_stock_securite'].iloc[0])
    st.metric("PREV QTE RESTE STOCK OPTIMAL", f"{prev_qte_reste_stock_optimal:}", prev_delta1, delta_color=prev_delta_color1)
# Plot des données de prévision
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(filtered_pred_df['date_mvt'], filtered_pred_df['prev_qte_stock_initial_optimal'], label='Prévision Stock Initial Optimal', marker='o')
ax.plot(filtered_pred_df['date_mvt'], filtered_pred_df['prev_qte_stock_securite'], label='Prévision Stock Sécurité', marker='o')
ax.plot(filtered_pred_df['date_mvt'], filtered_pred_df['prev_reapprovisionnement'], label='Prévision Réapprovisionnement', marker='o')
ax.plot(filtered_pred_df['date_mvt'], filtered_pred_df['prev_qte_reste_stock_reel'], label='Prévision Reste Stock Réel', marker='o')
ax.plot(filtered_pred_df['date_mvt'], filtered_pred_df['prev_qte_reste_stock_optimal'], label='Prévision Reste Stock Optimal', marker='o')
ax.set_xlabel("Date MVT")
ax.set_ylabel("Quantité")
ax.set_title(f"Prévisions pour {selected_pred_nom_article} en {selected_pred_year}")
ax.legend(loc='upper left')
st.pyplot(fig)