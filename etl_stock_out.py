    #!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import APIs
import os
import numpy as np
import datetime
from sqlalchemy import create_engine, BigInteger, Integer, String, Date
import pandas as pd
from sqlalchemy import text
import psycopg2
from psycopg2 import sql


###################################################################### Extract data ###################################################################################################
#Extract sales data
def extract_vente(chemin_repertoire):
    dfs = []
    for fichier in os.listdir(chemin_repertoire):
        if fichier.startswith('vente') and fichier.endswith('.csv'):
            chemin_fichier = os.path.join(chemin_repertoire, fichier)
            df = pd.read_csv(chemin_fichier,sep=';',encoding='utf-8')
            dfs.append(df)
    df_vente = pd.concat(dfs, ignore_index=True)  
    return df_vente

#Extract sales data
def extract_stock(chemin_repertoire):
    dfs = []
    for fichier in os.listdir(chemin_repertoire):
        if fichier.startswith('stock') and fichier.endswith('.csv'):
            chemin_fichier = os.path.join(chemin_repertoire, fichier)
            df = pd.read_csv(chemin_fichier,sep=';',encoding='utf-8')
            dfs.append(df)
    df_stock = pd.concat(dfs, ignore_index=True)  
    return df_stock

###################################################################### Transform data ###################################################################################################
#Transform sales data
def transform_vente(df_vente):
    # Supprimer les doublons et les lignes avec des valeurs manquantes
    df_vente = df_vente.drop_duplicates()
    df_vente = df_vente.dropna(how='all')
    df_vente = df_vente.dropna(subset=['DATE_MVT'] )
    df_vente = df_vente.dropna(subset=['NUM_MVT'] )
    # Conversion des types de données
    df_vente['DATE_MVT'] = pd.to_datetime(df_vente['DATE_MVT'], format='%d/%m/%Y', errors='coerce')
    df_vente['DATE_PER'] = pd.to_datetime(df_vente['DATE_PER'], format='%d/%m/%Y', errors='ignore')
    df_vente['QTE_SORT'] = df_vente['QTE_SORT'].fillna(0)
    df_vente['QTE_SORT'] = df_vente['QTE_SORT'].astype(int)
    df_vente['NUM_MVT'] = df_vente['NUM_MVT'].fillna(0)
    df_vente['NUM_MVT'] = df_vente['NUM_MVT'].astype(int)
    df_vente['ID_ARTICLE']=df_vente['ID_ARTICLE'].astype('int64')                  
    return df_vente

#Transform stock data
def transform_stock(df_stock):
    # Supprimer les doublons et les lignes avec des valeurs manquantes
    df_stock = df_stock.drop_duplicates()
    df_stock = df_stock.dropna(how='all')
    df_stock= df_stock.dropna(subset=['NUM_MVT'])
    # Conversion des types de données
    df_stock['ID_ARTICLE'] = df_stock['ID_ARTICLE'].astype('int64')
    df_stock['NOM_ARTICLE'] = df_stock['NOM_ARTICLE'].astype(str)
    df_stock['DATE_MVT'] = pd.to_datetime(df_stock['DATE_MVT'] ,format='%d/%m/%Y', errors='coerce')
    df_stock['NUM_MVT'] = df_stock['NUM_MVT'].astype('int32')
    df_stock['QTE_ENT'] = df_stock['QTE_ENT'].fillna(0)
    df_stock['QTE_ENT'] = df_stock['QTE_ENT'].astype('int32')
    df_stock['QTE_STOCK_INITIAL'] = df_stock['QTE_STOCK_INITIAL'].fillna(0)
    df_stock['QTE_STOCK_INITIAL']= df_stock['QTE_STOCK_INITIAL'].astype('int32')
    df_stock['DATE_PER'] = pd.to_datetime(df_stock['DATE_PER'] ,format='%d/%m/%Y', errors='coerce')
    df_stock['NUM_LOT'] = df_stock['NUM_LOT'].astype(str)
    df_stock['DESTINATION_ORIGINE'] = df_stock['DESTINATION_ORIGINE'].astype(str)
    return df_stock

#Create and Transform product data
def transform_produit(df_stock):
    df_produit = df_stock[['ID_ARTICLE', 'NOM_ARTICLE']].drop_duplicates()
    return df_produit


#Create and Transform fournisseur data
def transform_fournisseur(df_stock):
    fournisseur_uniques = df_stock['DESTINATION_ORIGINE'].unique()
    df_fournisseur = pd.DataFrame({"DESTINATION_ORIGINE": fournisseur_uniques})
    df_fournisseur.rename(columns={'DESTINATION_ORIGINE': 'NOM_FOURNISSEUR'}, inplace=True)
    df_fournisseur = df_fournisseur.dropna(subset=["NOM_FOURNISSEUR"])
    #df_fournisseur = df_fournisseur[~(df_fournisseur["NOM_FOURNISSEUR"] == 'nan')]
    return df_fournisseur

#Create and Transform fact_inventaire data
#consolidate vente dataframes
def consolidate_vente(df_vente):
    df_fact_vente = df_vente.groupby(['ID_ARTICLE','NOM_ARTICLE', 'DATE_MVT'])['QTE_SORT'].sum().reset_index()
    df_fact_vente.rename(columns={'QTE_SORT': 'SOMME_QTE_VENTE'}, inplace=True)
    df_vente_consolid = pd.DataFrame()
    df_vente_consolid['DATE_MVT'] = df_fact_vente['DATE_MVT']
    df_vente_consolid['ID_ARTICLE'] = df_fact_vente['ID_ARTICLE']
    df_vente_consolid['NOM_ARTICLE'] = df_fact_vente['NOM_ARTICLE']
    df_vente_consolid['SOMME_QTE_VENTE'] = df_fact_vente['SOMME_QTE_VENTE']
    df_vente_consolid['QTE_STOCK_INITIAL'] = None
    df_vente_consolid['QTE_ENT'] = None
    return df_vente_consolid

#consolidate stock dataframe
def consolidate_stock(df_stock):
    df_stock_consolid = pd.DataFrame()
    df_stock_consolid['DATE_MVT'] = df_stock['DATE_MVT']
    df_stock_consolid['ID_ARTICLE'] = df_stock['ID_ARTICLE']
    df_stock_consolid['NOM_ARTICLE'] = df_stock['NOM_ARTICLE']
    df_stock_consolid['SOMME_QTE_VENTE'] = None
    df_stock_consolid['QTE_STOCK_INITIAL'] = df_stock['QTE_STOCK_INITIAL']
    df_stock_consolid['QTE_ENT'] = df_stock['QTE_ENT']
    return df_stock_consolid

#union stock and vente dataframes
def union_stock_vente(df_vente_consolid,df_stock_consolid):
    df_stock_vente = pd.concat([df_stock_consolid, df_vente_consolid], ignore_index=True)
    return df_stock_vente

#union stock and vente dataframes
def transform_stock_vente(df_stock_vente):
    df_stock_vente[['SOMME_QTE_VENTE', 'QTE_ENT', 'QTE_STOCK_INITIAL']] = df_stock_vente[['SOMME_QTE_VENTE', 'QTE_ENT', 'QTE_STOCK_INITIAL']].apply(pd.to_numeric, errors='coerce')
    df_stock_vente['SOMME_QTE_VENTE']=df_stock_vente['SOMME_QTE_VENTE'].fillna(0)
    df_stock_vente['QTE_ENT']=df_stock_vente['QTE_ENT'].fillna(0)
    df_stock_vente['QTE_STOCK_INITIAL']=df_stock_vente['QTE_STOCK_INITIAL'].fillna(0)
    df_stock_vente['SOMME_QTE_VENTE']=df_stock_vente['SOMME_QTE_VENTE'].astype(int)
    df_stock_vente['QTE_ENT']=df_stock_vente['QTE_ENT'].astype(int)
    df_stock_vente['QTE_STOCK_INITIAL']=df_stock_vente['QTE_STOCK_INITIAL'].astype(int)
    return df_stock_vente

#create and calculate inventaire dataframe
def calculate_fact_inventaire(df_stock_vente):
    #Trier le DataFrame par ID_ARTICLE et DATE_MVT
    df_fact_inventaire = df_stock_vente.sort_values(by=['ID_ARTICLE', 'DATE_MVT'])
    #Calcul de la somme cumulative de QTE_ENT pour chaque article
    df_fact_inventaire['QTE_ENT_CUMUL'] = df_fact_inventaire.groupby('ID_ARTICLE')['QTE_ENT'].cumsum()
    #Initialiser le stock cumulé avec la première valeur de stock initial pour chaque article
    df_fact_inventaire['QTE_STOCK_CUMULE'] = df_fact_inventaire.groupby('ID_ARTICLE')['QTE_STOCK_INITIAL'].transform('first') + df_fact_inventaire['QTE_ENT_CUMUL']
    #df_fact_inventaire['QTE_STOCK_CUMULE'] = df_fact_inventaire.groupby(['ID_ARTICLE'])['QTE_STOCK_CUMULE'].cumsum()
    df_fact_inventaire['QTE_VENTE_CUMULE'] = df_fact_inventaire.groupby('ID_ARTICLE')['SOMME_QTE_VENTE'].cumsum()
    #Calcul de reste de stock réel : la différence entre le cumul de QTE_SORT et Stock_cumule
    df_fact_inventaire['QTE_RESTE_STOCK_REEL'] =   df_fact_inventaire['QTE_STOCK_CUMULE'] - df_fact_inventaire['QTE_VENTE_CUMULE']
    #calcul de stock de securité
    demande_annuelle = df_fact_inventaire.groupby([df_fact_inventaire['DATE_MVT'].dt.year,'ID_ARTICLE'])['SOMME_QTE_VENTE'].sum()
    stock_de_securite = 0.1 * demande_annuelle
    stock_de_securite_dict = (stock_de_securite.round().astype(int)).to_dict() 
    df_fact_inventaire['YEAR'] = df_fact_inventaire['DATE_MVT'].dt.year
    #calcul de quantité de cumule de vente annuelle
    df_fact_inventaire['QTE_VENTE_CUMULE_ANNUELLE'] = df_fact_inventaire.groupby(['ID_ARTICLE', 'YEAR'])['SOMME_QTE_VENTE'].cumsum()
    df_fact_inventaire['QTE_STOCK_SECURITE'] = df_fact_inventaire.apply(lambda row: stock_de_securite_dict.get((row['YEAR'], row['ID_ARTICLE'])), axis=1)
    df_fact_inventaire.drop('YEAR', axis=1, inplace=True)
    #calcul de stock de securité initial optimal
    df_fact_inventaire['QTE_STOCK_INITIAL_OPTIMAL'] = df_fact_inventaire['QTE_STOCK_INITIAL'].iloc[0] + df_fact_inventaire['QTE_STOCK_SECURITE']
    #calcul de quantité de reste de stock optimal
    df_fact_inventaire['QTE_RESTE_STOCK_OPTIMAL'] = df_fact_inventaire['QTE_STOCK_INITIAL_OPTIMAL'] - df_fact_inventaire['QTE_VENTE_CUMULE_ANNUELLE']
    return df_fact_inventaire

#division inventaire data par année
def devide_fact_inventaire(df_fact_inventaire) :
    df_fact_inventaire['YEAR'] = df_fact_inventaire['DATE_MVT'].dt.year
    dfs_par_annee = {annee: df_group for annee, df_group in df_fact_inventaire.groupby('YEAR')}
    return dfs_par_annee

#calculer de reaprovisionnement optimal
def calcul_reaprovisionnement_necessaire(df_fact_inventaire):
    df_fact_inventaire['REAPROVISIONNEMENT'] = 0
    df_fact_inventaire.drop('YEAR', axis=1, inplace=True)
    df_fact_inventaire.sort_values(by=['ID_ARTICLE', 'DATE_MVT'], inplace=True)
    for i in df_fact_inventaire.index:
        if df_fact_inventaire.at[i, 'QTE_RESTE_STOCK_OPTIMAL'] < df_fact_inventaire.at[i, 'QTE_STOCK_SECURITE']:
            df_fact_inventaire.at[i, 'REAPROVISIONNEMENT'] = df_fact_inventaire.at[i, 'QTE_STOCK_INITIAL_OPTIMAL'] - df_fact_inventaire.at[i, 'QTE_RESTE_STOCK_OPTIMAL']
            # Mise à jour de QTE_RESTE_STOCK_OPTIMAL pour les enregistrements futurs
            df_fact_inventaire.loc[i:, 'QTE_RESTE_STOCK_OPTIMAL'] += df_fact_inventaire.at[i, 'REAPROVISIONNEMENT']
    return df_fact_inventaire

#union de inventaire data
def union_fact_inventaire(df_fact_inventaire_2021,df_fact_inventaire_2022,df_fact_inventaire_2023) :
    df_fact_inventaire = pd.concat([df_fact_inventaire_2021, df_fact_inventaire_2022, df_fact_inventaire_2023], ignore_index=True)
    df_fact_inventaire.sort_values(by=['ID_ARTICLE', 'DATE_MVT'], inplace=True)
    return df_fact_inventaire

#Transform temps data
def transform_temps(df_stock_vente):
    dates_uniques = df_stock_vente['DATE_MVT'].unique()
    df_dates = pd.DataFrame({"DATE_MVT": dates_uniques})
    df_dates = df_dates.dropna(how='all')
    df_dates['DATE_MVT'] = pd.to_datetime(df_dates['DATE_MVT'])
    df_dates['YEAR'] = df_dates['DATE_MVT'].dt.year
    df_dates['MONTH'] = df_dates['DATE_MVT'].dt.month
    df_dates['DAY'] = df_dates['DATE_MVT'].dt.day
    return df_dates


#####################################################################Load data###############################################################################################
#Load product data
def load_produit(user,pwd,host,port,dbname,schema,table,df):
    engine = create_engine('postgresql://'+user+':'+pwd+'@'+host+':'+port+'/'+dbname+'')
    connexion = engine.connect()
   
        
    df.to_sql(table, engine, schema= schema, if_exists='replace', index=False, dtype={
    'ID_ARTICLE': BigInteger,
    'NOM_ARTICLE': String(256),
    })
   
    print('chargement de données avec succèes')
    connexion.close()

def load_fournisseur(user, pwd, host, port, dbname, schema, table, df):
    engine = create_engine(f'postgresql://{user}:{pwd}@{host}:{port}/{dbname}')
    connexion = engine.connect()
    df.to_sql(table, engine, schema=schema, if_exists='replace', index=False, dtype={
    'NOM_FOURNISSEUR': String(256)
    })
    
    connexion.close()

#Load stock data
def load_stock(user,pwd,host,port,dbname,schema,table,df):
    engine = create_engine('postgresql://'+user+':'+pwd+'@'+host+':'+port+'/'+dbname+'')
    connexion = engine.connect()
    #df.to_sql(table, engine,schema = schema,if_exists='replace', index=False)
    df['ID_STOCK'] = np.arange(len(df)) + 1
    # Convertir la colonne DATE_MVT au format YYYY-MM-DD
    df['DATE_MVT'] = pd.to_datetime(df['DATE_MVT'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    df.to_sql(table, engine, schema= schema, if_exists='replace', index=False, dtype={
    'ID_STOCK': BigInteger,
    'ID_ARTICLE': BigInteger,
    'NOM_ARTICLE': String(256),
    'DATE_MVT': Date,
    'NUM_MVT': Integer,
    'QTE_STOCK_INITIAL': Integer,
    'QTE_ENT': Integer,
    'DATE_PER': Date,
    'NUM_LOT': String(256),
    'DESTINATION_ORIGINE': String(256)
    })
         
    print('chargement de données avec succèes')
    connexion.close()

#Load vente data
def load_vente(user,pwd,host,port,dbname,schema,table,df):
    engine = create_engine('postgresql://'+user+':'+pwd+'@'+host+':'+port+'/'+dbname+'')
    connexion = engine.connect()
   #df.to_sql(table, engine,schema = schema,if_exists='replace', index=False)
    df['ID_VENTE'] = np.arange(len(df)) + 1
    df['DATE_MVT'] = pd.to_datetime(df['DATE_MVT'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    df.to_sql(table, engine, schema= schema, if_exists='replace', index=False, dtype={
    'ID_VENTE': BigInteger,
    'ID_ARTICLE': BigInteger,
    'NOM_ARTICLE': String(256),
    'DATE_MVT': Date,
    'NUM_MVT': Integer,
    'QTE_SORT': Integer,
    'DATE_PER': Date,
    'NUM_LOT': String(256),
    'DESTINATION_ORIGINE': String(256)
    })
    print('chargement de données avec succèes')
    connexion.close()   


#Load temps data
def load_temps(user, pwd, host, port, dbname, schema, table, df_dates):
    engine = create_engine(f'postgresql://{user}:{pwd}@{host}:{port}/{dbname}')
    connexion = engine.connect()
    df_dates.to_sql(table, engine, schema=schema, if_exists='replace', index=False, dtype={
    'DATE_MVT': Date,
    'YEAR': Integer,
    'MONTH': Integer,
    'DAY': Integer,
    })
    print('chargement de données avec succèes')
    connexion.close()

#Load inventaire data
def load_fact_inventaire(user, pwd, host, port, dbname, schema, table, df_inventaire):
    engine = create_engine(f'postgresql://{user}:{pwd}@{host}:{port}/{dbname}')
    connexion = engine.connect()
    df_inventaire['ID_INVENTAIRE'] = np.arange(len(df_inventaire)) + 1
    df_inventaire['DATE_MVT'] = pd.to_datetime(df_inventaire['DATE_MVT'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    df_inventaire.to_sql(table, engine, schema=schema, if_exists='replace', index=False, dtype={
    'ID_INVENTAIRE': BigInteger,
    'DATE_MVT': Date,
    'ID_ARTICLE': BigInteger,
    'NOM_ARTICLE': String(256),
    'SOMME_QTE_VENTE': Integer,
    'QTE_STOCK_INITIAL' :Integer,
    'QTE_ENT': Integer,
    'QTE_ENT_CUMULE': BigInteger,
    'QTE_STOCK_CUMULE' : Integer,
    'QTE_VENTE_CUMULE':Integer,
    'QTE_STOCK_REEL':Integer,
    'QTE_STOCK_SECURITE':Integer,
    'QTE_STOCK_INITIAL_OPTIMAL':Integer,
    'REAPROVISIONNEMENT' : Integer
    })   
    print('chargement de données avec succèes')
    connexion.close()
#****************************************************************************************************************************
def relation_table(user,pwd,host,port,dbname):
    conn_string = f"dbname='{dbname}' user='{user}' host='{host}' password='{pwd}' port='{port}'"
    # Connexion à la base de données
    conn = psycopg2.connect(conn_string)
    try:
        # Création d'un cursor pour exécuter des requêtes
        cur = conn.cursor()
    # Votre requête SQL
        query_key_Produit=sql.SQL("""ALTER TABLE public."Dim_Produit" ADD PRIMARY KEY ("ID_ARTICLE");""")
        cur.execute(query_key_Produit)
        query_key_stock=sql.SQL("""ALTER TABLE public."Dim_Stock" ADD PRIMARY KEY ("ID_STOCK");""")
        cur.execute(query_key_stock)
        query_key_vente=sql.SQL("""ALTER TABLE public."Dim_Vente" ADD PRIMARY KEY ("ID_VENTE");""")
        cur.execute(query_key_vente)
        query_key_temps=sql.SQL("""ALTER TABLE public."Dim_Temps" ADD PRIMARY KEY ("DATE_MVT");""")
        cur.execute(query_key_temps)
        query_key_invetaire=sql.SQL("""ALTER TABLE public."Fact_Inventaire" ADD PRIMARY KEY ("ID_INVENTAIRE");""")
        cur.execute(query_key_invetaire)
        query_key_fournisseur=sql.SQL("""ALTER TABLE public."Dim_Fournisseur" ADD PRIMARY KEY ("NOM_FOURNISSEUR");""")
        cur.execute(query_key_fournisseur)

        query_stock_produit = sql.SQL("""
        ALTER TABLE public."Dim_Stock"
        ADD FOREIGN KEY ("ID_ARTICLE")
        REFERENCES public."Dim_Produit"("ID_ARTICLE");
        """)
        cur.execute(query_stock_produit)
        quer_stock_fournisseur=sql.SQL("""ALTER TABLE public."Dim_Stock" ADD FOREIGN KEY ("DESTINATION_ORIGINE") REFERENCES public."Dim_Fournisseur"("NOM_FOURNISSEUR");""")
        cur.execute(quer_stock_fournisseur)
        query_vente_produit=sql.SQL("""ALTER TABLE public."Dim_Vente" ADD FOREIGN KEY ("ID_ARTICLE") REFERENCES public."Dim_Produit"("ID_ARTICLE");""")
        cur.execute(query_vente_produit)
        query_inventaire_produit=sql.SQL("""ALTER TABLE public."Fact_Inventaire" ADD FOREIGN KEY ("ID_ARTICLE") REFERENCES public."Dim_Produit"("ID_ARTICLE");""")
        cur.execute(query_inventaire_produit)
        query_inventaire_temps=sql.SQL("""ALTER TABLE public."Fact_Inventaire" ADD FOREIGN KEY ("DATE_MVT") REFERENCES public."Dim_Temps"("DATE_MVT");""")
        cur.execute(query_inventaire_temps)
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
#***************************************************************************************************      
def drop_table(user,pwd,host,port,dbname):
    conn_string = f"dbname='{dbname}' user='{user}' host='{host}' password='{pwd}' port='{port}'"
    # Connexion à la base de données
    conn = psycopg2.connect(conn_string)
    try:
        # Création d'un cursor pour exécuter des requêtes
        cur = conn.cursor()
    # Votre requête SQL
        query_drop_fact_inventaire=sql.SQL("""DROP TABLE IF EXISTS public."Fact_Inventaire";""")
        cur.execute(query_drop_fact_inventaire)
        query_drop_pred_fact_inventaire=sql.SQL("""DROP TABLE IF EXISTS public."Fact_Pred_Inventaire";""")
        cur.execute(query_drop_pred_fact_inventaire)
        query_drop_stock=sql.SQL("""DROP TABLE IF EXISTS public."Dim_Stock";""")
        cur.execute(query_drop_stock)
        query_drop_vente=sql.SQL("""DROP TABLE IF EXISTS public."Dim_Vente";""")
        cur.execute(query_drop_vente)
        query_drop_temps=sql.SQL("""DROP TABLE IF EXISTS public."Dim_Temps";""")
        cur.execute(query_drop_temps)
        query_drop_produit=sql.SQL("""DROP TABLE IF EXISTS public."Dim_Produit";""")
        cur.execute(query_drop_produit)
        query_drop_fournisseur=sql.SQL("""DROP TABLE IF EXISTS dwh."Dim_Fournisseur";""")
        cur.execute(query_drop_fournisseur)
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
         
#*********************************************************************************************************************


    
def main():
    chemin_repertoire_vente = 'data/vente'
    chemin_repertoire_stock = 'data/stock'
    # Extraction
    df_vente = extract_vente(chemin_repertoire_vente)
    df_stock = extract_stock(chemin_repertoire_stock)
    # Transformation
    df_vente_transforme = transform_vente(df_vente)
    df_stock_transforme = transform_stock(df_stock)
    df_produit = transform_produit(df_stock_transforme)
    df_fournisseur = transform_fournisseur(df_stock_transforme)
    df_vente_consolid = consolidate_vente(df_vente_transforme)
    df_stock_consolid = consolidate_stock(df_stock_transforme)
    df_union_stock_vente = union_stock_vente(df_vente_consolid, df_stock_consolid)
    df_union_stock_vente_transforme = transform_stock_vente(df_union_stock_vente)
    df_fact_inventaire = calculate_fact_inventaire(df_union_stock_vente_transforme)
    dfs_par_annee = devide_fact_inventaire(df_fact_inventaire)
    df_fact_inventaire_2021 = dfs_par_annee[2021]
    df_fact_inventaire_2021 = calcul_reaprovisionnement_necessaire(df_fact_inventaire_2021)
    df_fact_inventaire_2022 = dfs_par_annee[2022]
    df_fact_inventaire_2022 = calcul_reaprovisionnement_necessaire(df_fact_inventaire_2022)
    df_fact_inventaire_2023 = dfs_par_annee[2023]
    df_fact_inventaire_2023 = calcul_reaprovisionnement_necessaire(df_fact_inventaire_2023)
    df_fact_inventaire = union_fact_inventaire(df_fact_inventaire_2021,df_fact_inventaire_2022,df_fact_inventaire_2023)
    df_dates = transform_temps(df_union_stock_vente_transforme)

 # Utilisation des paramètres de la base de données Django
    dbname = 'postgres'
    user = 'postgres'
    pwd = 'zied1990'
    host = 'localhost'
    port = '5432'
    schema = 'public'    
    
# Chargement
    drop_table(user,pwd,host,port,dbname)
    load_produit(user, pwd, host, port, dbname, schema, 'Dim_Produit', df_produit)
    load_fournisseur(user, pwd, host, port, dbname, schema, 'Dim_Fournisseur', df_fournisseur)
    load_temps(user, pwd, host, port, dbname, schema, 'Dim_Temps', df_dates)
    load_stock(user, pwd, host, port, dbname, schema, 'Dim_Stock', df_stock)
    load_vente(user, pwd, host, port, dbname, schema, 'Dim_Vente', df_vente)
    load_fact_inventaire(user, pwd, host, port, dbname, schema, 'Fact_Inventaire', df_fact_inventaire)
    relation_table(user,pwd,host,port,dbname)

if __name__ == "__main__":
    main()