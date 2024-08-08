from django.db import models

class FactInventaire(models.Model):
    date_mvt = models.DateField(db_column='DATE_MVT')
    id_article = models.BigIntegerField(db_column='ID_ARTICLE')
    nom_article = models.CharField(max_length=256, db_column='NOM_ARTICLE')
    somme_qte_vente = models.IntegerField(db_column='SOMME_QTE_VENTE')
    qte_stock_initial = models.IntegerField(db_column='QTE_STOCK_INITIAL')
    qte_ent = models.IntegerField(db_column='QTE_ENT')
    qte_ent_cumule = models.BigIntegerField(db_column='QTE_ENT_CUMUL')
    qte_stock_cumule = models.IntegerField(db_column='QTE_STOCK_CUMULE')
    qte_vente_cumule = models.IntegerField(db_column='QTE_VENTE_CUMULE')
    qte_reste_stock_reel = models.BigIntegerField(db_column='QTE_RESTE_STOCK_REEL')
    qte_vente_cumule_annuelle = models.BigIntegerField(db_column='QTE_VENTE_CUMULE_ANNUELLE')
    qte_stock_securite = models.IntegerField(db_column='QTE_STOCK_SECURITE')
    qte_stock_initial_optimal = models.IntegerField(db_column='QTE_STOCK_INITIAL_OPTIMAL')
    qte_reste_stock_optimal = models.BigIntegerField(db_column='QTE_RESTE_STOCK_OPTIMAL')
    reaprovisionnement = models.IntegerField(db_column='REAPROVISIONNEMENT')
    id_inventaire = models.BigIntegerField(primary_key=True, db_column='ID_INVENTAIRE')

    class Meta:
        db_table = '"public"."Fact_Inventaire"'
        managed = False

    def __str__(self):
        return f"Inventaire de {self.nom_article} ({self.id_article}) le {self.date_mvt}"

class DimStock(models.Model):
    ID_ARTICLE = models.BigIntegerField()
    NOM_ARTICLE = models.CharField(max_length=256)
    DATE_MVT = models.DateField()
    NUM_MVT = models.IntegerField()
    QTE_STOCK_INITIAL = models.IntegerField()
    QTE_ENT = models.IntegerField()
    DATE_PER = models.DateField()
    NUM_LOT = models.CharField(max_length=256)
    DESTINATION_ORIGINE = models.CharField(max_length=256)
    ID_STOCK = models.BigAutoField(primary_key=True, db_column='ID_STOCK',null=False)

    class Meta:
        db_table = '"public"."Dim_Stock"'
        managed = False


class DimFournisseur(models.Model):
    nom_fournisseur = models.CharField(max_length=256,primary_key=True, db_column='NOM_FOURNISSEUR')

    class Meta:
        db_table = '"public"."Dim_Fournisseur"'
        managed = False

class DimProduit(models.Model):
    ID_ARTICLE = models.BigIntegerField(primary_key=True, db_column='ID_ARTICLE')
    NOM_ARTICLE = models.CharField(max_length=256)

    class Meta:
        db_table = '"public"."Dim_Produit"'
        managed = False

class DimVente(models.Model):
    ID_ARTICLE = models.BigIntegerField()
    NOM_ARTICLE = models.CharField(max_length=256)
    DATE_MVT = models.DateField()
    NUM_MVT = models.IntegerField()
    QTE_SORT = models.IntegerField()
    DATE_PER = models.DateField()
    NUM_LOT = models.CharField(max_length=256)
    DESTINATION_ORIGINE = models.CharField(max_length=256)
    ID_VENTE = models.BigAutoField(primary_key=True, db_column='ID_VENTE',null=False)

    class Meta:
        db_table = '"public"."Dim_Vente"'
        managed = False

    def __str__(self):
        return f"{self.NOM_ARTICLE} ({self.ID_ARTICLE})"

class PredFactInventaire(models.Model):
    date_mvt = models.DateField(db_column='DATE_MVT')
    prev_somme_qte_vente = models.IntegerField(db_column='PREV_SOMME_QTE_VENTE')
    id_article = models.IntegerField(db_column='ID_ARTICLE')
    nom_article = models.CharField(max_length=256, db_column='NOM_ARTICLE')
    prev_somme_qte_stock = models.IntegerField(db_column='PREV_SOMME_QTE_STOCK')
    prev_somme_qte_stock_cumul = models.IntegerField(db_column='PREV_SOMME_QTE_STOCK_CUMUL')
    prev_somme_qte_vente_cumul = models.IntegerField(db_column='PREV_SOMME_QTE_VENTE_CUMUL')
    prev_qte_reste_stock_reel = models.IntegerField(db_column='PREV_QTE_RESTE_STOCK_REEL')
    prev_qte_stock_initial = models.IntegerField(db_column='PREV_QTE_STOCK_INITIAL')
    prev_qte_vente_cumule_annuelle = models.BigIntegerField(db_column='PREV_QTE_VENTE_CUMULE_ANNUELLE')
    prev_qte_stock_securite = models.IntegerField(db_column='PREV_QTE_STOCK_SECURITE')
    prev_qte_stock_initial_optimal = models.IntegerField(db_column='PREV_QTE_STOCK_INITIAL_OPTIMAL')
    prev_qte_reste_stock_optimal = models.IntegerField(db_column='PREV_QTE_RESTE_STOCK_OPTIMAL')
    prev_reapprovisionnement = models.IntegerField(db_column='PREV_REAPROVISIONNEMENT')
    id_fact_pred_inventaire = models.BigAutoField(primary_key=True, db_column='ID_FACT_PRED_INVENTAIRE')

    class Meta:
        db_table = '"public"."Fact_Pred_Inventaire"'
        managed = False

    def __str__(self):
        return f"Prévision d'inventaire pour {self.nom_article} ({self.id_article}) au {self.date_mvt}"