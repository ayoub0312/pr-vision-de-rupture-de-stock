# forms.py
from django import forms
from .models import DimStock, DimFournisseur, DimProduit,DimVente  # Assuming DimProduit exists

class DimStockForm(forms.ModelForm):
    DESTINATION_ORIGINE = forms.ChoiceField(choices=[])
    ID_ARTICLE = forms.ChoiceField(choices=[])
    NOM_ARTICLE = forms.ChoiceField(choices=[])

    class Meta:
        model = DimStock
        fields = ['ID_ARTICLE', 'NOM_ARTICLE', 'DATE_MVT', 'NUM_MVT', 'QTE_STOCK_INITIAL', 'QTE_ENT', 'DATE_PER', 'NUM_LOT', 'DESTINATION_ORIGINE']

    def __init__(self, *args, **kwargs):
        super(DimStockForm, self).__init__(*args, **kwargs)
        # Initialize choice fields dynamically if needed
        self.fields['DESTINATION_ORIGINE'].choices = [(f.nom_fournisseur, f.nom_fournisseur) for f in DimFournisseur.objects.using('postgres').all()]
        self.fields['ID_ARTICLE'].choices = [(p.ID_ARTICLE, p.ID_ARTICLE) for p in DimProduit.objects.using('postgres').all()]
        self.fields['NOM_ARTICLE'].choices = [(p.ID_ARTICLE, p.NOM_ARTICLE) for p in DimProduit.objects.using('postgres').all()]

    def clean(self):
        cleaned_data = super().clean()
        id_article = cleaned_data.get("ID_ARTICLE")
        # Ensure NOM_ARTICLE corresponds to the selected ID_ARTICLE
        corresponding_nom_article = DimProduit.objects.using('postgres').get(ID_ARTICLE=id_article).NOM_ARTICLE
        cleaned_data['NOM_ARTICLE'] = corresponding_nom_article
        return cleaned_data
    
    def clean_ID_ARTICLE(self):
        id_article = self.cleaned_data['ID_ARTICLE']
        try:
            DimProduit.objects.using('postgres').get(ID_ARTICLE=id_article)
        except DimProduit.DoesNotExist:
            raise forms.ValidationError("L'ID_ARTICLE sélectionné n'existe pas.")
        return id_article
    
class DimVenteForm(forms.ModelForm):
    DESTINATION_ORIGINE = forms.ChoiceField(choices=[])
    ID_ARTICLE = forms.ChoiceField(choices=[])
    NOM_ARTICLE = forms.ChoiceField(choices=[])

    class Meta:
        model = DimVente
        fields = ['ID_ARTICLE', 'NOM_ARTICLE', 'DATE_MVT', 'NUM_MVT', 'QTE_SORT', 'DATE_PER', 'NUM_LOT', 'DESTINATION_ORIGINE']

    def __init__(self, *args, **kwargs):
        super(DimVenteForm, self).__init__(*args, **kwargs)
        # Initialize choice fields dynamically if needed
        self.fields['DESTINATION_ORIGINE'].choices = [(f.nom_fournisseur, f.nom_fournisseur) for f in DimFournisseur.objects.using('postgres').all()]
        self.fields['ID_ARTICLE'].choices = [(p.ID_ARTICLE, p.ID_ARTICLE) for p in DimProduit.objects.using('postgres').all()]
        self.fields['NOM_ARTICLE'].choices = [(p.ID_ARTICLE, p.NOM_ARTICLE) for p in DimProduit.objects.using('postgres').all()]

    def clean(self):
        cleaned_data = super().clean()
        id_article = cleaned_data.get("ID_ARTICLE")
        # Ensure NOM_ARTICLE corresponds to the selected ID_ARTICLE
        corresponding_nom_article = DimProduit.objects.using('postgres').get(ID_ARTICLE=id_article).NOM_ARTICLE
        cleaned_data['NOM_ARTICLE'] = corresponding_nom_article
        return cleaned_data