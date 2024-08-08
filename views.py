from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from .models import FactInventaire
from django.http import JsonResponse
from .models import DimStock
from django.db.utils import OperationalError
from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from .forms import DimStockForm
from .models import DimVente
from .forms import DimVenteForm
from django.core.paginator import Paginator
from django.db import connections
from django.db.models import Max
import os
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import logging
import subprocess
from .models import PredFactInventaire



logger = logging.getLogger(__name__)


@csrf_exempt
def home(request):
    if request.method == 'POST':
        file = request.FILES['file']
        year = request.POST['year']
        file_type = request.POST['fileType']
        article_name = request.POST['articleName']

        # Définir les répertoires de destination
        if file_type == 'stock':
            directory = os.path.join('data', 'stock')
        elif file_type == 'vente':
            directory = os.path.join('data', 'vente')
        else:
            return JsonResponse({'message': 'Type de fichier invalide.'}, status=400)

        # Créer le répertoire s'il n'existe pas
        os.makedirs(directory, exist_ok=True)

        # Nommer le fichier
        file_name = f'{file_type}_{article_name}_{year}.csv'
        file_path = os.path.join(directory, file_name)

        # Sauvegarder le fichier
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        return JsonResponse({'message': f'Fichier {file_type} téléchargé avec succès!'})

    return render(request, 'home.html')

def handle_uploaded_file(f, path):
    os.makedirs(path, exist_ok=True)  # Create the directory if it does not exist
    with open(os.path.join(path, f.name), 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def validate_filename(filename, file_type, nom_article, year):
    expected_filename = f"{file_type}_{nom_article}_{year}.csv"
    return filename.lower() == expected_filename.lower()

@login_required
def dashboard(request):
 return render(request,"dashboard.html")



@login_required
def settings(request):
  return render(request,"settings.html")

@login_required
def stock(request):
    try:
        query = request.GET.get('search', '')
        if query:
            stocks = DimStock.objects.filter(NOM_ARTICLE__icontains=query).using('postgres')
        else:
            stocks = DimStock.objects.using('postgres')
        
        # Setup pagination
        paginator = Paginator(stocks, 20)  # Correct number of items per page
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        return render(request, 'stock.html', {'page_obj': page_obj})
    except Exception as e:
        return HttpResponse(f"Error: {e}")

  

def authView(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)  # Initialize form with POST data
        if form.is_valid():
            form.save()
            return redirect("base:login")  # Assuming 'base:login' is a valid URL name for your login route
    else:
        form = UserCreationForm()  # Initialize an empty form for a GET request

    return render(request, "registration/signup.html", {"form": form})



def fact_inventaire_data(request):
    try:
        data = list(FactInventaire.objects.using('postgres').values())
        return JsonResponse(data, safe=False)
    except OperationalError as e:
        return JsonResponse({'error': str(e)}, status=500)
    
#def list_stocks(request):
#    stocks = DimStock.objects.all()
#    return render(request, 'stock_list.html', {'stocks': stocks})

def add_stock(request):
    if request.method == 'POST':
        form = DimStockForm(request.POST)
        if form.is_valid():
            # Récupérer la dernière valeur de ID_STOCK et l'incrémenter de 1
            last_id_stock = DimStock.objects.all().using('postgres').aggregate(Max('ID_STOCK'))['ID_STOCK__max']
            new_id_stock = last_id_stock + 1 if last_id_stock is not None else 1
            new_stock = form.save(commit=False)
            new_stock.ID_STOCK = new_id_stock
            new_stock.save(using='postgres')
            return redirect('base:stock')  # Assurez-vous que la redirection est correcte
    else:
        form = DimStockForm()
    return render(request, 'add_stock.html', {'form': form})


def edit_stock(request, pk):
    # Using 'postgres' to ensure we are querying the correct database
    stock = get_object_or_404(DimStock.objects.using('postgres'), pk=pk)
    if request.method == 'POST':
        form = DimStockForm(request.POST, instance=stock)
        if form.is_valid():
            # Save the form and continue using the 'postgres' database
            form.save()
            return redirect('base:stock')  # Ensure this redirect is correct
    else:
        form = DimStockForm(instance=stock)
    return render(request, 'edit_stock.html', {'form': form})


def delete_stock(request, pk):
    # Retrieve the stock using the 'postgres' database
    stock = get_object_or_404(DimStock.objects.using('postgres'), pk=pk)
    if request.method == 'POST':
        stock.delete(using='postgres')  # Specify database for deletion
        return redirect('base:stock')  # Make sure the redirect is namespace-aware
    return render(request, 'delete_stock.html', {'stock': stock})


def vente(request):
    try:
        query = request.GET.get('search', '')
        if query:
            ventes = DimVente.objects.filter(NOM_ARTICLE__icontains=query).using('postgres')
        else:
            ventes = DimVente.objects.using('postgres').all()

        # Setup pagination
        paginator = Paginator(ventes, 30)  # 20 ventes par page
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        return render(request, 'vente.html', {'page_obj': page_obj})
    except Exception as e:
        return HttpResponse(f"Error: {e}")
    

def add_vente(request):
    if request.method == 'POST':
        form = DimVenteForm(request.POST)
        if form.is_valid():
            last_id_vente = DimVente.objects.all().using('postgres').aggregate(Max('ID_VENTE'))['ID_VENTE__max']
            new_id_vente = last_id_vente + 1 if last_id_vente is not None else 1
            new_vente = form.save(commit=False)
            new_vente.ID_VENTE = new_id_vente
            new_vente.save(using='postgres')
            return redirect('base:vente')  # Make sure the redirect is namespace-aware
    else:
        form = DimVenteForm()
    return render(request, 'add_vente.html', {'form': form})


def edit_vente(request, pk):
    # Using 'postgres' to ensure we are querying the correct database
    vente = get_object_or_404(DimVente.objects.using('postgres'), pk=pk)
    if request.method == 'POST':
        form = DimVenteForm(request.POST, instance=vente)
        if form.is_valid():
            # Save the form and continue using the 'postgres' database
            form.save()
            return redirect('base:vente')  # Ensure this redirect is correct
    else:
        form = DimVenteForm(instance=vente)
    return render(request, 'edit_vente.html', {'form': form})


def delete_vente(request, pk):
    # Retrieve the stock using the 'postgres' database
    vente = get_object_or_404(DimVente.objects.using('postgres'), pk=pk)
    if request.method == 'POST':
        vente.delete(using='postgres')  # Specify database for deletion
        return redirect('base:vente')  # Make sure the redirect is namespace-aware
    return render(request, 'delete_vente.html', {'vente': vente})





@login_required
def execute_etl(request):
    if request.method == 'POST':
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'base', 'scripts', 'etl_stock_out.py')
        try:
            # Using 'python3' instead of 'python'
            result = subprocess.run(['python3', script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return JsonResponse({'status': 'success', 'message': 'ETL process executed successfully!', 'output': result.stdout.decode()})
        except subprocess.CalledProcessError as e:
            return JsonResponse({'status': 'error', 'message': str(e), 'stderr': e.stderr.decode()})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=405)

def pred_fact_inventaire_data(request):
    try :
        data = list(PredFactInventaire.objects.using('postgres').values())
        return JsonResponse(data, safe=False)
    except OperationalError as e:
        return JsonResponse({'error': str(e)}, status=500)