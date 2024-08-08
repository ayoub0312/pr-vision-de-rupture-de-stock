from django.urls import path, include
from django.contrib import admin
from .views import authView, home,vente,stock,dashboard
from base import views
from .views import add_stock, edit_stock, delete_stock
from .views import add_vente,edit_vente,delete_vente,execute_etl

app_name = 'base'

urlpatterns = [
 path("", views.home, name="home"),
 path("signup/", authView, name="authView"),
 path("accounts/", include("django.contrib.auth.urls")),
 path("dashboard/", views.dashboard, name="dashboard"),
 path("vente/", views.vente, name="vente"),
 path("stock/", views.stock, name="stock"),
 path("settings/",views.settings,name="settings"),
 path('api/fact_inventaire/', views.fact_inventaire_data, name='fact_inventaire_data'),
 # Example URL patterns for CRUD operations
 path('stock/add/', views.add_stock, name='add_stock'),
 path('stock/edit/<int:pk>/', views.edit_stock, name='edit_stock'),
 path('stock/delete/<int:pk>/', views.delete_stock, name='delete_stock'),
  # Example URL patterns for CRUD operations
 path('vente/add/', views.add_vente, name='add_vente'),
 path('vente/edit/<int:pk>/', views.edit_vente, name='edit_vente'),
 path('vente/delete/<int:pk>/', views.delete_vente, name='delete_vente'),
 path('run-etl/', views.execute_etl, name='execute_etl'),
 path('api/pred_fact_inventaire/', views.pred_fact_inventaire_data, name='pred_fact_inventaire_data'),
]

