from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/<int:file_id>/', views.upload, name='view_csv_content'),
    path('delete-file/<int:file_id>/', views.delete_file, name='delete_file'),
    path('dash/', views.list_files, name='dash'),
    path('dash/', views.dash, name='dash'),
    path('pretraitement/', views.pretraitement_dataset, name='pretraitement'),
    path('pretraitement/', views.pretraitement, name='pretraitement'),
    path('algo/', views.algo, name='algo'),
    path('perforamnces/', views.perforamnces, name='perforamnces'),
    path('parametres/', views.parametres, name='parametres'),
]
