import os
import pandas as pd
import openpyxl
import csv
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .forms import UploadFileForm
from .models import UploadedFile
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

def pretraitement(request):
    return render(request, 'pages/pretraitement.html')

def dash(request):
    return render(request, 'pages/dash.html')

def algo(request):
    return render(request, 'pages/algo.html')

def parametres(request):
    return render(request, 'pages/parametres.html')

def perforamnces(request):
    return render(request, 'pages/perforamnces.html')
def index(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('index')
    else:
        form = UploadFileForm()
    
    # Récupérer tous les fichiers téléchargés pour l'historique
    uploaded_files = UploadedFile.objects.all().order_by('-uploaded_at')
    
    # Filtrer les fichiers pour ne garder que ceux qui existent dans le système de fichiers
    valid_files = []
    for uploaded_file in uploaded_files:
        if os.path.exists(uploaded_file.file.path):
            valid_files.append(uploaded_file)
    
    context = {
        'form': form,
        'uploaded_files': valid_files,
    }
    
    return render(request, 'pages/index.html', context)

def delete_file(request, file_id):
    file_to_delete = get_object_or_404(UploadedFile, id=file_id)
    
    # Supprimer physiquement le fichier du système de fichiers
    if os.path.exists(file_to_delete.file.path):
        os.remove(file_to_delete.file.path)
    
    # Supprimer l'objet de la base de données
    file_to_delete.delete()
    
    # Redirection vers la page index après suppression
    return redirect('index')


def upload(request, file_id):
    uploaded_file = get_object_or_404(UploadedFile, id=file_id)
    file_path = uploaded_file.file.path
    file_name = uploaded_file.file.name

    # Détecter le type de fichier en fonction de l'extension
    if file_name.endswith('.csv'):
        csv_content = read_csv(file_path)
        return render(request, 'pages/upload.html', {'csv_content': csv_content, 'file_name': file_name})
    elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        excel_content = read_excel(file_path)
        return render(request, 'pages/upload.html', {'excel_content': excel_content, 'file_name': file_name})
    else:
        # Gérer les autres types de fichiers ou retourner une erreur
        return render(request, 'pages/error.html', {'error_message': 'Format de fichier non pris en charge'})

def read_csv(file_path):
    csv_content = []
    with open(file_path, 'r', encoding='latin-1') as f:
        reader = csv.reader(f)
        for row in reader:
            csv_content.append(row)
    return csv_content

def read_excel(file_path):
    excel_content = []
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active
    for row in sheet.iter_rows(values_only=True):
        excel_content.append(row)
    return excel_content





def list_files(request):
    # Récupérer la liste des fichiers dans le répertoire media/uploads
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    uploaded_files = [f for f in os.listdir(upload_dir) if f.endswith(('.csv', '.xls', '.xlsx'))]

    if request.method == 'POST':
        selected_file = request.POST.get('file_name', '')
        file_path = os.path.join(upload_dir, selected_file)

        if selected_file and os.path.isfile(file_path):
            try:
                # Lire le fichier selon son extension
                if selected_file.endswith('.csv'):
                    # Essayer plusieurs encodages pour lire le fichier CSV
                    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding)
                            break  # Sortir de la boucle si la lecture est réussie
                        except Exception as e:
                            last_exception = e
                    else:
                        raise last_exception  # Lever la dernière exception si toutes les lectures échouent
                elif selected_file.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path, engine='openpyxl')
                else:
                    raise ValueError("Format de fichier non pris en charge")

                # Calculer les statistiques descriptives
                num_rows, num_cols = df.shape
                column_names = df.columns.tolist()
                summary = df.describe().transpose().to_html(classes='table table-striped')

                # Générer un graphique de distribution pour la première colonne numérique
                numeric_columns = df.select_dtypes(include='number').columns
                if not numeric_columns.empty:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(df[numeric_columns[0]], kde=True)
                    plt.title(f'Distribution of {numeric_columns[0]}')
                    plt.xlabel(numeric_columns[0])
                    plt.ylabel('Frequency')

                    # Convertir le graphique en image encodée en base64
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    image_png = buffer.getvalue()
                    buffer.close()
                    graphic = base64.b64encode(image_png).decode('utf-8')

                    return render(request, 'pages/dash.html', {
                        'uploaded_files': uploaded_files,
                        'selected_file': selected_file,
                        'num_rows': num_rows,
                        'num_cols': num_cols,
                        'column_names': column_names,
                        'summary': summary,
                        'graphic': graphic
                    })
                else:
                    return render(request, 'pages/dash.html', {
                        'uploaded_files': uploaded_files,
                        'selected_file': selected_file,
                        'num_rows': num_rows,
                        'num_cols': num_cols,
                        'column_names': column_names,
                        'summary': summary,
                        'error_message': 'No numeric columns found for plotting.'
                    })
            except Exception as e:
                return render(request, 'pages/dash.html', {
                    'uploaded_files': uploaded_files,
                    'error_message': f"Erreur lors de la lecture du fichier {selected_file}: {str(e)}"
                })

    return render(request, 'pages/dash.html', {'uploaded_files': uploaded_files})