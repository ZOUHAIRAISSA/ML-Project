# Modules Django
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .forms import UploadFileForm
from .models import Dataset, UploadedFile

# Modules pour le traitement des données et le machine learning
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Autres modules standard
import os
import openpyxl
import csv
from io import BytesIO
import base64
import os
from django.conf import settings
from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os
from django.conf import settings
from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
import os
from django.conf import settings
from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

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
    file_name1 = os.path.basename(file_path)


    # Détecter le type de fichier en fonction de l'extension
    if file_name.endswith('.csv'):
        csv_content = read_csv(file_path)
        return render(request, 'pages/upload.html', {'csv_content': csv_content, 'file_name': file_name1})
    elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        excel_content = read_excel(file_path)
        return render(request, 'pages/upload.html', {'excel_content': excel_content, 'file_name': file_name1})
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
                 # Calculer les statistiques descriptives
                num_rows, num_cols = df.shape
                column_names = df.columns.tolist()
                summary = df.describe().transpose().to_html(classes='table table-striped')

                # Nettoyer les colonnes non numériques
                numeric_columns = df.select_dtypes(include='number').columns
                graphics = []
                if not numeric_columns.empty:
                    for column in numeric_columns:
                        plt.figure(figsize=(12, 8))
                        sns.histplot(df[column].dropna(), kde=True)
                        plt.title(f'Distribution of {column}')
                        plt.xlabel(column)
                        plt.ylabel('Frequency')

                        # Convertir le graphique en image encodée en base64
                        buffer = BytesIO()
                        plt.savefig(buffer, format='png')
                        buffer.seek(0)
                        image_png = buffer.getvalue()
                        buffer.close()
                        graphic = base64.b64encode(image_png).decode('utf-8')
                        graphics.append(graphic)
                        plt.close()

                    # Générer une Heatmap de corrélation
                    heatmap_graphic = generate_heatmap(df[numeric_columns])
                    
                    # Generate scatter plots
                    scatter_plots = generate_scatter_plots(df)                    # Diviser les graphiques en groupes de trois
                    graphics_grouped = [graphics[i:i + 2] for i in range(0, len(graphics), 2)]

                    return render(request, 'pages/dash.html', {
                        'uploaded_files': uploaded_files,
                        'selected_file': selected_file,
                        'num_rows': num_rows,
                        'num_cols': num_cols,
                        'column_names': column_names,
                        'summary': summary,
                        'graphics_grouped': graphics_grouped,
                        'heatmap_graphic': heatmap_graphic,
                        'scatter_plots': scatter_plots
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

def generate_heatmap(df):
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    heatmap_graphic = base64.b64encode(image_png).decode('utf-8')
    plt.close()

    return heatmap_graphic



# generate scatler plot 
import itertools

def generate_scatter_plots(df):
    numeric_columns = df.select_dtypes(include='number').columns
    scatter_plots = []

    # Generate scatter plots for all pairs of numeric columns
    for x_col, y_col in itertools.combinations(numeric_columns, 2):
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=df[x_col], y=df[y_col])
        plt.title(f'Scatter Plot: {x_col} vs {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        scatter_plot = base64.b64encode(image_png).decode('utf-8')
        plt.close()
        
        scatter_plots.append({
            'x_col': x_col,
            'y_col': y_col,
            'scatter_plot': scatter_plot
        })

    return scatter_plots

def pretraitement_dataset(request):
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    uploaded_files = [f for f in os.listdir(upload_dir) if f.endswith(('.csv', '.xls', '.xlsx'))]

    if request.method == 'POST':
        selected_file = request.POST.get('file_name', '')
        file_path = os.path.join(upload_dir, selected_file)

        if selected_file and os.path.isfile(file_path):
            try:
                # Lecture du fichier en fonction de son extension
                if selected_file.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif selected_file.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path, engine='openpyxl')
                else:
                    raise ValueError("Format de fichier non pris en charge")

                # Supprimer les doublons
                df.drop_duplicates(inplace=True)

                # Nettoyage des données
                # Remplacer les valeurs non numériques par NaN pour la conversion
                df.replace(['NAN', 'nan', 'Nan', 'nAn'], pd.NA, inplace=True)
                
                # Conversion en valeurs numériques si possible
                df = df.apply(pd.to_numeric, errors='ignore')

                # Gestion des données manquantes avec imputation
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        imputer = SimpleImputer(strategy='mean')
                        df[col] = imputer.fit_transform(df[[col]]).ravel()

                # Encodage des variables catégorielles (One-Hot Encoding)
                categorical_columns = df.select_dtypes(include=['object']).columns
                if not categorical_columns.empty:
                    encoder = OneHotEncoder(handle_unknown='ignore')
                    encoded_cols = encoder.fit_transform(df[categorical_columns])
                    encoded_df = pd.DataFrame(encoded_cols.toarray(), columns=encoder.get_feature_names_out(categorical_columns))
                    df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)

                # Encodage TF-IDF pour les colonnes textuelles si nécessaire
                text_columns = df.select_dtypes(include=['object']).columns
                if not text_columns.empty:
                    for col in text_columns:
                        tfidf = TfidfVectorizer(max_features=1000)
                        tfidf_result = tfidf.fit_transform(df[col].values.astype('U'))
                        tfidf_df = pd.DataFrame(tfidf_result.toarray(), columns=tfidf.get_feature_names_out())
                        df = pd.concat([df.drop(columns=[col]), tfidf_df], axis=1)

                # Séparation des données en ensembles d'entraînement et de test
                X = df.drop(columns=df.columns[-1])  # Exclure la dernière colonne comme colonne cible
                y = df.iloc[:, -1]  # Sélectionner la dernière colonne comme colonne cible

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Mise à l'échelle des fonctionnalités numériques
                numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
                scaler = StandardScaler()
                X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
                X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

                # Sauvegarde du fichier nettoyé et transformé si nécessaire
                save_dir = os.path.join(settings.MEDIA_ROOT, 'Enregistrement')
                save_file_path = os.path.join(save_dir, selected_file)
                if selected_file.endswith('.csv'):
                    df.to_csv(save_file_path, index=False)
                elif selected_file.endswith(('.xlsx', '.xls')):
                    df.to_excel(save_file_path, index=False, engine='openpyxl')

                # Préparation des informations à passer au template
                X_train_shape = X_train.shape
                X_test_shape = X_test.shape

                # Informations supplémentaires à afficher dans le template
                processed_data_info = {
                    'X_train_shape': X_train_shape,
                    'X_test_shape': X_test_shape,
                    # Ajoutez d'autres informations au besoin
                }

                return render(request, 'pages/pretraitement.html', {
                    'uploaded_files': uploaded_files,
                    'message': f"Le fichier {selected_file} a été prétraité avec succès.",
                    'processed_data_info': processed_data_info,
                })
            except Exception as e:
                return render(request, 'pages/pretraitement.html', {
                    'uploaded_files': uploaded_files,
                    'error_message': f"Erreur lors du traitement du fichier {selected_file}: {str(e)}"
                })

    return render(request, 'pages/pretraitement.html', {'uploaded_files': uploaded_files})