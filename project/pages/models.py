# models.py
import csv
from django.db import models

def dataset_upload_to(instance, filename):
    return 'datasets/{filename}'.format(filename=filename)

class Dataset(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to=dataset_upload_to)

    def __str__(self):
        return self.name

# apps/home/models.py
from django.db import models

class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name
    
    def read_csv_content(self):
        try:
            with self.file.open('r') as f:
                reader = csv.reader(f)
                return list(reader)
        except Exception as e:
            return [[str(e)]]
