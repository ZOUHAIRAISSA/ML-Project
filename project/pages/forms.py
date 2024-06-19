from django import forms
from .models import UploadedFile  # Modèle à définir

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ['file']

    def clean_file(self):
        file = self.cleaned_data.get('file')

        if file:
            if not file.name.endswith(('.csv', '.json', '.xlsx', '.xls')):
                raise forms.ValidationError("Invalid file format. Only CSV, JSON, and Excel files are allowed.")
            # Remove or adjust the file size limit as needed
            if file.size > 50 * 1024 * 1024:  # Optional: limit file size to 10MB
                 raise forms.ValidationError("The file is too large. The size should not exceed 10MB.")

        return file