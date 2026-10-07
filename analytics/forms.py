from django import forms
from .models import Dataset

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['title', 'file']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., Customer Segmentation Q4'}),
            'file': forms.FileInput(attrs={'class': 'form-control', 'accept': '.csv'}),
        }