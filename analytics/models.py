from django.db import models

class Dataset(models.Model):
    title = models.CharField(max_length=100)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    summary_insight = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.title