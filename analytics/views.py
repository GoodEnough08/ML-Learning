from django.shortcuts import render, redirect
from .forms import DatasetUploadForm
from .models import Dataset
from .services import process_and_cluster_data, generate_ai_insights

def upload_and_analyze(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save()
            
            try:
                stats, cluster_summary = process_and_cluster_data(dataset.file.path)
                ai_report = generate_ai_insights(cluster_summary)
                dataset.summary_insight = ai_report
                dataset.save()

                context = {
                    'dataset': dataset,
                    'stats': stats,
                    'cluster_summary': cluster_summary,
                    'ai_report': ai_report
                }
                return render(request, 'analytics/dashboard.html', context)
            except Exception as err:
                return render(request, 'analytics/upload.html', {'form': form, 'error': str(err)})
    else:
        form = DatasetUploadForm()

    return render(request, 'analytics/upload.html', {'form': form})