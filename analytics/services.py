import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from google import genai

def process_and_cluster_data(file_path, n_clusters=3):
    """
    1. Reads CSV via Pandas.
    2. Selects numeric columns and handles missing values.
    3. Scales features and fits a KMeans clustering model.
    4. Returns summary dictionary and cluster statistics.
    """
    # 1. Load CSV
    df = pd.read_csv(file_path)

    # 2. Select numeric columns (exclude ID/Index columns)
    raw_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [
        col for col in raw_numeric 
        if not (col.lower() in ['id', 'index'] or col.lower().endswith('_id') or col.lower().endswith('id') or col.lower().startswith('id_'))
    ]
    if len(numeric_cols) < 2:
        numeric_cols = raw_numeric

    if len(numeric_cols) < 2:
        raise ValueError("The dataset must contain at least 2 numeric feature columns for clustering.")


    if len(df) < 2:
        raise ValueError("The dataset must contain at least 2 rows for clustering.")

    actual_clusters = min(n_clusters, len(df))

    # 3. Clean nulls using Pandas/NumPy (fill with median)
    df_clean = df[numeric_cols].copy()
    for col in numeric_cols:
        median_val = df_clean[col].median()
        if pd.isna(median_val):
            median_val = 0
        df_clean[col] = df_clean[col].fillna(median_val)

    # 4. Standardize data for K-Means (ML Step)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_clean)

    # 5. Fit KMeans
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(scaled_features)

    # 6. Aggregate cluster profiles using Pandas
    cluster_summary = df.groupby('Cluster')[numeric_cols].mean().round(2).to_dict(orient='index')

    # General dataset stats
    overall_stats = {
        "total_rows": int(len(df)),
        "columns_analyzed": numeric_cols,
        "clusters_found": actual_clusters
    }

    return overall_stats, cluster_summary


def generate_ai_insights(cluster_summary):
    """
    Feeds cluster centroids to an LLM to generate plain-English takeaways.
    """
    import dotenv
    from pathlib import Path
    
    # Reload environment variables from .env dynamically
    base_dir = Path(__file__).resolve().parent.parent
    dotenv.load_dotenv(base_dir / '.env', override=True)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key.strip() == "" or api_key.strip() == "YOUR_GEMINI_API_KEY_HERE":
        return "⚠️ AI insights unavailable: GEMINI_API_KEY is not configured. Please add your GEMINI_API_KEY in the .env file."

    client = genai.Client(api_key=api_key.strip())

    prompt = f"""
    You are an expert business and data analyst.
    Below are the mean metrics for {len(cluster_summary)} customer segments generated via K-Means clustering:
    
    {cluster_summary}

    Provide:
    1. A short descriptive name for each cluster (e.g., 'High-Value VIPs', 'At-Risk Inactive').
    2. A 2-sentence explanation of what characterizes each group.
    3. Exactly 1 actionable business recommendation for each group.
    Keep the tone concise, professional, and clear.
    """

    models_to_try = ['gemini-3.8-flash', 'gemini-flash-latest', 'gemini-3.5-flash-lite', 'gemini-3.5-flash']
    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if response and response.text:
                return response.text
        except Exception as e:
            last_error = e
            continue

    return f"Error generating insights: {str(last_error)}"