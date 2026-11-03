import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
catalog_data = {
    'product_id': [101, 102, 103, 104, 105],
    'title': ['Wireless Ergonomic Mouse', 'Mechanical Gaming Keyboard', 'UltraWide 4K Monitor', 'Bluetooth Headphones', 'Ergonomic Office Chair'],
    'description': [
        'Wireless optical mouse with ergonomic design, adjustable DPI, and long battery life for office work.',
        'RGB backlit mechanical keyboard with tactile blue switches, anti-ghosting, designed for gaming.',
        '34-inch ultrawide 4K monitor with IPS display, 144Hz refresh rate, perfect for gaming and multitasking.',
        'Over-ear wireless headphones with active noise cancellation, deep bass, and 40-hour battery life.',
        'High-back ergonomic office chair with lumbar support, adjustable armrests, and breathable mesh fabric.'
    ]
}
df_catalog = pd.DataFrame(catalog_data)
interaction_data = {
    'user_id': ['U1', 'U1', 'U1', 'U2', 'U2', 'U2', 'U3', 'U3', 'U4', 'U4', 'U5'],
    'product_id': [101, 102, 103, 101, 103, 104, 102, 105, 101, 104, 105],
    'rating': [5.0, 4.0, 1.0, 4.5, 5.0, 2.0, 4.0, 5.0, 5.0, 4.0, 5.0]
}
df_interactions = pd.DataFrame(interaction_data)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_catalog['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
reader = Reader(rating_scale=(1.0, 5.0))
dataset = Dataset.load_from_df(df_interactions[['user_id', 'product_id', 'rating']], reader)
trainset = dataset.build_full_trainset()

svd_model = SVD(n_factors=20, lr_all=0.007, reg_all=0.02, random_state=42)
svd_model.fit(trainset)

def get_hybrid_recommendations(user_id, current_product_id, df_cat, sim_matrix, collab_model, top_n=2):
    current_idx = df_cat[df_cat['product_id'] == current_product_id].index[0]
    content_scores = list(enumerate(sim_matrix[current_idx]))
    
    hybrid_rankings = []
    
    for idx, content_score in content_scores:
        target_product_id = df_cat.iloc[idx]['product_id']
        if target_product_id == current_product_id:
            continue
        predicted_rating = collab_model.predict(user_id, target_product_id).est
        normalized_collab_score = (predicted_rating - 1) / 4.0
        final_hybrid_score = (0.5 * content_score) + (0.5 * normalized_collab_score)
        
        hybrid_rankings.append({
            'product_id': target_product_id,
            'title': df_cat.iloc[idx]['title'],
            'hybrid_score': round(final_hybrid_score, 4)
        })
    df_rankings = pd.DataFrame(hybrid_rankings)
    df_rankings = df_rankings.sort_values(by='hybrid_score', ascending=False)
    return df_rankings.head(top_n)
active_user = 'U1'
active_product = 101

recommendation_tray = get_hybrid_recommendations(
    user_id=active_user, 
    current_product_id=active_product, 
    df_cat=df_catalog, 
    sim_matrix=cosine_sim, 
    collab_model=svd_model, 
    top_n=2
)

print(f"🛒 Generating Hybrid Recommendations for {active_user} viewing item {active_product}...")
print(recommendation_tray.to_string(index=False))
