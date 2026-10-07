import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Classifiers & Metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Step 1: Load Dataset
print("=== Step 1: Loading aug_train.csv ===")
df = pd.read_csv('aug_train.csv')
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
print("First 5 rows:")
print(df.head())

# Step 2: Data Cleaning & Feature Drop
print("\n=== Step 2: Checking Missing Values ===")
print(df.isnull().sum())

# Drop enrollee_id as it is just an ID column
if 'enrollee_id' in df.columns:
    df = df.drop(columns=['enrollee_id'])

# Step 3: Separate Features (X) and Target (y)
X = df.drop(columns=['target'])
y = df['target'].astype(int)

print(f"\nTarget distribution (0: Not looking for job change, 1: Looking for job change):")
print(y.value_counts())

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical features ({len(num_cols)}): {num_cols}")
print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

# Step 4: Build Preprocessing Pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ]
)

# Step 5: Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n=== Step 5: Preprocessing Features ===")
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)
print(f"Preprocessed training data shape: {X_train_prep.shape}")

# Step 6: Train & Evaluate Multiple Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42)
}

best_model = None
best_auc = 0
best_model_name = ""

print("\n=== Step 6: Model Training & Evaluation ===")
for name, model in models.items():
    if name == "Hist Gradient Boosting":
        # HistGradientBoosting handles raw data with categorical dtype directly
        X_train_hgb = X_train.copy()
        X_test_hgb = X_test.copy()
        for c in cat_cols:
            X_train_hgb[c] = X_train_hgb[c].astype('category')
            X_test_hgb[c] = X_test_hgb[c].astype('category')
        
        hgb_model = HistGradientBoostingClassifier(categorical_features='from_dtype', random_state=42)
        hgb_model.fit(X_train_hgb, y_train)
        preds = hgb_model.predict(X_test_hgb)
        probs = hgb_model.predict_proba(X_test_hgb)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc * 100:.2f}%")
        print(f"ROC-AUC Score: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_model = hgb_model
            best_model_name = name
    else:
        model.fit(X_train_prep, y_train)
        preds = model.predict(X_test_prep)
        probs = model.predict_proba(X_test_prep)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc * 100:.2f}%")
        print(f"ROC-AUC Score: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_model_name = name

print(f"\nWinner Model: {best_model_name} (ROC-AUC: {best_auc:.4f})")

# Step 7: Save Preprocessor & Best Model
with open('aug_preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

with open('aug_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("\nModel and Preprocessor saved as 'aug_model.pkl' and 'aug_preprocessor.pkl'")

# Step 8: Sample Prediction on New Candidate Profile
sample_candidate = pd.DataFrame([{
    'city': 'city_103',
    'city_development_index': 0.920,
    'gender': 'Male',
    'relevent_experience': 'Has relevent experience',
    'enrolled_university': 'no_enrollment',
    'education_level': 'Graduate',
    'major_discipline': 'STEM',
    'experience': '>20',
    'company_size': '50-99',
    'company_type': 'Pvt Ltd',
    'last_new_job': '1',
    'training_hours': 36
}])

print("\n=== Step 8: Sample Prediction ===")
if best_model_name == "Hist Gradient Boosting":
    for c in cat_cols:
        sample_candidate[c] = sample_candidate[c].astype('category')
    sample_pred = best_model.predict(sample_candidate)[0]
    sample_prob = best_model.predict_proba(sample_candidate)[0][1]
else:
    sample_prep = preprocessor.transform(sample_candidate)
    sample_pred = best_model.predict(sample_prep)[0]
    sample_prob = best_model.predict_proba(sample_prep)[0][1]

status = "Looking for a job change (1)" if sample_pred == 1 else "Not looking for a job change (0)"
print(f"Candidate Profile: Graduate, STEM, >20 yrs exp, CDI=0.920")
print(f"Predicted Result: {status}")
print(f"Probability of changing job: {sample_prob * 100:.2f}%")
