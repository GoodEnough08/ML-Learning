import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
df = pd.read_csv('placement.csv')
print("--- Initial Data Snapshot ---")
print(df.head())
if 'index' in df.columns:
    df = df.drop(columns=['index'])
print("\nMissing values in dataset:\n", df.isnull().sum())
duplicate_count = df.duplicated().sum()
print("\nNumber of duplicate rows:", duplicate_count)

df = df.drop_duplicates()
X = df[['iq', 'cgpa']]
y = df['placement']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("\n--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("\nModel and Scaler successfully saved as 'model.pkl' and 'scaler.pkl'")
new_students = pd.DataFrame({
    'iq': [120, 95, 110],
    'cgpa': [8.5, 5.5, 7.2]
})

new_students_scaled = scaler.transform(new_students)
predictions = model.predict(new_students_scaled)

print("\n--- Sample Predictions ---")
for i, pred in enumerate(predictions):
    iq_val = new_students.iloc[i]['iq']
    cgpa_val = new_students.iloc[i]['cgpa']
    status = "Placed (1)" if pred == 1 else "Not Placed (0)"
    print(f"Student (IQ: {iq_val}, CGPA: {cgpa_val}) -> Predicted Result: {status}")
