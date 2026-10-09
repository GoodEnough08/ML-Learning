import numpy as np 
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('placement.csv', usecols=['cgpa','placement_exam_marks','placed'])
print(df.sample(5))
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['cgpa'])

plt.subplot(1,2,2)
sns.distplot(df['placement_exam_marks'])

# plt.show()
'''Trimming'''
print("Highest allowed",df['cgpa'].mean() + 3*df['cgpa'].std())
print("Lowest allowed",df['cgpa'].mean() - 3*df['cgpa'].std())
df[(df['cgpa'] > 8.80) | (df['cgpa'] < 5.11)]
new_df = df[(df['cgpa'] < 8.80) & (df['cgpa'] > 5.11)]
print(new_df)
'''Calculating Z score'''
df['cgpa_zscore'] = (df['cgpa'] - df['cgpa'].mean())/df['cgpa'].std()
df[(df['cgpa_zscore'] > 3) | (df['cgpa_zscore'] < -3)]
new_df = df[(df['cgpa_zscore'] < 3) & (df['cgpa_zscore'] > -3)]
print(new_df)
'''capping'''
upper_limit = df['cgpa'].mean() + 3*df['cgpa'].std()
lower_limit = df['cgpa'].mean() - 3*df['cgpa'].std()
df['cgpa'] = np.where(
    df['cgpa']>upper_limit,
    upper_limit,
    np.where(
        df['cgpa']<lower_limit,
        lower_limit,
        df['cgpa']
    )
)