import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('placement.csv',usecols=['cgpa','placement_exam_marks','placed'])
percentile25 = df['placement_exam_marks'].quantile(0.25)
percentile75 = df['placement_exam_marks'].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
new_df = df[df['placement_exam_marks'] < upper_limit]
# plt.figure(figsize=(16,8))
# plt.subplot(2,2,1)
# sns.distplot(df['placement_exam_marks'])

# plt.subplot(2,2,2)
# sns.boxplot(df['placement_exam_marks'])

# plt.subplot(2,2,3)
# sns.distplot(new_df['placement_exam_marks'])

# plt.subplot(2,2,4)
# sns.boxplot(new_df['placement_exam_marks'])

# plt.show()
new_df_cap = df.copy()

new_df_cap['placement_exam_marks'] = np.where(
    new_df_cap['placement_exam_marks'] > upper_limit,
    upper_limit,
    np.where(
        new_df_cap['placement_exam_marks'] < lower_limit,
        lower_limit,
        new_df_cap['placement_exam_marks']
    )
)
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])

plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'])

plt.subplot(2,2,3)
sns.distplot(new_df_cap['placement_exam_marks'])

plt.subplot(2,2,4)
sns.boxplot(new_df_cap['placement_exam_marks'])

plt.show()