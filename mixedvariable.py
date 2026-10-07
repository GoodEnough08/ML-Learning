import numpy as np
import pandas as pd

df=pd.read_csv('titanic.csv')
print(df.head())
print(df['number'].unique())
df['numerical_num']=pd.to_numeric(df['number'],errors='coerce',downcast='integer')
df['number_cat']=np.where(df['numerical_num'].isnull(),df['number'],np.nan)
print(df.head())
df['cabin_num']=df['Cabin'].str.extract(r'(\d+)')
df['cabin_num'] = pd.to_numeric(df['cabin_num'],errors='coerce',downcast='integer')
df['cabin_cat']=df['Cabin'].str[0]
df['ticket_num'] = df['Ticket'].apply(lambda s: s.split()[-1])
df['ticket_num'] = pd.to_numeric(df['ticket_num'],errors='coerce',downcast='integer')
df['ticket_cat'] = df['Ticket'].apply(lambda s: s.split()[0])
df['ticket_cat'] = np.where(df['ticket_cat'].str.isdigit(), np.nan,df['ticket_cat'])
print(df.head(20))