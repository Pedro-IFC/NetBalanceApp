import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("history_advanced_complete.csv")
df = df.drop(columns=['Individual_ID', 'Generation'])

df = df.drop_duplicates()

X = df.drop(columns=['Fitness'])
y = df['Fitness']

gb = GradientBoostingRegressor(random_state=42)
gb.fit(X, y)

#gb.predict