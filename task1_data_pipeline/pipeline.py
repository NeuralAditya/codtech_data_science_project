import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data.csv')

# Preprocessing
df = df.dropna()  # remove missing values

# Transformation
scaler = StandardScaler()
df[['col1', 'col2']] = scaler.fit_transform(df[['col1', 'col2']])

# Save processed data
df.to_csv('processed_data.csv', index=False)

print("Pipeline completed!")