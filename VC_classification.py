import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

investments_df = pd.read_csv('data/investments.csv')

investments_df.dropna(inplace=True)

investment_counts = investments_df.groupby('investor_object_id').size().reset_index(name='investment_count')

funds_df = pd.read_csv('data/funds.csv')

funds_df = pd.merge(funds_df, investment_counts, left_on='object_id', right_on='investor_object_id', how='inner')

X = funds_df[['raised_amount', 'investment_count']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
funds_df['cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(funds_df['raised_amount'], funds_df['investment_count'], c=funds_df['cluster'], cmap='viridis', marker='o', s=50)
plt.title('Clustering of VC Funds based on Investments')
plt.xlabel('Raised Amount')
plt.ylabel('Number of Investments')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

for cluster in sorted(funds_df['cluster'].unique()):
    cluster_data = funds_df[funds_df['cluster'] == cluster]
    print(f"Cluster {cluster}:")
    print(cluster_data[['fund_id', 'raised_amount', 'investment_count']])
    print("\n")
