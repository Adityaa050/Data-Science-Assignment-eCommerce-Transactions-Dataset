import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the datasets
customers = pd.read_csv("Customers.csv")
transactions = pd.read_csv("Transactions.csv")

# Merge the datasets
customer_transactions = pd.merge(transactions, customers, on="CustomerID")

# Aggregate the features
customer_features = customer_transactions.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'TransactionID': 'count',
    'Region': 'first'
}).reset_index()
customer_features.rename(columns={'TransactionID': 'NumTransactions'}, inplace=True)

# Normalize numerical features
scaler = MinMaxScaler()
numerical_cols = ['TotalValue', 'Quantity', 'NumTransactions']
customer_features[numerical_cols] = scaler.fit_transform(customer_features[numerical_cols])

# One-hot encode the 'Region' column
customer_features = pd.get_dummies(customer_features, columns=['Region'])

# Clustering and DB Index Calculation
feature_matrix = customer_features.drop('CustomerID', axis=1)
cluster_metrics = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(feature_matrix)
    db_index = davies_bouldin_score(feature_matrix, labels)
    cluster_metrics.append((k, db_index))

# Find optimal clusters
optimal_clusters = min(cluster_metrics, key=lambda x: x[1])
print(f"Optimal Number of Clusters: {optimal_clusters[0]}, DB Index: {optimal_clusters[1]}")

# Final model with optimal clusters
kmeans = KMeans(n_clusters=optimal_clusters[0], random_state=42)
customer_features['Cluster'] = kmeans.fit_predict(feature_matrix)

# Save clustering results to CSV
customer_features.to_csv("Customer_Segments.csv", index=False)
print("Clustering results saved to 'Customer_Segments.csv'")

# PCA for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(feature_matrix)
reduced_df = pd.DataFrame(reduced_features, columns=['PCA1', 'PCA2'])
reduced_df['Cluster'] = customer_features['Cluster']

# Plot clusters
plt.figure(figsize=(10, 6))
for cluster in reduced_df['Cluster'].unique():
    cluster_data = reduced_df[reduced_df['Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}')
plt.title("Customer Segmentation Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()
