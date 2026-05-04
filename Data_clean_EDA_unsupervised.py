import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as mtick

# ==========================================
# Step 1: Data Acquisition & Logical Preparation
# ==========================================
print("--- Step 1: Data Acquisition & Preparation ---")
data_dir = 'processed'

# Load relational datasets
orders = pd.read_csv(os.path.join(data_dir, 'olist_orders_dataset.csv'))
reviews = pd.read_csv(os.path.join(data_dir, 'olist_order_reviews_dataset.csv'))
items = pd.read_csv(os.path.join(data_dir, 'olist_order_items_dataset.csv'))
products = pd.read_csv(os.path.join(data_dir, 'olist_products_dataset.csv'))
sellers = pd.read_csv(os.path.join(data_dir, 'olist_sellers_dataset.csv'))
customers = pd.read_csv(os.path.join(data_dir, 'olist_customers_dataset.csv'))
geo = pd.read_csv(os.path.join(data_dir, 'olist_geolocation_dataset.csv'))

# Multi-table merge & Geospatial aggregation (Handling many-to-many conflicts)
geo_dict = geo.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()

df = orders.merge(reviews[['order_id', 'review_score']], on='order_id', how='inner')
df = df.merge(items, on='order_id', how='left')
df = df.merge(products, on='product_id', how='left')
df = df.merge(sellers, on='seller_id', how='left')
df = df.merge(customers, on='customer_id', how='left')

df = df.merge(geo_dict, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
df.rename(columns={'geolocation_lat': 'buyer_lat', 'geolocation_lng': 'buyer_lng'}, inplace=True)
df = df.merge(geo_dict, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
df.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}, inplace=True)

# Data Cleaning & Temporal Consistency Check
time_cols = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
for col in time_cols:
    df[col] = pd.to_datetime(df[col])

# Remove missing deliveries and physically impossible records (delivered before purchased)
df = df.dropna(subset=['order_delivered_customer_date'])
df = df[df['order_delivered_customer_date'] >= df['order_purchase_timestamp']]

# Define the Target Variable Early (1 = Bad Review, 0 = Good/Neutral)
df['is_bad_review'] = (df['review_score'] <= 2).astype(int)

print(f"Preparation Complete. Master dataset records: {df.shape[0]}")


# ==========================================
# Step 2: Core Feature Engineering (Spatial & Temporal)
# ==========================================
print("\n--- Step 2: Core Feature Engineering ---")

# Spatial Feature: Haversine Distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return R * 2 * np.arcsin(np.sqrt(a))

df['distance_km'] = haversine(df['buyer_lat'], df['buyer_lng'], df['seller_lat'], df['seller_lng'])

# Temporal & Financial Features
df['delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
df['delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
df['freight_ratio'] = df['freight_value'] / (df['price'] + df['freight_value'])

# Dynamic Rolling Window: Seller's 30-day historical delay average
df = df.sort_values(['seller_id', 'order_purchase_timestamp']).reset_index(drop=True)
df['seller_recent_delay_avg'] = df.groupby('seller_id').rolling('30D', on='order_purchase_timestamp')['delay_days'].mean().values
df['seller_recent_delay_avg'].fillna(0, inplace=True)


# ==========================================
# Step 3: Advanced Preprocessing & Unsupervised Synergy
# ==========================================
print("\n--- Step 3: Advanced Preprocessing & Clustering ---")

# Advanced Imputation & Anomaly Detection (KNN & Isolation Forest)
dim_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'distance_km']
knn_imputer = KNNImputer(n_neighbors=5)
df[dim_cols] = knn_imputer.fit_transform(df[dim_cols])
df['product_volume_cm3'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']

# Filter high-dimensional outliers to protect ML models
iso_features = ['price', 'freight_value', 'product_volume_cm3', 'distance_km']
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df['is_outlier'] = iso_forest.fit_predict(df[iso_features].fillna(0))
clean_df = df[df['is_outlier'] == 1].drop(columns=['is_outlier'])

# Unsupervised Synergy: PCA-KMeans Logistics Profiling
cluster_features = ['product_weight_g', 'product_volume_cm3', 'freight_ratio']
cluster_log = np.log1p(clean_df[cluster_features])
cluster_scaled = StandardScaler().fit_transform(cluster_log)

# Denoise via PCA before clustering
pca_cluster = PCA(n_components=2, random_state=42)
cluster_pca_proj = pca_cluster.fit_transform(cluster_scaled)

kmeans = KMeans(n_clusters=4, random_state=42)
clean_df['logistics_cluster'] = kmeans.fit_predict(cluster_pca_proj)


# ==========================================
#  Comprehensive Visualizations
# ==========================================
print("\n--- Generating Comprehensive Visualizations ---")

# Plot 1: Logistics Cluster Boundaries in PCA Space (Structural Discovery)
plt.figure(figsize=(10, 6))
plt.scatter(cluster_pca_proj[:, 0], cluster_pca_proj[:, 1], c=clean_df['logistics_cluster'], cmap='viridis', alpha=0.5, s=8)
plt.title('Logistics Clusters in Principal Component Space')
plt.xlabel('PC1 (Size/Mass Factor)')
plt.ylabel('PC2 (Cost/Efficiency Factor)')
plt.colorbar(label='Logistics Cluster ID')
plt.show()

# Plot 2: Unsupervised Cluster Risk Profiling (Business Impact of Clusters)
plt.figure(figsize=(10, 6))
cluster_bad_rate = clean_df.groupby('logistics_cluster')['is_bad_review'].mean().reset_index()
sns.barplot(x='logistics_cluster', y='is_bad_review', data=cluster_bad_rate, palette='magma')
plt.title('Probability of Bad Review by Logistics Profile (PCA-KMeans Clusters)', fontsize=14)
plt.xlabel('Logistics Cluster ID')
plt.ylabel('Bad Review Probability')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.show()

# Plot 3: Global Correlation Matrix (Identifying primary drivers)
plt.figure(figsize=(10, 8))
corr_cols = ['is_bad_review', 'delivery_days', 'delay_days', 'freight_ratio', 'distance_km', 'seller_recent_delay_avg']
sns.heatmap(clean_df[corr_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix: Drivers of Customer Dissatisfaction', fontsize=15)
plt.show()

# Plot 4: Delivery Time vs. Exact Review Score (Temporal Impact)
plt.figure(figsize=(10, 6))
plot_data = clean_df[(clean_df['delivery_days'] >= 0) & (clean_df['delivery_days'] <= 60)]
sns.boxplot(x='review_score', y='delivery_days', data=plot_data, palette='RdYlGn')
plt.title('Actual Delivery Days vs. Customer Review Score', fontsize=14)
plt.xlabel('Review Score (1 = Worst, 5 = Best)')
plt.ylabel('Delivery Time (Days)')
plt.show()

# Plot 5: Freight Ratio Distribution (Financial Impact - The "Freight Assassin")
plt.figure(figsize=(10, 6))
sns.violinplot(x='is_bad_review', y='freight_ratio', data=clean_df, palette='Set2', inner='quartile')
plt.title('Freight Cost Ratio: Good vs. Bad Reviews', fontsize=14)
plt.xlabel('Review Type (0: Good/Neutral, 1: Bad Review)')
plt.ylabel('Freight Ratio (Shipping / Total Cost)')
plt.show()

# Plot 6: Seller Reliability KDE (Validating the Rolling Window Feature)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=clean_df[clean_df['is_bad_review']==0], x='seller_recent_delay_avg', label='Good Reviews', fill=True, color='seagreen', common_norm=False)
sns.kdeplot(data=clean_df[clean_df['is_bad_review']==1], x='seller_recent_delay_avg', label='Bad Reviews', fill=True, color='crimson', common_norm=False)
plt.xlim(-1, 10)
plt.title('Probability Density: Recent Seller Delays vs. Future Dissatisfaction')
plt.xlabel('Average Seller Delay in Past 30 Days')
plt.legend()
plt.show()


# ==========================================
# Final Scaling, Encoding & Exporting
# ==========================================
print("\n--- Finalizing Preprocessing for ML Models ---")
clean_df['logistics_cluster'] = clean_df['logistics_cluster'].astype(str)

num_features = ['price', 'freight_value', 'distance_km', 'delivery_days', 'delay_days',
                'freight_ratio', 'product_volume_cm3', 'seller_recent_delay_avg']

# Final scaling to Standard Normal Distribution
clean_df[num_features] = StandardScaler().fit_transform(clean_df[num_features])
# One-Hot Encoding for Cluster IDs
clean_df = pd.get_dummies(clean_df, columns=['logistics_cluster'], drop_first=True)

# Save final dataset
output_file = os.path.join(data_dir, 'clean_df_final.csv')
clean_df.to_csv(output_file, index=False)

print(f"Final Feature Matrix Shape: {clean_df.shape}")
print(f"Data saved to: {output_file}")