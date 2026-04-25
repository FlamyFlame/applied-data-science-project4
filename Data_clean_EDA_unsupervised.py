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

# ==========================================
# Data Acquisition & Preparation
# ==========================================
print("--- Step 1: Data Acquisition & Preparation ---")
data_dir = 'archive'

# Load relational datasets
orders = pd.read_csv(os.path.join(data_dir, 'olist_orders_dataset.csv'))
reviews = pd.read_csv(os.path.join(data_dir, 'olist_order_reviews_dataset.csv'))
items = pd.read_csv(os.path.join(data_dir, 'olist_order_items_dataset.csv'))
products = pd.read_csv(os.path.join(data_dir, 'olist_products_dataset.csv'))
sellers = pd.read_csv(os.path.join(data_dir, 'olist_sellers_dataset.csv'))
customers = pd.read_csv(os.path.join(data_dir, 'olist_customers_dataset.csv'))
geo = pd.read_csv(os.path.join(data_dir, 'olist_geolocation_dataset.csv'))

# Data Merging Logic
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

# Data Cleaning & Formatting
time_cols = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
for col in time_cols:
    df[col] = pd.to_datetime(df[col])

df = df.dropna(subset=['order_delivered_customer_date'])
df = df[df['order_delivered_customer_date'] >= df['order_purchase_timestamp']]

print(f"Preparation Complete. Shape: {df.shape}")


# ==========================================
# Exploratory Data Analysis (EDA)
# ==========================================
print("\n--- Step 2: Exploratory Data Analysis (EDA) ---")

# Distribution of Target
plt.figure(figsize=(8, 4))
sns.countplot(x='review_score', data=df, palette='viridis')
plt.title('Distribution of Customer Review Scores')
plt.show()

# Unsupervised Discovery (PCA)
print("Applying PCA to discover latent structures...")
physical_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
# Log-transform and Scale for better PCA visualization
pca_temp = np.log1p(df[physical_cols].fillna(df[physical_cols].median()))
pca_scaled = StandardScaler().fit_transform(pca_temp)

pca_discovery = PCA(n_components=2, random_state=42)
pca_proj = pca_discovery.fit_transform(pca_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(pca_proj[:, 0], pca_proj[:, 1], alpha=0.1, c='#3498db', s=7)
plt.title('PCA Projection of Product Physical Dimensions')
plt.xlabel(f'PC1: Size Factor ({pca_discovery.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2: Shape Factor ({pca_discovery.explained_variance_ratio_[1]*100:.1f}%)')
plt.show()


# ==========================================
# Feature Engineering & Preprocessing
# ==========================================
print("\n--- Step 3: Feature Engineering & Preprocessing ---")

# Advanced Feature Creation
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return R * 2 * np.arcsin(np.sqrt(a))

df['distance_km'] = haversine(df['buyer_lat'], df['buyer_lng'], df['seller_lat'], df['seller_lng'])
df['delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
df['delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
df['freight_ratio'] = df['freight_value'] / (df['price'] + df['freight_value'])

# Dynamic Rolling Window (30-day performance)
df = df.sort_values(['seller_id', 'order_purchase_timestamp']).reset_index(drop=True)
df['seller_recent_delay_avg'] = df.groupby('seller_id').rolling('30D', on='order_purchase_timestamp')['delay_days'].mean().values
df['seller_recent_delay_avg'].fillna(0, inplace=True)

# Advanced Imputation & Outlier Removal
print("Applying KNN Imputation and Isolation Forest...")
dim_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'distance_km']
knn_imputer = KNNImputer(n_neighbors=5)
df[dim_cols] = knn_imputer.fit_transform(df[dim_cols])
df['product_volume_cm3'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']

iso_features = ['price', 'freight_value', 'product_volume_cm3', 'distance_km']
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df['is_outlier'] = iso_forest.fit_predict(df[iso_features].fillna(0))
clean_df = df[df['is_outlier'] == 1].drop(columns=['is_outlier'])

# Define binary target before visualization
clean_df['is_bad_review'] = (clean_df['review_score'] <= 2).astype(int)

# Unsupervised Feature Engineering: Clustering in PCA Space
print("Optimizing Clusters via PCA-KMeans...")
cluster_features = ['product_weight_g', 'product_volume_cm3', 'freight_ratio']
cluster_log = np.log1p(clean_df[cluster_features])
cluster_scaled = StandardScaler().fit_transform(cluster_log)

# Reduce noise with PCA before clustering
pca_cluster = PCA(n_components=2, random_state=42)
cluster_pca_proj = pca_cluster.fit_transform(cluster_scaled)

kmeans = KMeans(n_clusters=4, random_state=42)
clean_df['logistics_cluster'] = kmeans.fit_predict(cluster_pca_proj)

# Visualizing the Advanced Insights
print("Generating visualizations for engineered features...")

# Plot 1: Distance vs Delivery Time
plt.figure(figsize=(10, 5))
sns.regplot(x='distance_km', y='delivery_days', data=clean_df.sample(2000),
            scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
plt.title('Impact of Spherical Distance on Delivery Time')
plt.show()


# Plot 2: Rolling Window Effectiveness (KDE Plot)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=clean_df[clean_df['is_bad_review']==0], x='seller_recent_delay_avg', label='Good Reviews', fill=True, color='seagreen', common_norm=False)
sns.kdeplot(data=clean_df[clean_df['is_bad_review']==1], x='seller_recent_delay_avg', label='Bad Reviews', fill=True, color='crimson', common_norm=False)
plt.xlim(-1, 10)
plt.title('Density: Recent Seller Delay vs. Review Quality')
plt.legend()
plt.show()


# Plot 3: PCA Space Clusters
plt.figure(figsize=(10, 6))
plt.scatter(cluster_pca_proj[:, 0], cluster_pca_proj[:, 1], c=clean_df['logistics_cluster'], cmap='viridis', alpha=0.5, s=8)
plt.title('Logistics Clusters in PCA Space')
plt.show()


# Final Scaling & Encoding
print("Finalizing preprocessing for modeling...")
clean_df['logistics_cluster'] = clean_df['logistics_cluster'].astype(str)
num_features = ['price', 'freight_value', 'distance_km', 'delivery_days', 'delay_days',
                'freight_ratio', 'product_volume_cm3', 'seller_recent_delay_avg']

clean_df[num_features] = StandardScaler().fit_transform(clean_df[num_features])
clean_df = pd.get_dummies(clean_df, columns=['logistics_cluster'], drop_first=True)

# Export Data
output_file = os.path.join(data_dir, 'clean_df_final.csv')
clean_df.to_csv(output_file, index=False)

print(f"Final data saved to: {output_file}")


