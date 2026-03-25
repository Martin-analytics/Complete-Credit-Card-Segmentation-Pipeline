import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.impute import SimpleImputer as SI
import numpy as np, matplotlib.pyplot as plt, seaborn as sns, math
from scipy.stats import shapiro
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as SS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

print("Initializing dataset extraction from KaggleAPI...\n")
api = KaggleApi()
api.authenticate()

dataset_name = "arjunbhasin2013/ccdata"
print(f"Fetching {dataset_name}...")

api.dataset_download_files(dataset_name, path=".", unzip=True)
print("Data successfully extracted!")
print("\nLoading the data for Analysis...\n")
df_data = pd.read_csv('CC GENERAL.csv')
print("Data Information...")
print(df_data.info())

#4. Clean the Data for Machine Learning
customer_ids = df_data['CUST_ID']
df = df_data.drop('CUST_ID', axis=1)

print("Data Pipeline Complete.")
print("Below is the Data Structure...")
print(df.dtypes)

#Counting
total_rows = len(df)

#---Missing Values
print("Columns and their corresponding number of missing values are listed below...\n")
missn = df.isnull().sum()
print(missn)
print("\nChecking for missing values...\n")
missing = df.isna().sum().sum()
if missing == 0:
    print(f"There are no missing values in the dataset\n")
elif missing < (0.05 * total_rows * len(df.columns)):
    print(f"Minor cleaning needed. There are {missing} missing values in the dataset\n")
else:
    print(f"Large missing values: {missing} in the dataset. Consider dropping rows...\n")
    
for col in df.columns:
    missing_col = df[col].isnull().sum()
    if missing_col > 0:
        print(f"Missing columns: {missing_col}")
        perc_missing_col = round((missing_col/total_rows * 100), 1)
        if perc_missing_col > 20:
            print(f"Missing values in the dataset on {col} is {perc_missing_col}%. Consider dropping the row.\n")
        else:
            print(f"Missing values are not significant. Consider filling it up.\n")
#Simple Imputer
print("Filling the missing values for {CREDIT_LIMIT} and {MINIMUM_PAYMENTS} columns...")
print("Missing values are insignificant, so they are replaced by the averages of their corresponding columns")
imputer = SI(strategy='mean')
imputer.fit(df[['CREDIT_LIMIT', 'MINIMUM_PAYMENTS']])
df[['CREDIT_LIMIT', 'MINIMUM_PAYMENTS']] = imputer.transform(df[['CREDIT_LIMIT', 'MINIMUM_PAYMENTS']])
print("Missing Values fixed sucessfully!")
print(df.isnull().sum())

#Duplicates
print("Checking the dataset for Duplicates...\n")
dupl = df.duplicated()
dup = df.duplicated().sum()
print(dupl.sort_values(ascending=True))
if dup == 0:
    print(f"There are no duplicates in the dataset\n")
else:
    print(f"There are {dup} duplicated values in the dataset.\n")
    
#Desciptive Analysis
print("Desciptive Analysis...")
desc = df.describe()
print(desc)
print(df.info())

#Dealing with Outliers
print("Dealing with Outliers...")
print("Visualizing outliers")
out_col = [
    'BALANCE',
    'PURCHASES',
    'ONEOFF_PURCHASES', 
    'INSTALLMENTS_PURCHASES',
    'CASH_ADVANCE', 
    'CASH_ADVANCE_FREQUENCY', 
    'CASH_ADVANCE_TRX', 
    'PURCHASES_TRX', 
    'CREDIT_LIMIT',
    'PAYMENTS', 
    'MINIMUM_PAYMENTS'
]
num_col = df.copy()
num_plot = len(out_col)
num_row = math.ceil(num_plot / 2)

for col in out_col:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(f"Outlier Check: {col}", fontsize=14, fontweight='bold')
    plt.ylabel("Amount")
    plt.show()

for col in out_col:
    plt.figure(figsize=(10, 4))
    num_col[col] = np.log1p(num_col[col])
    #df[f"{col}_log"] = np.log1p(df[col])
    stat, p_value = shapiro(num_col[col])
    sns.kdeplot(num_col[col], fill=True, color="seagreen")
    plt.title(f"{col} (Log Transformed)  |  Shapiro p-value: {p_value:.4f}", fontsize=12)
    plt.xlabel("") 
    plt.show()
print("Due to the magnitude of our dataset, Shapiro-wilk Test could not perfectly visualize the outcome of the transformation\n")

#------Correlation Matrix--------to see if any column is related to another
print("Correlation Matrix to identify redundant columns...\n")
num_col_corr = num_col.corr()

sns.heatmap(num_col_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap on Num_Data', fontsize=16)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.show()
print("Dropping redundant columns...\n")

redundant_col = set()
for i in range(len(num_col_corr.columns)): #y axis
    for j in range(i):#x axis
        if abs(num_col_corr.iloc[i, j]) >= 0.80: #finds cols from both axis that has corr > 0.8
            col_i = num_col_corr.columns[i]
            col_j = num_col_corr.columns[j]
            if abs(num_col_corr[col_i]).mean() > abs(num_col_corr[col_j]).mean():
                redundant_col.add(col_i)
            else:
                redundant_col.add(col_j)
print(f"The algorith authomatically flagged these for deletion: {redundant_col}")

print("\nDue to correlation threshold of 0.8 to reduce multicollinearity while preserving meaningful features, we would normally drop columns that are above the thresold and are not the 'parent' column!")
print("\nThe automated script flagged the 'Parent' columns because they had the highest overall correlation.\n")
print("We had to resort to Principal Componenet Analysis due to the magnitude of the dataset...\n")

#Standard Scaler
print("\nStandard Scaler to compress extremely large values to decimals to allow our model make proper decisions\n")
scaler = SS()
num_col_scaled = scaler.fit_transform(num_col)

#Scree Plot
#PCA - Principal Component Analysis
print("\nBy plotting the 'Columns' against their 'Variance(information of each column)'; we aim to retain the information in the dataset while dropping noisy columns that may interfere with our model.\n")
pca = PCA()
temp_PCA = pca.fit_transform(num_col_scaled)
evr = pca.explained_variance_ratio_
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(evr) +1), evr, marker='o', color='purple')
plt.title('PCA Scree Plot - Expalined Variance Ratio for Each Component')
plt.xlabel('Components')
plt.ylabel('Explained Variance Ratio')
plt.show()

pca = PCA(n_components=5)
pca_data = pca.fit_transform(num_col_scaled)
df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
print("Data successfully compressed to 5 dimensions.")
print(df_pca.head())

loadings = pd.DataFrame(
    pca.components_.T, 
    columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], 
    index=num_col.columns
)
plt.figure(figsize=(12, 8))
sns.heatmap(loadings, cmap='coolwarm', annot=True, fmt='.2f', center=0, linewidths=0.5)

plt.title('PCA Loadings: What makes up our PCA Columns?', fontsize=16, fontweight='bold')
plt.show()

#KMeans Clustering
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(pca_data)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.title('K-Means Elbow Method to Find Customer Segments', fontsize=14, fontweight='bold')
plt.xlabel('Number of Clusters (Customer Groups)')
plt.ylabel('WCSS (Inertia)')
plt.xticks(range(1, 11))
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("\nBecause our elbow plot is hard to decide the elbow point, we use 'The Silhouette Score' to decide!\n")
cluster_options = [3, 4, 5, 6]
silhouette_scores = []

for k in cluster_options:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(pca_data)
    score = silhouette_score(pca_data, cluster_labels)
    silhouette_scores.append(score)
    print(f"Silhouette Score for {k} clusters: {score:.4f}")

plt.figure(figsize=(8, 5))
plt.bar(cluster_options, silhouette_scores, color='teal')
plt.title('Silhouette Scores', fontsize=14, fontweight='bold')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()
print("\nWe pick the 'Silhouette' value that have the highest score and fit to KMeans and then visualize...\n")

final_kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
cluster_labels = final_kmeans.fit_predict(pca_data)

df__ = df.copy()
df__['Customer_Persona'] = cluster_labels

persona = df__.groupby('Customer_Persona').mean()
print("\n====== THE 3 CUSTOMER PERSONAS ======\n")
print(persona.T.round(2)) #Display the transpose (.T) so it's easier to read top-to-bottom

print("\nIntegrating 'CUST_ID' into the result...\n")
final_mapping = pd.DataFrame({
    'CUST_ID': customer_ids,
    'Customer_Persona': cluster_labels
})
print(final_mapping.head(10))
                       
print("\nExporting the Machine Learning Pipeline for Deployment...")
joblib.dump(scaler, 'cc_scaler.joblib')
joblib.dump(pca, 'cc_pca.joblib')
joblib.dump(final_kmeans, 'cc_kmeans.joblib')
print("Pipeline successfully saved to disk!\n")
                       
# How to load them back later:
# loaded_scaler = joblib.load('cc_scaler.joblib')
# loaded_pca = joblib.load('cc_pca.joblib')
# loaded_kmeans = joblib.load('cc_kmeans.joblib')

# Then you just run: loaded_kmeans.predict(loaded_pca.transform(loaded_scaler.transform(new_customer_data)))