# Credit Card Customer Segmentation 

## Project Overview
The objective of this project is to develop a customer segmentation model for a bank's credit card portfolio. By analyzing the spending habits, balance behaviors, and payment histories of thousands of active credit card users, this model automatically categorizes customers into distinct, actionable marketing personas.

## The Problem
Banks sit on massive amounts of transactional data, but raw numbers don't tell a story. Treating all customers the same leads to wasted marketing spend and missed revenue opportunities. The goal is to use unsupervised machine learning to find hidden groupings in the data so the bank can deploy targeted financial products to the right people.

## Methodology & Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Data Preprocessing:** * Addressed heavy skewness in financial data using Log Transformations (properly isolated to prevent data bleed).
  * Identified and tackled multicollinearity (removing redundant "child" features while preserving "parent" features).
  * Standardized all features using `StandardScaler` to ensure equal weighting.
* **Dimensionality Reduction (PCA):** * Applied Principal Component Analysis (PCA) to compress 17 highly correlated features down to 5 Principal Components.
  * Analyzed the Cumulative Explained Variance and Scree Plot to retain ~70% of the dataset's core signal while eliminating noise.
* **Clustering (K-Means):** * Utilized the Elbow Method and Silhouette Scoring to mathematically determine the optimal number of clusters (`k=3`).
  * Deployed `k-means++` initialization for stable, distinct customer groupings.

## Key Business Insights: The 3 Personas
Through reverse-engineering the PCA Loadings and aggregating the final cluster data, the model successfully identified three distinct customer profiles:

### Persona 0: "The Premium Swipers"
* **Profile:** The VIPs. They have the highest purchase volume (~$2,360) and make over 33 transactions a month. Because of their high engagement, they hold the highest average credit limits (~$5,895).
* **Business Strategy:** Highly profitable via merchant swipe fees. Actionable strategy: Upgrade to premium travel/rewards tiers to ensure brand loyalty and prevent churn to competitors.

### Persona 1: "The Cash-Dependent Revolvers"
* **Profile:** They carry the highest average balances (~$2,342) and have astronomical cash advance usage (~$2,153). They rarely use the card for retail purchases (~$108) and pay off their full statement only ~3% of the time.
* **Business Strategy:** High interest revenue, but high default risk. Actionable strategy: Target with debt-consolidation loans and restrict automatic credit limit increases.

### Persona 2: "The Frugal Casuals"
* **Profile:** They maintain the lowest balances by far (~$249), barely touch the cash advance feature (~$27), and make minimal purchases (~$476 with ~8 transactions a month). 
* **Business Strategy:** Safe, but low-profit. Actionable strategy: Deploy targeted cash-back incentives for everyday spending categories (gas, groceries) to build swiping habits.

## How to Run the Project
1. Clone the repository.
2. Ensure you have `scikit-learn`, `pandas`, `matplotlib`, and `seaborn` installed (or run `pip install -r requirements.txt`).
3. Run the Python script (or Jupyter Notebook) step-by-step to view the visualizations (Heatmaps, Scree Plots, Elbow Curves) and the final cluster profiles.