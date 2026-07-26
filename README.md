# 📊 E-commerce Customer Segmentation & Churn Predictor

A complete end-to-end machine learning project that applies **RFM Analysis**, **Customer Segmentation**, and **Churn Prediction** to an e-commerce transaction dataset — deployed as an interactive web app using Streamlit.

---

## 🔗 Live Demo

> [Launch App](https://ecomcust-segmentation-prediction-tky.streamlit.app/)

**Note:** the analysis notebook below was recently revised (leakage fix + simplification, see "Recent Changes"). If you retrain and redeploy, double-check `app.py`'s input construction still matches the model — see the warning in that section.

---

## 📌 Project Overview

This project analyses e-commerce transaction data to help a business understand its customer base and take targeted action — by segmenting customers based on purchasing behavior and predicting which ones are likely to churn.

### What This Project Does

- Cleans and preprocesses raw transactional data
- Computes **RFM metrics** (Recency, Frequency, Monetary value) per customer
- Applies and compares **three clustering techniques** — K-Means, DBSCAN, and Agglomerative Clustering — using identical, consistently-scaled input
- Trains a **Random Forest Classifier** to predict customer churn, using only features that are independent of how the churn label itself is defined
- Deploys the model via a **Streamlit web app** with single and batch prediction support

---

## 🆕 Recent Changes (Notebook Rewrite)

The analysis notebook was rewritten to fix a data leakage issue and simplify the pipeline:

- **Removed quartile-based RFM scoring and business-label segmentation** (e.g. "Loyal Customers", "Champions" naming, treemap/bar chart visuals). That step was built purely for stakeholder presentation and wasn't needed for the technical pipeline — removing it also removes the root cause of the leakage below.
- **Fixed data leakage in the churn model.** The previous version excluded raw `Recency` from the model's features (correctly reasoning that using the same field the label is derived from is leakage) but still included a composite `RFM_Score` — which is `R + F + M`, where `R` is just Recency binned into quartiles. That composite score re-introduced much of the same leaked signal through the back door. The churn model is now trained only on `Frequency` and `Value`, both independent of the churn label's definition.
- **Fixed inconsistent scaling across clustering methods.** K-Means previously ran on `MinMaxScaler`-scaled data while DBSCAN/Agglomerative ran on `StandardScaler`-scaled data — different geometries, making the three-method comparison less meaningful. All three now use the same `StandardScaler` output.
- **DBSCAN's `eps` is chosen from an actual k-distance plot** rather than a hardcoded value.
- **Removed dead code** (an unused `k_means` import, an unused `segment_labels` list).
- **Metrics are no longer restated as fixed numbers in markdown.** The previous notebook had two different accuracy figures written in two different narrative sections (91% vs. 92%) that had drifted out of sync with the model's actual output. This version only ever prints metrics live from code, so the notebook can't silently go stale again.
- **`app.py` updated to match**: input form now asks only for Frequency and Value (Recency input removed, since it's not a model feature), the RFM segment/business-label UI is gone, and batch CSV upload now expects `Frequency`/`Value` columns instead of `Frequency`/`Value`/`RFM_Score`.

**Expect the retrained model's accuracy to be noticeably lower than the previously reported ~92%** — that's the leakage being corrected, not a regression. Whatever number the notebook prints when you run it is the honest, current one.

---

## 📂 Repository Structure

```
├── app.py                                          # Streamlit web application
├── churn_prediction_model.pkl                      # Trained Random Forest model
├── Ecom_CustomerSegmentation_Prediction.ipynb      # Full analysis & model training notebook
├── requirements.txt                                # Dependencies for running app.py
├── requirements-notebook.txt                       # Additional dependencies to reproduce the notebook
└── README.md
```

---

## 📊 Dataset

| Column | Description |
|---|---|
| `InvoiceNo` | Unique transaction identifier |
| `StockCode` | Product code |
| `Description` | Product name |
| `Quantity` | Number of items purchased |
| `InvoiceDate` | Date and time of the transaction |
| `UnitPrice` | Price per unit |
| `CustomerID` | Unique customer identifier |
| `Country` | Country where the transaction occurred |

---

## 🔬 Methodology

### 1. Data Preprocessing
- Dropped rows with missing `CustomerID` (can't be attributed to any customer for RFM analysis)
- Parsed `InvoiceDate` to datetime
- Engineered `TotalRevenue = Quantity × UnitPrice`

### 2. RFM Analysis
Per-customer raw values only — no quartile scoring or labeling:

| Metric | Definition |
|---|---|
| **Recency** | Days since the customer's last purchase |
| **Frequency** | Number of distinct invoices |
| **Monetary (Value)** | Total spend |

### 3. Clustering Analysis (all three methods on the same `StandardScaler`-scaled RFM data)

- **K-Means (primary):** `k` selected via the elbow method and silhouette scores, evaluated together rather than by silhouette score alone (the highest silhouette score can occur at an uninteresting `k`, e.g. one that just separates a single outlier from everyone else).
- **DBSCAN (attempted):** `eps` chosen from a k-distance plot on the actual data. If your dataset's k-distance curve is flat with no clear elbow, that's informative — it suggests the data is too densely/uniformly packed for density-based clustering to find meaningful separated regions (this was the case on the original ~4,300-customer dataset).
- **Agglomerative Clustering:** applied as a hierarchical alternative; a dendrogram visualizes the merge structure and supports (or challenges) the chosen number of clusters.

### 4. Churn Prediction
- **Churn definition:** a customer is churned if their last purchase was more than **60 days** ago
- **Model:** Random Forest Classifier (100 estimators, balanced class weights)
- **Features:** `Frequency`, `Value` only — see "Recent Changes" above for why `Recency` and any Recency-derived composite score are excluded
- **Train/Test Split:** 80/20, stratified on the churn label

---

## 🖥️ Streamlit App Features

### Single Prediction
Enter a customer's Frequency and Value — the app returns:
- Churn / No-Churn prediction
- Churn probability percentage
- Risk badge (High / Medium / Low)
- Recommended action, tied to the risk tier

### Batch Prediction
Upload a CSV with columns `Frequency`, `Value` to predict churn for multiple customers at once, then download the results.

> **`app.py` has been updated to match the new 2-feature model** (`Frequency`, `Value` — no more `RFM_Score`). The old RFM segment/business-label UI (Loyal Customer, Potential Loyalist, etc. and the "Segment Guide" sidebar table) has been removed, since it depended on the leaky `RFM_Score` and the labeling step this project no longer uses. Recommendations are now tied to the churn-risk tier (derived directly from the model's own probability output) instead. I verified this version actually runs — launched it with Streamlit, confirmed the page loads and the health check passes, then used Streamlit's `AppTest` framework to simulate clicking "Predict" and confirmed no exceptions, plus tested the batch-CSV logic directly against a real (test) model.

---

## 📦 Requirements

**To run the deployed app:**
```
pip install -r requirements.txt
```

**To reproduce the analysis notebook** (adds plotting/scientific libraries not needed by the app):
```
pip install -r requirements-notebook.txt
```

---

## 🚀 Deployment

This app is deployed on **Streamlit Community Cloud**. To deploy your own:

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo, set `app.py` as the entry point
4. Click **Deploy**

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas)
