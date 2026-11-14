# Recommendation Systems

End-to-end recommender system projects built on public datasets. Each project goes from **business understanding** and **exploratory data analysis (EDA)** to **modeling, evaluation and business recommendations**.

The goal of this repository is to cover the main **learning paradigms** (supervised, unsupervised and self-supervised) and different **business domains** and **types of feedback** used in real-world recommender systems.

---

## 📂 Projects

| # | Project | Business case | Learning paradigms | Feedback type | Dataset | Status |
|---|---|---|---|---|---|---|
| 1 | [Instacart Market Basket](./01_instacart_market_basket/) | 🛒 Retail / grocery | **Supervised** + **Unsupervised** | Implicit (purchases) | [Instacart Market Basket Analysis (Kaggle)](https://www.kaggle.com/c/instacart-market-basket-analysis) | ✅ Done |
| 2 | MovieLens 1M | 🎬 Streaming | **Supervised** + **Unsupervised** | Explicit (ratings) | [MovieLens 1M (GroupLens)](https://grouplens.org/datasets/movielens/1m/) | 🔜 Next |
| 3 | Amazon Reviews 2023 | 📦 E-commerce | **Self-supervised** + **Unsupervised** | Sequential (interactions over time) | [Amazon Reviews 2023 (Hugging Face)](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) | 🗓️ Planned |
| 4 | Criteo Display Advertising | 📢 Advertising | **Supervised** | Implicit (clicks) | [Criteo Display Advertising Challenge (Kaggle)](https://www.kaggle.com/c/criteo-display-ad-challenge) | ⏳ Optional |

---

## 🎯 What This Repository Demonstrates

**Learning paradigms**
- **Unsupervised learning:** association rules, collaborative filtering, matrix factorization, clustering.
- **Supervised learning:** gradient boosting ranking models, rating prediction, click prediction.
- **Self-supervised learning** *(planned)*: sequential next-item recommendation with deep learning.

**Business domains**
- **Retail** (grocery delivery), and upcoming **streaming**, **e-commerce** and **advertising** use cases.

**Recommender system skills**
- Translating a business problem into a recommendation task and offline metrics.
- EDA focused on decisions for the recommender (repetition, popularity bias, timing, basket size).
- Two-stage recommenders: **candidate generation + ranking**.
- Evaluation with ranking metrics (Precision@K, Recall@K, F1, NDCG, Hit Rate), **theoretical ceilings**, catalog coverage and **segment analysis**.
- Leakage-free validation, ablation studies and model calibration.
- Turning results into **business recommendations** and A/B testing proposals.

---

## 1️⃣ Instacart Market Basket — Grocery Recommendations

> **Demonstrates:** ✅ Supervised learning · ✅ Unsupervised learning · 🛒 Retail

**Business problem:** given a customer's purchase history, which products will they buy in their next order? This powers surfaces such as *"Buy it again"*, a pre-filled cart, *"Frequently bought together"* and reorder reminders.

**Data:** ~3.4M orders from ~206k customers across ~50k products.

**Approach**
| Stage | Models | Paradigm |
|---|---|---|
| Baselines | Popularity, frequency and recency-based "Buy it again" | Heuristic |
| Discovery | Item-item collaborative filtering, SVD, association rules (support, confidence, lift) | **Unsupervised** |
| Ranking | LightGBM reorder model (30 engineered features: frequency, recency, purchase cycles, product and customer profiles) | **Supervised** |
| List length | Personalized K via expected-F1 maximization on calibrated probabilities | Supervised + decision rule |

**Key results** (30k evaluation customers)
| Model | F1 | Improvement |
|---|---|---|
| Popularity baseline (K = 10) | 0.063 | — |
| Buy-again: frequency + recency (K = 10) | 0.270 | 4.3× popularity |
| LightGBM ranking (K = 10) | 0.286 | +6% vs. heuristic |
| **LightGBM + personalized K** | **0.325** | **+13.6% vs. fixed K** |
| Kaggle metric, local estimate (incl. `None`) | **0.3975** | ≈ top 20% of the leaderboard (winner: 0.4091) |

**Key insights**
- Grocery recommendation is mainly a **reorder problem**: ~60% of the next order comes from the customer's own history.
- **Deciding how many products to recommend** (+13.6%) mattered more than a better ranking model (+6%).
- **Discovery is an order of magnitude harder** than predicting reorders, and mixing it into "Buy it again" lowers accuracy for every customer segment.
- An **ablation study** showed that purchase-cycle features add only ~0.3% once frequency and recency are included: feature importance shows what a model *uses*, not what it *needs*.

📓 Notebook: [`instacart_market_basket_recsys.ipynb`](./01_instacart_market_basket/instacart_market_basket_recsys.ipynb)

---

## 2️⃣ MovieLens 1M — Movie Rating Prediction *(coming soon)*

> **Demonstrates:** ✅ Supervised learning · ✅ Unsupervised learning · 🎬 Streaming

Predict how a user would rate movies they have not seen yet, using **explicit feedback** (1–5 star ratings). Planned: matrix factorization and clustering of users and movies by taste (unsupervised), and rating prediction with user, movie and genre features (supervised).

---

## 3️⃣ Amazon Reviews 2023 — Sequential Next-Item Recommendation *(planned)*

> **Demonstrates:** ✅ Self-supervised learning · ✅ Unsupervised learning · 📦 E-commerce

Predict the next product a user will interact with, based on the **order of their past interactions**. Planned: item-similarity baseline (unsupervised) and sequential deep learning models such as GRU4Rec / SASRec (self-supervised).

---

## 4️⃣ Criteo Display Advertising — Click-Through Rate Prediction *(optional)*

> **Demonstrates:** ✅ Supervised learning · 📢 Advertising

Predict whether a user will click on an ad, using large-scale data with high-cardinality categorical features. Planned: logistic regression, factorization machines and DeepFM.

---

## 🗂️ Repository Structure

```
Recommendation_Systems/
├── README.md
├── 01_instacart_market_basket/
│   ├── README.md
│   └── instacart_market_basket_recsys.ipynb
├── 02_movielens/                 # coming soon
├── 03_amazon_reviews_sequential/ # planned
└── 04_criteo_ctr/                # optional
```

---

## ⚙️ Tech Stack

Python · pandas · NumPy · SciPy · scikit-learn · LightGBM · Matplotlib · seaborn · Google Colab

---

## ▶️ How to Run

1. Open the notebook in **Google Colab** (or a local Jupyter environment with ≥12 GB RAM).
2. Download the dataset with the Kaggle API (instructions inside each notebook) into a `data/` folder.
3. Run all cells from top to bottom. Each notebook reports its **total execution time** at the end.

> Datasets are **not included** in this repository; they must be downloaded from their original sources, subject to their respective licenses and terms of use.
