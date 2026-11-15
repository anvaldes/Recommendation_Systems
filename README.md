# Recommendation Systems

End-to-end recommender system projects built on public datasets. Each project goes from **business understanding** and **exploratory data analysis (EDA)** to **modeling, evaluation and business recommendations**.

The goal of this repository is to cover the main **learning paradigms** (supervised, unsupervised and self-supervised) and different **business domains** and **types of feedback** used in real-world recommender systems.

---

## 📂 Projects

| # | Project | Business case | Learning paradigms | Feedback type | Dataset | Status |
|---|---|---|---|---|---|---|
| 1 | [Instacart Market Basket](./01_instacart_market_basket/) | 🛒 Retail / grocery | **Supervised** + **Unsupervised** | Implicit (purchases) | [Instacart Market Basket Analysis (Kaggle)](https://www.kaggle.com/c/instacart-market-basket-analysis) | ✅ Done |
| 2 | [MovieLens 1M](./02_movielens/) | 🎬 Media & entertainment (streaming) | **Supervised** + **Unsupervised** | Explicit (1–5 star ratings) | [MovieLens 1M (GroupLens)](https://grouplens.org/datasets/movielens/1m/) | ✅ Done |
| 3 | Amazon Reviews 2023 | 📦 E-commerce | **Self-supervised** + **Unsupervised** | Sequential (interactions over time) | [Amazon Reviews 2023 (Hugging Face)](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) | 🔜 Next |
| 4 | Criteo Display Advertising | 📢 Advertising | **Supervised** | Implicit (clicks) | [Criteo Display Advertising Challenge (Kaggle)](https://www.kaggle.com/c/criteo-display-ad-challenge) | ⏳ Optional |

---

## 🎯 What This Repository Demonstrates

**Learning paradigms**
- **Unsupervised learning:** association rules, item-item collaborative filtering, matrix factorization (SVD, ALS, PureSVD), user clustering into taste segments.
- **Supervised learning:** gradient boosting reorder models, rating prediction with stacking, learning-to-rank (LambdaRank).
- **Self-supervised learning** *(planned)*: sequential next-item recommendation with deep learning.

**Business domains**
- **Retail** (grocery delivery) and **media & entertainment** (movie streaming), with upcoming **e-commerce** and **advertising** use cases.

**Types of feedback**
- **Implicit** (purchases) and **explicit** (star ratings), with **sequential** interactions planned.

**Recommender system skills**
- Translating a business problem into a recommendation task and offline metrics.
- EDA focused on decisions for the recommender (repetition, popularity bias, rating biases, timing, basket size, taste structure).
- Two-stage recommenders: **unsupervised candidate generation + supervised ranking**.
- **Stacking / blending** of unsupervised models inside a supervised model, with leakage-free validation.
- Evaluation with **rating metrics** (RMSE, MAE, star-level confusion matrix) and **ranking metrics** (Precision@K, Recall@K, F1, NDCG, Hit Rate), **theoretical ceilings**, catalog coverage, novelty and **segment / fairness analysis**.
- Leakage-free temporal validation, ablation studies and model calibration.
- Turning results into **business recommendations**, customer segments and A/B testing proposals.

---

## 1️⃣ Instacart Market Basket — Grocery Recommendations

> **Demonstrates:** ✅ Supervised learning · ✅ Unsupervised learning · 🛒 Retail

**Business problem:** given a customer's purchase history, which products will they buy in their next order? This powers surfaces such as *"Buy it again"*, a pre-filled cart, *"Frequently bought together"* and reorder reminders.

**Data:** \~3.4M orders from \~206k customers across \~50k products.

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
- Grocery recommendation is mainly a **reorder problem**: \~60% of the next order comes from the customer's own history.
- **Deciding how many products to recommend** (+13.6%) mattered more than a better ranking model (+6%).
- **Discovery is an order of magnitude harder** than predicting reorders, and mixing it into "Buy it again" lowers accuracy for every customer segment.
- An **ablation study** showed that purchase-cycle features add only \~0.3% once frequency and recency are included: feature importance shows what a model *uses*, not what it *needs*.

📓 Notebook: [`instacart_market_basket_recsys.ipynb`](./01_instacart_market_basket/instacart_market_basket_recsys.ipynb)

---

## 2️⃣ MovieLens 1M — Movie Recommendations from Explicit Ratings

> **Demonstrates:** ✅ Supervised learning · ✅ Unsupervised learning · 🎬 Media & entertainment

**Business problem:** given a user's past ratings, which movies they have not seen yet will they watch and like the most? This powers surfaces such as *"Top picks for you"*, *"Because you watched…"*, *"Similar movies"* and themed rows by taste segment.

**Data:** \~1M ratings (1–5 stars) from 6,040 users on \~3,700 movies, with user demographics and movie genres. Evaluated with a **per-user temporal split** (last 20% of each user's ratings as test).

**Two tasks**
- **Rating prediction:** how many stars would the user give? → RMSE, MAE, star-level accuracy.
- **Top-10 recommendation:** which unseen movies should we show? → Precision@10, Recall@10, NDCG@10, Hit Rate, coverage.

**Approach**
| Stage | Models | Paradigm |
|---|---|---|
| Baselines | Global mean, popularity, Bayesian best-rated, regularized user + movie biases | Heuristic / statistical |
| Neighborhoods | Item-item kNN on residual ratings (rating prediction) and on co-watching (ranking) | **Unsupervised** |
| Latent factors | Truncated SVD, biased matrix factorization (ALS), PureSVD on implicit "watched" signal | **Unsupervised** |
| Segmentation | K-means user clustering into **8 interpretable taste segments** | **Unsupervised** |
| Rating model | LightGBM **stacking** on collaborative predictions + content and demographic features | **Supervised** |
| Recommender | **Two-stage:** hybrid candidate generation + LightGBM **LambdaRank** re-ranking | **Unsupervised + supervised** |

**Key results** (test set)
| Task | Model | Result | Improvement |
|---|---|---|---|
| Rating prediction | Bias model | RMSE 0.921 | −20% vs. global mean |
| Rating prediction | Item-kNN (best single model) | RMSE 0.875 | −5.0% vs. biases |
| Rating prediction | **LightGBM stacking** | **RMSE 0.857 · MAE 0.673** | **−6.9% vs. biases** |
| Rating prediction | LightGBM stacking (rounded to stars) | **46% exact star · 92% within ±1 star** | AUC 0.825 for "liked" (≥ 4) |
| Top-10 | Most rated movies | NDCG@10 0.090 | — |
| Top-10 | Item-kNN co-watching | NDCG@10 0.113 | +25% vs. popularity |
| Top-10 | **Two-stage recommender (LambdaRank)** | **NDCG@10 0.142 · Hit Rate 60% · coverage 33%** | **+26% vs. item-kNN** |

**Key insights**
- **Predicting ratings and recommending are different problems:** the bias model cuts RMSE by 20% but is the worst ranker. Offline top-K rewards *"will they watch it?"*, while RMSE rewards *"how much will they like it?"*.
- **The best recommender combines both questions:** unsupervised watch signals generate candidates and a supervised re-ranker adds taste signals → +26% NDCG, and **78%** of the recommended movies users watch are rated 4–5 stars (vs. \~70%).
- **Interaction data beats explicit attributes:** a supervised model with genres, year and demographics only is worse than item-kNN; demographics explain 0.05–0.37 stars of variation vs. 1.07 across individual users.
- **Unsupervised models discover real taste structure:** item similarities recover franchises and directors, latent factors separate mainstream from auteur cinema, and clustering finds named segments (*Classic cinema lovers*, *Sci-fi fans*, *Rom-com & family*, *90s action blockbusters*…).
- **Fairness gap:** light users, young users and women (28% of users) are consistently harder to predict.

📓 Notebook: [`movielens_1m_recsys.ipynb`](./02_movielens/movielens_1m_recsys.ipynb)

---

## 3️⃣ Amazon Reviews 2023 — Sequential Next-Item Recommendation *(next)*

> **Demonstrates:** ✅ Self-supervised learning · ✅ Unsupervised learning · 📦 E-commerce

Predict the next product a user will interact with, based on the **order of their past interactions**. Planned: item-similarity baseline (unsupervised) and sequential deep learning models such as GRU4Rec / SASRec (self-supervised). This complements MovieLens, where ratings were mostly made in bulk sessions and their order carried little information.

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
├── 02_movielens/
│   ├── README.md
│   └── movielens_1m_recsys.ipynb
├── 03_amazon_reviews_sequential/ # next
└── 04_criteo_ctr/                # optional
```

---

## ⚙️ Tech Stack

Python · pandas · NumPy · SciPy · scikit-learn · LightGBM · Matplotlib · seaborn · Google Colab

---

## ▶️ How to Run

1. Open the notebook in **Google Colab** (or a local Jupyter environment with ≥12 GB RAM).
2. Download the data into a `data/` folder:
   - **Instacart:** with the Kaggle API (instructions inside the notebook).
   - **MovieLens 1M:** downloaded automatically from GroupLens by the notebook (no account needed).
3. Run all cells from top to bottom. Each notebook reports its **total execution time** at the end.

> Datasets are **not included** in this repository; they must be downloaded from their original sources, subject to their respective licenses and terms of use.
