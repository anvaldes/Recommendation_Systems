# Recommendation Systems

End-to-end recommender system projects built on public datasets. Each project goes from **business understanding** and **exploratory data analysis (EDA)** to **modeling, evaluation and business recommendations**.

The goal of this repository is to cover the main **learning paradigms** (supervised, unsupervised and self-supervised) and different **business domains** and **types of feedback** used in real-world recommender systems.

---

## 📂 Projects

| # | Project | Business case | Learning paradigms | Feedback type | Dataset | Status |
|---|---|---|---|---|---|---|
| 1 | [Instacart Market Basket](./01_instacart_market_basket/) | 🛒 Retail / grocery | **Supervised** + **Unsupervised** | Implicit (purchases) | [Instacart Market Basket Analysis (Kaggle)](https://www.kaggle.com/c/instacart-market-basket-analysis) | ✅ Done |
| 2 | [MovieLens 1M](./02_movielens/) | 🎬 Media & entertainment (streaming) | **Supervised** + **Unsupervised** | Explicit (1–5 star ratings) | [MovieLens 1M (GroupLens)](https://grouplens.org/datasets/movielens/1m/) | ✅ Done |
| 3 | [Amazon Reviews 2023 — Video Games](./03_amazon_reviews_sequential/) | 📦 E-commerce | **Self-supervised** + **Unsupervised** | Sequential (interactions over time) | [Amazon Reviews 2023 (Hugging Face)](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) | ✅ Done |
| 4 | Criteo Display Advertising | 📢 Advertising | **Supervised** | Implicit (clicks) | [Criteo Display Advertising Challenge (Kaggle)](https://www.kaggle.com/c/criteo-display-ad-challenge) | 🔜 Next (optional) |

---

## 🎯 What This Repository Demonstrates

**Learning paradigms**
- **Unsupervised learning:** association rules, item-item collaborative filtering, matrix factorization (SVD, ALS, PureSVD), user clustering into taste segments, Markov transitions and sequential item-kNN.
- **Supervised learning:** gradient boosting reorder models, rating prediction with stacking, learning-to-rank (LambdaRank).
- **Self-supervised learning:** sequential next-item prediction with deep learning (GRU4Rec, SASRec Transformer) in PyTorch on GPU.

**Business domains**
- **Retail** (grocery delivery), **media & entertainment** (movie streaming) and **e-commerce** (video games marketplace), with an optional **advertising** use case.

**Types of feedback**
- **Implicit** (purchases), **explicit** (star ratings) and **sequential** (ordered interactions over time).

**Recommender system skills**
- Translating a business problem into a recommendation task and offline metrics.
- EDA focused on decisions for the recommender (repetition, popularity bias, rating biases, timing, basket size, taste structure).
- Two-stage recommenders: **unsupervised candidate generation + supervised ranking**.
- **Hybrid recommenders:** score fusion of unsupervised and deep self-supervised models, and context-aware routing (gap-aware switch).
- **Stacking / blending** of unsupervised models inside a supervised model, with leakage-free validation.
- Evaluation with **rating metrics** (RMSE, MAE, star-level confusion matrix) and **ranking metrics** (Precision@K, Recall@K, F1, NDCG, Hit Rate), **full-catalog ranking**, **theoretical ceilings**, catalog coverage, long-tail share and **segment / fairness analysis**.
- Leakage-free temporal validation, leave-one-out benchmarks (and their leakage caveats), fair refitting, ablation studies and model calibration.
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

## 3️⃣ Amazon Reviews 2023 (Video Games) — Sequential Next-Item Recommendation

> **Demonstrates:** ✅ Self-supervised learning · ✅ Unsupervised learning · 📦 E-commerce

**Business problem:** given the **ordered sequence** of products a customer has bought and reviewed, which product will they buy next? This powers surfaces such as *"Customers who bought this also bought"*, *"Recommended for you"* on the home page and post-purchase emails.

**Data:** official 5-core benchmark of the *Video Games* category: **814,586 interactions** from **94,762 users** on **25,612 products** (1999–2023, density 0.034%). Evaluated with the official **leave-one-out** split (last item = test, second-to-last = validation) and **full-catalog ranking** over all 25,612 products (no negative sampling).

**Approach**
| Stage | Models | Paradigm |
|---|---|---|
| Baselines | All-time popularity, **recent popularity** (last *W* months, leakage-free) | Heuristic |
| Item-to-item | Markov transitions, **sequential item-kNN** (symmetric co-occurrence window, popularity-aware similarity, recency decay over the last items) | **Unsupervised** |
| Deep sequential | **GRU4Rec** (recurrent) and **SASRec** (Transformer with causal self-attention), full-softmax next-item objective, PyTorch on GPU | **Self-supervised** |
| Hybrid | Z-score fusion of item-kNN + SASRec | Unsupervised + self-supervised |
| Routing | **Gap-aware switch:** hybrid for active customers, trending products for customers inactive > 180 days | Hybrid + business rule |

**Key results** (test set, 94,762 users)
| Model | HR@10 | NDCG@10 | Coverage@10 | Improvement (NDCG@10) |
|---|---|---|---|---|
| Popularity (all-time) | 2.5% | 0.0127 | 0.1% | — |
| Recent popularity (last month) | 6.0% | 0.0318 | 5.7% | 2.5× all-time popularity |
| Sequential item-kNN | 9.5% | 0.0570 | 91.5% | +79% vs. recent popularity |
| SASRec (refit on train + valid) | 10.6% | 0.0592 | 55.1% | +4% vs. item-kNN |
| Hybrid item-kNN + SASRec | 11.3% | 0.0651 | 69.8% | +10% vs. SASRec |
| **Gap-aware switch (final)** | **12.3%** | **0.0711** | — | **5.6× popularity · +25% vs. item-kNN** |

**Key insights**
- **Recency drives demand:** popularity from the *last month* beats all-time popularity by 2.4×, even though the all-time version sees the future. Console generations reshape the market (PlayStation's share fell from \~33% to \~20% while Nintendo's rose to \~35% after the Switch launch).
- **Order matters:** consecutive purchases share the same console 3.8× and the same publisher 4.5× more often than chance, but the long-term profile also matters, so the best models use the **whole history with strong recency weighting**.
- **Unsupervised vs. self-supervised are complementary:** item-kNN wins inside shopping sessions, SASRec wins after longer gaps. Their fusion reaches 83% of the oracle that always picks the right model, and raises coverage from 55% to 70%.
- **Time since the last purchase is the strongest driver:** NDCG@10 varies 10× between same-session targets (0.129) and customers returning after a year (0.013), where *no* personalized model beats trending products → a one-line routing rule adds +9%.
- **Protocol details change conclusions:** a fair refit of the neural models on train + valid reversed their ranking against item-kNN, and leave-one-out leaks future data (19.4% of training interactions happen after the median user's test target), which is documented as a limitation.

📓 Notebook: [`amazon_reviews_2023_video_games_recsys.ipynb`](./03_amazon_reviews_sequential/amazon_reviews_2023_video_games_recsys.ipynb)

---

## 4️⃣ Criteo Display Advertising — Click-Through Rate Prediction *(next, optional)*

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
├── 03_amazon_reviews_sequential/
│   ├── README.md
│   └── amazon_reviews_2023_video_games_recsys.ipynb
└── 04_criteo_ctr/                # next (optional)
```

---

## ⚙️ Tech Stack

Python · pandas · NumPy · SciPy · scikit-learn · LightGBM · PyTorch · Hugging Face Hub · Matplotlib · seaborn · Google Colab

---

## ▶️ How to Run

1. Open the notebook in **Google Colab** (or a local Jupyter environment with ≥12 GB RAM). The Amazon notebook trains neural models and should run on a **GPU runtime** (a Colab T4 is enough).
2. Download the data into a `data/` folder:
   - **Instacart:** with the Kaggle API (instructions inside the notebook).
   - **MovieLens 1M:** downloaded automatically from GroupLens by the notebook (no account needed).
   - **Amazon Reviews 2023:** downloaded automatically from the Hugging Face Hub by the notebook (no account needed).
3. Run all cells from top to bottom. Each notebook reports its **total execution time** at the end.

> Datasets are **not included** in this repository; they must be downloaded from their original sources, subject to their respective licenses and terms of use.
