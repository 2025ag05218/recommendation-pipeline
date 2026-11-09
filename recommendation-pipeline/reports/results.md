# Model Evaluation Results (K = 10)

Evaluated on 904 users with at least one relevant (rating >= 4) item in the held-out, most-recent part of their history. Best value per column in **bold** (RMSE / MAE: lower is better, all others: higher is better).

| Model | precision_at_10 | recall_at_10 | ndcg_at_10 | coverage | diversity | rmse | mae |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0729 | 0.0688 | 0.0917 | 0.0416 | 0.6946 | 1.0808 | 0.8683 |
| item_cf | **0.1054** | 0.1174 | **0.1448** | 0.0957 | **0.7155** | **1.0571** | **0.8379** |
| content_based | 0.0244 | 0.0257 | 0.0339 | **0.2711** | 0.1301 | n/a | n/a |
| hybrid | 0.1051 | **0.1181** | 0.1440 | 0.1136 | 0.5565 | **1.0571** | **0.8379** |
