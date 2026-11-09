# Feature Catalog

Generated from `src/features/feature_store.py::FEATURES`. All features are computed from the training split only, point-in-time at `as_of` (= last training timestamp).

Genres used for the `genre_*` columns: action, adventure, animation, children's, comedy, crime, documentary, drama, fantasy, film-noir, horror, musical, mystery, romance, sci-fi, thriller, unknown, war, western.

| Feature | Entity | Type | Description | Computation |
|---|---|---|---|---|
| `user_activity_count` | user | int | Number of rated items in the training history | count(interactions) per user |
| `user_avg_rating` | user | float | Mean rating given by the user | mean(rating) per user |
| `user_rating_std` | user | float | Spread of the user's ratings (0 if one rating) | std(rating) per user, NaN -> 0 |
| `user_positive_ratio` | user | float | Share of the user's ratings that are likes (rating >= threshold) | mean(rating >= 4) |
| `days_since_last_interaction` | user | float | Recency: days between the user's last interaction and `as_of` | (as_of - max(timestamp)) in days |
| `user_active_span_days` | user | int | Days between first and last interaction | max(ts) - min(ts) + 1 |
| `preferred_genre` | user | string | Genre with the highest share in the user's liked items | argmax(genre_pref_*) |
| `user_distinct_genres` | user | int | Number of genres the user has liked at least once | count(genre_pref_* > 0) |
| `genre_pref_<genre>  (19 cols)` | user | float | Share of the user's liked items that belong to <genre> (rows sum to 1) | L1-normalised P @ item_genre matrix |
| `item_popularity` | item | int | Number of training interactions with the item | count(interactions) per item |
| `item_popularity_pct` | item | float | Popularity percentile (1.0 = most popular) | rank(item_popularity) / n_items |
| `item_avg_rating` | item | float | Raw mean rating (global mean if unseen) | mean(rating) per item |
| `item_bayes_rating` | item | float | Shrunk mean rating, robust for items with few ratings | (sum + m*mu) / (n + m) |
| `item_positive_ratio` | item | float | Share of the item's ratings that are likes | mean(rating >= 4) |
| `item_age_years` | item | int | Age of the item at `as_of` | year(as_of) - release_year |
| `item_n_genres` | item | int | Number of genres attached to the item | sum(genre_*) |
| `genre_<genre>  (19 cols)` | item | int(0/1) | Multi-hot genre indicator | split(genres, '|') |
| `rating_minus_user_mean` | interaction | float | How much the rating deviates from the user's own average | rating - user_avg_rating |
| `rating_minus_item_mean` | interaction | float | How much the rating deviates from the item's average | rating - item_avg_rating |
| `genre_affinity` | interaction | float | Cosine similarity between user genre profile and item genres | cos(genre_pref_*, genre_*) |
| `interaction_age_days` | interaction | float | Age of the interaction at `as_of` | (as_of - timestamp) in days |
| `time_decay_weight` | interaction | float | Exponential recency weight for training | 0.5 ** (age_days / half_life) |

## Storage

| Table | Path | Key |
|---|---|---|
| User features | `data/features/user_features.parquet` | `user_id` |
| Item features | `data/features/item_features.parquet` | `item_id` |
| Interaction features | `data/features/interaction_features.parquet` | `(user_id, item_id)` |

## Reuse for training and prediction

`FeatureStore.load()` returns the stored tables; `compute_user_features()` can be called on a fresh interaction history to produce identical features at prediction time (see `src/inference/predict.py`).
