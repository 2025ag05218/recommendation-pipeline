# Data Validation Report

Automated checks run by `src/validation/validate.py` on the raw CSVs.

## 1. Summary

| Table | Rows in | Valid rows | Rejected rows | Rejected % |
|---|---:|---:|---:|---:|
| users | 953 | 943 | 10 | 1.05% |
| items | 1,687 | 1,682 | 5 | 0.30% |
| interactions | 100,150 | 99,540 | 610 | 0.61% |

## 2. Checks executed

| Table | Check | Severity | Rows flagged |
|---|---|---|---:|
| users | `duplicate_key:user_id` | REJECT (moved to error dataset) | 10 |
| users | `missing_value:user_id` | REJECT (moved to error dataset) | 0 |
| users | `wrong_dtype:user_id` | REJECT (moved to error dataset) | 0 |
| users | `invalid_or_missing_age` | WARN (repaired in preparation) | 8 |
| users | `invalid_or_missing_gender` | WARN (repaired in preparation) | 5 |
| users | `missing_occupation` | WARN (repaired in preparation) | 6 |
| items | `duplicate_key:item_id` | REJECT (moved to error dataset) | 5 |
| items | `missing_value:item_id` | REJECT (moved to error dataset) | 0 |
| items | `wrong_dtype:item_id` | REJECT (moved to error dataset) | 0 |
| items | `invalid_or_missing_release_date` | WARN (repaired in preparation) | 1 |
| items | `missing_genres` | WARN (repaired in preparation) | 5 |
| items | `missing_title` | WARN (repaired in preparation) | 8 |
| interactions | `duplicate_row` | REJECT (moved to error dataset) | 150 |
| interactions | `invalid_rating` | REJECT (moved to error dataset) | 100 |
| interactions | `invalid_timestamp` | REJECT (moved to error dataset) | 64 |
| interactions | `missing_value:item_id` | REJECT (moved to error dataset) | 0 |
| interactions | `missing_value:rating` | REJECT (moved to error dataset) | 60 |
| interactions | `missing_value:timestamp` | REJECT (moved to error dataset) | 0 |
| interactions | `missing_value:user_id` | REJECT (moved to error dataset) | 30 |
| interactions | `orphan_item_id` | REJECT (moved to error dataset) | 70 |
| interactions | `orphan_user_id` | REJECT (moved to error dataset) | 70 |
| interactions | `wrong_dtype:item_id` | REJECT (moved to error dataset) | 0 |
| interactions | `wrong_dtype:rating` | REJECT (moved to error dataset) | 50 |
| interactions | `wrong_dtype:timestamp` | REJECT (moved to error dataset) | 16 |
| interactions | `wrong_dtype:user_id` | REJECT (moved to error dataset) | 0 |

## 3. Injected faults vs. detected

Faults were injected into the raw CSVs by the ingestion stage (seeded, see `configs/config.yaml`). A good validator must find all of them.

| Table | Fault | Injected | Detected | Result |
|---|---|---:|---:|---|
| interactions | invalid_rating_value | 100 | 100 | OK |
| interactions | non_numeric_rating | 50 | 50 | OK |
| interactions | missing_rating | 60 | 60 | OK |
| interactions | missing_user_id | 30 | 30 | OK |
| interactions | invalid_timestamp | 80 | 80 | OK |
| interactions | orphan_user_id | 70 | 70 | OK |
| interactions | orphan_item_id | 70 | 70 | OK |
| interactions | duplicate_rows | 150 | 150 | OK |
| users | duplicate_rows | 10 | 10 | OK |
| users | invalid_age | 8 | 8 | OK |
| users | invalid_gender | 5 | 5 | OK |
| users | missing_occupation | 6 | 6 | OK |
| items | duplicate_rows | 5 | 5 | OK |
| items | missing_title | 8 | 8 | OK |
| items | missing_genres | 5 | 5 | OK |

_Detected can exceed injected when the original MovieLens data already contained a natural instance of the same issue (e.g. one item with no release date)._

## 4. Where to find rejected records

* `data/errors/interactions_errors.parquet`
* `data/errors/users_errors.parquet`
* `data/errors/items_errors.parquet`
* `reports/validation_errors_sample.csv` (first 40 rejected rows per table, readable sample)

Each file holds the original raw row, `source_row` (row position in the raw CSV) and `error_reason` (one or more `;`-separated check codes).
