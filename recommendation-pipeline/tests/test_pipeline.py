"""Unit tests: metrics, validation checks, split logic and feature leakage. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.common import load_config
from src.evaluation import metrics as M
from src.evaluation.evaluate import top_k_unseen
from src.preparation.prepare import time_based_split
from src.validation import validate


# ---------------------------------------------------------------- metrics
def test_precision_recall_ndcg_perfect_and_empty():
    recs, rel = [1, 2, 3, 4, 5], {1, 2}
    assert M.precision_at_k(recs, rel, 5) == pytest.approx(0.4)
    assert M.recall_at_k(recs, rel, 5) == 1.0
    assert M.ndcg_at_k(recs, rel, 5) == pytest.approx(1.0)          # hits ranked first = ideal
    assert M.ndcg_at_k([9, 8, 7], rel, 3) == 0.0
    assert M.recall_at_k(recs, set(), 5) == 0.0


def test_ndcg_penalises_late_hit():
    early = M.ndcg_at_k([1, 9, 8], {1}, 3)
    late = M.ndcg_at_k([9, 8, 1], {1}, 3)
    assert early > late > 0


def test_coverage_and_diversity():
    assert M.catalog_coverage([[1, 2], [2, 3]], 10) == pytest.approx(0.3)
    vec = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
    assert M.intra_list_diversity([0, 1], vec) == pytest.approx(0.0)    # identical genres
    assert M.intra_list_diversity([0, 2], vec) == pytest.approx(1.0)    # orthogonal genres


def test_rmse_mae():
    y, p = np.array([4.0, 2.0]), np.array([3.0, 2.0])
    assert M.rmse(y, p) == pytest.approx(np.sqrt(0.5))
    assert M.mae(y, p) == pytest.approx(0.5)


def test_top_k_excludes_seen():
    scores = np.array([[0.9, 0.8, 0.7, 0.1]])
    seen = np.array([[True, False, False, False]])
    assert top_k_unseen(scores, seen, 2).tolist() == [[1, 2]]


# ---------------------------------------------------------------- validation
def test_interaction_validation_flags_each_defect():
    cfg = load_config()["validation"]
    df = pd.DataFrame({
        "user_id":   ["1", "1", "2", "3", "4", "99", "1", "1"],
        "item_id":   ["10", "10", "10", "10", "10", "10", "55", "10"],
        "rating":    ["5", "5", "9", "abc", None, "3", "3", "4"],
        "timestamp": ["900000000"] * 5 + ["900000000", "900000000", "0"],
    })
    chk = validate.validate_interactions(df, cfg, valid_users={1, 2, 3, 4}, valid_items={10})
    c = chk.reject_counts
    assert c["duplicate_row"] == 1
    assert c["invalid_rating"] == 1
    assert c["wrong_dtype:rating"] == 1
    assert c["missing_value:rating"] == 1
    assert c["orphan_user_id"] == 1
    assert c["orphan_item_id"] == 1
    assert c["invalid_timestamp"] == 1
    ok, err = chk.split()
    assert len(ok) == 1 and len(err) == 7 and "error_reason" in err.columns


# ---------------------------------------------------------------- split
def _toy():
    rows = [(u, i, 4.0, pd.Timestamp("2000-01-01", tz="UTC") + pd.Timedelta(days=i)) for u in (1, 2) for i in range(10)]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def test_per_user_split_is_chronological():
    train, test = time_based_split(_toy(), {"strategy": "per_user", "test_fraction": 0.2})
    assert len(test) == 4 and len(train) == 16
    for u in (1, 2):
        assert train[train.user_id == u].timestamp.max() < test[test.user_id == u].timestamp.min()


def test_global_split_has_no_overlap_in_time():
    train, test = time_based_split(_toy(), {"strategy": "global", "global_cutoff_quantile": 0.8})
    assert train.timestamp.max() <= test.timestamp.min()
    assert len(train) + len(test) == 20
