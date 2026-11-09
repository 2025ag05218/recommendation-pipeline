"""Stage 2 - Data validation.

Runs automated quality checks on the raw CSVs written by the ingestion stage.

Severity levels
  * REJECT  - the record is unusable (broken key, wrong type, invalid rating/timestamp,
              duplicate, orphan foreign key). It is removed from the pipeline and written to
              data/errors/<table>_errors.parquet together with the reason(s).
  * WARN    - the record is usable but a non-key attribute is bad (e.g. age = 250,
              missing title). It is kept, counted in the report, and repaired in the
              preparation stage.

Outputs: data/processed/validated/*.parquet, data/errors/*_errors.parquet,
         reports/validation_report.md and reports/validation_report.json
"""
from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from src.common import get_logger, get_path, logged_stage, read_json, write_json

log = get_logger("validation")


class ValidationError(RuntimeError):
    pass


class Checker:
    """Collects reject reasons per row and counts of every check."""

    def __init__(self, df: pd.DataFrame, table: str):
        self.df, self.table = df, table
        self.reasons: dict[int, list[str]] = defaultdict(list)
        self.reject_counts: Counter = Counter()
        self.warn_counts: Counter = Counter()

    def reject(self, mask: pd.Series, code: str) -> None:
        mask = mask.fillna(False)
        for idx in self.df.index[mask]:
            self.reasons[idx].append(code)
        self.reject_counts[code] += int(mask.sum())

    def warn(self, mask: pd.Series, code: str) -> None:
        self.warn_counts[code] += int(mask.fillna(False).sum())

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        bad_idx = list(self.reasons.keys())
        errors = self.df.loc[bad_idx].copy()
        errors["error_reason"] = [";".join(self.reasons[i]) for i in bad_idx]
        errors["source_row"] = bad_idx
        return self.df.drop(index=bad_idx).copy(), errors.sort_values("source_row")


def _is_int_like(num: pd.Series) -> pd.Series:
    return num.notna() & (num == num.round())


def _check_int_key(chk: Checker, col: str) -> pd.Series:
    """Missing / non-integer check for an id column. Returns the numeric (coerced) series."""
    raw = chk.df[col]
    num = pd.to_numeric(raw, errors="coerce")
    chk.reject(raw.isna(), f"missing_value:{col}")
    chk.reject(raw.notna() & ~_is_int_like(num), f"wrong_dtype:{col}")
    return num


# --------------------------------------------------------------------------- per-table validation
def validate_users(df: pd.DataFrame, v: dict) -> Checker:
    chk = Checker(df, "users")
    uid = _check_int_key(chk, "user_id")
    chk.reject(uid.notna() & uid.duplicated(keep="first"), "duplicate_key:user_id")
    age = pd.to_numeric(df["age"], errors="coerce")
    chk.warn(df["age"].isna() | age.isna() | (age < v["age_min"]) | (age > v["age_max"]), "invalid_or_missing_age")
    chk.warn(~df["gender"].isin(v["allowed_genders"]), "invalid_or_missing_gender")
    chk.warn(df["occupation"].isna(), "missing_occupation")
    return chk


def validate_items(df: pd.DataFrame, v: dict) -> Checker:
    chk = Checker(df, "items")
    iid = _check_int_key(chk, "item_id")
    chk.reject(iid.notna() & iid.duplicated(keep="first"), "duplicate_key:item_id")
    chk.warn(df["title"].isna(), "missing_title")
    chk.warn(df["genres"].isna(), "missing_genres")
    year = pd.to_datetime(df["release_date"], format="%d-%b-%Y", errors="coerce").dt.year
    chk.warn(year.isna() | (year < v["release_year_min"]) | (year > v["release_year_max"]), "invalid_or_missing_release_date")
    return chk


def validate_interactions(df: pd.DataFrame, v: dict, valid_users: set, valid_items: set) -> Checker:
    chk = Checker(df, "interactions")
    uid = _check_int_key(chk, "user_id")
    iid = _check_int_key(chk, "item_id")

    rating = pd.to_numeric(df["rating"], errors="coerce")
    chk.reject(df["rating"].isna(), "missing_value:rating")
    chk.reject(df["rating"].notna() & rating.isna(), "wrong_dtype:rating")
    chk.reject(rating.notna() & ((rating < v["rating_min"]) | (rating > v["rating_max"])), "invalid_rating")

    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    chk.reject(df["timestamp"].isna(), "missing_value:timestamp")
    chk.reject(df["timestamp"].notna() & ~_is_int_like(ts), "wrong_dtype:timestamp")
    lo = pd.Timestamp(v["timestamp_min"], tz="UTC").timestamp()
    hi = pd.Timestamp(v["timestamp_max"], tz="UTC").timestamp()
    chk.reject(_is_int_like(ts) & ((ts < lo) | (ts > hi)), "invalid_timestamp")

    chk.reject(df.duplicated(keep="first"), "duplicate_row")

    # referential integrity against the *validated* parent tables
    chk.reject(_is_int_like(uid) & ~uid.isin(valid_users), "orphan_user_id")
    chk.reject(_is_int_like(iid) & ~iid.isin(valid_items), "orphan_item_id")
    return chk


# --------------------------------------------------------------------------- reporting
# how each injected fault maps to detection codes (used for the "injected vs detected" table)
FAULT_TO_CODES = {
    ("interactions", "invalid_rating_value"): ["invalid_rating"],
    ("interactions", "non_numeric_rating"): ["wrong_dtype:rating"],
    ("interactions", "missing_rating"): ["missing_value:rating"],
    ("interactions", "missing_user_id"): ["missing_value:user_id"],
    ("interactions", "invalid_timestamp"): ["invalid_timestamp", "wrong_dtype:timestamp"],
    ("interactions", "orphan_user_id"): ["orphan_user_id"],
    ("interactions", "orphan_item_id"): ["orphan_item_id"],
    ("interactions", "duplicate_rows"): ["duplicate_row"],
    ("users", "duplicate_rows"): ["duplicate_key:user_id"],
    ("users", "invalid_age"): ["invalid_or_missing_age"],
    ("users", "invalid_gender"): ["invalid_or_missing_gender"],
    ("users", "missing_occupation"): ["missing_occupation"],
    ("items", "duplicate_rows"): ["duplicate_key:item_id"],
    ("items", "missing_title"): ["missing_title"],
    ("items", "missing_genres"): ["missing_genres"],
}


def _write_report(cfg: dict, stats: dict, checkers: dict[str, Checker], manifest) -> None:
    lines = ["# Data Validation Report", "",
             "Automated checks run by `src/validation/validate.py` on the raw CSVs.", "",
             "## 1. Summary", "",
             "| Table | Rows in | Valid rows | Rejected rows | Rejected % |", "|---|---:|---:|---:|---:|"]
    for t, s in stats.items():
        lines.append(f"| {t} | {s['rows_in']:,} | {s['rows_valid']:,} | {s['rows_rejected']:,} | {s['rejected_pct']:.2f}% |")

    lines += ["", "## 2. Checks executed", "",
              "| Table | Check | Severity | Rows flagged |", "|---|---|---|---:|"]
    for t, chk in checkers.items():
        for code, n in sorted(chk.reject_counts.items()):
            lines.append(f"| {t} | `{code}` | REJECT (moved to error dataset) | {n:,} |")
        for code, n in sorted(chk.warn_counts.items()):
            lines.append(f"| {t} | `{code}` | WARN (repaired in preparation) | {n:,} |")

    if isinstance(manifest, dict):
        lines += ["", "## 3. Injected faults vs. detected", "",
                  "Faults were injected into the raw CSVs by the ingestion stage (seeded, see `configs/config.yaml`). "
                  "A good validator must find all of them.", "",
                  "| Table | Fault | Injected | Detected | Result |", "|---|---|---:|---:|---|"]
        for (t, fault), codes in FAULT_TO_CODES.items():
            injected = manifest.get(t, {}).get(fault)
            if injected is None:
                continue
            counts = {**checkers[t].reject_counts, **checkers[t].warn_counts}
            detected = sum(counts.get(c, 0) for c in codes)
            ok = "OK" if detected >= injected else "MISSED"
            lines.append(f"| {t} | {fault} | {injected} | {detected} | {ok} |")
        lines += ["", "_Detected can exceed injected when the original MovieLens data already contained a "
                      "natural instance of the same issue (e.g. one item with no release date)._"]

    lines += ["", "## 4. Where to find rejected records", "",
              "* `data/errors/interactions_errors.parquet`", "* `data/errors/users_errors.parquet`",
              "* `data/errors/items_errors.parquet`", "* `reports/validation_errors_sample.csv` (first 40 rejected rows per table, readable sample)", "",
              "Each file holds the original raw row, `source_row` (row position in the raw CSV) and `error_reason` "
              "(one or more `;`-separated check codes)."]
    get_path(cfg, "reports_dir", "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- stage entrypoint
@logged_stage("validation")
def run(cfg: dict) -> dict:
    v = cfg["validation"]
    raw = get_path(cfg, "raw_dir", "csv", "x.csv").parent
    # Only a truly empty cell counts as missing; text such as "N/A" must reach the type checks.
    read = dict(dtype=str, keep_default_na=False, na_values=[""])
    users = pd.read_csv(raw / "users.csv", **read)
    items = pd.read_csv(raw / "items.csv", **read)
    inter = pd.read_csv(raw / "interactions.csv", **read)

    cu = validate_users(users, v)
    ci = validate_items(items, v)
    users_ok, users_err = cu.split()
    items_ok, items_err = ci.split()
    ct = validate_interactions(
        inter, v,
        valid_users=set(pd.to_numeric(users_ok["user_id"]).astype(int)),
        valid_items=set(pd.to_numeric(items_ok["item_id"]).astype(int)),
    )
    inter_ok, inter_err = ct.split()

    out = get_path(cfg, "processed_dir", "validated", "x.parquet").parent
    err_dir = get_path(cfg, "errors_dir", "x.parquet").parent
    users_ok.to_parquet(out / "users.parquet", index=False)
    items_ok.to_parquet(out / "items.parquet", index=False)
    # typed interaction table for downstream stages
    inter_ok = inter_ok.assign(
        user_id=pd.to_numeric(inter_ok["user_id"]).astype("int64"),
        item_id=pd.to_numeric(inter_ok["item_id"]).astype("int64"),
        rating=pd.to_numeric(inter_ok["rating"]).astype("float64"),
        timestamp=pd.to_numeric(inter_ok["timestamp"]).astype("int64"),
    )
    inter_ok.to_parquet(out / "interactions.parquet", index=False)
    for name, e in (("users", users_err), ("items", items_err), ("interactions", inter_err)):
        e.to_parquet(err_dir / f"{name}_errors.parquet", index=False)

    sample = pd.concat([e.head(40).assign(table=n) for n, e in
                        (("interactions", inter_err), ("users", users_err), ("items", items_err))], ignore_index=True)
    sample.to_csv(get_path(cfg, "reports_dir", "validation_errors_sample.csv"), index=False)   # human-readable sample

    stats = {}
    for name, raw_df, ok_df in (("users", users, users_ok), ("items", items, items_ok), ("interactions", inter, inter_ok)):
        rej = len(raw_df) - len(ok_df)
        stats[name] = {"rows_in": len(raw_df), "rows_valid": len(ok_df), "rows_rejected": rej,
                       "rejected_pct": 100 * rej / max(len(raw_df), 1)}
        if rej / max(len(raw_df), 1) > v["max_rejected_fraction"]:
            raise ValidationError(f"{name}: {rej} of {len(raw_df)} rows rejected (> {v['max_rejected_fraction']:.0%})")

    checkers = {"users": cu, "items": ci, "interactions": ct}
    manifest = read_json(get_path(cfg, "reports_dir", "ingestion_summary.json")).get("fault_injection")
    _write_report(cfg, stats, checkers, manifest)
    summary = {"tables": stats,
               "reject_counts": {t: dict(c.reject_counts) for t, c in checkers.items()},
               "warn_counts": {t: dict(c.warn_counts) for t, c in checkers.items()}}
    write_json(summary, get_path(cfg, "reports_dir", "validation_report.json"))
    log.info("Validation summary: %s", {t: s["rows_rejected"] for t, s in stats.items()})
    return summary
