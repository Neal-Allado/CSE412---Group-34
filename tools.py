"""Agent tools: find_dataset, verify_result, get_provenance.

Setup:  pip install "psycopg[binary]"
        set DATABASE_URL to your Postgres connection string
        run meta_schema.sql once (adds data_source, ingestion_run, verification_rule, discrepancy_log)
Use:    config = types.GenerateContentConfig(tools=[find_dataset, verify_result, get_provenance])
"""
import json
import math
import os
import uuid
from decimal import Decimal

import psycopg
from psycopg.rows import dict_row


def _db():
    return psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)


def _clean(x):
    """Make DB values (Decimal, date, ...) JSON-friendly."""
    return json.loads(json.dumps(x, default=lambda v: float(v) if isinstance(v, Decimal) else str(v)))


def find_dataset(description: str) -> dict:
    """Find which data source and tables hold the data needed for a question.

    Args:
        description: Plain-English description of the data needed,
            e.g. "consumer complaints by company" or "quarterly GDP revisions".

    Returns:
        {"matches": [{"source_id", "name", "access_method", "tables": {table: ["col type", ...]}}]}
        best match first.
    """
    words = {w.strip(".,?!").rstrip("s") for w in description.lower().split()}
    words = {w for w in words if len(w) > 2}
    matches = []
    with _db() as db:
        for s in db.execute("SELECT * FROM data_source").fetchall():
            text = " ".join([s["name"], s["description"], *s["keywords"], *s["tables"]]).lower()
            score = sum(w in text for w in words)
            if score:
                tables = {}
                for t in s["tables"]:
                    cols = db.execute(
                        "SELECT column_name, data_type FROM information_schema.columns "
                        "WHERE table_schema='public' AND table_name=%s ORDER BY ordinal_position", (t,))
                    tables[t] = [f"{c['column_name']} {c['data_type']}" for c in cols]
                matches.append({"source_id": s["source_id"], "name": s["name"],
                                "access_method": s["access_method"], "tables": tables, "score": score})
    matches.sort(key=lambda m: -m["score"])
    return _clean({"matches": matches})


def verify_result(result: list, rule_name: str = None) -> dict:
    """Sanity-check a prior execute_sql result, and optionally re-run a stored comparison.

    Checks that the result is not empty, has no all-NULL columns, and has no NaN or
    absurdly large numbers. With rule_name, also re-runs a stored comparison from the
    verification_rule table: 'balance_sheet_identity' (Assets vs Liabilities + Equity),
    'vintage_revision' (first vs latest value of a macro series), or 'ohlc_consistent'
    (daily price bars). Findings are written to the discrepancy_log table.

    Args:
        result: Rows returned by execute_sql, as a list of dicts.
        rule_name: Optional stored rule name.

    Returns:
        {"passed": bool, "problems": [...], "mismatches": [first 20], "check_id": str}
        Look up details in SQL: SELECT * FROM discrepancy_log WHERE check_id = '<check_id>'
    """
    rows = result.get("rows", []) if isinstance(result, dict) else result
    problems = []
    if not rows:
        problems.append("Result is empty.")
    for col in (rows[0] if rows else {}):
        vals = [r[col] for r in rows if r.get(col) is not None]
        nums = [float(v) for v in vals if isinstance(v, (int, float, Decimal)) and not isinstance(v, bool)]
        if not vals:
            problems.append(f"Column '{col}' is all NULL.")
        if any(math.isnan(n) or math.isinf(n) for n in nums):
            problems.append(f"Column '{col}' has NaN or infinity.")
        if any(abs(n) > 1e15 for n in nums):
            problems.append(f"Column '{col}' has implausibly large values (unit error?).")

    mismatches = []
    check_id = str(uuid.uuid4())
    with _db() as db:
        if rule_name:
            rule = db.execute("SELECT * FROM verification_rule WHERE rule_name=%s", (rule_name,)).fetchone()
            if not rule:
                return {"error": f"Unknown rule '{rule_name}'."}
            tol = float(rule["tolerance_rel"])
            filters = {"cik": None, "ticker": None, "series_id": None}
            for r in db.execute(rule["comparison_sql"], filters).fetchall():
                if r["expected"] is None or r["actual"] is None:
                    continue
                if abs(r["actual"] - r["expected"]) > tol * max(abs(r["expected"]), 1e-9):
                    mismatches.append({"record_key": r["record_key"],
                                       "expected": r["expected"], "actual": r["actual"]})

        sql = ("INSERT INTO discrepancy_log (check_id, rule_name, status, record_key, expected, actual, note) "
               "VALUES (%s, %s, %s, %s, %s, %s, %s)")
        for p in problems:
            db.execute(sql, (check_id, "generic_checks", "fail", None, None, None, p))
        for m in mismatches[:100]:
            db.execute(sql, (check_id, rule_name, "fail", m["record_key"], m["expected"], m["actual"], None))
        if not problems and not mismatches:
            db.execute(sql, (check_id, rule_name or "generic_checks", "pass", None, None, None, None))

    return _clean({"passed": not problems and not mismatches, "problems": problems,
                   "mismatches": mismatches[:20], "mismatch_count": len(mismatches), "check_id": check_id})


def get_provenance(table: str, record_id: str) -> dict:
    """Trace a record back to its source: record -> ingestion_run -> data_source.

    Use this for any number you report so it can be cited.

    Args:
        table: Table name, e.g. 'complaint', 'financial_fact', 'series_obs'.
        record_id: Primary key value. For a composite key, join the parts in key order
            with '|', e.g. 'JPM|2024-03-01' for price_bar.

    Returns:
        {"record": {...}, "source": {...}, "ingestion_run": {...}}
        The ingestion run is the latest one logged for the table's source.
    """
    with _db() as db:
        # Only tables registered in data_source are allowed (also keeps the table name safe).
        source = db.execute("SELECT * FROM data_source WHERE %s = ANY(tables)", (table,)).fetchone()
        if not source:
            return {"error": f"Unknown table '{table}'."}
        pk = [r["column_name"] for r in db.execute(
            "SELECT kcu.column_name FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu USING (constraint_name, table_schema) "
            "WHERE tc.table_schema='public' AND tc.table_name=%s AND tc.constraint_type='PRIMARY KEY' "
            "ORDER BY kcu.ordinal_position", (table,))]
        key = str(record_id).split("|")
        if len(key) != len(pk):
            return {"error": f"'{table}' key is {pk}; give {len(pk)} value(s) joined by '|'."}
        where = " AND ".join(f'"{c}"::text = %s' for c in pk)
        record = db.execute(f'SELECT * FROM public."{table}" WHERE {where}', key).fetchone()
        if not record:
            return {"error": "Record not found."}
        run = db.execute("SELECT * FROM ingestion_run WHERE source_id=%s ORDER BY started_at DESC LIMIT 1",
                         (source["source_id"],)).fetchone()
    return _clean({"record": record,
                   "source": {k: source[k] for k in ("source_id", "name", "access_method", "base_url")},
                   "ingestion_run": run or "No ingestion_run logged for this source yet."})
