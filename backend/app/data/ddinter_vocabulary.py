"""
DDInter vocabulary table (build once, deterministic)
==================================================
Populates `ddinter_drug_names` in drugbank.db from ddinter_code_*.csv
"""

from __future__ import annotations

import csv
import os
import sqlite3
from typing import List, Optional

from app.data.kb_normalization import normalize_kb_text


def list_ddinter_csvs(data_dir: str) -> List[str]:
    return [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().startswith("ddinter_code_") and f.lower().endswith(".csv")
    ]


def rebuild_ddinter_vocabulary(drugbank_db_path: str, data_dir: str) -> int:
    """
    Rebuilds ddinter_drug_names from all DDInter CSVs in data_dir.
    Returns number of unique normalized names inserted.
    """
    paths = list_ddinter_csvs(data_dir)
    names = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = {c.lower(): c for c in (reader.fieldnames or [])}
            a_col = cols.get("drug_a") or cols.get("drug a")
            b_col = cols.get("drug_b") or cols.get("drug b")
            if not (a_col and b_col):
                continue
            for row in reader:
                a = (row.get(a_col) or "").strip()
                b = (row.get(b_col) or "").strip()
                if a:
                    names.add(normalize_kb_text(a))
                if b:
                    names.add(normalize_kb_text(b))
    # drop empties
    names = {n for n in names if n}

    conn = sqlite3.connect(drugbank_db_path)
    try:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS ddinter_drug_names (
              norm_name TEXT PRIMARY KEY
            )
            """
        )
        c.execute("DELETE FROM ddinter_drug_names")
        c.executemany("INSERT OR IGNORE INTO ddinter_drug_names(norm_name) VALUES (?)", [(n,) for n in sorted(names)])
        conn.commit()
    finally:
        conn.close()
    return len(names)
