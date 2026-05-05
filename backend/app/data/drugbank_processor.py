"""
DrugBank XML Processor
=======================
Parses the DrugBank full database XML and builds a SQLite lookup database.

map_severity() — ACADEMIC DOCUMENTATION
-----------------------------------------
This function assigns severity scores (0-3) to drug-drug interactions using
keyword matching on DrugBank interaction description text.

This is a HEURISTIC / SILVER-LABELING approach. It is NOT:
  - Human-annotated ground truth
  - Validated against a gold-standard severity scale

It IS:
  - A form of distant supervision (Mintz et al., 2009)
  - Suitable for training when gold labels are unavailable
  - Should be reported in the paper as "silver labels derived from
    DrugBank interaction descriptions via keyword matching"

Severity scale (0-3):
  0 = safe      — no severity keywords found
  1 = caution   — mild/possible/minor keywords
  2 = warning   — moderate/significant/monitor keywords
  3 = danger    — major/severe/fatal/contraindicated keywords

Reference:
  Mintz et al. (2009). Distant supervision for relation extraction without
  labeled data. ACL 2009.
"""

import os
import sqlite3
import xml.etree.ElementTree as ET
from typing import Dict, List

from app.data.kb_normalization import normalize_kb_text

NS = {'db': 'http://www.drugbank.ca'}


def map_severity(description: str) -> int:
    """
    Heuristic severity mapping from DrugBank interaction description text.
    Silver-labeling strategy — see module docstring for academic context.

    Returns:
        0 = safe      (no keywords matched)
        1 = caution   (minor / mild / possible)
        2 = warning   (moderate / significant / risk)
        3 = danger    (major / severe / fatal / contraindicated)
    """
    if not description:
        return 0
    desc_lower = description.lower()

    if any(word in desc_lower for word in [
        'major', 'contraindicated', 'severe', 'serious', 'fatal',
        'dangerous', 'life-threatening', 'toxic', 'toxicity',
    ]):
        return 3

    if any(word in desc_lower for word in [
        'moderate', 'significant', 'monitor', 'increase', 'decrease',
        'risk', 'bleeding', 'anticoagulant', 'inhibit', 'enhance',
    ]):
        return 2

    if any(word in desc_lower for word in [
        'minor', 'mild', 'slight', 'may', 'possible', 'potential',
    ]):
        return 1

    return 0


def parse_drugbank_xml(xml_path: str) -> List[Dict]:
    """Parse DrugBank XML and extract drug + interaction data."""
    print(f"Parsing DrugBank XML from {xml_path} ...")
    drugs = []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for drug in root.findall('db:drug', NS):
        drugbank_id_el = drug.find('db:drugbank-id[@primary="true"]', NS)
        name_el        = drug.find('db:name', NS)
        description_el = drug.find('db:description', NS)
        mechanism_el   = drug.find('db:mechanism-of-action', NS)

        drug_id     = drugbank_id_el.text if drugbank_id_el is not None else ''
        drug_name   = name_el.text        if name_el        is not None else ''
        description = description_el.text if description_el is not None else ''
        mechanism   = mechanism_el.text   if mechanism_el   is not None else ''

        interactions = []
        interactions_el = drug.find('db:drug-interactions', NS)
        if interactions_el is not None:
            for interaction in interactions_el.findall('db:drug-interaction', NS):
                interacting_id_el   = interaction.find('db:drugbank-id', NS)
                interacting_name_el = interaction.find('db:name', NS)
                desc_el             = interaction.find('db:description', NS)

                interacting_id   = interacting_id_el.text   if interacting_id_el   is not None else ''
                interacting_name = interacting_name_el.text if interacting_name_el is not None else ''
                desc             = desc_el.text             if desc_el             is not None else ''
                severity         = map_severity(desc)

                interactions.append({
                    'drug_b_id':   interacting_id,
                    'drug_b_name': interacting_name,
                    'description': desc,
                    'severity':    severity,
                })

        # ── Layer 1: synonyms / brands / product names (for entity linking) ──
        synonyms: List[Dict] = []
        int_brands: List[Dict] = []
        product_names: List[Dict] = []

        syns_el = drug.find("db:synonyms", NS)
        if syns_el is not None:
            for s in syns_el.findall("db:synonym", NS):
                t = (s.text or "").strip()
                if t:
                    synonyms.append({"text": t, "source": "synonym"})

        ib_el = drug.find("db:international-brands", NS)
        if ib_el is not None:
            for b in ib_el.findall("db:international-brand", NS):
                t = (b.text or "").strip()
                if t:
                    int_brands.append({"text": t, "source": "international_brand"})

        prod_el = drug.find("db:products", NS)
        if prod_el is not None:
            for p in prod_el.findall("db:product", NS):
                n_el = p.find("db:name", NS)
                t = (n_el.text or "").strip() if n_el is not None else ""
                if t:
                    product_names.append({"text": t, "source": "product"})

        drugs.append({
            'id':           drug_id,
            'name':         drug_name,
            'description':  description,
            'mechanism':    mechanism,
            'interactions': interactions,
            'link_synonyms':   synonyms,
            'link_int_brands': int_brands,
            'link_products':   product_names,
        })

    print(f"Parsed {len(drugs)} drugs from DrugBank")
    return drugs


def build_sqlite_db(drugs: List[Dict], db_path: str):
    """Insert parsed DrugBank data into SQLite for fast inference lookup."""
    print(f"Building SQLite database at {db_path} ...")
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS drugs (
        id          TEXT PRIMARY KEY,
        name        TEXT,
        description TEXT,
        mechanism   TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS interactions (
        drug_a_id   TEXT,
        drug_b_id   TEXT,
        drug_b_name TEXT,
        description TEXT,
        severity    INTEGER,
        PRIMARY KEY (drug_a_id, drug_b_id)
    )''')

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS drugbank_synonyms (
            drug_id       TEXT,
            synonym_text  TEXT,
            source        TEXT,
            norm_text     TEXT,
            PRIMARY KEY (drug_id, norm_text, source)
        )
        """
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_drugbank_synonyms_norm ON drugbank_synonyms(norm_text)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_drugbank_synonyms_drug ON drugbank_synonyms(drug_id)")

    for drug in drugs:
        c.execute('INSERT OR REPLACE INTO drugs VALUES (?, ?, ?, ?)',
                  (drug['id'], drug['name'], drug['description'], drug['mechanism']))
        for iact in drug['interactions']:
            c.execute('INSERT OR REPLACE INTO interactions VALUES (?, ?, ?, ?, ?)',
                      (drug['id'], iact['drug_b_id'], iact['drug_b_name'],
                       iact['description'], iact['severity']))

        # Rebuild per-drug linking rows
        c.execute("DELETE FROM drugbank_synonyms WHERE drug_id = ?", (drug["id"],))
        for row in drug.get("link_synonyms", []) or []:
            t = (row.get("text") or "").strip()
            if t:
                nt = normalize_kb_text(t)
                c.execute(
                    "INSERT OR IGNORE INTO drugbank_synonyms(drug_id, synonym_text, source, norm_text) VALUES (?, ?, ?, ?)",
                    (drug["id"], t, "synonym", nt),
                )
        for row in drug.get("link_int_brands", []) or []:
            t = (row.get("text") or "").strip()
            if t:
                nt = normalize_kb_text(t)
                c.execute(
                    "INSERT OR IGNORE INTO drugbank_synonyms(drug_id, synonym_text, source, norm_text) VALUES (?, ?, ?, ?)",
                    (drug["id"], t, "international_brand", nt),
                )
        for row in drug.get("link_products", []) or []:
            t = (row.get("text") or "").strip()
            if t:
                nt = normalize_kb_text(t)
                c.execute(
                    "INSERT OR IGNORE INTO drugbank_synonyms(drug_id, synonym_text, source, norm_text) VALUES (?, ?, ?, ?)",
                    (drug["id"], t, "product", nt),
                )
        if drug.get("name"):
            t = (drug["name"] or "").strip()
            nt = normalize_kb_text(t)
            c.execute(
                "INSERT OR IGNORE INTO drugbank_synonyms(drug_id, synonym_text, source, norm_text) VALUES (?, ?, ?, ?)",
                (drug["id"], t, "primary_name", nt),
            )

    # Layer 2: anchor DDInter vocabulary in the same DB (rebuild from CSVs)
    data_dir = os.path.dirname(db_path) or "."
    try:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS ddinter_drug_names (
              norm_name TEXT PRIMARY KEY
            )
            """
        )
        c.execute("DELETE FROM ddinter_drug_names")
        # Collect names in-process (avoids a second pass through CSV here)
        from app.data.ddinter_vocabulary import list_ddinter_csvs
        import csv as _csv
        all_names = set()
        for p in list_ddinter_csvs(data_dir):
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8", newline="") as f:
                reader = _csv.DictReader(f)
                cols = {c.lower(): c for c in (reader.fieldnames or [])}
                a_col = cols.get("drug_a") or cols.get("drug a")
                b_col = cols.get("drug_b") or cols.get("drug b")
                if not (a_col and b_col):
                    continue
                for row in reader:
                    a = (row.get(a_col) or "").strip()
                    b = (row.get(b_col) or "").strip()
                    if a:
                        all_names.add(normalize_kb_text(a))
                    if b:
                        all_names.add(normalize_kb_text(b))
        all_names = {n for n in all_names if n}
        c.executemany("INSERT OR IGNORE INTO ddinter_drug_names(norm_name) VALUES (?)", [(n,) for n in sorted(all_names)])
        print(f"Rebuilt ddinter_drug_names: {len(all_names)} unique normalized names")
    except Exception as e:
        print(f"⚠  ddinter_drug_names rebuild failed: {e}")

    conn.commit()
    conn.close()
    print("SQLite database built successfully")


def lookup_interaction(drug_a_name: str, drug_b_name: str, db_path: str) -> Dict:
    """Look up interaction between two drugs by name."""
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()

    c.execute('''
        SELECT i.description, i.severity, i.drug_b_name
        FROM   interactions i
        JOIN   drugs da ON i.drug_a_id = da.id
        JOIN   drugs db ON i.drug_b_id = db.id
        WHERE  (LOWER(da.name) = LOWER(?) AND LOWER(db.name) = LOWER(?))
        OR     (LOWER(da.name) = LOWER(?) AND LOWER(db.name) = LOWER(?))
    ''', (drug_a_name, drug_b_name, drug_b_name, drug_a_name))

    row = c.fetchone()
    conn.close()

    if row:
        sev_map = {0: 'safe', 1: 'caution', 2: 'warning', 3: 'danger'}
        return {
            'found':          True,
            'description':    row[0],
            'severity':       row[1],
            'severity_label': sev_map.get(row[1], 'unknown'),
        }
    return {'found': False}


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(base_dir, 'drugbank_full.xml', 'full database.xml')
    db_path  = os.path.join(base_dir, 'drugbank.db')

    if os.path.exists(xml_path):
        drugs = parse_drugbank_xml(xml_path)
        build_sqlite_db(drugs, db_path)
        print(f"Done! {len(drugs)} drugs → {db_path}")
    else:
        print(f"DrugBank XML not found at: {xml_path}")
