import sqlite3
from collections import defaultdict
import re

def parse_t_layer(t_layer: str):
    """Convert /t layer string to a dict {index: +1/-1}, None if placeholder."""
    if not t_layer or t_layer == "NoDo":
        return None
    out = {}
    for item in t_layer[2:].split(","):  # skip '/t'
        m = re.match(r"\d+", item)
        if not m:
            continue  # skip malformed entries
        idx = int(m.group())
        sign = +1 if "+" in item else -1 if "-" in item else 0
        out[idx] = sign
    return out

def t_layer_hash(t_dict):
    """Canonical sorted tuple for t_layer dict."""
    return tuple(sorted(t_dict.items()))

def main(db_path="bigdata.sqlite", batch_size=10000):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Check for tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    if "main" not in tables:
        raise ValueError("Table 'main' does not exist in database.")
    if "inchikey_blocks" not in tables:
        raise ValueError("Table 'inchikey_blocks' does not exist in database.")

    # 2. Add enantiomer_id column if missing
    cur.execute("PRAGMA table_info(main)")
    columns = [info[1] for info in cur.fetchall()]
    if "enantiomer_id" not in columns:
        cur.execute("ALTER TABLE main ADD COLUMN enantiomer_id INTEGER DEFAULT -1")
        conn.commit()

    # 3. Create indexes for speed
    cur.execute("CREATE INDEX IF NOT EXISTS idx_main_formula ON main(formula_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_main_inchi ON main(inchi_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_inchikey_block ON inchikey_blocks(value)")
    conn.commit()

    # 4. Fetch all distinct formula_ids
    cur.execute("SELECT DISTINCT formula_id FROM main")
    formula_ids = [row[0] for row in cur.fetchall()]

    update_batch = []

    total_processed = 0
    total_skipped = 0
    total_enantiomers = 0

    for f_idx, f_id in enumerate(formula_ids, start=1):
        # Fetch all entries for this formula_id
        cur.execute("""
            SELECT main.id, inchikey_blocks.value, main.t_layer
            FROM main
            LEFT JOIN inchikey_blocks ON main.inchi_id = inchikey_blocks.id
            WHERE main.formula_id = ?
        """, (f_id,))
        rows = cur.fetchall()

        if not rows:
            continue

        # Group by InChIKey block1
        block_groups = defaultdict(list)
        skipped_rows = 0
        for row_id, block1, t_layer in rows:
            if t_layer == "NoDo" or block1 is None:
                skipped_rows += 1
                update_batch.append((row_id, -1))
                continue
            block_groups[block1].append((row_id, parse_t_layer(t_layer)))

        total_skipped += skipped_rows
        total_processed += len(rows) - skipped_rows

        # Assign enantiomers within each block1 group
        for block1, entries in block_groups.items():
            hash_to_id = {}
            for row_id, t_dict in entries:
                if t_dict is None:
                    update_batch.append((row_id, -1))
                    continue
                inv_hash = tuple((k, -v) for k, v in t_layer_hash(t_dict))
                if inv_hash in hash_to_id:
                    other_id = hash_to_id[inv_hash]
                    # Assign each other
                    update_batch.append((row_id, other_id))
                    update_batch.append((other_id, row_id))
                    total_enantiomers += 2
                else:
                    hash_to_id[t_layer_hash(t_dict)] = row_id

        # Commit batch periodically
        if len(update_batch) >= batch_size:
            cur.executemany("UPDATE main SET enantiomer_id = ? WHERE id = ?", update_batch)
            conn.commit()
            update_batch = []

        if f_idx % 1000 == 0:
            print(f"Processed {f_idx}/{len(formula_ids)} formula_ids...")

    # Final commit
    if update_batch:
        cur.executemany("UPDATE main SET enantiomer_id = ? WHERE id = ?", update_batch)
        conn.commit()

    print(f"Total rows processed (with valid t_layer): {total_processed}")
    print(f"Total rows skipped (NoDo or missing block1): {total_skipped}")
    print(f"Total enantiomer assignments made: {total_enantiomers}")

    conn.close()

if __name__ == "__main__":
    main("organized_data_train.sqlite3")
