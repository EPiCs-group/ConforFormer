#!/usr/bin/env python3
"""
Compute Tanimoto distance for (smi_1_id, smi_2_id) pairs in sim_scores.

- Per-SMILES (177k): compute Open Babel fingerprints once (default FP2)
  and store as compact BLOBs (variable-length big-endian bitvectors).
- Per-pair (163M): update sim_scores.tanimoto_dist = 1 - TanimotoSimilarity
  via a fast SQLite UDF and chunked indexed joins.

Requirements:
  pip install openbabel  # or openbabel-wheel
"""

import argparse
import sqlite3
import sys
import time
from multiprocessing import Pool, cpu_count

# ------------ Open Babel (pybel) ------------
try:
    from openbabel import pybel
except Exception:
    sys.stderr.write(
        "ERROR: Could not import openbabel.pybel. Install Open Babel Python bindings.\n"
        "  pip install openbabel   # official bindings (may build)\n"
        "  or: pip install openbabel-wheel  # prebuilt wheels where available\n"
    )
    raise

# ------------ Pragmas & indices -------------
PRAGMAS = [
    "PRAGMA journal_mode=WAL;",
    "PRAGMA synchronous=NORMAL;",
    "PRAGMA temp_store=MEMORY;",
    "PRAGMA cache_size=-1500000;",   # ~1.5 GiB page cache; adjust to your RAM
]

IDX_CMDS = [
    # We join smiles twice on id for each update, so pk index on smiles(id) is enough.
    "CREATE INDEX IF NOT EXISTS idx_sim_scores_1_2 ON sim_scores(smi_1_id, smi_2_id);",
    "CREATE INDEX IF NOT EXISTS idx_sim_scores_2_1 ON sim_scores(smi_2_id, smi_1_id);",
]

# ------------ Column helpers ----------------
def _fp_colnames(fptype: str):
    # Store per-type so you can recompute with different fingerprints later if needed
    tag = fptype.lower()
    return (f"fp_{tag}_bytes", f"fp_{tag}_popcnt")

def add_missing_columns(conn: sqlite3.Connection, fptype: str):
    cur = conn.cursor()

    # smiles.* fingerprint storage
    cur.execute("PRAGMA table_info(smiles)")
    s_cols = {r[1] for r in cur.fetchall()}
    fp_bytes_col, fp_popcnt_col = _fp_colnames(fptype)
    if fp_bytes_col not in s_cols:
        cur.execute(f"ALTER TABLE smiles ADD COLUMN {fp_bytes_col} BLOB;")
    if fp_popcnt_col not in s_cols:
        cur.execute(f"ALTER TABLE smiles ADD COLUMN {fp_popcnt_col} INTEGER;")

    # sim_scores.tanimoto_dist
    cur.execute("PRAGMA table_info(sim_scores)")
    ss_cols = {r[1] for r in cur.fetchall()}
    if "tanimoto_dist" not in ss_cols:
        cur.execute("ALTER TABLE sim_scores ADD COLUMN tanimoto_dist REAL;")

    conn.commit()

def ensure_indexes(conn: sqlite3.Connection):
    cur = conn.cursor()
    for p in PRAGMAS:
        cur.execute(p)
    for s in IDX_CMDS:
        cur.execute(s)
    conn.commit()

# ------------ Fingerprint computation --------
def _bits_to_bytes(bits):
    """bits: iterable of 1-based bit indices (as returned by pybel calcfp) -> big-endian bytes."""
    if not bits:
        return b"", 0
    n = 0
    for b in bits:
        if b > 0:
            n |= (1 << (b - 1))
    popcnt = n.bit_count()
    length = (n.bit_length() + 7) // 8
    return n.to_bytes(length, "big"), popcnt

def _fp_one(args):
    """
    args: (row_id, smi, fptype)
    returns: (fp_bytes, popcnt, row_id)
    """
    row_id, smi, fptype = args
    try:
        mol = pybel.readstring("smi", smi)
        fp = mol.calcfp(fptype=fptype)
        fp_bytes, popcnt = _bits_to_bytes(getattr(fp, "bits", []))
        return (fp_bytes, popcnt, row_id)
    except Exception:
        return (None, None, row_id)

def compute_smiles_fingerprints(db_path: str, fptype: str, nprocs: int):
    fp_bytes_col, fp_popcnt_col = _fp_colnames(fptype)
    conn = sqlite3.connect(db_path)
    ensure_indexes(conn)
    add_missing_columns(conn, fptype)

    todo = conn.execute(
        f"SELECT id, smi FROM smiles WHERE {fp_bytes_col} IS NULL"
    ).fetchall()
    total = len(todo)
    if total == 0:
        print(f"All SMILES already have {fp_bytes_col}.")
        conn.close()
        return

    print(f"Computing {fptype} fingerprints for {total:,} SMILES ...")
    t0 = time.time()

    # Multiprocessing over molecules; each worker uses pybel in its own process
    BATCH = 5000
    args = ((row_id, smi, fptype) for row_id, smi in todo)
    if nprocs > 1:
        with Pool(processes=nprocs) as pool:
            batch, done = [], 0
            for fp_bytes, popcnt, row_id in pool.imap_unordered(_fp_one, args, chunksize=1000):
                batch.append((fp_bytes, popcnt, row_id))
                if len(batch) >= BATCH:
                    with conn:
                        conn.executemany(
                            f"UPDATE smiles SET {fp_bytes_col}=?, {fp_popcnt_col}=? WHERE id=?",
                            batch
                        )
                    done += len(batch); batch.clear()
                    if done % 50000 == 0:
                        print(f"  processed {done:,}/{total:,}")
            if batch:
                with conn:
                    conn.executemany(
                        f"UPDATE smiles SET {fp_bytes_col}=?, {fp_popcnt_col}=? WHERE id=?",
                        batch
                    )
    else:
        batch, done = [], 0
        for row_id, smi in todo:
            fp_bytes, popcnt, _ = _fp_one((row_id, smi, fptype))
            batch.append((fp_bytes, popcnt, row_id))
            if len(batch) >= BATCH:
                with conn:
                    conn.executemany(
                        f"UPDATE smiles SET {fp_bytes_col}=?, {fp_popcnt_col}=? WHERE id=?",
                        batch
                    )
                done += len(batch); batch.clear()
                if done % 50000 == 0:
                    print(f"  processed {done:,}/{total:,}")
        if batch:
            with conn:
                conn.executemany(
                    f"UPDATE smiles SET {fp_bytes_col}=?, {fp_popcnt_col}=? WHERE id=?",
                    batch
                )

    n_ok = conn.execute(f"SELECT COUNT(*) FROM smiles WHERE {fp_bytes_col} IS NOT NULL").fetchone()[0]
    print(f"Done fingerprints: {n_ok:,}/{total:,}  (elapsed {time.time()-t0:.1f}s)")
    conn.close()

# ------------ Tanimoto UDF & chunked update ---
def _tanimoto_from_blobs(b1: bytes, b2: bytes) -> float | None:
    """
    SQLite UDF: returns Tanimoto distance in [0,1].
    Input: two BLOBs produced by _bits_to_bytes; empty bytes => zero vector.
    """
    if b1 is None or b2 is None:
        return None
    a = int.from_bytes(b1, "big") if b1 else 0
    b = int.from_bytes(b2, "big") if b2 else 0
    if a == 0 and b == 0:
        return 0.0
    c = (a & b).bit_count()
    pa = a.bit_count()
    pb = b.bit_count()
    denom = pa + pb - c
    if denom == 0:
        return 0.0
    sim = c / denom
    return 1.0 - sim

def _rowid_bounds(conn: sqlite3.Connection):
    lo, hi = conn.execute(
        "SELECT MIN(rowid), MAX(rowid) FROM sim_scores WHERE tanimoto_dist IS NULL"
    ).fetchone()
    return lo, hi

def update_tanimoto_distances(db_path: str, fptype: str, chunk_rows: int):
    fp_bytes_col, _ = _fp_colnames(fptype)
    conn = sqlite3.connect(db_path)
    ensure_indexes(conn)
    add_missing_columns(conn, fptype)
    conn.create_function("tanimoto_blob", 2, _tanimoto_from_blobs)

    lo, hi = _rowid_bounds(conn)
    if lo is None:
        print("All rows in sim_scores already have tanimoto_dist.")
        conn.close(); return

    print(f"Updating tanimoto_dist using {fptype} over rowid [{lo}, {hi}] in chunks of {chunk_rows:,} ...")
    t0_all = time.time()

    start = lo
    while start <= hi:
        end = min(start + chunk_rows - 1, hi)
        t0 = time.time()
        with conn:
            # Build a small TEMP table for this range: (rowid, distance)
            conn.execute("DROP TABLE IF EXISTS temp.tmp_tani;")
            conn.execute("CREATE TEMP TABLE tmp_tani(rowid INTEGER PRIMARY KEY, d REAL);")

            # Compute distances via indexed joins once per row
            conn.execute(f"""
                INSERT INTO tmp_tani(rowid, d)
                SELECT s.rowid,
                       tanimoto_blob(f1.{fp_bytes_col}, f2.{fp_bytes_col}) AS d
                  FROM sim_scores AS s
                  JOIN smiles AS f1 ON f1.id = s.smi_1_id
                  JOIN smiles AS f2 ON f2.id = s.smi_2_id
                 WHERE s.rowid BETWEEN ? AND ?
                   AND s.tanimoto_dist IS NULL;
            """, (start, end))

            # Apply to sim_scores for this range
            conn.execute("""
                UPDATE sim_scores AS s
                   SET tanimoto_dist = (SELECT d FROM tmp_tani WHERE rowid = s.rowid)
                 WHERE s.rowid BETWEEN ? AND ?
                   AND s.tanimoto_dist IS NULL
                   AND EXISTS (SELECT 1 FROM tmp_tani t WHERE t.rowid = s.rowid);
            """, (start, end))

            # (implicitly drops tmp_tani next iteration)
        n_upd = conn.execute(
            "SELECT COUNT(*) FROM sim_scores WHERE rowid BETWEEN ? AND ? AND tanimoto_dist IS NOT NULL",
            (start, end),
        ).fetchone()[0]
        took = time.time() - t0
        print(f"  chunk {start:,}-{end:,}: updated {n_upd:,} rows in {took:.1f}s")
        start = end + 1

    print(f"All chunks done in {time.time()-t0_all:.1f}s.")
    conn.close()

# ------------ CLI ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Add Tanimoto distance to sim_scores using Open Babel fingerprints.")
    ap.add_argument("db", help="Path to Seed3_contrast_sim.sqlite3")
    ap.add_argument("--fingerprint", default="FP2", choices=["FP2", "FP3", "FP4", "MACCS"],
                    help="Open Babel fingerprint to use (default: FP2)")
    ap.add_argument("--chunk", type=int, default=2_000_000,
                    help="Number of sim_scores rows per chunk (default: 2,000,000)")
    ap.add_argument("--nprocs", type=int, default=max(1, cpu_count() // 2),
                    help="Worker processes for per-SMILES fingerprinting (default: half cores)")
    args = ap.parse_args()

    compute_smiles_fingerprints(args.db, args.fingerprint, args.nprocs)
    update_tanimoto_distances(args.db, args.fingerprint, args.chunk)

if __name__ == "__main__":
    main()
