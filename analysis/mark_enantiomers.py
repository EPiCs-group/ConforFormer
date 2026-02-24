#!/usr/bin/env python3
import argparse
import sqlite3
import sys
import time
from multiprocessing import Pool, cpu_count

# ---- Open Babel setup -------------------------------------------------------
# Requires Open Babel 3.x Python bindings: `from openbabel import openbabel as ob`
try:
    from openbabel import openbabel as ob
except Exception as e:
    sys.stderr.write(
        "ERROR: Could not import Open Babel. Install openbabel (>=3.0) first.\n"
        "On conda:   conda install -c conda-forge openbabel\n"
        "On pip:     pip install openbabel\n"
    )
    raise

# Worker-local converters (created once per process)
_worker_conv_in = None
_worker_conv_inchi_rel = None
_worker_conv_inchikey_std = None

def _worker_init():
    global _worker_conv_in, _worker_conv_inchi_rel, _worker_conv_inchikey_std
    _worker_conv_in = ob.OBConversion()
    assert _worker_conv_in.SetInFormat("smi")

    _worker_conv_inchi_rel = ob.OBConversion()
    assert _worker_conv_inchi_rel.SetOutFormat("inchi")
    # Pass InChI extra option to encode *relative* stereochemistry
    _worker_conv_inchi_rel.AddOption("X", _worker_conv_inchi_rel.OUTOPTIONS, "SRel")

    _worker_conv_inchikey_std = ob.OBConversion()
    assert _worker_conv_inchikey_std.SetOutFormat("inchikey")

def _compute_one(record):
    """
    record: (id, smi)
    returns (inchi_simrel, inchikey_std, id)  -- or (None, None, id) on parse failure
    """
    _id, smi = record
    mol = ob.OBMol()
    try:
        if not _worker_conv_in.ReadString(mol, smi):
            return (None, None, _id)
        inchi_rel = _worker_conv_inchi_rel.WriteString(mol).strip()
        inchikey_std = _worker_conv_inchikey_std.WriteString(mol).strip()
        # Normalize empties to None for SQL NULL
        if not inchi_rel:
            inchi_rel = None
        if not inchikey_std:
            inchikey_std = None
        return (inchi_rel, inchikey_std, _id)
    except Exception:
        return (None, None, _id)

# ---- Database helpers -------------------------------------------------------

DDL_SMILES_COLS = [
    "ALTER TABLE smiles ADD COLUMN inchi_simrel   TEXT;",
    "ALTER TABLE smiles ADD COLUMN inchikey_std   TEXT;"
]

DDL_SIM_SCORES_COL = "ALTER TABLE sim_scores ADD COLUMN is_enantiomer INTEGER NOT NULL DEFAULT 0;"

IDX_CMDS = [
    "CREATE INDEX IF NOT EXISTS idx_smiles_inchi_simrel  ON smiles(inchi_simrel);",
    "CREATE INDEX IF NOT EXISTS idx_smiles_inchikey_std  ON smiles(inchikey_std);",
    # Two directions help us hit both (smi_1_id, smi_2_id) and the swapped case efficiently
    "CREATE INDEX IF NOT EXISTS idx_sim_scores_1_2 ON sim_scores(smi_1_id, smi_2_id);",
    "CREATE INDEX IF NOT EXISTS idx_sim_scores_2_1 ON sim_scores(smi_2_id, smi_1_id);",
]

PRAGMAS = [
    "PRAGMA journal_mode=WAL;",
    "PRAGMA synchronous=NORMAL;",
    # Negative cache_size is in KiB; increase if you have RAM (here ~1.5 GiB)
    "PRAGMA cache_size=-1500000;",
    "PRAGMA temp_store=MEMORY;",
]

def add_missing_columns(conn):
    cur = conn.cursor()
    # Add columns if missing
    existing = set(r[1] for r in cur.execute("PRAGMA table_info(smiles)"))
    for cmd in DDL_SMILES_COLS:
        colname = cmd.split("ADD COLUMN")[1].split()[0]
        if colname not in existing:
            cur.execute(cmd)

    existing = set(r[1] for r in cur.execute("PRAGMA table_info(sim_scores)"))
    if "is_enantiomer" not in existing:
        cur.execute(DDL_SIM_SCORES_COL)
    conn.commit()

def ensure_indexes(conn):
    cur = conn.cursor()
    for cmd in IDX_CMDS:
        cur.execute(cmd)
    conn.commit()

# ---- Main steps -------------------------------------------------------------

def compute_smiles_inchi_fields(db_path, nprocs):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for p in PRAGMAS:
        cur.execute(p)
    add_missing_columns(conn)

    # Pull only rows missing either field (idempotent)
    to_compute = list(cur.execute(
        "SELECT id, smi FROM smiles WHERE inchi_simrel IS NULL OR inchikey_std IS NULL"
    ))
    total = len(to_compute)
    if total == 0:
        print("All smiles have inchi_simrel and inchikey_std. Skipping recompute.")
        conn.close()
        return

    print(f"Computing InChI (relative stereo) and standard InChIKey for {total:,} SMILES...")

    # Use multiprocessing if requested
    nprocs = nprocs or 1
    t0 = time.time()

    if nprocs > 1:
        with Pool(processes=nprocs, initializer=_worker_init) as pool:
            # Stream results and batch updates to keep memory bounded
            batch, B = [], 5000
            for i, res in enumerate(pool.imap_unordered(_compute_one, to_compute, chunksize=1000), 1):
                batch.append(res)
                if len(batch) >= B:
                    with conn:
                        conn.executemany(
                            "UPDATE smiles SET inchi_simrel=?, inchikey_std=? WHERE id=?",
                            batch
                        )
                    batch.clear()
                    if i % (50_000) == 0:
                        print(f"  processed {i:,}/{total:,}")
            if batch:
                with conn:
                    conn.executemany(
                        "UPDATE smiles SET inchi_simrel=?, inchikey_std=? WHERE id=?",
                        batch
                    )
    else:
        # Single-process path (no pool)
        _worker_init()
        batch, B = [], 5000
        for i, rec in enumerate(to_compute, 1):
            batch.append(_compute_one(rec))
            if len(batch) >= B:
                with conn:
                    conn.executemany(
                        "UPDATE smiles SET inchi_simrel=?, inchikey_std=? WHERE id=?",
                        batch
                    )
                batch.clear()
                if i % (50_000) == 0:
                    print(f"  processed {i:,}/{total:,}")
        if batch:
            with conn:
                conn.executemany(
                    "UPDATE smiles SET inchi_simrel=?, inchikey_std=? WHERE id=?",
                    batch
                )

    # Indexes after population (faster than maintaining during writes)
    ensure_indexes(conn)

    # Sanity check counts
    n_rel = conn.execute("SELECT COUNT(*) FROM smiles WHERE inchi_simrel IS NOT NULL").fetchone()[0]
    n_std = conn.execute("SELECT COUNT(*) FROM smiles WHERE inchikey_std IS NOT NULL").fetchone()[0]
    print(f"Done: inchi_simrel={n_rel:,}, inchikey_std={n_std:,} (elapsed {time.time()-t0:.1f}s)")
    conn.close()

def build_enantiomer_pairs(db_path):
    import sqlite3, time
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for p in PRAGMAS:
        cur.execute(p)

    ensure_indexes(conn)

    print("Building enantiomer pairs from the smiles table (persistent)...")
    t0 = time.time()
    with conn:
        cur.execute("DROP TABLE IF EXISTS enant_pairs;")
        cur.execute("""
            CREATE TABLE enant_pairs(
                id1 INTEGER NOT NULL,
                id2 INTEGER NOT NULL,
                PRIMARY KEY (id1, id2)
            ) WITHOUT ROWID;
        """)
        cur.execute("""
            INSERT INTO enant_pairs(id1, id2)
            SELECT s1.id, s2.id
            FROM smiles s1
            JOIN smiles s2
              ON s1.inchi_simrel = s2.inchi_simrel
             AND s1.inchikey_std <> s2.inchikey_std
             AND s1.id < s2.id
            WHERE s1.inchi_simrel IS NOT NULL
              AND s1.inchikey_std IS NOT NULL
              AND s2.inchikey_std IS NOT NULL;
        """)
        # PK gives us (id1,id2) index; create reverse for the second pass.
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ep_21 ON enant_pairs(id2, id1);")

    n_pairs = cur.execute("SELECT COUNT(*) FROM enant_pairs;").fetchone()[0]
    print(f"Identified {n_pairs:,} enantiomer pairs. Elapsed {time.time()-t0:.1f}s")
    conn.close()
    return n_pairs

def update_sim_scores(db_path):
    import sqlite3, time
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for p in PRAGMAS:
        cur.execute(p)

    add_missing_columns(conn)
    ensure_indexes(conn)

    print("Updating sim_scores.is_enantiomer via indexed EXISTS joins...")
    t0 = time.time()
    with conn:
        cur.execute("""
            UPDATE sim_scores AS s
               SET is_enantiomer = 1
             WHERE EXISTS (
                 SELECT 1 FROM enant_pairs p
                 WHERE p.id1 = s.smi_1_id AND p.id2 = s.smi_2_id
             );
        """)
        cur.execute("""
            UPDATE sim_scores AS s
               SET is_enantiomer = 1
             WHERE EXISTS (
                 SELECT 1 FROM enant_pairs p
                 WHERE p.id1 = s.smi_2_id AND p.id2 = s.smi_1_id
             );
        """)

    n_marked = cur.execute("SELECT COUNT(*) FROM sim_scores WHERE is_enantiomer=1;").fetchone()[0]
    print(f"Marked {n_marked:,} sim_scores rows. Elapsed {time.time()-t0:.1f}s")
    conn.close()

# ---- CLI --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute relative InChI for SMILES and mark enantiomer rows in sim_scores.")
    ap.add_argument("db", help="Path to Seed3_contrast_sim.sqlite3")
    ap.add_argument("--nprocs", type=int, default=max(1, cpu_count() // 2),
                    help="Parallel worker processes for SMILES conversion (default: half of CPU cores)")
    args = ap.parse_args()

    compute_smiles_inchi_fields(args.db, nprocs=args.nprocs)
    n_pairs = build_enantiomer_pairs(args.db)
    if n_pairs == 0:
        print("No enantiomer pairs found; sim_scores.is_enantiomer will remain 0.")
        return
    update_sim_scores(args.db)

if __name__ == "__main__":
    main()
