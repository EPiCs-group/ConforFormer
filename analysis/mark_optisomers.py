#!/usr/bin/env python3
import argparse
import sqlite3
import sys
import time
from multiprocessing import Pool, cpu_count

# ---- Open Babel (OBConversion) ----------------------------------------------
try:
    from openbabel import openbabel as ob
except Exception:
    sys.stderr.write(
        "ERROR: Could not import Open Babel. Install openbabel (>=3.0).\n"
        "Conda: conda install -c conda-forge openbabel\n"
        "Pip:   pip install openbabel\n"
    )
    raise

# Worker-local converters
_worker_conv_in = None
_worker_conv_inchi_rel = None
_worker_conv_inchi_nostereo = None
_worker_conv_inchikey_std = None

def _worker_init():
    """Create OBConversion objects once per worker."""
    global _worker_conv_in, _worker_conv_inchi_rel, _worker_conv_inchi_nostereo, _worker_conv_inchikey_std
    _worker_conv_in = ob.OBConversion()
    assert _worker_conv_in.SetInFormat("smi")

    _worker_conv_inchi_rel = ob.OBConversion()
    assert _worker_conv_inchi_rel.SetOutFormat("inchi")
    _worker_conv_inchi_rel.AddOption("X", _worker_conv_inchi_rel.OUTOPTIONS, "SRel")      # relative stereo

    _worker_conv_inchi_nostereo = ob.OBConversion()
    assert _worker_conv_inchi_nostereo.SetOutFormat("inchi")
    _worker_conv_inchi_nostereo.AddOption("X", _worker_conv_inchi_nostereo.OUTOPTIONS, "SNon")  # no stereo

    _worker_conv_inchikey_std = ob.OBConversion()
    assert _worker_conv_inchikey_std.SetOutFormat("inchikey")   # standard InChIKey

def _compute_one(record):
    """
    record: (id, smi)
    returns (inchi_simrel, inchi_nostereo, inchikey_std, id)
    """
    _id, smi = record
    mol = ob.OBMol()
    try:
        if not _worker_conv_in.ReadString(mol, smi):
            return (None, None, None, _id)
        inchi_rel = _worker_conv_inchi_rel.WriteString(mol).strip() or None
        inchi_nostereo = _worker_conv_inchi_nostereo.WriteString(mol).strip() or None
        inchikey_std = _worker_conv_inchikey_std.WriteString(mol).strip() or None
        return (inchi_rel, inchi_nostereo, inchikey_std, _id)
    except Exception:
        return (None, None, None, _id)

# ---- DB helpers -------------------------------------------------------------

PRAGMAS = [
    "PRAGMA journal_mode=WAL;",
    "PRAGMA synchronous=NORMAL;",
    "PRAGMA temp_store=MEMORY;",
    "PRAGMA cache_size=-1000000;",   # ~1 GiB page cache; adjust to taste
]

DDL_SMILES_COLS = [
    "ALTER TABLE smiles ADD COLUMN inchi_simrel    TEXT;",
    "ALTER TABLE smiles ADD COLUMN inchi_nostereo  TEXT;",
    "ALTER TABLE smiles ADD COLUMN inchikey_std    TEXT;",
]

DDL_SIM_SCORES_COLS = [
    "ALTER TABLE sim_scores ADD COLUMN is_enantiomer   INTEGER NOT NULL DEFAULT 0;",
    "ALTER TABLE sim_scores ADD COLUMN is_stereoisomer INTEGER NOT NULL DEFAULT 0;",
    "ALTER TABLE sim_scores ADD COLUMN is_diastereomer INTEGER NOT NULL DEFAULT 0;",
]

IDX_CMDS = [
    "CREATE INDEX IF NOT EXISTS idx_smiles_inchi_simrel    ON smiles(inchi_simrel);",
    "CREATE INDEX IF NOT EXISTS idx_smiles_inchi_nostereo  ON smiles(inchi_nostereo);",
    "CREATE INDEX IF NOT EXISTS idx_smiles_inchikey_std    ON smiles(inchikey_std);",
    "CREATE INDEX IF NOT EXISTS idx_sim_scores_1_2 ON sim_scores(smi_1_id, smi_2_id);",
    "CREATE INDEX IF NOT EXISTS idx_sim_scores_2_1 ON sim_scores(smi_2_id, smi_1_id);",
]

def add_missing_columns(conn):
    cur = conn.cursor()
    existing = {r[1] for r in cur.execute("PRAGMA table_info(smiles)")}
    for cmd in DDL_SMILES_COLS:
        col = cmd.split("ADD COLUMN")[1].split()[0]
        if col not in existing:
            cur.execute(cmd)

    existing = {r[1] for r in cur.execute("PRAGMA table_info(sim_scores)")}
    for cmd in DDL_SIM_SCORES_COLS:
        col = cmd.split("ADD COLUMN")[1].split()[0]
        if col not in existing:
            cur.execute(cmd)
    conn.commit()

def ensure_indexes(conn):
    cur = conn.cursor()
    for cmd in IDX_CMDS:
        cur.execute(cmd)
    conn.commit()

# ---- Steps ------------------------------------------------------------------

def compute_smiles_identifiers(db_path, nprocs):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for p in PRAGMAS:
        cur.execute(p)

    add_missing_columns(conn)

    todo = cur.execute("""
        SELECT id, smi
        FROM smiles
        WHERE inchi_simrel   IS NULL
           OR inchi_nostereo IS NULL
           OR inchikey_std   IS NULL
    """).fetchall()

    if not todo:
        print("All SMILES already have inchi_simrel, inchi_nostereo, inchikey_std.")
        conn.close()
        return

    total = len(todo)
    print(f"Computing identifiers for {total:,} SMILES (SRel, SNon, std InChIKey)...")
    t0 = time.time()

    if nprocs > 1:
        with Pool(processes=nprocs, initializer=_worker_init) as pool:
            batch, B = [], 5000
            for i, res in enumerate(pool.imap_unordered(_compute_one, todo, chunksize=1000), 1):
                batch.append(res)
                if len(batch) >= B:
                    with conn:
                        conn.executemany(
                            "UPDATE smiles SET inchi_simrel=?, inchi_nostereo=?, inchikey_std=? WHERE id=?",
                            batch
                        )
                    batch.clear()
                if i % 50_000 == 0:
                    print(f"  processed {i:,}/{total:,}")
            if batch:
                with conn:
                    conn.executemany(
                        "UPDATE smiles SET inchi_simrel=?, inchi_nostereo=?, inchikey_std=? WHERE id=?",
                        batch
                    )
    else:
        _worker_init()
        batch, B = [], 5000
        for i, rec in enumerate(todo, 1):
            batch.append(_compute_one(rec))
            if len(batch) >= B:
                with conn:
                    conn.executemany(
                        "UPDATE smiles SET inchi_simrel=?, inchi_nostereo=?, inchikey_std=? WHERE id=?",
                        batch
                    )
                batch.clear()
            if i % 50_000 == 0:
                print(f"  processed {i:,}/{total:,}")
        if batch:
            with conn:
                conn.executemany(
                    "UPDATE smiles SET inchi_simrel=?, inchi_nostereo=?, inchikey_std=? WHERE id=?",
                    batch
                )

    ensure_indexes(conn)
    n_rel  = conn.execute("SELECT COUNT(*) FROM smiles WHERE inchi_simrel   IS NOT NULL").fetchone()[0]
    n_non  = conn.execute("SELECT COUNT(*) FROM smiles WHERE inchi_nostereo IS NOT NULL").fetchone()[0]
    n_std  = conn.execute("SELECT COUNT(*) FROM smiles WHERE inchikey_std   IS NOT NULL").fetchone()[0]
    print(f"Done: inchi_simrel={n_rel:,}, inchi_nostereo={n_non:,}, inchikey_std={n_std:,} (elapsed {time.time()-t0:.1f}s)")
    conn.close()

def build_enantiomer_pairs(db_path):
    """Persistent (id1,id2) with id1<id2: same SRel, different std InChIKey."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for p in PRAGMAS:
        cur.execute(p)
    ensure_indexes(conn)

    print("Building persistent enant_pairs...")
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
             WHERE s1.inchi_simrel   IS NOT NULL
               AND s1.inchikey_std   IS NOT NULL
               AND s2.inchikey_std   IS NOT NULL;
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ep_21 ON enant_pairs(id2, id1);")
    n_pairs = cur.execute("SELECT COUNT(*) FROM enant_pairs;").fetchone()[0]
    print(f"  enant_pairs: {n_pairs:,} (elapsed {time.time()-t0:.1f}s)")
    conn.close()
    return n_pairs

def build_stereoisomer_pairs(db_path):
    """
    Persistent (id1,id2) with id1<id2: same SNon (no-stereo InChI), different std InChIKey.
    This set includes both enantiomers and diastereomers.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for p in PRAGMAS:
        cur.execute(p)
    ensure_indexes(conn)

    print("Building persistent stereo_pairs (all stereoisomers)...")
    t0 = time.time()
    with conn:
        cur.execute("DROP TABLE IF EXISTS stereo_pairs;")
        cur.execute("""
            CREATE TABLE stereo_pairs(
                id1 INTEGER NOT NULL,
                id2 INTEGER NOT NULL,
                PRIMARY KEY (id1, id2)
            ) WITHOUT ROWID;
        """)
        cur.execute("""
            INSERT INTO stereo_pairs(id1, id2)
            SELECT s1.id, s2.id
              FROM smiles s1
              JOIN smiles s2
                ON s1.inchi_nostereo = s2.inchi_nostereo
               AND s1.inchikey_std  <> s2.inchikey_std
               AND s1.id < s2.id
             WHERE s1.inchi_nostereo IS NOT NULL
               AND s1.inchikey_std   IS NOT NULL
               AND s2.inchikey_std   IS NOT NULL;
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sp_21 ON stereo_pairs(id2, id1);")
    n_pairs = cur.execute("SELECT COUNT(*) FROM stereo_pairs;").fetchone()[0]
    print(f"  stereo_pairs: {n_pairs:,} (elapsed {time.time()-t0:.1f}s)")
    conn.close()
    return n_pairs

def update_sim_scores(db_path):
    """
    Mark flags in sim_scores using efficient joins once per flag and
    compute diastereomers from the two flags (no extra lookups):

      is_enantiomer   := 1 if (smi_1_id, smi_2_id) in enant_pairs (any order)
      is_stereoisomer := 1 if (smi_1_id, smi_2_id) in stereo_pairs (any order)
      is_diastereomer := is_stereoisomer * (1 - is_enantiomer)
    """
    import sqlite3, time
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Pragmas + indexes help the joins stay index-driven
    for p in PRAGMAS:
        cur.execute(p)
    add_missing_columns(conn)
    ensure_indexes(conn)

    print("Updating sim_scores flags (no repeated lookups for diastereomers)...")
    t0 = time.time()
    with conn:
        # ENANTIOMERS: set once using a UNION to cover both (id1,id2) and (id2,id1)
        cur.execute("""
            UPDATE sim_scores
               SET is_enantiomer = 1
             WHERE rowid IN (
                SELECT s.rowid
                  FROM sim_scores AS s
                  JOIN enant_pairs AS p
                    ON p.id1 = s.smi_1_id AND p.id2 = s.smi_2_id
                UNION
                SELECT s.rowid
                  FROM sim_scores AS s
                  JOIN enant_pairs AS p
                    ON p.id1 = s.smi_2_id AND p.id2 = s.smi_1_id
             );
        """)

        # STEREOISOMERS: likewise, one UNION query for both orientations
        cur.execute("""
            UPDATE sim_scores
               SET is_stereoisomer = 1
             WHERE rowid IN (
                SELECT s.rowid
                  FROM sim_scores AS s
                  JOIN stereo_pairs AS p
                    ON p.id1 = s.smi_1_id AND p.id2 = s.smi_2_id
                UNION
                SELECT s.rowid
                  FROM sim_scores AS s
                  JOIN stereo_pairs AS p
                    ON p.id1 = s.smi_2_id AND p.id2 = s.smi_1_id
             );
        """)

        # DIASTEREOMERS: derive purely from the two flags (no joins)
        # Use arithmetic so result is 0/1 and avoid rewriting unchanged rows.
        cur.execute("""
            UPDATE sim_scores
               SET is_diastereomer = is_stereoisomer * (1 - is_enantiomer)
             WHERE is_diastereomer <> is_stereoisomer * (1 - is_enantiomer);
        """)

    # Quick counts
    n_en  = cur.execute("SELECT COUNT(*) FROM sim_scores WHERE is_enantiomer=1").fetchone()[0]
    n_st  = cur.execute("SELECT COUNT(*) FROM sim_scores WHERE is_stereoisomer=1").fetchone()[0]
    n_dia = cur.execute("SELECT COUNT(*) FROM sim_scores WHERE is_diastereomer=1").fetchone()[0]
    print(f"Marked rows  enantiomer={n_en:,},  stereoisomer={n_st:,},  diastereomer={n_dia:,} "
          f"(elapsed {time.time()-t0:.1f}s)")
    conn.close()





# def update_sim_scores(db_path):
#     """
#     Set sim_scores flags using EXISTS joins (order-independent), driven by pair tables.
#       - is_enantiomer   = 1 for pairs in enant_pairs
#       - is_stereoisomer = 1 for pairs in stereo_pairs
#       - is_diastereomer = 1 for pairs in stereo_pairs but NOT in enant_pairs
#     """
#     conn = sqlite3.connect(db_path)
#     cur = conn.cursor()
#     for p in PRAGMAS:
#         cur.execute(p)

#     add_missing_columns(conn)
#     ensure_indexes(conn)

#     print("Updating sim_scores flags via indexed EXISTS joins...")
#     t0 = time.time()
#     with conn:
#         # ENANTIOMERS (both orders)
#         cur.execute("""
#             UPDATE sim_scores AS s
#                SET is_enantiomer = 1
#              WHERE EXISTS (SELECT 1 FROM enant_pairs p
#                             WHERE p.id1 = s.smi_1_id AND p.id2 = s.smi_2_id);
#         """)
#         cur.execute("""
#             UPDATE sim_scores AS s
#                SET is_enantiomer = 1
#              WHERE EXISTS (SELECT 1 FROM enant_pairs p
#                             WHERE p.id1 = s.smi_2_id AND p.id2 = s.smi_1_id);
#         """)

#         # STEREOISOMERS (includes enantiomers and diastereomers)
#         cur.execute("""
#             UPDATE sim_scores AS s
#                SET is_stereoisomer = 1
#              WHERE EXISTS (SELECT 1 FROM stereo_pairs p
#                             WHERE p.id1 = s.smi_1_id AND p.id2 = s.smi_2_id);
#         """)
#         cur.execute("""
#             UPDATE sim_scores AS s
#                SET is_stereoisomer = 1
#              WHERE EXISTS (SELECT 1 FROM stereo_pairs p
#                             WHERE p.id1 = s.smi_2_id AND p.id2 = s.smi_1_id);
#         """)

#         # DIASTEREOMERS: in stereo_pairs but NOT in enant_pairs (both orders)
#         cur.execute("""
#             UPDATE sim_scores AS s
#                SET is_diastereomer = 1
#              WHERE EXISTS (SELECT 1 FROM stereo_pairs sp
#                             WHERE sp.id1 = s.smi_1_id AND sp.id2 = s.smi_2_id)
#                AND NOT EXISTS (SELECT 1 FROM enant_pairs ep
#                                 WHERE ep.id1 = s.smi_1_id AND ep.id2 = s.smi_2_id);
#         """)
#         cur.execute("""
#             UPDATE sim_scores AS s
#                SET is_diastereomer = 1
#              WHERE EXISTS (SELECT 1 FROM stereo_pairs sp
#                             WHERE sp.id1 = s.smi_2_id AND sp.id2 = s.smi_1_id)
#                AND NOT EXISTS (SELECT 1 FROM enant_pairs ep
#                                 WHERE ep.id1 = s.smi_2_id AND ep.id2 = s.smi_1_id);
#         """)

#     n_en  = cur.execute("SELECT COUNT(*) FROM sim_scores WHERE is_enantiomer=1").fetchone()[0]
#     n_st  = cur.execute("SELECT COUNT(*) FROM sim_scores WHERE is_stereoisomer=1").fetchone()[0]
#     n_dia = cur.execute("SELECT COUNT(*) FROM sim_scores WHERE is_diastereomer=1").fetchone()[0]
#     print(f"Marked rows  enantiomer={n_en:,},  stereoisomer={n_st:,},  diastereomer={n_dia:,} (elapsed {time.time()-t0:.1f}s)")
#     conn.close()

# ---- CLI --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute InChI identifiers and mark enantiomer/stereo/diastereo flags in sim_scores.")
    ap.add_argument("db", help="Path to Seed3_contrast_sim.sqlite3")
    ap.add_argument("--nprocs", type=int, default=max(1, cpu_count() // 2),
                    help="Parallel worker processes for SMILES conversion (default: half cores)")
    args = ap.parse_args()

    compute_smiles_identifiers(args.db, nprocs=args.nprocs)
    build_stereoisomer_pairs(args.db)   # must run before diastereo logic
    build_enantiomer_pairs(args.db)
    update_sim_scores(args.db)

if __name__ == "__main__":
    main()
