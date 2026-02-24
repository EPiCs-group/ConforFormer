import lmdb
import sqlite3
import pickle
from tqdm import tqdm


def iter_formula_to_main(db_path):
    """
    Generator that yields (formula_id, lmdb_id, [main_ids])
    for each formula in the database.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Ensure index exists for performance
    cur.execute("CREATE INDEX IF NOT EXISTS idx_main_formula_id ON main(formula_id);")
    conn.commit()

    # Iterate through all formulas
    cur.execute("SELECT id, new_id FROM formulas;")
    for formula in cur:
        formula_id = formula["id"]
        lmdb_id = formula["new_id"]

        # Collect all main IDs for this formula
        cur2 = conn.cursor()
        cur2.execute("SELECT id FROM main WHERE formula_id = ?;", (formula_id,))
        main_ids = [row["id"] for row in cur2.fetchall()]
        cur2.close()

        yield (formula_id, lmdb_id, main_ids)

    conn.close()


def main():
    sqlite_db_name = "organized_data_train.sqlite3"
    lmdb_name = "isomers_train.lmdb"

    env = lmdb.open(lmdb_name, subdir=False, lock=False, map_size=20 * 1024**3)
    with env.begin(write=True) as txn:
        for formula_id, lmdb_id, main_ids in tqdm(iter_formula_to_main(sqlite_db_name), desc="Processed"):
            if formula_id != -1:
                key = f"{lmdb_id}".encode("utf-8")
                value = pickle.dumps(main_ids)
                txn.put(key, value)

    print("done")
if __name__ == "__main__":
    main()