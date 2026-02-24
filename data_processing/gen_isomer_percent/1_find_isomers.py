from openbabel import openbabel as ob
import multiprocessing as mp
from tqdm import tqdm
import sqlite3
import pickle
import lmdb
import re

ob.obErrorLog.SetOutputLevel(0)


def smiles_to_inchi(smiles: str) -> str:
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("smi", "inchi")
    mol = ob.OBMol()
    conv.ReadString(mol, smiles)
    return conv.WriteString(mol).strip()

def smiles_to_inchikey(smiles: str) -> str:
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("smi", "inchikey")
    mol = ob.OBMol()
    conv.ReadString(mol, smiles)
    return conv.WriteString(mol).strip()

def get_block1(inchikey: str) -> str:
    # InChIKey format: AAAAAAAAAAAAAA-BBBBBBBB-C
    return inchikey.split("-")[0]

def get_t_layer(inchi: str) -> str:
    m = re.search(r"/t[^/]+", inchi)
    return m.group(0) if m else ""

def smiles_info(smiles: str):
    inchi = smiles_to_inchi(smiles)
    inchikey = smiles_to_inchikey(smiles)
    block1 = get_block1(inchikey)
    t_layer = get_t_layer(inchi)
    return block1, t_layer


def worker(worker_id: int, input_queue: mp.Queue, output_queue: mp.Queue):
    print(f"starting {worker_id=}", flush=True)
    while True:
        p = input_queue.get()
        if p is None:
            break
        
        
        lmdb_id, pkl_data = p
        data = pickle.loads(pkl_data)
        
        lmdb_id = int(lmdb_id.decode())
        atoms = data["atoms"]
        formula = "".join(sorted(atoms))
        smi = data["smi"]

        if "." in smi:
            inchikey_block = "NoDo"
            t_layer = "NoDo"
        else:
            inchikey_block, t_layer = smiles_info(smi)
    
        output_queue.put((lmdb_id, formula, smi, inchikey_block, t_layer))
    print("worker killed", flush=True)

def writer(writer_id, input_queue: mp.Queue, db_name):
    print(f"starting {writer_id=}")
    with sqlite3.connect(db_name + ".sqlite3") as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS formulas (
                id INTEGER PRIMARY KEY,
                value TEXT UNIQUE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS inchikey_blocks (
                id INTEGER PRIMARY KEY,
                value TEXT UNIQUE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS main (
                id INTEGER PRIMARY KEY,
                formula_id INTEGER,
                smi TEXT,
                inchi_id INTEGER,
                t_layer TEXT,
                FOREIGN KEY (formula_id) REFERENCES formulas(id),
                FOREIGN KEY (inchi_id) REFERENCES inchikey_blocks(id)
            )
        """)
    
    def write_batch(batch: list[tuple], form_set, inchi_set):
        with sqlite3.connect(db_name + ".sqlite3") as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            cur = conn.cursor()

            cur.executemany("""
                INSERT OR IGNORE INTO formulas(value) VALUES (?)
            """, ((f,) for f in form_set)
            )
            cur.executemany("""
                INSERT OR IGNORE INTO inchikey_blocks(value) VALUES (?)
            """, ((c,) for c in inchi_set)
            )

            # formulas_set and connectivity_set are sets of unique strings in the batch
            cur.execute(
                f"SELECT value, id FROM formulas WHERE value IN ({','.join(['?']*len(form_set))})",
                tuple(form_set)
            )
            formula_ids = {v: i for v, i in cur.fetchall()}

            cur.execute(
                f"SELECT value, id FROM inchikey_blocks WHERE value IN ({','.join(['?']*len(inchi_set))})",
                tuple(inchi_set)
            )
            inchi_ids = {v: i for v, i in cur.fetchall()}

            new_rows = [
                (db_id, formula_ids[formula], smi, inchi_ids[inchi_block], t_layer) 
                for db_id, formula, smi, inchi_block, t_layer in batch
            ]
            cur.executemany("""
                INSERT INTO main (id, formula_id, smi, inchi_id, t_layer) VALUES (?, ?, ?, ?, ?)""", 
                new_rows
            )

    batch = []
    form_set = set()
    inchi_set = set()
    while True:
        p = input_queue.get()
        if p is None:
            if len(batch) != 0:
                write_batch(batch=batch, form_set=form_set, inchi_set=inchi_set)
                batch = []
                form_set = set()
                inchi_set = set()
            break
        
        lmdb_id, formula, smi, inchikey_block, t_layer = p
        
        form_set.add(formula)
        inchi_set.add(inchikey_block)
        batch.append((lmdb_id, formula, smi, inchikey_block, t_layer))
        if len(batch) >= 50_000:
            write_batch(batch=batch, form_set=form_set, inchi_set=inchi_set)
            batch = []
            form_set = set()
            inchi_set = set()
    print("writer killed", flush=True)

def main():
    lmdb_name = "train.lmdb"
    sqlite_name = "organized_data_train"

    n_workers = 30
    to_worker_queue = mp.Queue(maxsize=n_workers*10)
    to_writer_queue = mp.Queue(maxsize=n_workers*10)

    workers = []
    for i in range(n_workers):
        w = mp.Process(target=worker, args=(i, to_worker_queue, to_writer_queue))
        w.start()
        workers.append(w)
    
    write = mp.Process(target=writer, args=(0, to_writer_queue, sqlite_name))
    write.start()

    env = lmdb.open(lmdb_name, subdir=False, lock=False, readonly=True)
    with env.begin() as txn:
        for key, value in tqdm(txn.cursor()):
            to_worker_queue.put((key, value))

    for _ in workers:
        to_worker_queue.put(None)
    for w in workers:
        w.join()

    to_writer_queue.put(None)
    write.join()

    print("processed full db", flush=True)


if __name__ == "__main__":
    main()