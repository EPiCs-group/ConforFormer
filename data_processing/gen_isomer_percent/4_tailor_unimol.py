import pickle
import lmdb
import sqlite3
from tqdm import tqdm
import multiprocessing as mp


def init_lmdb_env(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    return env


def worker(worker_id, input_queue, output_queue, sqlite_db):
    print(f"started {worker_id}", flush=True)
    conn = sqlite3.connect(f"file:{sqlite_db}?mode=ro", uri=True)
    cur = conn.cursor()

    while True:
        p = input_queue.get()
        if p is None:
            break
        
        db_id, pkl_data = p
        cur.execute("""
            SELECT m.id, m.formula_id, m.inchi_id, m.t_layer, m.enantiomer_id, f.value, f.new_id
            FROM main m
            JOIN formulas f ON m.formula_id = f.id
            WHERE m.id = ?
        """, (db_id,))
        row = cur.fetchone()
        _, _, _, _, enantiomer_id, _, isomer_id = row
        pkl_data["isomer_id"] = isomer_id
        pkl_data["enantiomer_id"] = enantiomer_id if enantiomer_id is not None else -1

        altered_data = pickle.dumps(pkl_data)
        str_key = f"{db_id}".encode("utf-8")
        output_queue.put((str_key, altered_data))

def writer(writer_id, input_queue, new_db_name):
    print(f"started {writer_id=}", flush=True)

    env = lmdb.open(new_db_name, subdir=False, lock=False, map_size=200 * 1024**3)
    def write_batch(batch):
        with env.begin(write=True) as txn:
            for key, val in batch.items():
                txn.put(key, val)

    batch = {}
    while True:
        p = input_queue.get()
        if p is None:
            if len(batch) != 0:
                write_batch(batch)
                batch = {}
            break

        key, val = p
        batch[key] = val
        if len(batch) >= 25_000:
            write_batch(batch)
            batch = {}
    

def main():
    new_lmdb_name = "altered_train.lmdb"
    sql_db_name = "organized_data_train.sqlite3"
    n_workers = 30
    to_workers_queue = mp.Queue(maxsize=n_workers * 10)
    to_writer_queue = mp.Queue(maxsize=n_workers * 10)

    workers = []
    for i in range(n_workers):
        w = mp.Process(target=worker, args=(i, to_workers_queue, to_writer_queue, sql_db_name))
        w.start()
        workers.append(w)

    writing = mp.Process(target=writer, args=(0, to_writer_queue, new_lmdb_name))
    writing.start()

    env = init_lmdb_env("train.lmdb")
    with env.begin() as txn:
        for key, val in tqdm(txn.cursor(), desc="Process"):
            key = int(key.decode())
            val = pickle.loads(val)
            to_workers_queue.put((key, val))
    
    for _ in range(n_workers):
        to_workers_queue.put(None)
    for w in workers:
        w.join()

    to_writer_queue.put(None)
    writing.join()

    print("done", flush=True)

if __name__ == "__main__":
    main()