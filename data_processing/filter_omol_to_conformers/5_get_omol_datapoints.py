import pickle
import glob
import os
import argparse
import pandas as pd
from tqdm import tqdm
import lmdb
from functools import lru_cache
import numpy as np
from fairchem.core.datasets import AseDBDataset
from multiprocessing import Process, Queue
import sqlite3

PATH_TO_BASE_DIR = "/home/mpklein/scratch/combined_dataset"
BATCHSIZE= 10_000

import contextlib

def generator_func(db_name, table_name):
    """
    Generator function to iterate over all rows in a given SQLite table.

    Args:
        db_name (str): Path to the SQLite database file.
        table_name (str): Name of the table to iterate over.

    Yields:
        tuple: Each row in the table as a tuple.
    """
    # Connect to the database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Execute query to fetch all rows
    query = f"SELECT * FROM {table_name}"
    cursor.execute(query)
    
    # Yield one row at a time
    for row in cursor:
        yield row
    
    # Close the connection
    conn.close()


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def args_load_codex(parser):
    parser.add_argument(
        "--codex-dir",
        help="Directory which contains the codex for routing",
        )
    parser.add_argument(
        "--codex-name",
        default="codex",
        help="name of the codex (without the .pkl extension on the end)"
    )
    return parser


def args_route_key(parser):
    parser.add_argument(
        "--db-dir",
        help="directory with all of the data files",
    )
    parser.add_argument(
        "--db-extension",
        default="conformer.pkl",
        help="extension uniquely identifying the database files"
    )
    return parser

def load_codex(args):
    path_to_parser = os.path.join(args.codex_dir, args.codex_name + ".pkl")
    return dict(pickle.load(open(path_to_parser, "rb")))


@lru_cache(maxsize=16)
def route_key(db_dir, db_name):
    path_to_db_part = os.path.join(db_dir, db_name)
    return pickle.load(open(path_to_db_part, "rb"))


def get_db_id(codex, smi):
    try:
        db_id = codex[smi]
    except KeyError:
        db_id = None
    return db_id


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


def get_omol_ids(smi_string: str, codex: dict[str, str], db_dir: str, db_extension: str) -> list[int]:
    """
    Gets the IDs of a smiles string within the omol database.

    args:
      codex (dict): dictionary for which smiles string is in which database part
      smi_string (str): smiles string to query
      db_dir (str): directory of the database. From arguments parser
      db_extension (str): Extension of the database, 

    returns:
      list of omol IDs. If the smiles string is not in the omol dataset the the list is empty 
    """

    db_id = get_db_id(codex, smi_string)
    if db_id is None:
        return []
    else:
        output = []
        db_name = db_id + "-" + db_extension
        db_part = route_key(db_dir, db_name)
        
        data = db_part[smi_string]
        for conf in data:
            output.append(conf["OMol_db_id"])
        return output


def process_to_lmdb_format(smi, omol_atoms: list) -> dict:
    # get the atoms to output
    atoms_output = sorted(omol_atoms[0].get_chemical_symbols())
    
    # get the coords to output
    coords_list = []
    for a in omol_atoms:
        sorted_indicies = sorted(range(len(a.get_chemical_symbols())), key=lambda i: a.get_chemical_symbols()[i])
        coords = a.get_positions()
        sorted_coords = coords[sorted_indicies]
        coords_list.append(sorted_coords.astype(np.float32))

    return {'atoms': atoms_output, 'coordinates': coords_list, 'smi': smi}


def worker(worker_id, input_queue, output_queue, train_dataset, valid_dataset, codex, args):
    print(f"started worker number {worker_id}", flush=True)
    aggregated_smi = []
    conn = sqlite3.connect("codex.sqlite3")
    cur = conn.cursor()

    while True:
        d = input_queue.get()
        if d is None:
            print(f"killed worker number {worker_id}", flush=True)
            break
        
        smi, t = d
        partition = t
        dataset_name = "train"
        if dataset_name == "train":
            dataset: AseDBDataset = train_dataset
            aggregated_smi.append(smi)

            if len(aggregated_smi):
            omol_id_to_get = []
            cur.execute("""
                SELECT p.partition
                FROM codex c
                JOIN partitions p ON c.p_id = p.id
                WHERE c.smi = ?
            """, (smi,))
            results = [row[0] for row in cur.fetchall()]
            partition = results[0]
            db_name = partition + "-" + args.db_extension
            
            db_part = route_key(args.db_dir, db_name)
            data = db_part[smi]
            for conf in data:
                omol_id_to_get.append(conf["OMol_db_id"])
        
            if len(omol_id_to_get) > 10:
                with numpy_seed(42, len(omol_id_to_get)):
                    omol_id_to_get = np.random.choice(omol_id_to_get, size=10, replace=False)

            if len(omol_id_to_get) > 1:
                omol_atoms_objects = [dataset.get_atoms(i) for i in omol_id_to_get]
                lmdb_dict = process_to_lmdb_format(smi, omol_atoms_objects)
                lmdb_dict['origin'] = f"{dataset_name}_{partition}"
                output_queue.put((smi, pickle.dumps(lmdb_dict)))
        


def writer(writer_id, lmdb_name, output_queue):
    print(f"started writer {writer_id}", flush=True)
    env = lmdb.open(lmdb_name, subdir=False, lock=False, map_size=100 *1024**3)

    def batch_write(batch):
        """
        assumes everything is already in bytes
        """
        with env.begin(write=True) as txn:
            for key, val in batch.items():
                txn.put(key, val)
    
    batch = {}
    while True:
        d = output_queue.get()
        if d is None:
            print(f"killed writer {writer_id}", flush=True)
            break
            
        key, val = d
        batch[key.encode("utf-8")] = val
        if len(batch) >= BATCHSIZE:
            print("write", flush=True)
            batch_write(batch)
            batch = {}
    
    if len(batch) != 0:
        batch_write(batch)


def main(args):
    # foreplay
    codex = None
    # data_to_get = pickle.load(open(os.path.join(PATH_TO_BASE_DIR, "NewCombinedDataset_OMol_portion.pkl"), "rb"))
    print("loaded file", flush=True)

    dataset_path_train = "./train"
    dataset_train = AseDBDataset({"src": dataset_path_train})
    print("loaded omol train", flush=True)
    dataset_path_valid = "./val"
    # dataset_valid = AseDBDataset({"src": dataset_path_valid})
    dataset_valid = None
    print("loaded omol valid", flush=True)


    # start of processing
    input_queue = Queue(maxsize=1_000)
    output_queue = Queue(maxsize=1_000)

    writer_process_1 = Process(target=writer, args=(1, os.path.join(PATH_TO_BASE_DIR, "OMOL_part1_take2.lmdb"), output_queue))
    writer_process_1.start()
    # writer_process_2 = Process(target=writer, args=(1, os.path.join(PATH_TO_BASE_DIR, "OMOL_part2.lmdb"), output_queue))
    # writer_process_2.start()

    n_workers = 1
    worker_processes =[]
    for i in range(n_workers):
        p = Process(target=worker, args=(i, input_queue, output_queue, dataset_train, dataset_valid, codex, args))
        p.start()
        worker_processes.append(p)
    
    # get the jobs going
    for t in tqdm(generator_func(os.path.join(PATH_TO_BASE_DIR, "NewCombinedDataset_OMol_portion.splite3"), "datapoints"), desc="Processed"):
        smi = t[0]
        t = t[1:]
        input_queue.put((smi, t))
    
    # wait for them to finish and kill processes
    for _ in range(n_workers):
        input_queue.put(None)
    
    for w in worker_processes:
        w.join()
    
    output_queue.put(None)
    writer_process_1.join()

    print("everything done", flush=True)


if __name__ == "__main__":
    omol_dir = "path/to/omol_dataset"
    os.chdir(omol_dir)
    parser = argparse.ArgumentParser()
    parser_functions = [args_route_key, args_load_codex]
    for f in parser_functions:
        parser = f(parser)
    args = parser.parse_args()

    main(args)