from argparse import ArgumentParser
import logging

from multiprocessing import Process, Queue
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import lmdb
import pickle

from conf_gen.rdkit_gen import generate_conformers
from conf_gen.rdkit_write import write_to_np_array


def get_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger(name)


def init_lmdb_env(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=256,
    )
    return env


# lmdb structure we aim for:
# lmdb[formula] = pickle.loads({smi: (atoms, coords)})

def worker(worker_id, input_queue, output_queue):
    log = get_logger("worker")
    log.info(f"Starting number {worker_id}")

    while True:
        p = input_queue.get()
        if p is None:
            break

        _, smi_string, _ = p
        if "." in smi_string:
            continue
        
        rdkit_confs = generate_conformers(smi_string, num_threads=8)
        if not rdkit_confs is None:
            atoms, coords = write_to_np_array(rdkit_confs)
            output_queue.put((smi_string, atoms, coords))


def writer(output_queue, env, formula):
    log = get_logger("writer")
    log.info("Starting process")

    data = {}
    summary = defaultdict(int)
    while True:
        p = output_queue.get()
        if p is None:
            break

        smi, atoms, coords = p
        data[smi] = (atoms, coords)
        summary["total"] += 1
        summary[len(coords)] += 1

    log.info(f"Writing lmdb entry for {formula}")
    with env.begin(write=True) as txn:
        txn.put(formula.encode("utf-8"), pickle.dumps(data))
    log.info(f"Write done")


def main(partition_number, formulas, num_workers):
    log = get_logger("main")
    log.info(
        f"Start dataset parition {partition_number} with {len(formulas)} chemical formulas",
    )

    lmdb_name = f"dataset_partition_{partition_number}.lmdb"
    env_for_write = lmdb.open(lmdb_name, map_size=75*1024**3, lock=False, readonly=False, subdir=False)

    tot = 0
    env = init_lmdb_env("ZINC_SMILES.lmdb")
    for formula in formulas:
        input_queue = Queue()
        output_queue = Queue()

        workers = []
        for i in range(num_workers):
            p = Process(target=worker, args=(i, input_queue, output_queue))
            p.start()
            workers.append(p)

        writer_process = Process(target=writer, args=(output_queue, env_for_write, formula))
        writer_process.start()

        with env.begin() as txn:
            list_smi = pickle.loads(txn.get(formula.encode("utf-8")))
            for smi in list_smi:
                input_queue.put(smi)
               
        # kill and wait for workers
        for _ in range(num_workers):
            input_queue.put(None)

        for w in workers:
            w.join()

        # kill and wait for writer
        output_queue.put(None)
        writer_process.join()

        log.info(f"Done with formula {formula}")
    
    log.info("job done")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--partition-number",
        type=int,
        help="key partition in the 'split_of_formulas.pkl' file"
        )
    parser.add_argument(
        "--num-workers",
        type=int
    )
    args = parser.parse_args()

    formula_split = pickle.load(open("split_of_formulas.pkl", "rb"))
    partition = formula_split[args.partition_number]
    del formula_split

    main(args.partition_number, partition, args.num_workers)
