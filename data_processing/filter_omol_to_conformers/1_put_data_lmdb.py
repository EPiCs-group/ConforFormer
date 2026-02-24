import logging
import os
import pickle
import lmdb
import numpy as np
import multiprocessing as mp
from fairchem.core.datasets import AseDBDataset

# -------------------- Logger Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(processName)s | %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# ------------------------------------------------------

d_and_f_block_elements = {
    # d-block (transition metals)
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
    # f-block (lanthanides)
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
    'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    # f-block (actinides)
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'
}

main_group_elements = {
    'Li', 'Be', 'B', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Ga', 'Ge', 'As', 'Se',
    'Rb', 'Sr', 'In', 'Sn', 'Sb', 'Te', 'Cs', 'Ba', 'Tl', 'Pb', 'Bi', 'Po',
    'At', 'Rn', 'Fr', 'Ra', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
}

dataset_path = "./val"
dataset = AseDBDataset({"src": dataset_path})
MOL_IDS = [val for val in range(len(dataset))]

db_names = {
    0: "all_data.lmdb",
    1: "all_data_No_H.lmdb",
    2: "larger_than_5.lmdb",
    3: "d_f_block.lmdb",
    4: "spicy_main_group.lmdb",
    5: "main_and_d.lmdb"
}
map_sizing_GB = {0: 500, 1: 500, 2: 500, 3: 150, 4: 150, 5: 150}
BYTES_TO_GB = 1024 * 1024 * 1024
BATCH_SIZE = 1_000


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def writer(db_id, queue, db_dir):
    db_path = os.path.join(db_dir, db_names[db_id])
    env = lmdb.open(db_path, subdir=False, lock=False, map_size=map_sizing_GB[db_id] * BYTES_TO_GB)
    key = 0
    buffer = []

    while True:
        item = queue.get()
        if item is None:
            break
        buffer.append(item)

        if len(buffer) >= BATCH_SIZE:
            with env.begin(write=True) as txn:
                for value in buffer:
                    txn.put(f"{key}".encode('ascii'), value)
                    key += 1
                    if key % 100_000 == 0:
                        logger.info(f"Writer {db_id} has written {key} entries")
            buffer.clear()

    if len(buffer) != 0:
        with env.begin(write=True) as txn:
            for value in buffer:
                txn.put(f"{key}".encode('ascii'), value)
                key += 1

    logger.info(f"Writer {db_id} has closed")
    env.close()


def router(main_queue, dq_queues):
    while True:
        item = main_queue.get()
        if item is None:
            break
        db_id, value = item
        dq_queues[db_id].put(value)

    # Stop each writer
    for q in dq_queues.values():
        q.put(None)


def worker(main_queue, chunk, worker_id):
    logger.info(f"Opening worker {worker_id}")
    for i, mol_id in enumerate(chunk):
        atoms = dataset.get_atoms(mol_id)
        syms = atoms.get_chemical_symbols()
        if not np.allclose(atoms.pbc, [False, False, False]):
            continue

        to_route = {'atoms': syms, "coordinates": [np.array(atoms.get_positions(), dtype=np.float32)], 'smi': "C", 'OMol_id': mol_id}
        to_route = pickle.dumps(to_route)

        no_H = [sym for sym in syms if sym != "H"]
        has_d_block = any(sym in d_and_f_block_elements for sym in syms)
        has_main_group = any(sym in main_group_elements for sym in syms)

        main_queue.put((0, to_route))
        if no_H:
            main_queue.put((1, to_route))
        if len(syms) > 4 and len(no_H) >= 3:
            main_queue.put((2, to_route))
        if has_d_block:
            main_queue.put((3, to_route))
        if has_main_group:
            main_queue.put((4, to_route))
        if has_d_block or has_main_group:
            main_queue.put((5, to_route))

        if i % 500_000 == 0 and i != 0:
            logger.info(f"Worker {worker_id} has finished {i}/{len(chunk)}")

    logger.info(f"Worker {worker_id} has finished")


if __name__ == "__main__":
    db_count = 6
    db_dir = "./val_lmdbs"
    os.makedirs(db_dir, exist_ok=True)

    main_queue = mp.Queue(maxsize=1_000_000)
    db_queues = {i: mp.Queue(maxsize=500_00) for i in range(db_count)}

    # Start writer processes
    writers = []
    for db_id in range(db_count):
        p = mp.Process(target=writer, args=(db_id, db_queues[db_id], db_dir), name=f"Writer-{db_id}")
        p.start()
        writers.append(p)

    # Start router process
    num_routers = 2
    routers = []
    for i in range(num_routers):
        p = mp.Process(target=router, args=(main_queue, db_queues), name=f"Router-{i}")
        p.start()
        routers.append(p)

    # Start worker processes
    num_workers = 9
    mol_id_chunks = split_list(MOL_IDS, num_workers)
    workers = []
    for i in range(num_workers):
        p = mp.Process(target=worker, args=(main_queue, mol_id_chunks[i], i), name=f"Worker-{i}")
        p.start()
        workers.append(p)

    # Wait for workers to finish
    for w in workers:
        w.join()

    main_queue.put(None)
    for r in routers:
        r.join()

    for w in writers:
        w.join()