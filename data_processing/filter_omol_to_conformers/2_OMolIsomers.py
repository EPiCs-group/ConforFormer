from multiprocessing import Process, Queue, cpu_count
from collections import defaultdict
import pickle
import lmdb
import os
from tqdm import tqdm
from scipy.spatial import KDTree
import sys
import time
# from fairchem.core.datasets import AseDBDataset


symbol_to_atomic_number = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58,
    "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
    "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
    "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98,
    "Es": 99, "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105,
    "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112,
    "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
}


OUTPUT_DIR = "."
FINAL_OUTPUT = "Formulas_OMol.pkl"
LMDB_PATH = "."
# dataset_path = "./val"
# dataset = AseDBDataset({"src": dataset_path})
# MOL_IDS = [val for val in range(len(dataset))]
    

def stringify_dp(atoms):
    return "".join(sym for sym in sorted(atoms))


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


def process_key_worker(worker_id, input_queue, output_queue):
    env = init_lmdb_env("val_lmdbs/larger_than_5.lmdb")
    with env.begin() as txn:
        while True:
            key = input_queue.get()
            if key is None:
                break

            atoms = pickle.loads(txn.get(key))['atoms']
            output_queue.put((atoms, key))
        print("process killed")


def collector(result_queue, output_queue, n_workers):
    finished = 0
    result_dict = defaultdict(list)
    while finished < n_workers:
        item = output_queue.get()
        if item is None:
            finished += 1
        else:
            atoms, key = item
            result_dict[stringify_dp(atoms)].append(key)
    
    print("collector killed")
    result_queue.put(result_dict)


def main():
    # keys = MOL_IDS
    n_workers = 4

    input_queue = Queue(maxsize=1000)
    output_queue = Queue(maxsize=10_000)
    result_queue = Queue()

    # Start workers
    workers = []
    for i in range(n_workers):
        p = Process(target=process_key_worker, args=(i, input_queue, output_queue))
        p.start()
        workers.append(p)

    # Start collector
    collector_proc = Process(target=collector, args=(result_queue, output_queue, n_workers))
    collector_proc.start()

    # Feed keys into input queue
    env = init_lmdb_env("val_lmdbs/larger_than_5.lmdb")
    with env.begin() as txn:
        for key, _ in tqdm(txn.cursor(), desc="Queuing keys"):
            input_queue.put(key)

    # Send sentinel to workers
    for _ in range(n_workers):
        input_queue.put(None)

    # Wait for workers to finish
    for p in workers:
        p.join()

    # Send sentinel to collector
    for _ in range(n_workers):
        output_queue.put(None)
    time.sleep(1)
    
    result_dict = result_queue.get()
    time.sleep(1)
    collector_proc.join()
    
    # Save result
    print("Writing final set")
    with open(FINAL_OUTPUT, "wb") as f:
        pickle.dump(result_dict, f)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")  # Especially important for macOS or Windows
    main()
