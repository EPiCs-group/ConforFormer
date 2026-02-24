from multiprocessing import Process, Queue, cpu_count
from collections import defaultdict
import pickle
import lmdb
import os
from tqdm import tqdm
from openbabel import openbabel as ob
import contextlib
import sys
import time
import re
from collections import Counter
from glob import glob

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


@contextlib.contextmanager
def suppress_stderr():
    # Save original stderr file descriptor
    stderr_fileno = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fileno)

    # Redirect stderr to /dev/null
    with open(os.devnull, 'w') as devnull:
        os.dup2(devnull.fileno(), stderr_fileno)
        try:
            yield
        finally:
            # Restore original stderr
            os.dup2(saved_stderr_fd, stderr_fileno)
            os.close(saved_stderr_fd)
    

def convert_to_obmol(symbols, coordinates):
    with suppress_stderr():
        obmol = ob.OBMol()
        for sym, coord in zip(symbols, coordinates):
            obatom = obmol.NewAtom()
            obatom.SetAtomicNum(symbol_to_atomic_number[sym])
            obatom.SetVector(float(coord[0]), float(coord[1]), float(coord[2]))
        
        obmol.ConnectTheDots()
        obmol.AddHydrogens()
        obmol.PerceiveBondOrders()
        return obmol
    

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


def compress_chemical_formula(s):
    # Match chemical symbols: Capital letter followed by optional lowercase
    symbols = re.findall(r'[A-Z][a-z]?', s)
    
    # Count while preserving order
    counts = Counter()
    order = []
    for sym in symbols:
        if sym not in counts:
            order.append(sym)
        counts[sym] += 1
    
    return ''.join(f"{sym}{counts[sym]}" for sym in order)


def process_key_worker(worker_id, input_queue, output_queue):
    worker_id += 64
    conv = ob.OBConversion()
    conv.SetOutFormat("can")
    
    conformer_dict = defaultdict(list)
    lmdb_to_OMol = defaultdict(list)

    env = init_lmdb_env("val_lmdbs/larger_than_5.lmdb")
    
    num_files = 0
    with env.begin() as txn:
        while True:
            key_val_pair = input_queue.get()
            if key_val_pair is None:
                break
            
            key, db_idx = key_val_pair
            if len(key) <= 3: # If there are less than 3 atoms there should be no conformers. 
                continue

            for idx in db_idx:
                datapoint = pickle.loads(txn.get(idx))
                atoms = datapoint['atoms']
                coordinates = datapoint['coordinates'][0]

                obmol = convert_to_obmol(atoms, coordinates)
                if len(atoms) != obmol.NumAtoms():
                    print(f"!!!found bad {mol_id=}", flush=True)
                    continue
                
                smiles = conv.WriteString(obmol).strip()
                conformer_dict[smiles].append(idx)
            
                OMol_id = datapoint["OMol_id"]
                lmdb_to_OMol[smiles].append({"lmdb": idx, "OMol_db_id": OMol_id})

            if len(conformer_dict) >= 50_000:
                with open(f"conformer_pkl_files/{worker_id}_{num_files}-conformer.pkl", "wb") as f:
                    pickle.dump(conformer_dict, f)
                with open(f"lmdb_to_OMol_ids/{worker_id}_{num_files}-conformer.omol", "wb") as f:
                    pickle.dump(lmdb_to_OMol, f)
                
                num_files += 1
                conformer_dict = defaultdict(list)
                lmdb_to_OMol = defaultdict(list)
        
        if len(conformer_dict) != 0:
            with open(f"conformer_pkl_files/{worker_id}_{num_files}-conformers.pkl", "wb") as f:
                pickle.dump(conformer_dict, f)
            with open(f"lmdb_to_OMol_ids/{worker_id}_{num_files}-conformer.omol", "wb") as f:
                pickle.dump(lmdb_to_OMol, f)
                
        print("process killed")


def main():
    n_workers = 8
    data = pickle.load(open("Formulas_OMol.pkl", "rb"))

    input_queue = Queue(maxsize=1000)
    output_queue = Queue(maxsize=10_000)
    result_queue = Queue()
    
    # Start workers
    workers = []
    for i in range(n_workers):
        p = Process(target=process_key_worker, args=(i, input_queue, output_queue))
        p.start()
        workers.append(p)


    # Feed keys into input queue
    for key, vals in tqdm(data.items(), desc="Queuing keys"):
        input_queue.put((key, vals))

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
    
    codex = defaultdict(str)
    file_names = glob("conformer_pkl_files/*.pkl")
    for file_name in file_names:
        data = pickle.load(open(file_name, "rb"))
        partition = file_name.strip().split("/")[-1].split("-")[0] # extract the numbers
        for key in data.keys():
            codex[key] = partition
    
    with open("conformer_pkl_files/codex.pkl", "wb") as f:
        pickle.dump(codex, f)
    with open("lmdb_to_OMol_ids/codex.pkl", "wb") as f:
        pickle.dump(codex, f)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")  # Especially important for macOS or Windows
    main()
