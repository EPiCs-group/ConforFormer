import csv
from multiprocessing import Process, Queue
from pathlib import Path
from openbabel import openbabel as ob
from collections import defaultdict
import lmdb
import pickle
import os
from tqdm import tqdm

atomic_number_to_symbol = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
    21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
    31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr',
    41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
    61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
    71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
    81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
    91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
    101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt',
    110: 'Ds', 111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'
}


conv = ob.OBConversion()
conv.SetInFormat("smi")
def get_chemical_formula(smi_string):
    obmol = ob.OBMol()
    success = conv.ReadString(obmol, smi_string)
    if not success:
        return None
    obmol.AddHydrogens()
    return obmol.GetFormula()


def process_file(worker_id, input_queue: Queue, writer_queue: Queue):
    print(f"starting worker {worker_id}", flush=True)
    file = pickle.load(open("get_these_chemical_formulas.pkl", "rb"))
    wanted_chem_formulas = [t[0] for t in file]
    while True:
        smi_file = input_queue.get()
        if smi_file is None:
            break

        with open(smi_file, "r") as f:
            reader = csv.reader(f, delimiter=" ")
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                
                smi, ZINC_index = row
                chem_formula = get_chemical_formula(smi)
                if chem_formula in wanted_chem_formulas:
                    print("found")
                    writer_queue.put((chem_formula, smi_file, smi, ZINC_index))  # put into the lmdb
        print(f"!!! done {smi_file} !!!", flush=True)
    print(f"Worker {worker_id} killed", flush=True)


def init_lmdb_env(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    return env

BYTES_TO_GB = 1024**3
def lmdb_writer(writer_id, writer_queue, lmdb_path):
    print(f"Starting lmdb writer number {writer_id}", flush=True)
    map_size = 15
    env = lmdb.open(f"{lmdb_path}_{writer_id}.lmdb", subdir=False, lock=False, map_size=map_size*BYTES_TO_GB)

    # helper function
    def batch_write(batch_dict):
        with env.begin(write=True) as txn:
            for key, val in batch_dict.items():
                data_pickled = txn.get(key.encode("utf-8"))
                if data_pickled is not None:
                    old_data = pickle.loads(data_pickled)
                else:
                    old_data = set()
                
                new_data = old_data|val
                txn.put(key.encode("utf-8"), pickle.dumps(new_data))

    # main loop of process
    elem_in_buffer = 0
    buffer = defaultdict(set)
    while True:
        new_input = writer_queue.get()
        if new_input is None:
            break
        
        formula, file, smi, ZINC_id = new_input
        
        buffer[formula].add((file, smi, ZINC_id))
        elem_in_buffer += 1
        if elem_in_buffer >= 20_000:
            batch_write(buffer)
            buffer.clear()
            elem_in_buffer = 0

    # write anything remaining in the buffer
    if len(buffer) != 0:
        batch_write(buffer)
        buffer.clear()
        elem_in_buffer = 0
    print("writer killed", flush=True)


def main():
    path_to_lmdb = "ZINC_SMILES"
    num_workers = 30
    num_db_writers = 1

    input_queue = Queue(maxsize=1)
    writer_queue = Queue(maxsize=1_000_000)

    # start the database writer
    lmdb_writers = []
    for i in range(num_db_writers):
        p = Process(target=lmdb_writer, args=(i, writer_queue, path_to_lmdb))
        p.start()
        lmdb_writers.append(p)

    # start the workers
    workers = []
    for i in range(num_workers):
        p = Process(target=process_file, args=(i, input_queue, writer_queue))
        p.start()
        workers.append(p)
    
    # Put the files in the queue
    smi_files = list(Path(".").rglob("*.smi"))
    for file_path in tqdm(smi_files, desc="files processed"):
        input_queue.put(file_path)

    # signal to workers that finished
    for _ in workers:
        input_queue.put(None)

    # Wait for workers to finish
    for w in workers:
        w.join()

    # signal to summary and lmdb writer to finish
    for _ in lmdb_writers:
        writer_queue.put(None)

    for w in lmdb_writers:
        w.join()

    print("Everything is finished", flush=True)


if __name__ == "__main__":
    main()