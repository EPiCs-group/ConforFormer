from utils.benchmarking import numpy_seed, string_to_seed
from utils.data_utils import init_lmdb_env, merge_lowercase_with_prev

import lmdb
import pickle
import numpy as np
import itertools
import multiprocessing as mp
import logging

from tqdm import tqdm


manager = mp.Manager()
NUM_ENERTIES_IN_LMDB = manager.Value('i', 0)

def get_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger(name)


def setup_worker(d: dict[str, tuple[list[str], np.ndarray[float]]]) -> tuple[dict[str, int], dict[int, str], list[int]]:
    """
    generates the intitial dictionaries for the worker to function after it has been started. Goes through all
    of the smiles strings in the dictionary and makes a dicitonary to track their availability. If there are not
    enough conformers to be used then they are already flagged for no use.

    args:
      d: unpickled dictionary in those lmdb databases with the SMILES and their conformers

    return:
      tuple[dict[str, int], dict[int, str], list[int]]
      dict 1: to track how many conformers have been used and remain for use. Has keys: "used" and "remaining"
      dict 2: to convert between the indices and the smiles strings
      list: the smiles string where there are not enough to be used
    """

    num_used_conf = {}
    convert_int_to_smi: dict[int, str] = {}
    fully_used_smi = []

    for idx, (smi, (atoms, coords)) in enumerate(d.items()):
        num_used_conf[smi] = {"used": 0, "remaining": len(coords)}
        convert_int_to_smi[idx] = smi
        if len(coords) < 2:
            fully_used_smi.append(idx)
    return num_used_conf, convert_int_to_smi, fully_used_smi


def get_conf_atoms_and_coords(
    smi: str, d: dict, conf_id_1: int, conf_id_2: int
) -> tuple[tuple[list[str], np.ndarray], tuple[list[str], np.ndarray]]:
    """
    Helper function for getting the atoms and coordinates to a conformer of a specific index.

    args:
      smi (str): smiles string belonging to the desired molecule
      d (dict): dictionary with the atoms and coordinates of molecules
      conf_id_1 (int): index of one of the conformers
      conf_id_2 (int): index of the other conformers
    
    returns:
      two tuples of the (atoms, coordinates)
    """
    atoms, coords = d[smi]
    try:
        return (atoms, coords[conf_id_1]), (atoms, coords[conf_id_2])
    except IndexError:
        print(conf_id_1)
        print(conf_id_2)
        print(smi)
        print(len(coords))


def update_num_conf_used(
    smi: str, num_used_conf: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    """
    Updates the "used" and "remaining" tags of the dictionary num_used_conf.

    args
      smi (str): SMILES string which acts as the key for the update
      num_used_conf (dict): dictionary which has SMILES strings as keys and values of {"same": *int*, "remaining": *int*}
    
    returns:
      The updated version of num_used_conf
    """
    num_used_conf[smi]["used"] += 2
    num_used_conf[smi]["remaining"] -= 2
    return num_used_conf


def worker(worker_id, input_queue, output_queue, points_to_choose: int=128, **kwargs):
    """
    Worker to process on entry of a "dataset_partition_i" lmdb file containing conformers to the ZINC20 SMILES string.
    Takes all of the smiles string and selects points_to_choose number of indicies. Uses every conformer only once so
    checks that there are enough reamining conformers and also makes sure that there are above the desired number of unique
    pairs of isomers. If either condition is not met then the block is not used. Finally puts everything into the output 
    queue to go to the writer process.

    args:
      worker_id (int): arbritrary workers id
      input_queue (mp.Queue): Queue where the database entries are put into
      output_queue (mp.Queue): Queue where the new datapoints are put into
      points_to_choose (int): number of smiles strings to take for the nwe dataset entry

    kwargs:
      max_attempts (int): maximum number of times the loop to generate new points can fail in a row before process is killed
      seed (int): initial see for the rng
      uniqueness_factor (float): Fraction of pairs within the dataset that can be duplicates before the block is rejected 
    """
    logger = get_logger(f"Worker {worker_id}")
    logger.debug(f"Starting")

    # initializing some values
    total_made = 0

    max_attempts = kwargs.get("max_attemps", 1000)
    seed = kwargs.get("seed", 42)
    uniqueness_factor = kwargs.get("uniqueness_factor", 0)

    while True:
        current_attempts = 0
        total_attemps = 0

        p = input_queue.get()
        if p is None:
            break
        formula, smi_dict = p
        logger.debug(f"{formula=}")

        try:
            formula = formula.decode("utf-8")
        except AttributeError:
            pass

        num_used_conf, convert_int_to_smi, fully_used_smi = setup_worker(smi_dict)
        # print(setup_worker(smi_dict))
        used_pairs = set()
        logger.debug(f"{len(used_pairs)=}")
        logger.debug(f"{len(fully_used_smi)=}")

        num_points = len(num_used_conf)
        while current_attempts <= max_attempts:
            total_attemps += 1
            logger.debug(f"{total_attemps=}")

            with numpy_seed(
                seed,
                string_to_seed(formula),
                total_attemps,
            ):
                block = np.random.choice(np.arange(num_points), points_to_choose, replace=False)
            
            pairs = set(itertools.combinations(block, 2))
            overlap = len(pairs & used_pairs)

            ## just test if there are enough conformers to go around
            if np.isin(block, fully_used_smi).any():
                current_attempts += 1
                continue

            # for the extremems:
            #  if uniqueness_factor is 1
            #     then all pairs must be unique thus overlap must be 0
            #  if uniqueness_factor is 0
            #     then there can be as many as we want. >= is fine in this
            #     case because other wise we get a situation of
            #     [all point] > [all points]
            #     which is not possible but allowed
            uniqueness_condition = overlap == 0 if uniqueness_factor == 1 else overlap >= len(pairs) * (1 - uniqueness_factor)

            # check if there are enough unique pairs to be considered
            # if uniqueness_condition:  #
            #     current_attempts += 1 # DO NOT CONSIDER UNIQUENESS AT THE MOMENT
            #     continue              #

            # Made it past check so reset limits counter
            current_attempts = 0
            total_made += 1

            conformers_1, conformers_2, smi_list = [], [], []
            used_pairs.update(pairs)

            for i in block:
                smi = convert_int_to_smi[i]
                conf_id_1 = num_used_conf[smi]["used"]
                conf_id_2 = conf_id_1 + 1
                conf_1, conf_2 = get_conf_atoms_and_coords(
                    smi, smi_dict, conf_id_1, conf_id_2
                )

                conformers_1.append(conf_1)
                conformers_2.append(conf_2)
                smi_list.append(smi)

                num_used_conf = update_num_conf_used(smi, num_used_conf)
                if num_used_conf[smi]["remaining"] < 2:
                    fully_used_smi.append(i)
            
            logger.debug("Putting data in output queue")
            output_queue.put((formula, smi_list, conformers_1, conformers_2))
    
    logger.debug(f"Killing process")


def writer(output_queue, lmdb_env, bsz: int = 100):
    """
    Writer process for making a lmdb dataset. Pulls values put into the output queue by the worker process.
    Then organizes the ordering of the coordinates so that everything is done identically to save space on
    having to store all atoms with the coordinates. Adds to the entry for the database which is given a key
    and put into a batch. Once the batchsize (bsz) is reached the function writes to the lmdb database. If a
    None is recieved in the queue the the process is killed.

    args:
      output_queue (mp.Queue): queue from which values are collected.
      lmdb_env: LMDB environment which can be activated to write to
    
    kwargs:
      bsz (int): Batchsize for to be reached before writing to the database

    raises:
      Assertion error if the ordering of the atoms is not the same across all molecules.
    """
    logger = get_logger("Writer")
    logger.debug("Starting process")

    tracking = logging.getLogger("Tracking")
    tracking_handler = logging.FileHandler("dataset_distribution.log")
    tracking_handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
    tracking.addHandler(tracking_handler)
    tracking.setLevel(logging.INFO)

    batch = {}

    def sort_conf(atoms, coords):
        try:
            merged_atoms = merge_lowercase_with_prev(atoms) # need this due to minor fuck-up. ie CClH3 will give list of [C, C, l, H, H, H]
            sort_idx = np.argsort(merged_atoms)
            sorted_atoms = np.array(merged_atoms)[sort_idx]
            sorted_coords = coords[sort_idx]
            return sorted_atoms.tolist(), sorted_coords
        except IndexError:
            merged_atoms = merge_lowercase_with_prev(atoms)
            sort_idx = np.argsort(merged_atoms)
            print(f"{sort_idx=}")
            print(f"{merged_atoms=}")
            print(f"{coords=}")
            sorted_atoms = np.array(merged_atoms)[sort_idx]
            sorted_coords = coords[sort_idx]

    def batch_write(batch):
        # writing batch
        with lmdb_env.begin(write=True) as txn:
            for key, val in batch.items():
                txn.put(key, val)
        # done writing

    while True:
        p = output_queue.get()
        if p is None:
            if len(batch) != 0:
                batch_write(batch)
            break
        formula, smi_list, confs_1, confs_2 = p

        db_val = {"atoms": [], "coordinates_1": [], "coordinates_2": [], "formula": formula, "smi": smi_list}
        for conf_1, conf_2 in zip(confs_1, confs_2):
            sorted_atoms_1, sorted_coords_1 = sort_conf(*conf_1)
            sorted_atoms_2, sorted_coords_2 = sort_conf(*conf_2)

            if len(db_val["atoms"]) == 0:
                db_val["atoms"] = sorted_atoms_1
            # sanity checks
            assert db_val["atoms"] == sorted_atoms_1
            assert db_val["atoms"] == sorted_atoms_2

            db_val["coordinates_1"].append(sorted_coords_1.astype(np.float32))
            db_val["coordinates_2"].append(sorted_coords_2.astype(np.float32))

        batch[f"{NUM_ENERTIES_IN_LMDB.value}".encode("utf-8")] = pickle.dumps(db_val)
        tracking.info(f"{NUM_ENERTIES_IN_LMDB.value} :: {formula}")
        NUM_ENERTIES_IN_LMDB.value += 1

        if len(batch) >= bsz:
            batch_write(batch)
            batch = {}
    
    logger.debug("Killing process")

def process_partition(partition_name, lmdb_env, num_workers, worker_kwargs):
    logger = get_logger("Handeler")
    logger.info(f"Starting to process {partition_name}")

    input_queue = mp.Queue(maxsize=2)
    output_queue = mp.Queue(maxsize=10_000)

    worker_processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker,
            args=(i, input_queue, output_queue),
            kwargs=worker_kwargs,
        )
        p.start()
        worker_processes.append(p)

    writer_process = mp.Process(target=writer, args=(output_queue, lmdb_env))
    writer_process.start()

    env = init_lmdb_env(partition_name+".lmdb")
    with env.begin() as txn:
        for k, v in txn.cursor():
            input_queue.put((k, pickle.loads(v)))


    for _ in worker_processes:
        input_queue.put(None)
    for w in worker_processes:
        w.join()

    output_queue.put(None)
    writer_process.join()

    logger.info("done")


def main():
    logger = get_logger("Main")

    # sub-datasets to process
    partition_names = [f"dataset_partition_{i}" for i in range(8)]
    logger.info(" ".join(partition_names))

    # Final resting place of the datapoints
    total_dataset_size_GB = 30 * 1024**3
    lmdb_env = lmdb.open(
        "tmp.lmdb",
        map_size=total_dataset_size_GB,
        readonly=False,
        lock=False,
        subdir=False,
    )

    num_workers = 16
    worker_kwargs = {
        "max_attempts": 1000,
        "uniqueness_factor": 0
        }

    for part in partition_names:
        process_partition(part, lmdb_env, num_workers, worker_kwargs)


if __name__ == "__main__":
    main()
