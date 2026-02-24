from collections import defaultdict
import pickle
import lmdb
from tqdm import tqdm

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

def split_dict(dict_to_split):
    output_idx = {"train": [], "test": [], "valid": []}
    output_formulas = {"train": [], "test": [], "valid": []}
    count = {"train": 0, "test": 0, "valid": 0, "total": 0}


    for key, val_list in sorted(dict_to_split.items(), key=lambda x: len(x[1]), reverse=True):
        try:
            frac_train = count["train"]/count["total"]
        except ZeroDivisionError:
            frac_train = 0

        if frac_train <= 0.8:
            [output_idx["train"].append(val) for val in val_list]
            output_formulas["train"].append(key)
            count["train"] += len(val_list)

        else:
            try:
                frac_test = count["test"]/(count["test"] + count["valid"])
            except ZeroDivisionError:
                frac_test = 0

            if frac_test <= 0.5:
                [output_idx["test"].append(val) for val in val_list]
                output_formulas["test"].append(key)
                count["test"] += len(val_list)
            else:
                [output_idx["valid"].append(val) for val in val_list]
                output_formulas["valid"].append(key)
                count["valid"] += len(val_list)
        
        count["total"] += len(val_list)

    return output_formulas, output_idx

LOG_NAME = "../dataset_distribution.log"
def prelim():
    results = defaultdict(list)
    with open(LOG_NAME, "r") as f:
        for line in f.readlines():
            data = line.strip().split("|")[-1]
            number, formula = data.strip().split("::")
            results[formula].append(number)
    
    formulas, idx = split_dict(results)

    with open("train_dataset.pkl", "wb") as f:
        pickle.dump({"formulas": formulas["train"], "idx": idx["train"]}, f)
    
    with open("valid_dataset.pkl", "wb") as f:
        pickle.dump({"formulas": formulas["valid"], "idx": idx["valid"]}, f)
    
    with open("test_dataset.pkl", "wb") as f:
        pickle.dump({"formulas": formulas["test"], "idx": idx["test"]}, f)
    

def encode_list(l):
    return [f"{i}".strip().encode("utf-8") for i in l]


def load_and_encode(filename):
    dd = pickle.load(open(filename, "rb"))
    print(len(dd["formulas"]))
    return encode_list(dd["idx"])

def main():
    train_split = load_and_encode("train_dataset.pkl")
    test_split = load_and_encode("test_dataset.pkl")
    valid_split = load_and_encode("valid_dataset.pkl")

    env = init_lmdb_env("../full_dataset_with_smi.lmdb")
    
    train_env = lmdb.open("train_split.lmdb", map_size=25 * 1024**3, lock=False, subdir=False, readonly=False)
    test_env = lmdb.open("test_split.lmdb", map_size=5 * 1024**3, lock=False, subdir=False, readonly=False)
    valid_env = lmdb.open("valid_split.lmdb", map_size=5 * 1024**3, lock=False, subdir=False, readonly=False)

    train_count, test_count, valid_count = 0, 0, 0
    with env.begin() as txn:
        for key, val in tqdm(txn.cursor()):
            if key in test_split:
                with test_env.begin(write=True) as db:
                    db.put(f"{test_count}".encode("utf-8"), val)
                test_count += 1

            elif key in valid_split:
                with valid_env.begin(write=True) as db:
                    db.put(f"{valid_count}".encode("utf-8"), val)
                valid_count += 1
            
            else: # must be in the training set
                with train_env.begin(write=True) as db:
                    db.put(f"{train_count}".encode("utf-8"), val)
                train_count += 1
                
if __name__ == "__main__":
    prelim()
    main()