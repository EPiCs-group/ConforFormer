import pickle
import sqlite3
from tqdm import tqdm


conn = sqlite3.connect("NewCombinedDataset_OMol_portion.splite3")
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS datapoints (
        key TEXT PRIMARY KEY,
        partition TEXT
    )
""")

number = 0
data_to_get = pickle.load(open("NewCombinedDataset_OMol_portion.pkl", "rb"))
for k, v in tqdm(data_to_get.items()):
    if v[1] == "train":
        cur.execute("INSERT OR REPLACE INTO datapoints (key, partition) VALUES (?, ?)", (k, v[0]))
        number += 1
        if number > 1_000:
            number = 0
            conn.commit()