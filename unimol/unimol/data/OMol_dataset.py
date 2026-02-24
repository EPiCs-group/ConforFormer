import os
from functools import lru_cache
import logging
import numpy as np

from fairchem.core.datasets import AseDBDataset

logger = logging.getLogger(__name__)

class OMolDataset:
    def __init__(self, mid_path, db_path):
        self.db_path = db_path
        self._keys = np.load(mid_path)
    
    def connect_db(self, dataset_path, save_to_self=False):
        dataset = AseDBDataset({"src": dataset_path})
        if save_to_self:
            self.omol_db = dataset
        else:
            return dataset
    
    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        mol_id = self._keys[idx]

        if not hasattr(self, "omol_db"):
            self.connect_db(self.db_path, save_to_self=True)
        
        atoms = self.omol_db.get_atoms(mol_id)
        return {'atoms': atoms.get_chemical_symbols(), "coordinates": [np.array(atoms.get_positions(), dtype=np.float32)], 'smi': "C"}
        