import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from . import data_utils


class IsomerSampleDatset(BaseWrapperDataset):
    def __init__(self, raw_dataset, isomer_lookup_dataset, isomer_id_dataset, seed, atoms, coordinates):
        self.dataset = raw_dataset
        self.isomer_lookup_dataset = isomer_lookup_dataset
        self.isomer_id_dataset = isomer_id_dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        iso_id = self.isomer_id_dataset[index]
        if iso_id == -1:
            # just put a random non-sense molcule that will make its way through dataset processing
            filler_atoms = np.array(self.dataset[index][self.atoms]) # to make sure that the dataset is processed properly
            filler_coords = np.array([[0, 0, 0] for _ in range(len(filler_atoms))], dtype=np.float32) # obvious signal that something has gone wrong
            filler_smi = "!!filler" # For identification of filler data later.
            return {"atoms": filler_atoms, "coordinates_1": filler_coords, "coordinates_2": filler_coords, "smi": filler_smi}
        else:
            isomers: list[int] = self.isomer_lookup_dataset[iso_id]
            isomers.remove(index)
            with data_utils.numpy_seed(self.seed, epoch, index, iso_id):
                isomer_idx_to_sample = np.random.choice(isomers, size=1, replace=False)
                isomer_idx_to_sample = isomer_idx_to_sample[0]

            atoms = np.array(self.dataset[isomer_idx_to_sample][self.atoms])
            smi = self.dataset[isomer_idx_to_sample]["smi"]
            num_choices = len(self.dataset[isomer_idx_to_sample][self.coordinates])
            if num_choices == 1:
                sample_idx = np.array([0, 0])
            else:
                with data_utils.numpy_seed(self.seed, epoch, isomer_idx_to_sample):
                    sample_idx = np.random.choice(range(num_choices), size=2, replace=False)
            coordinates_1 = self.dataset[isomer_idx_to_sample][self.coordinates][sample_idx[0]]
            coordinates_2 = self.dataset[isomer_idx_to_sample][self.coordinates][sample_idx[1]]
            return {"atoms": atoms, "coordinates_1": coordinates_1.astype(np.float32), "coordinates_2": coordinates_2.astype(np.float32), "smi": smi}

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)