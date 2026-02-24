# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from . import data_utils


class AllCoordsDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates_1: str, coordinates_2: str, total_points=None):
        """
        coordinates_i is for the key to get the coordinates from the dataset
        """
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.key_coordinates_1 = coordinates_1
        self.key_coordinates_2 = coordinates_2
        self.set_epoch(None)
        self.total_points = total_points

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        assert len(atoms) > 0

        coordinates_1 = self.dataset[index][self.key_coordinates_1]
        coordinates_1 = np.array(coordinates_1)[:16, :, :]
        coordinates_1 = np.transpose(coordinates_1, (1, 2, 0))

        coordinates_2 = self.dataset[index][self.key_coordinates_2]
        coordinates_2 = np.array(coordinates_2)[:16, :, :]
        coordinates_2 = np.transpose(coordinates_2, (1, 2, 0))
        return {"atoms": atoms, "coordinates_1": coordinates_1.astype(np.float32), "coordinates_2": coordinates_2.astype(np.float32)}

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)

    def __len__(self):
        return self.total_points if self.total_points is not None else len(self.dataset)
