# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    TokenizeDataset,
    RightPadDataset2D,
    FromNumpyDataset,
    RawArrayDataset,
)
from unimol.data import (
    KeyDataset,
    ContrastConformerSampleWithIdDataset,
    DistanceDataset,
    EdgeTypeDataset,
    MaskPointsDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    Add2DConformerDataset,
    LMDBDataset,
    TTADataset,
    HMDBDataset,
    OMolDataset,
    IsomerSampleDatset,
)
from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)


@register_task("unimol_isomer_percent")
class UniMolTaskIsomerPercent(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.05,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.05,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--noise-type",
            default="uniform",
            choices=["trunc_normal", "uniform", "normal", "none"],
            help="noise type in coordinate noise",
        )
        parser.add_argument(
            "--noise",
            default=1.0,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=512,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only polar hydrogen ; -1: all hydrogen ; 0: remove all hydrogen ",
        )
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help="number of conformers generated with each molecule",
        )
        parser.add_argument(
            "--so-path",
            type=str,
            help="Path to the .so file for HugeMDB"
        )
        parser.add_argument(
            "--train-db-type",
            type=str,
            help="The type of database the training data is stored in"
        )
        parser.add_argument(
            "--valid-db-type",
            type=str,
            help="The type of database the validation data is stored in"
        )
        parser.add_argument(
            "--num-isomers-to-add",
            default=0,
            type=int,
            help="Num of isomer datapoints to add to the sample",
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True

        self.update_freq = args.update_freq[0] if len(args.update_freq) == 1 else None

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def give_raw_dataset(self, split):
        """Handle the logic of selecting and loading the raw dataset. There is 
        Args:
            split (str): name of the split (e.g., train, valid, test)
        Returns: Raw dataset in the format {'atoms': List[str], 'coordinates':List[np.array], 'smi': str}
        """
        if split in ["train", "train.small"]:
            db_type = self.args.train_db_type
        else:
            db_lookup = dict(zip(self.args.valid_subset.split(","), self.args.valid_db_type.split(",")))
            db_type = db_lookup[split]

        if db_type == "lmdb":
            split_path = os.path.join(self.args.data, split + ".lmdb")
            raw_dataset = LMDBDataset(split_path)
        elif db_type == "omol":
            mid_path = os.path.join(self.args.data, split + ".omol")
            db_path = os.path.join(self.args.data, "neutral_train")
            raw_dataset = OMolDataset(mid_path=mid_path, db_path=db_path)
        return raw_dataset
    
    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        
        raw_dataset = self.give_raw_dataset(split)
        coord_seed = self.args.seed
        mask_seed = self.args.seed

        if self.args.mode =='train':
            base_dataset = ContrastConformerSampleWithIdDataset( # select a specific conformer
                raw_dataset, coord_seed, "atoms", "coordinates"
            )
             # convert the atoms into their respective tokens

        elif self.args.mode == 'infer': # 
            base_dataset = TTADataset(
                raw_dataset, self.args.seed, "atoms", "coordinates", self.args.conf_size
            )

        smi_dataset = KeyDataset(raw_dataset, "smi")
        net_input_1, target_1 = self.process_dataset(base_dataset, smi_dataset, mask_seed=mask_seed, atoms_key="atoms", coordinates_key="coordinates_1")
        net_input_2, target_2 = self.process_dataset(base_dataset, smi_dataset, mask_seed=mask_seed+1, atoms_key="atoms", coordinates_key="coordinates_2")

        
        split_path = os.path.join(self.args.data, split + "_isomers.lmdb")
        isomer_lookup = LMDBDataset(split_path)
        isomer_id = KeyDataset(raw_dataset, "isomer_id")
        isomer_sample_dataset = IsomerSampleDatset(
            raw_dataset=raw_dataset,
            isomer_lookup_dataset=isomer_lookup,
            isomer_id_dataset=isomer_id,
            seed=self.args.seed,
            atoms="atoms",
            coordinates="coordinates"
            )
        isomer_smi = KeyDataset(isomer_sample_dataset, "smi")
        
        isomer_input_1, isomer_target_1 = self.process_dataset(isomer_sample_dataset, isomer_smi, mask_seed=mask_seed, atoms_key="atoms", coordinates_key="coordinates_1")
        isomer_input_2, isomer_target_2 = self.process_dataset(isomer_sample_dataset, isomer_smi, mask_seed=mask_seed+1, atoms_key="atoms", coordinates_key="coordinates_2")

        dataset = {
            "net_input_1": net_input_1,
            "target_1": target_1,
            "net_input_2": net_input_2,
            "target_2": target_2,
            "isomer_input_1": isomer_input_1,
            "isomer_target_1": isomer_target_1,
            "isomer_input_2": isomer_input_2,
            "isomer_target_2": isomer_target_2,
        }

        dataset = NestedDictionaryDataset(dataset)
        if split in ["train", "train.small"]:
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)
        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        return model

    def disable_shuffling(self) -> bool:
        return True
    
    
    def process_dataset(self, initial_dataset, smi_dataset, mask_seed, atoms_key, coordinates_key):
        """
        helper function for the creation of the chain of classes to process the data for training
        """
        dataset = AtomTypeDataset(initial_dataset, initial_dataset, atoms=atoms_key, coordinates=coordinates_key)
        dataset = RemoveHydrogenDataset(
            initial_dataset,
            atoms_key,
            coordinates_key,
            self.args.remove_hydrogen,
            self.args.remove_polar_hydrogen,
        ) # remove hydrogen if so desired

        dataset = CroppingDataset(
            dataset, self.seed, atoms_key, coordinates_key, self.args.max_atoms
        ) # Safety for incase there are too many atoms within the molecule. Stictly speaking this in unecessary
        dataset = NormalizeDataset(dataset, coordinates_key, normalize_coord=True) # "normalizing" in this case means centering the atom
        token_dataset = KeyDataset(dataset, atoms_key) # extract the atoms of the molecule
        token_dataset = TokenizeDataset(
            token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        ) # embed the tokens 
        coord_dataset = KeyDataset(dataset, coordinates_key) # extract the coordinates of the chosen atom
        expand_dataset = MaskPointsDataset(
            token_dataset,
            coord_dataset,
            self.dictionary,
            pad_idx=self.dictionary.pad(),
            mask_idx=self.mask_idx,
            noise_type=self.args.noise_type,
            noise=self.args.noise,
            seed=mask_seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
        ) # add noise to some of the coordinates of the atoms and mask some atoms as well

        def PrependAndAppend(dataset, pre_token, app_token): # make sure everything is padded to the necessary length
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        encoder_token_dataset = KeyDataset(expand_dataset, "atoms")
        encoder_target_dataset = KeyDataset(expand_dataset, "targets")
        encoder_coord_dataset = KeyDataset(expand_dataset, "coordinates")

        src_dataset = PrependAndAppend(
            encoder_token_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        tgt_dataset = PrependAndAppend(
            encoder_target_dataset, self.dictionary.pad(), self.dictionary.pad()
        )
        encoder_coord_dataset = PrependAndAppend(encoder_coord_dataset, 0.0, 0.0)
        encoder_distance_dataset = DistanceDataset(encoder_coord_dataset) # pre-calculate the distance between atoms

        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary)) # determine the square matrix describing the graph edge types
        coord_dataset = FromNumpyDataset(coord_dataset) # convert the numpy array to pytorch tensors
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = DistanceDataset(coord_dataset)

        return {
            "src_tokens": RightPadDataset(
                src_dataset,
                pad_idx=self.dictionary.pad(),
            ),
            "src_coord": RightPadDatasetCoord(
                encoder_coord_dataset,
                pad_idx=0,
            ),
            "src_distance": RightPadDataset2D(
                encoder_distance_dataset,
                pad_idx=0,
            ),
            "src_edge_type": RightPadDataset2D(
                edge_type,
                pad_idx=0,
            ),
        }, {
            "tokens_target": RightPadDataset(
                tgt_dataset, pad_idx=self.dictionary.pad()
            ),
            "distance_target": RightPadDataset2D(distance_dataset, pad_idx=0),
            "coord_target": RightPadDatasetCoord(coord_dataset, pad_idx=0),
            "smi_name": RawArrayDataset(smi_dataset),
        }