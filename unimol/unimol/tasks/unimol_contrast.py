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
    AllCoordsDataset,
    FlattenDistanceDataset,
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
    OMolDataset,
    FlattenRightPadDatasetCoord,
    FlattenRightPadDataset2D,
)
from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)


@register_task("unimol_contrast")
class UniMolContrast(UnicoreTask):
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
        def one_dataset(raw_dataset, coord_seed, mask_seed):
            """
            helper function for the creation of the chain of classes to process the data for training
            """
            if self.args.mode =='train':
                dataset = AllCoordsDataset( # select a specific conformer
                    raw_dataset, coord_seed, "atoms", "coordinates_1", "coordinates_2"
                )
                dataset = AtomTypeDataset(raw_dataset, dataset, coordinates="coordinates_1") # convert the atoms into their respective tokens
            
            elif self.args.mode == 'infer': # 
                dataset = AllCoordsDataset( # select a specific conformer
                    raw_dataset, coord_seed, "atoms", "coordinates_1", "coordinates_2"
                )
                dataset = AtomTypeDataset(raw_dataset, dataset, coordinates="coordinates_1")
            smi = KeyDataset(raw_dataset, "smi")
            formula = KeyDataset(raw_dataset, "formula")
            raw_atoms = KeyDataset(raw_dataset, "atoms")

            dataset_1 = RemoveHydrogenDataset(
                dataset,
                "atoms",
                "coordinates_1",
                self.args.remove_hydrogen,
                self.args.remove_polar_hydrogen,
            ) # remove hydrogen if so desired
            
            dataset_2 = RemoveHydrogenDataset(
                dataset,
                "atoms",
                "coordinates_2",
                self.args.remove_hydrogen,
                self.args.remove_polar_hydrogen,
            ) # remove hydrogen if so desired

            def PrependAndAppend(dataset, pre_token, app_token): # make sure everything is padded to the necessary length
                dataset = PrependTokenDataset(dataset, pre_token)
                return AppendTokenDataset(dataset, app_token)
            
            
            def process_coords(dataset, coordinates_key):
                dataset = NormalizeDataset(dataset, coordinates_key, normalize_coord=True, dim2=False) # "normalizing" in this case means centering the atom

                coord_dataset = KeyDataset(dataset, coordinates_key) # extract the coordinates of the chosen atom
                coord_dataset = FromNumpyDataset(coord_dataset) # convert the numpy array to pytorch tensors

                # dataset = CroppingDataset(
                #     dataset, self.seed, "atoms", "coordinates", self.args.max_atoms
                # ) # Safety for incase there are too many atoms within the molecule. Stictly speaking this in unecessary

            
                encoder_coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
                encoder_distance_dataset = FlattenDistanceDataset(encoder_coord_dataset) # pre-calculate the distance between atoms

                return encoder_coord_dataset, encoder_distance_dataset

            encoder_coord_dataset_1, encoder_distance_dataset_1 = process_coords(dataset_1, "coordinates_1")
            encoder_coord_dataset_2, encoder_distance_dataset_2 = process_coords(dataset_2, "coordinates_2")

            token_dataset = KeyDataset(dataset_1, "atoms") # extract the atoms of the molecule
            token_dataset = TokenizeDataset(
                token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
            ) # embed the tokens 
            src_dataset = PrependAndAppend(
                token_dataset, self.dictionary.bos(), self.dictionary.eos()
            )
            edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary)) # determine the square matrix describing the graph edge types
            


            return {
                "src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "src_coord": FlattenRightPadDatasetCoord(
                    encoder_coord_dataset_1,
                    pad_idx=0,
                ),
                "src_distance": FlattenRightPadDataset2D(
                    encoder_distance_dataset_1,
                    pad_idx=0,
                ),
                "src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
            },{
                "src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "src_coord": FlattenRightPadDatasetCoord(
                    encoder_coord_dataset_2,
                    pad_idx=0,
                ),
                "src_distance": FlattenRightPadDataset2D(
                    encoder_distance_dataset_2,
                    pad_idx=0,
                ),
                "src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
            }, {
                "all_smi": smi,
                "formula": formula,
                "atoms": raw_atoms,
            }

        
        raw_dataset = self.give_raw_dataset(split)
        net_input_set_1, net_input_set_2, smi = one_dataset(raw_dataset, self.args.seed, self.args.seed)
        dataset = {"net_input_set_1": net_input_set_1, "net_input_set_2": net_input_set_2, "misc": smi}

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
    
