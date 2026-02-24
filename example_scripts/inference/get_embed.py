#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import pickle
import torch
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks
import torch.multiprocessing as mp
import torch.nn.functional as F
import sqlite3
import zlib
import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


def xyz_string(atoms, coords, comment=""):
    """
    Create an XYZ-formatted string from atoms and coordinates.
    """
    atoms = np.asarray(atoms)
    coords = np.asarray(coords)

    # if atoms.shape[0] != coords.shape[0]:
    #     raise ValueError("Number of atoms must match number of coordinate rows")

    lines = [str(len(atoms)), comment]
    for atom, (x, y, z) in zip(atoms, coords):
        lines.append(f"{atom:2s} {x:15.8f} {y:15.8f} {z:15.8f}")
    return "\n".join(lines)



def write_to_sqlDB(db_name, datapoint_id, smiles_list, atoms, coords_1, cls_1, coords_2, cls_2):
    """
    Write molecules with coordinates, embeddings, and SMILES into sqlite3 database.
    """
    with sqlite3.connect(db_name + "_embed.sqlite3") as conn:

        # Performance tweaks
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("PRAGMA journal_mode=MEMORY;")

        # Create tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS smiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                smi TEXT UNIQUE NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS main (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dp_id INTEGER,
                set_id INTEGER,
                smi_id INTEGER NOT NULL,
                xyz_file TEXT,
                embedding BLOB,
                FOREIGN KEY(smi_id) REFERENCES smiles(id)
            )
        """)

        # Insert SMILES (unique)
        conn.executemany(
            "INSERT OR IGNORE INTO smiles (smi) VALUES (?)",
            [(smi,) for smi in smiles_list]
        )
        conn.commit()

        # Map SMILES to IDs
        smi_to_id = {row[1]: row[0] for row in conn.execute("SELECT id, smi FROM smiles")}

         # 🔑 Convert embeddings (cls_1, cls_2) to numpy *before* all_data
        cls_1 = cls_1.detach().cpu().numpy() if torch.is_tensor(cls_1) else np.asarray(cls_1)
        cls_2 = cls_2.detach().cpu().numpy() if torch.is_tensor(cls_2) else np.asarray(cls_2)

        # Bundle coords + cls together
        all_data = [
            (coords_1, cls_1),
            (coords_2, cls_2)
        ]

        insert_rows = []
        for i, (coords_batch, cls_batch) in enumerate(all_data):
            for smi, coords, embedding in zip(smiles_list, coords_batch, cls_batch):
                # Convert tensor coords -> xyz string
                xyz = xyz_string(atoms, coords, comment=smi)
                smi_id = smi_to_id[smi]

                # Ensure embedding is numpy float32
                emb_blob = np.asarray(embedding, dtype=np.float32).tobytes()

                insert_rows.append((datapoint_id, i+1, smi_id, xyz, emb_blob))

        # Bulk insert
        conn.executemany("""
            INSERT INTO main (dp_id, set_id, smi_id, xyz_file, embedding)
            VALUES (?, ?, ?, ?, ?)
        """, insert_rows)

        conn.commit()


def xyz_string(atoms, coords, comment=""):
    """
    Create an XYZ-formatted string from atoms and coordinates.
    """
    atoms = np.asarray(atoms)
    coords = coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)

    # if atoms.shape[0] != coords.shape[0]:
    #     raise ValueError("Number of atoms must match number of coordinate rows")

    lines = [str(len(atoms)), comment]
    for atom, (x, y, z) in zip(atoms, coords):
        lines.append(f"{atom:2s} {x:15.8f} {y:15.8f} {z:15.8f}")
    return "\n".join(lines)


def make_similarity_matrix(cls_1, cls_2):
    print(cls_1.shape, cls_2.shape)
    norm_1 = F.normalize(cls_1, dim=1)
    norm_2 = F.normalize(cls_2, dim=1)

    batch = torch.cat([norm_1, norm_2], dim=0)
    sim_mat = batch @ batch.T
    return sim_mat, batch

def main(args):
    DB_NAME = args.db_name
    assert (
        args.batch_size is not None
    ), "Must specify batch size either with --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if args.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    if use_cuda:
        model.cuda()
        # fp16 only supported on CUDA for fused kernels
        if use_fp16:
            model.half()

    # Print args
    logger.info(args)

    for subset in args.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)


        progress = progress_bar.progress_bar(
            range(len(dataset)),
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )

        model.eval()
        print(f"total num_datapoints: {range(len(dataset))}", flush=True)
        for i, sample in enumerate(progress):
            if i == 2500:
                break

            print(i, flush=True)
            sample = dataset[i]
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if len(sample) == 0:
                continue
            
            smi_strings = sample["misc.all_smi"]
            formula = sample["misc.formula"]
            atoms = sample["misc.atoms"]

            with torch.no_grad():
                encoder_rep_1, _, _ = model(
                    src_tokens = sample["net_input_set_1.src_tokens"].unsqueeze(0),
                    src_distance = sample["net_input_set_1.src_distance"],
                    src_coord = sample["net_input_set_1.src_coord"].permute(2, 0, 1),
                    src_edge_type = sample["net_input_set_1.src_edge_type"].unsqueeze(0),
                    features_only=True
                )
                cls_1 = encoder_rep_1[:, 0, :]

                encoder_rep_2, _, _ = model(
                    src_tokens = sample["net_input_set_2.src_tokens"].unsqueeze(0),
                    src_distance = sample["net_input_set_2.src_distance"],
                    src_coord = sample["net_input_set_2.src_coord"].permute(2, 0, 1),
                    src_edge_type = sample["net_input_set_2.src_edge_type"].unsqueeze(0),
                    features_only=True
                )
                cls_2 = encoder_rep_2[:, 0, :]
            
            write_to_sqlDB(
                db_name=DB_NAME, 
                datapoint_id=i, 
                smiles_list=smi_strings,
                atoms=atoms,
                coords_1=sample["net_input_set_1.src_coord"].permute(2, 0, 1),
                cls_1=cls_1,
                coords_2=sample["net_input_set_2.src_coord"].permute(2, 0, 1),
                cls_2=cls_2
                )
            torch.cuda.empty_cache()
            
        logger.info("Done inference! ")
    return None


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument("--db-name")
    parser.add_argument("--update-freq", default=[1,2])
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
