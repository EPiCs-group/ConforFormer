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



def compress_array(arr: np.ndarray) -> bytes:
    return zlib.compress(arr.tobytes())

def write_to_sqlDB(db_name, datapoint_id, sim_mat, smiles_list, cls_batch, formula):
    with sqlite3.connect(db_name + "_sim.sqlite3") as conn_sim:

        # Performance pragmas (tune if persistence is critical)
        conn_sim.execute("PRAGMA synchronous=OFF;")
        conn_sim.execute("PRAGMA journal_mode=MEMORY;")

        # Create normalized tables
        conn_sim.execute("""
            CREATE TABLE IF NOT EXISTS formulas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                formula TEXT UNIQUE NOT NULL
            )
        """)
        conn_sim.execute("""
            CREATE TABLE IF NOT EXISTS smiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                smi TEXT UNIQUE NOT NULL
            )
        """)
        conn_sim.execute("""
            CREATE TABLE IF NOT EXISTS sim_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dp_id INTEGER,
                formula_id INTEGER NOT NULL,
                smi_1_id INTEGER NOT NULL,
                smi_2_id INTEGER NOT NULL,
                tgt INTEGER,
                cos_sim REAL,
                FOREIGN KEY(formula_id) REFERENCES formulas(id),
                FOREIGN KEY(smi_1_id) REFERENCES smiles(id),
                FOREIGN KEY(smi_2_id) REFERENCES smiles(id)
            )
        """)


        # Insert unique formulas
        conn_sim.execute(
            "INSERT OR IGNORE INTO formulas (formula) VALUES (?)",
            (formula,)
        )

        # Insert unique SMILES
        conn_sim.executemany(
            "INSERT OR IGNORE INTO smiles (smi) VALUES (?)",
            [(smi,) for smi in smiles_list]
        )
        conn_sim.commit()

        # Map formulas and SMILES to IDs
        formula_id = conn_sim.execute("SELECT id FROM formulas WHERE formula = ?", (formula,)).fetchone()[0]
        smi_to_id = {row[1]: row[0] for row in conn_sim.execute("SELECT id, smi FROM smiles")}

        # Begin transaction for bulk insertion
        conn_sim.execute("BEGIN TRANSACTION;")
        sim_rows = []
        for i, row in enumerate(sim_mat):
            for j, data in enumerate(row):
                if i == j:
                    continue

                smi_1 = smiles_list[i % len(smiles_list)]
                smi_2 = smiles_list[j % len(smiles_list)]
                tgt = 1 if smi_1 == smi_2 else 0

                sim_rows.append((
                    datapoint_id,
                    formula_id,
                    smi_to_id[smi_1],
                    smi_to_id[smi_2],
                    tgt,
                    float(data)
                ))

        # Bulk insert sim_scores
        conn_sim.executemany("""
            INSERT INTO sim_scores (dp_id, formula_id, smi_1_id, smi_2_id, tgt, cos_sim)
            VALUES (?, ?, ?, ?, ?, ?)
        """, sim_rows)

        # Commit transaction
        conn_sim.commit()


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
            
            sim_mat, cls_batch = make_similarity_matrix(cls_1, cls_2)
            write_to_sqlDB(DB_NAME, i, sim_mat, smi_strings, cls_batch, formula)
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
