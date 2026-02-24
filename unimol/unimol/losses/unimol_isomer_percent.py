# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from torchmetrics.classification import BinaryPrecision


import numpy as np
import contextlib
@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)



def merge_batches(*batches):
    """
    Merge multiple batch dictionaries into one, flattening along the first dimension.
    
    For each key:
    - If values are tensors: concatenates along the first dimension (dim=0)
    - If values are lists: extends them
    - Otherwise (scalars/strings/etc): collects into a list
    """
    if not batches:
        return {}

    merged = {}
    keys = batches[0].keys()

    for key in keys:
        values = [b[key] for b in batches]
        
        # Case 1: all tensors
        if all(isinstance(v, torch.Tensor) for v in values):
            merged[key] = torch.cat(values, dim=0)  # flatten along first dimension

        # Case 2: all lists
        elif all(isinstance(v, list) for v in values):
            merged[key] = sum(values, [])

        # Case 3: scalars, strings, etc.
        else:
            merged[key] = values

    return merged



def identify_non_filler_data(full_sample):
    non_filler = []
    for i, smi in enumerate(full_sample["smi_name"]):
        if smi != "!!filler":
            non_filler.append(i)
    return non_filler


def select_subset(batch, indices):
    """
    Select a subset of a batch-dictionary given a list of indices.
    - If values are tensors: index them along dim=0
    - If values are lists: select corresponding elements
    - Otherwise: collect values into a list
    """
    subset = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            subset[key] = value[indices]  # keep as tensor
        elif isinstance(value, list):
            subset[key] = [value[i] for i in indices]
        else:
            # for scalars/strings/etc → return aligned list
            subset[key] = [value for _ in indices]
    return subset


@register_loss("unimol_isomer_percent")
class ContrastIsomerPercent(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.seed = task.seed
        self.dist_mean = 6.312581655060595 # random number and I have no idea where it comes from
        self.dist_std = 3.3899264663911888 # random number and I have no idea where it comes from
        
        self.agg_masked_cls_token = None
        self.agg_contrast_cls_token = None
        self.mini_batch_count = None

        self.metrics = {
            0.50: BinaryPrecision(threshold=0.50).to("cuda"),
            0.95: BinaryPrecision(threshold=0.95).to("cuda"),
            0.99: BinaryPrecision(threshold=0.99).to("cuda"),
        }

    def forward(self, model, sample, reduce=True):
        """
        Function to calculate the loss of the current model on the given batch. 
        """

        # origional samples keys
        input_key_1 = "net_input_1"
        target_key_1 = "target_1"
        input_key_2 = "net_input_2"
        target_key_2 = "target_2"
        
        # augmented_samples keys
        isomer_input_key_1 = "isomer_input_1"
        isomer_target_key_1 = "isomer_target_1"
        isomer_input_key_2 = "isomer_input_2"
        isomer_target_key_2 = "isomer_target_2"

        # determine which idx from augmented data to take
        non_fill_data_id = identify_non_filler_data(sample[isomer_target_key_1])
        num_to_select = self.task.args.num_isomers_to_add
        if len(non_fill_data_id) < num_to_select:
            num_to_select = len(non_fill_data_id)
        
        if len(non_fill_data_id) != 0:
            with numpy_seed(self.task.seed, non_fill_data_id[0]):
                id_to_merge = np.random.choice(non_fill_data_id, size=num_to_select, replace=False)
            
            # collect chosen ids
            isomer_1_for_merge = select_subset(sample[isomer_input_key_1], indices=id_to_merge)
            isomer_target_1_for_merge = select_subset(sample[isomer_target_key_1], indices=id_to_merge)
            isomer_2_for_merge = select_subset(sample[isomer_input_key_2], indices=id_to_merge)
            isomer_target_2_for_merge = select_subset(sample[isomer_target_key_2], indices=id_to_merge)

            # merge inputs and target datasets
            merged_input = merge_batches(sample[input_key_1], isomer_1_for_merge, sample[input_key_2], isomer_2_for_merge)
            merged_target = merge_batches(sample[target_key_1], isomer_target_1_for_merge, sample[target_key_2], isomer_target_2_for_merge)
            
        else: # there are no isomers to fetch so skip
            merged_input = merge_batches(sample[input_key_1], sample[input_key_2])
            merged_target = merge_batches(sample[target_key_1], sample[target_key_2])

        # print(merged_input["src_tokens"])
        # start forward pass
        masked_tokens = merged_target["tokens_target"].ne(self.padding_idx)
        sample_size = masked_tokens.long().sum()
        (
            logits_encoder,
            encoder_distance,
            encoder_coord,
            x_norm,
            delta_encoder_pair_rep_norm,
            cls_token
        ) = model(**merged_input, encoder_masked_tokens=masked_tokens)
        target = merged_target["tokens_target"]
        
        if masked_tokens is not None:
            target = target[masked_tokens]
        masked_token_loss = F.nll_loss(
            F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
            target,
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        masked_pred = logits_encoder.argmax(dim=-1)
        masked_hit = (masked_pred == target).long().sum()
        masked_cnt = sample_size
        loss = masked_token_loss * self.args.masked_token_loss
        logging_output = {
            "sample_size": 1,
            "bsz": merged_target["tokens_target"].size(0),
            "seq_len": merged_target["tokens_target"].size(1)
            * merged_target["tokens_target"].size(0),
            "masked_token_loss": masked_token_loss.data,
            "masked_token_hit": masked_hit.data,
            "masked_token_cnt": masked_cnt,
        }

        if encoder_coord is not None:
            # real = mask + delta
            coord_target = merged_target["coord_target"]
            masked_coord_loss = F.smooth_l1_loss(
                encoder_coord[masked_tokens].view(-1, 3).float(),
                coord_target[masked_tokens].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            loss = loss + masked_coord_loss * self.args.masked_coord_loss
            # restore the scale of loss for displaying
            logging_output["masked_coord_loss"] = masked_coord_loss.data

        if encoder_distance is not None:
            dist_masked_tokens = masked_tokens
            masked_dist_loss = self.cal_dist_loss(
                merged_input, merged_target, encoder_distance, dist_masked_tokens, normalize=True
            )
            loss = loss + masked_dist_loss * self.args.masked_dist_loss
            logging_output["masked_dist_loss"] = masked_dist_loss.data

        if self.args.x_norm_loss > 0 and x_norm is not None:
            loss = loss + self.args.x_norm_loss * x_norm
            logging_output["x_norm_loss"] = x_norm.data

        if (
            self.args.delta_pair_repr_norm_loss > 0
            and delta_encoder_pair_rep_norm is not None
        ):
            loss = (
                loss + self.args.delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
            )
            logging_output[
                "delta_pair_repr_norm_loss"
            ] = delta_encoder_pair_rep_norm.data

        # addtitional contrastive loss
        if self.args.contrastive_loss > 0:
            contrast_loss, precs = self.calc_contrast_loss(
                cls_token,  
                temperature=self.args.contrast_temperature, 
            )


            loss = loss + contrast_loss * self.args.contrastive_loss
            logging_output["contrast_loss"] = contrast_loss.data
            
            for t in precs:
                min_prec, metrics = t
                logging_output[f"min_recall_{min_prec}_prec"] = metrics.data

        logging_output["loss"] = loss.data
        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3, priority=1)
        metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)

        masked_loss = sum(log.get("masked_token_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "masked_token_loss", masked_loss / sample_size, sample_size, round=3
        )

        masked_acc = sum(
            log.get("masked_token_hit", 0) for log in logging_outputs
        ) / sum(log.get("masked_token_cnt", 0) for log in logging_outputs)
        metrics.log_scalar("masked_acc", masked_acc, sample_size, round=3)

        masked_coord_loss = sum(
            log.get("masked_coord_loss", 0) for log in logging_outputs
        )
        if masked_coord_loss > 0:
            metrics.log_scalar(
                "masked_coord_loss",
                masked_coord_loss / sample_size,
                sample_size,
                round=5,
            )

        masked_dist_loss = sum(
            log.get("masked_dist_loss", 0) for log in logging_outputs
        )
        if masked_dist_loss > 0:
            metrics.log_scalar(
                "masked_dist_loss", masked_dist_loss / sample_size, sample_size, round=5
            )

        x_norm_loss = sum(log.get("x_norm_loss", 0) for log in logging_outputs)
        if x_norm_loss > 0:
            metrics.log_scalar(
                "x_norm_loss", x_norm_loss / sample_size, sample_size, round=3
            )

        delta_pair_repr_norm_loss = sum(
            log.get("delta_pair_repr_norm_loss", 0) for log in logging_outputs
        )
        if delta_pair_repr_norm_loss > 0:
            metrics.log_scalar(
                "delta_pair_repr_norm_loss",
                delta_pair_repr_norm_loss / sample_size,
                sample_size,
                round=3,
            )

        contrast_loss = sum(
            log.get("contrast_loss", 0) for log in logging_outputs
        )
        if contrast_loss > 0:
            metrics.log_scalar(
                "contrast_loss",
                contrast_loss/sample_size,
                sample_size,
                round=3
            )

            min_precisions = [0.5, 0.95, 0.99]
            for prec in min_precisions:
                recalls = sum([log.get(f"min_recall_{prec}_prec", 0) for log in logging_outputs])
                metrics.log_scalar(f"recall_{prec}_prec", recalls/sample_size, sample_size, round=5, priority=11)



    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def cal_dist_loss(self, inputs, targets, dist, masked_tokens, normalize=False):
        dist_masked_tokens = masked_tokens
        masked_distance = dist[dist_masked_tokens, :]
        masked_distance_target = targets["distance_target"][
            dist_masked_tokens
        ]
        # padding distance
        nb_masked_tokens = dist_masked_tokens.sum(dim=-1)
        masked_src_tokens = inputs["src_tokens"].ne(self.padding_idx)
        masked_src_tokens_expanded = torch.repeat_interleave(masked_src_tokens, nb_masked_tokens, dim=0)
        #
        if normalize:
            masked_distance_target = (
                masked_distance_target.float() - self.dist_mean
            ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[masked_src_tokens_expanded].view(-1).float(),
            masked_distance_target[masked_src_tokens_expanded].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_dist_loss

    def calc_contrast_loss(self, z_i, temperature=0.01, reduction="mean"):
        embeddings = F.normalize(z_i, dim=1)
        bsz = embeddings.size(0)

        similarity_mat = embeddings @ embeddings.T
        precs = self.calc_precisions(similarity_mat.clone())
        
        mask = ~torch.eye(bsz, dtype=bool, device=z_i.device)
        logits = similarity_mat / temperature
        logits = logits.masked_select(mask).view(bsz, -1)

        half_bsz = bsz // 2
        positives = torch.sum(embeddings[:half_bsz] * embeddings[half_bsz:], dim=1) / temperature
        positives = torch.cat([positives, positives], dim=0)  # duplicate for both halves

        loss = -positives + torch.logsumexp(logits, dim=1)

        if reduction == "mean":
            return loss.mean(), precs
        elif reduction == "max":
            return loss.max(), precs
        else:
            return loss, precs
        
    def calc_precisions(self, sim_mat):
        output = []

        bsz = sim_mat.size(0)
        assert bsz % 2 == 0
        N = bsz // 2
        mask = ~torch.eye(bsz, dtype=torch.bool, device=sim_mat.device)

        labels = torch.zeros_like(sim_mat, dtype=torch.int, device=sim_mat.device)
        for i in range(N):
            labels[i ,i+N] =1 
            labels[i+N ,i] =1
        
        scores = sim_mat[mask]
        targets = labels[mask]

        for min_prec, metric in self.metrics.items():
            metric.update(scores, targets)
            output.append((min_prec, metric.compute()))

        return output