# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from sklearn.metrics import average_precision_score
from torchmetrics.classification import BinaryPrecision


@register_loss("unimol_contrast")
class ContrastLoss(UnicoreLoss):
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
        set_1_key = "net_input_set_1"
        set_2_key = "net_input_set_2"

        # sample_size = masked_tokens.long().sum()
        (
            logits_encoder,
            encoder_distance,
            encoder_coord,
            x_norm,
            delta_encoder_pair_rep_norm,
            cls_token_set_1
        ) = model(**sample[set_1_key], features_only=True)
        
        (
            logits_encoder,
            encoder_distance,
            encoder_coord,
            x_norm,
            delta_encoder_pair_rep_norm,
            cls_token_set_2
        ) = model(**sample[set_2_key], features_only=True)


        contrast_loss, precs = self.calc_contrast_loss(
            cls_token_set_1, 
            cls_token_set_2, 
            temperature=self.args.contrast_temperature
            )
        loss = contrast_loss * self.args.contrastive_loss

        logging_output = {
            "sample_size": 1,
            "bsz": sample[set_1_key]["src_tokens"].size(0),
            "seq_len": sample[set_1_key]["src_tokens"].size(1)
            * sample[set_1_key]["src_tokens"].size(0),
            "contrast_loss": contrast_loss.data,
        }
        
        for t in precs:
            min_prec, metrics = t
            logging_output[f"min_recall_{min_prec}_prec"] = metrics.data
            # logging_output[f"min_recall_{min_prec}_thresholds"] = metrics[1].data

        logging_output["loss"] = loss.data
        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        tot_samples = 0
        for log in logging_outputs:
            if log.get("contrast_loss") != 0:
                tot_samples += 1
        
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        pr_auc = sum(log.get("pr-auc", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / tot_samples, tot_samples, round=3)
        metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)
        metrics.log_scalar("pr_auc", pr_auc / bsz, 1, round=7)


        contrast_loss = sum(
            log.get("contrast_loss", 0) for log in logging_outputs
        )
        
        if contrast_loss > 0:
            metrics.log_scalar(
                "contrast_loss",
                contrast_loss / tot_samples,
                round=3
            )

        min_precisions = [0.5, 0.95, 0.99]
        for prec in min_precisions:
            recalls = sum([log.get(f"min_recall_{prec}_prec", 0) for log in logging_outputs])
            # thresholds = sum([log.get(f"min_recall_{prec}_thresholds", 0) for log in logging_outputs])

            metrics.log_scalar(f"recall_{prec}_prec", recalls/tot_samples, tot_samples, round=5, priority=11)
            # metrics.log_scalar(f"recall_{prec}_thresh", thresholds/tot_samples, tot_samples, round=5, priority=11)


    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


    def calc_contrast_loss(self, z_i, z_j, temperature=0.01, reduction="mean"):
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        embeddings = torch.cat([z_i, z_j], dim=0)
        bsz = embeddings.size(0)

        similarity_mat = embeddings @ embeddings.T
        precs = self.calc_precisions(similarity_mat.clone())
        
        mask = ~torch.eye(bsz, dtype=bool, device=z_i.device)
        logits = similarity_mat / temperature
        logits = logits.masked_select(mask).view(bsz, -1)

        positives = torch.cat([torch.sum(z_i * z_j, dim=1), torch.sum(z_j * z_i, dim=1)], dim=0)
        positives /= temperature

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

    @staticmethod
    def compute_pr_auc(similarity_mat):
        bsz = similarity_mat.size(0)
        assert bsz % 2 == 0, "Batch size should be even."
        N = bsz // 2

        # Build labels matrix: 1 for positive pairs, 0 for all other pairs
        labels = torch.zeros_like(similarity_mat, dtype=torch.int)

        # Positive pairs: (i, i+N) and (i+N, i)
        for i in range(N):
            labels[i, i + N] = 1
            labels[i + N, i] = 1

        # Flatten and remove diagonal (self-similarities)
        mask = ~torch.eye(bsz, dtype=torch.bool, device=similarity_mat.device)

        # Extract valid pairs
        scores = similarity_mat[mask].detach().flatten().cpu().numpy()
        targets = labels[mask].detach().flatten().cpu().numpy()

        # Compute PR-AUC
        pr_auc = average_precision_score(targets, scores)

        return pr_auc