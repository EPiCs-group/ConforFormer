# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from torchmetrics.classification import BinaryPrecision

@register_loss("unimol_contrast_head_double")
class ContrastHeadLossDouble(UnicoreLoss):
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
        
        # processing sample_1
        input_key_1 = "net_input_1"
        target_key_1 = "target_1"
        masked_tokens_1 = sample[target_key_1]["tokens_target"].ne(self.padding_idx)
        sample_size_1 = masked_tokens_1.long().sum()
        (
            logits_encoder_1,
            encoder_distance_1,
            encoder_coord_1,
            x_norm_1,
            delta_encoder_pair_rep_norm_1,
            cls_tokens_1
        ) = model(**sample[input_key_1], encoder_masked_tokens=masked_tokens_1)
        target_1 = sample[target_key_1]["tokens_target"]
        
        if masked_tokens_1 is not None:
            target_1 = target_1[masked_tokens_1]

        masked_token_loss_1 = None
        if logits_encoder_1 is not None:
            masked_token_loss_1 = F.nll_loss(
                F.log_softmax(logits_encoder_1, dim=-1, dtype=torch.float32),
                target_1,
                ignore_index=self.padding_idx,
                reduction="mean",
            )
            masked_pred_1 = logits_encoder_1.argmax(dim=-1)
            masked_hit_1 = (masked_pred_1 == target_1).long().sum()
        else:
            masked_hit_1 = torch.tensor(0, device=target_1.device)

        masked_cnt_1 = sample_size_1


        # processing sample_2
        input_key_2 = "net_input_2"
        target_key_2 = "target_2"
        masked_tokens_2 = sample[target_key_2]["tokens_target"].ne(self.padding_idx)
        sample_size_2 = masked_tokens_2.long().sum()
        (
            logits_encoder_2,
            encoder_distance_2,
            encoder_coord_2,
            x_norm_2,
            delta_encoder_pair_rep_norm_2,
            cls_tokens_2
        ) = model(**sample[input_key_2], encoder_masked_tokens=masked_tokens_2)
        target_2 = sample[target_key_2]["tokens_target"]
        
        if masked_tokens_2 is not None:
            target_2 = target_2[masked_tokens_2]

        masked_token_loss_2 = None
        if logits_encoder_2 is not None:
            masked_token_loss_2 = F.nll_loss(
                F.log_softmax(logits_encoder_2, dim=-1, dtype=torch.float32),
                target_2,
                ignore_index=self.padding_idx,
                reduction="mean",
            )
            masked_pred_2 = logits_encoder_2.argmax(dim=-1)
            masked_hit_2 = (masked_pred_2 == target_2).long().sum()
        else:
            masked_hit_2 = torch.tensor(0, device=target_1.device)
        masked_cnt_2 = sample_size_2


        # processing losses
        loss = 0

        masked_token_loss = None
        if masked_token_loss_1 is not None:
            masked_token_loss = masked_token_loss_1
        if masked_token_loss_2 is not None:
            if masked_token_loss_1 is not None:
                masked_token_loss = (masked_token_loss + masked_token_loss_2)/2
            else:
                masked_token_loss = masked_token_loss_2
        
        if masked_token_loss is not None:
            loss = loss + masked_token_loss * self.args.masked_token_loss
        else:
            masked_token_loss = torch.tensor(0, device=target_1.device)
        
        logging_output = {
            "sample_size": 1,
            "bsz": sample[target_key_1]["tokens_target"].size(0) + sample[target_key_2]["tokens_target"].size(0),
            "seq_len": sample[target_key_1]["tokens_target"].size(1)
            * sample[target_key_1]["tokens_target"].size(0),
            "masked_token_loss": masked_token_loss.data,
            "masked_token_hit": masked_hit_1.data + masked_hit_2.data,
            "masked_token_cnt": masked_cnt_1 + masked_cnt_2,
        }

        # masking coordinates loss
        masked_coord_loss = None
        if encoder_coord_1 is not None:
            # real = mask + delta
            coord_target = sample[target_key_1]["coord_target"]
            masked_coord_loss_1 = F.smooth_l1_loss(
                encoder_coord_1[masked_tokens_1].view(-1, 3).float(),
                coord_target[masked_tokens_1].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            
            if masked_coord_loss is None:
                masked_coord_loss = masked_coord_loss_1
            else:
                masked_coord_loss += masked_coord_loss_1
        
        if encoder_coord_2 is not None:
            # real = mask + delta
            coord_target = sample[target_key_2]["coord_target"]
            masked_coord_loss_2 = F.smooth_l1_loss(
                encoder_coord_2[masked_tokens_2].view(-1, 3).float(),
                coord_target[masked_tokens_2].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            
            if masked_coord_loss is None:
                masked_coord_loss = masked_coord_loss_2
            else:
                masked_coord_loss += masked_coord_loss_2
            # restore the scale of loss for displaying

        if masked_coord_loss is not None:
            if masked_coord_loss_1 is not None and masked_coord_loss_2 is not None:
                masked_coord_loss = masked_coord_loss/2
            loss = loss + masked_coord_loss * self.args.masked_coord_loss
            logging_output["masked_coord_loss"] = masked_coord_loss.data

        # masked_dist_loss
        masked_dist_loss = None
        if encoder_distance_1 is not None:
            dist_masked_tokens = masked_tokens_1
            masked_dist_loss_1 = self.cal_dist_loss(
                sample, encoder_distance_1, dist_masked_tokens, target_key_1, input_key_1, normalize=True
            )
            if masked_dist_loss is None:
                masked_dist_loss = masked_dist_loss_1
            else:
                masked_dist_loss += masked_dist_loss_1
        
        if encoder_distance_2 is not None:
            dist_masked_tokens = masked_tokens_2
            masked_dist_loss_2 = self.cal_dist_loss(
                sample, encoder_distance_2, dist_masked_tokens, target_key_2, input_key_2, normalize=True
            )
            if masked_dist_loss is None:
                masked_dist_loss = masked_dist_loss_2
            else:
                masked_dist_loss += masked_dist_loss_2

        if masked_dist_loss is not None:
            if masked_dist_loss_1 is not None and masked_dist_loss_2 is not None:
                masked_dist_loss = masked_dist_loss/2
            loss = loss + masked_dist_loss * self.args.masked_dist_loss
            logging_output["masked_dist_loss"] = masked_dist_loss.data
        
        # calculate the x norm loss
        if self.args.x_norm_loss > 0 and x_norm_1 is not None and x_norm_2 is not None:
            x_norm = (x_norm_1 + x_norm_2)/2
            loss = loss + self.args.x_norm_loss * x_norm
            logging_output["x_norm_loss"] = x_norm.data

        # calculate the encoder representation
        if (
            self.args.delta_pair_repr_norm_loss > 0
            and delta_encoder_pair_rep_norm_1 is not None
            and delta_encoder_pair_rep_norm_2 is not None
        ):
            delta_encoder_pair_rep_norm = (delta_encoder_pair_rep_norm_1 + delta_encoder_pair_rep_norm_2)/2
            loss = (
                loss + self.args.delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
            )
            logging_output[
                "delta_pair_repr_norm_loss"
            ] = delta_encoder_pair_rep_norm.data

        # addtitional contrastive loss
        if self.args.contrastive_loss > 0:
            contrast_loss, precs = self.calc_contrast_loss(
                cls_tokens_1, 
                cls_tokens_2, 
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

    def cal_dist_loss(self, sample, dist, masked_tokens, target_key, input_key, normalize=False):
        dist_masked_tokens = masked_tokens
        masked_distance = dist[dist_masked_tokens, :]
        masked_distance_target = sample[target_key]["distance_target"][
            dist_masked_tokens
        ]
        # padding distance
        nb_masked_tokens = dist_masked_tokens.sum(dim=-1)
        masked_src_tokens = sample[input_key]["src_tokens"].ne(self.padding_idx)
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