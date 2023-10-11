# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from omegaconf import II

_EPSILON = torch.finfo(torch.float32).eps
TARGET_DIST_NORM_CHOICES = ChoiceEnum(["none", "minmax"])


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    target_dist_norm: TARGET_DIST_NORM_CHOICES = field(
        default="none",
        metadata={"help": "method to normalize the range of target scores"},
    )
    loss_temperature: float = field(
        default=1.0,
        metadata={"help": "temperature in softmax for kl div"},
    )
    gate_loss_scaling: float = field(
        default=1.0,
        metadata={"help": "gate loss impact hyper param."}
    )
    gate_loss_type: str = field(
        default="mae",
        metadata={"help": "loss type for gates."}
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion(
    "label_smoothed_cross_entropy_with_gates", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyWithGatesCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            target_dist_norm="none",
            loss_temperature=1.0,
            gate_loss_scaling=1.0,
            gate_loss_type="kl_div",
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.target_dist_norm = target_dist_norm
        self.loss_temperature = loss_temperature
        self.gate_loss_scaling = gate_loss_scaling
        self.scaling = 1.0

        self.gate_loss_type = gate_loss_type
        if gate_loss_type == "kl_div":
            self.compute_gate_loss = self.compute_kl_div_loss
        elif gate_loss_type == "ce":
            self.compute_gate_loss = self.compute_ce_loss
        else:
            raise ValueError(f"Unknown `gate_loss_type`: {gate_loss_type}")

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        gate_loss, loss_per_gate = self.compute_gate_loss(net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        joint_loss = self.scaling * loss + self.gate_loss_scaling * gate_loss

        logging_output = {
            "joint_loss": joint_loss.data,
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "gate_loss": gate_loss.data,
        }
        if loss_per_gate is not None:
            logging_output["loss_per_gate"] = [l.data for l in loss_per_gate]
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return joint_loss, sample_size, logging_output

    def compute_kl_div_loss(self, net_output, sample, reduce=True):
        gates = net_output[1]["gates"]
        transformer_layer_count = int(len(gates) / 2)
        assert len(gates) == 2 * transformer_layer_count

        # Encoder
        gates = gates[:transformer_layer_count]
        bsz = gates[0].size(0)
        adapter_count = gates[0].size(1)
        # Target is target_dist as logits
        target_logits = (sample["net_input"]["domains"]
                         .unsqueeze(-1).repeat(1, transformer_layer_count, 1)
                         .reshape(-1, adapter_count))
        target_probs = torch.log_softmax(target_logits / self.loss_temperature, dim=-1)
        model_logits = torch.stack(gates, -2).reshape(-1, adapter_count)
        model_log_probs = torch.log_softmax(model_logits / self.loss_temperature, dim=-1)

        loss = (self.loss_temperature ** 2) * F.kl_div(model_log_probs, target_probs, log_target=True, reduction="none")
        loss_per_gate = [chunk.sum() for chunk in loss.split(bsz)]

        # Decoder
        gates = net_output[1]["gates"][transformer_layer_count:]
        seq_len = gates[0].size(0)
        bsz = gates[0].size(1)
        adapter_count = gates[0].size(2)
        transformer_layer_count = len(gates)
        target_logits = (sample["net_input"]["domains"]
                         .unsqueeze(-1).repeat(seq_len, transformer_layer_count, 1)
                         .reshape(-1, adapter_count))
        target_probs = torch.log_softmax(target_logits / self.loss_temperature, dim=-1)
        model_logits = torch.stack(gates, -2).reshape(-1, adapter_count)
        model_log_probs = torch.log_softmax(model_logits / self.loss_temperature, dim=-1)
        decoder_loss = (self.loss_temperature ** 2) * F.kl_div(model_log_probs, target_probs, log_target=True,
                                                               reduction="none")
        loss_per_gate = loss_per_gate + [chunk.sum() for chunk in decoder_loss.split(bsz * seq_len)]

        if reduce:
            loss = loss.sum()
            if decoder_loss is not None:
                loss += decoder_loss.sum()
        return loss, loss_per_gate

    def compute_ce_loss(self, net_output, sample, reduce=True):
        gates = net_output[1]["gates"]
        # Encoder
        gates = gates[:6]
        bsz = gates[0].size(0)
        adapter_count = gates[0].size(1)
        transformer_layer_count = len(gates)
        # Target is target_dist as logits
        target = (sample["net_input"]["domains"]
                  .unsqueeze(-1).repeat(1, transformer_layer_count, 1)
                  .reshape(-1)).long()
        model_logits = torch.stack(gates, -2).reshape(-1, adapter_count)
        loss = F.cross_entropy(model_logits, target, reduction="none")
        loss_per_gate = [chunk.sum() for chunk in loss.split(bsz)]
        decoder_loss = None
        if len(net_output[1]["gates"]) > 6:
            gates = net_output[1]["gates"][6:]
            seq_len = gates[0].size(0)
            bsz = gates[0].size(1)
            adapter_count = gates[0].size(2)
            transformer_layer_count = len(gates)
            target = (sample["net_input"]["domains"]
                      .unsqueeze(-1).repeat(seq_len, transformer_layer_count, 1)
                      .reshape(-1)).long()
            model_logits = torch.stack(gates, -2).reshape(-1, adapter_count)
            decoder_loss = F.cross_entropy(model_logits, target, reduction="none")

            loss_per_gate = loss_per_gate + [chunk.sum() for chunk in decoder_loss.split(bsz * seq_len)]

        if reduce:
            loss = loss.sum()
            if decoder_loss is not None:
                loss += decoder_loss.sum()
        return loss, loss_per_gate

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        joint_loss_sum = sum(log.get("joint_loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        gate_loss_sum = sum(log.get("gate_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        loss_per_gate_sum = {}
        for log in logging_outputs:
            for i in range(len(log.get("loss_per_gate", []))):
                gate_i_loss = log["loss_per_gate"][i]
                if i not in loss_per_gate_sum:
                    loss_per_gate_sum[i] = 0
                loss_per_gate_sum[i] = loss_per_gate_sum[i] + gate_i_loss

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "joint_loss", joint_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "gate_loss", gate_loss_sum / sample_size / math.log(2), ntokens, round=3
        )
        for k, v in loss_per_gate_sum.items():
            metrics.log_scalar(
                f"gate_loss/{k}_layer", v / ntokens / math.log(2), ntokens, round=3
            )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
