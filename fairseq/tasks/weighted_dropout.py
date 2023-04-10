# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
import json
import torch
import torch.nn.functional as F

from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II

from fairseq import metrics, models
from fairseq.data import encoders
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

logger = logging.getLogger(__name__)


@dataclass
class TranslationWeightedDropoutConfig(TranslationConfig):
    
    wd_iters: int = field(
            default=0,
            metadata={"help": "iteration interval for calculating weighted dropout probability"}
            )

@register_task("translation_weighted_dropout", dataclass=TranslationWeightedDropoutConfig)
class Translation_Weighted_Dropout(TranslationTask):
    """
    Translation task for Switch Transformer models.
    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language
    .. note::
        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    The translation task provides the following additional command-line
    arguments:
    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    def __init__(self, cfg: TranslationWeightedDropoutConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def build_model(self, cfg):
        model = models.build_model(cfg, self)

        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )

        return model

    
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        
        if self.cfg.wd_iters == 0 or update_num % self.cfg.wd_iters != 0:
            return loss, sample_size, logging_output
        
        else: 
            print("re-calculating weighted dropout probabilities")
            model.eval()
            
            for name, params in model.named_parameters():
                if 'embeddings' in name:
                    continue

                if params.requires_grad:
                    param_shape = params.shape
                    grad = params.grad.clone().detach().view(-1)
                    p = params.clone().view(-1)
                    scores = torch.abs(grad*p)
                    normalized_scores = scores - scores.min()
                    normalized_scores /= scores.max()
                    #weighted dropout
                    logits = torch.rand(len(normalized_scores), device='cuda')
                    mask = (logits >= normalized_scores).type(torch.float)
                    p = p*mask*len(mask)/torch.norm(mask, p=1)
                    params.data = p.view(param_shape)
            
            
            model.train()
            return loss, sample_size, logging_output
