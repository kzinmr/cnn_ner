# -*- coding: utf-8 -*-
# modified from https://github.com/jiesutd/NCRFpp

import argparse
import json
import multiprocessing
import os
import pickle
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import MeCab
import numpy as np
import pytorch_lightning as pl
import requests
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from seqeval.metrics import accuracy_score
from seqeval.metrics.v1 import precision_recall_fscore_support
from seqeval.scheme import BILOU
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

ListStr = List[str]
ListListStr = List[ListStr]
ListInt = List[int]
ListListInt = List[ListInt]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


@dataclass
class InputExample:
    words: ListStr
    labels: ListStr


@dataclass
class FeatureExample:
    words: ListStr
    features: ListStr
    characters: ListListStr
    labels: ListStr


@dataclass
class InputFeature:
    word_ids: ListInt
    feature_ids: ListInt
    character_ids: ListListInt
    label_ids: ListInt


def download_dataset(data_dir: Union[str, Path]):
    def _download_data(url, file_path):
        response = requests.get(url)
        if response.ok:
            with open(file_path, "w") as fp:
                fp.write(response.content.decode("utf8"))
            return file_path

    for mode in Split:
        mode = mode.value
        url = f"https://github.com/megagonlabs/UD_Japanese-GSD/releases/download/v2.6-NE/{mode}.bio"
        file_path = os.path.join(data_dir, f"{mode}.txt")
        if _download_data(url, file_path):
            print(f"{mode} data is successfully downloaded")


class WordDropout(nn.Module):

    """copied from flair.nn
    Implementation of word dropout. Randomly drops out entire words (or characters) in embedding space.
    """

    def __init__(self, dropout_rate=0.05, inplace=False):
        super(WordDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.dropout_rate)

        mask = torch.autograd.Variable(m, requires_grad=False)
        return mask * x

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)


def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(
        -1, 1, m_size
    )  # B * M
    return max_score.view(-1, m_size) + torch.log(
        torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)
    ).view(
        -1, m_size
    )  # B * M


class CRF(nn.Module):
    def __init__(self, tagset_size, gpu=False):
        super(CRF, self).__init__()
        self.gpu = gpu
        print("build CRF...")
        self.START_TAG = -2
        self.STOP_TAG = -1

        # Matrix of transition parameters.  Entry i,j is the score of transitioning from i to j.
        self.tagset_size = tagset_size
        # # We add 2 here, because of START_TAG and STOP_TAG
        # # transitions (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        init_transitions = torch.zeros(self.tagset_size + 2, self.tagset_size + 2)
        init_transitions[:, self.START_TAG] = -10000.0
        init_transitions[self.STOP_TAG, :] = -10000.0
        init_transitions[:, 0] = -10000.0
        init_transitions[0, :] = -10000.0
        if self.gpu:
            init_transitions = init_transitions.cuda()
        self.transitions = nn.Parameter(init_transitions)

    def forward(self, feats, mask):
        return self.viterbi_decode(feats, mask)

    def viterbi_decode(self, feats, mask):
        """
        input:
            feats: (batch, seq_len, self.tag_size+2)
            mask: (batch, seq_len)
        output:
            decode_idx: (batch, seq_len) decoded sequence
            path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert tag_size == self.tagset_size + 2
        ## calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        ## mask to (seq_len, batch_size)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = (
            feats.transpose(1, 0)
            .contiguous()
            .view(ins_num, 1, tag_size)
            .expand(ins_num, tag_size, tag_size)
        )
        ## need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(
            ins_num, tag_size, tag_size
        )
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        ## record the position of best score
        back_points = list()
        partition_history = list()
        ##  reverse mask (bug for mask = 1- mask, use this as alternative choice)
        mask = (1 - mask.long()).bool()
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = (
            inivalues[:, self.START_TAG, :].clone().view(batch_size, tag_size)
        )  # bat_size * to_target_size
        partition_history.append(partition)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: batch_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1
            ).expand(batch_size, tag_size, tag_size)
            ## forscores, cur_bp = torch.max(cur_values[:,:-2,:], 1) # do not consider START_TAG/STOP_TAG
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)
            ## cur_bp: (batch_size, tag_size) max source score position in current tag
            ## set padded label as 0, which will be filtered in post processing
            cur_bp.masked_fill_(
                mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0
            )
            back_points.append(cur_bp)
        ### add score to final STOP_TAG
        partition_history = (
            torch.cat(partition_history, 0)
            .view(seq_len, batch_size, -1)
            .transpose(1, 0)
            .contiguous()
        )  ## (batch_size, seq_len. tag_size)
        ### get the last position for each setences, and select the last partitions using gather()
        last_position = (
            length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        )
        last_partition = torch.gather(partition_history, 1, last_position).view(
            batch_size, tag_size, 1
        )
        ### calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(
            batch_size, tag_size, tag_size
        ) + self.transitions.view(1, tag_size, tag_size).expand(
            batch_size, tag_size, tag_size
        )
        _, last_bp = torch.max(last_values, 1)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long()
        if self.gpu:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        ## select end ids in STOP_TAG
        pointer = last_bp[:, self.STOP_TAG]
        insert_last = (
            pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        )
        back_points = back_points.transpose(1, 0).contiguous()
        ## move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        back_points.scatter_(1, last_position, insert_last)
        back_points = back_points.transpose(1, 0).contiguous()
        ## decode from the end, padded position ids are 0, which will be filtered if following evaluation
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        if self.gpu:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.detach()
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(
                back_points[idx], 1, pointer.contiguous().view(batch_size, 1)
            )
            decode_idx[idx] = pointer.detach().view(batch_size)
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def viterbi_decode_nbest(self, feats, mask, nbest):
        """
        input:
            feats: (batch, seq_len, self.tag_size+2)
            mask: (batch, seq_len)
        output:
            decode_idx: (batch, nbest, seq_len) decoded sequence
            path_score: (batch, nbest) corresponding score for each sequence (to be implementated)
            nbest decode for sentence with one token is not well supported, to be optimized
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert tag_size == self.tagset_size + 2
        ## calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        ## mask to (seq_len, batch_size)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = (
            feats.transpose(1, 0)
            .contiguous()
            .view(ins_num, 1, tag_size)
            .expand(ins_num, tag_size, tag_size)
        )
        ## need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(
            ins_num, tag_size, tag_size
        )
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        ## record the position of best score
        back_points = list()
        partition_history = list()
        ##  reverse mask (bug for mask = 1- mask, use this as alternative choice)
        mask = (1 - mask.long()).bool()
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.START_TAG, :].clone()  # bat_size * to_target_size
        ## initial partition [batch_size, tag_size]
        partition_history.append(
            partition.view(batch_size, tag_size, 1).expand(batch_size, tag_size, nbest)
        )
        # iter over last scores
        for idx, cur_values in seq_iter:
            if idx == 1:
                cur_values = cur_values.view(
                    batch_size, tag_size, tag_size
                ) + partition.contiguous().view(batch_size, tag_size, 1).expand(
                    batch_size, tag_size, tag_size
                )
            else:
                # previous to_target is current from_target
                # partition: previous results log(exp(from_target)), #(batch_size * nbest * from_target)
                # cur_values: batch_size * from_target * to_target
                cur_values = cur_values.view(batch_size, tag_size, 1, tag_size).expand(
                    batch_size, tag_size, nbest, tag_size
                ) + partition.contiguous().view(batch_size, tag_size, nbest, 1).expand(
                    batch_size, tag_size, nbest, tag_size
                )
                ## compare all nbest and all from target
                cur_values = cur_values.view(batch_size, tag_size * nbest, tag_size)
            partition, cur_bp = torch.topk(cur_values, nbest, 1)
            ## cur_bp/partition: [batch_size, nbest, tag_size], id should be normize through nbest in following backtrace step
            if idx == 1:
                cur_bp = cur_bp * nbest
            partition = partition.transpose(2, 1)
            cur_bp = cur_bp.transpose(2, 1)

            # partition: (batch_size * to_target * nbest)
            # cur_bp: (batch_size * to_target * nbest) Notice the cur_bp number is the whole position of tag_size*nbest, need to convert when decode
            partition_history.append(partition)
            ## cur_bp: (batch_size,nbest, tag_size) topn source score position in current tag
            ## set padded label as 0, which will be filtered in post processing
            ## mask[idx] ? mask[idx-1]
            cur_bp.masked_fill_(
                mask[idx].view(batch_size, 1, 1).expand(batch_size, tag_size, nbest), 0
            )
            back_points.append(cur_bp)
        ### add score to final STOP_TAG
        partition_history = (
            torch.cat(partition_history, 0)
            .view(seq_len, batch_size, tag_size, nbest)
            .transpose(1, 0)
            .contiguous()
        )  ## (batch_size, seq_len, nbest, tag_size)
        ### get the last position for each setences, and select the last partitions using gather()
        last_position = (
            length_mask.view(batch_size, 1, 1, 1).expand(batch_size, 1, tag_size, nbest)
            - 1
        )
        last_partition = torch.gather(partition_history, 1, last_position).view(
            batch_size, tag_size, nbest, 1
        )
        ### calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(
            batch_size, tag_size, nbest, tag_size
        ) + self.transitions.view(1, tag_size, 1, tag_size).expand(
            batch_size, tag_size, nbest, tag_size
        )
        last_values = last_values.view(batch_size, tag_size * nbest, tag_size)
        end_partition, end_bp = torch.topk(last_values, nbest, 1)
        ## end_partition: (batch, nbest, tag_size)
        end_bp = end_bp.transpose(2, 1)
        # end_bp: (batch, tag_size, nbest)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size, nbest)).long()
        if self.gpu:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size, nbest)

        ## select end ids in STOP_TAG
        pointer = end_bp[:, self.STOP_TAG, :]  ## (batch_size, nbest)
        insert_last = (
            pointer.contiguous()
            .view(batch_size, 1, 1, nbest)
            .expand(batch_size, 1, tag_size, nbest)
        )
        back_points = back_points.transpose(1, 0).contiguous()
        ## move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        ## copy the ids of last position:insert_last to back_points, though the last_position index
        ## last_position includes the length of batch sentences
        back_points.scatter_(1, last_position, insert_last)
        ## back_points: [batch_size, seq_length, tag_size, nbest]
        """
        back_points: in simple demonstratration
        x,x,x,x,x,x,x,x,x,7
        x,x,x,x,x,4,0,0,0,0
        x,x,6,0,0,0,0,0,0,0
        """

        back_points = back_points.transpose(1, 0).contiguous()
        ## back_points: (seq_len, batch, tag_size, nbest)
        ## decode from the end, padded position ids are 0, which will be filtered in following evaluation
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size, nbest))
        if self.gpu:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.data.true_divide(nbest)
        # use old mask, let 0 means has token
        for idx in range(len(back_points) - 2, -1, -1):
            new_pointer = torch.gather(
                back_points[idx].view(batch_size, tag_size * nbest),
                1,
                pointer.contiguous().view(batch_size, nbest),
            )
            decode_idx[idx] = new_pointer.data.true_divide(nbest)
            # use new pointer to remember the last end nbest ids for non longest
            pointer = (
                new_pointer
                + pointer.contiguous().view(batch_size, nbest)
                * mask[idx].view(batch_size, 1).expand(batch_size, nbest).long()
            )

        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        ## decode_idx: [batch, seq_len, nbest]
        ### calculate probability for each sequence
        scores = end_partition[:, :, self.STOP_TAG]
        ## scores: [batch_size, nbest]
        max_scores, _ = torch.max(scores, 1)
        minus_scores = scores - max_scores.view(batch_size, 1).expand(batch_size, nbest)
        path_score = F.softmax(minus_scores, 1)
        ## path_score: [batch_size, nbest]
        return path_score, decode_idx

    def _calculate_PZ(self, feats, mask):
        """
        input:
            feats: (batch, seq_len, self.tag_size+2)
            masks: (batch, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert tag_size == self.tagset_size + 2
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = (
            feats.transpose(1, 0)
            .contiguous()
            .view(ins_num, 1, tag_size)
            .expand(ins_num, tag_size, tag_size)
        )
        ## need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(
            ins_num, tag_size, tag_size
        )
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # build iter
        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = (
            inivalues[:, self.START_TAG, :].clone().view(batch_size, tag_size, 1)
        )  # bat_size * to_target_size

        ## add start score (from start to all tag, duplicate to batch_size)
        # partition = partition + self.transitions[START_TAG,:].view(1, tag_size, 1).expand(batch_size, tag_size, 1)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target

            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1
            ).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)
            # (bat_size * from_target * to_target) -> (bat_size * to_target)
            # partition = utils.switch(partition, cur_partition, mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size)).view(bat_size, -1)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)

            ## effective updated partition part, only keep the partition value of mask value = 1
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            ## let mask_idx broadcastable, to disable warning
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)

            ## replace the partition where the maskvalue=1, other partition value keeps the same
            partition.masked_scatter_(mask_idx, masked_cur_partition)
        # until the last state, add transition score for all partition (and do log_sum_exp) then select the value in STOP_TAG
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(
            batch_size, tag_size, tag_size
        ) + partition.contiguous().view(batch_size, tag_size, 1).expand(
            batch_size, tag_size, tag_size
        )
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, self.STOP_TAG]
        return final_partition.sum(), scores

    def _score_sentence(self, scores, mask, tags):
        """
        input:
            scores: variable (seq_len, batch, tag_size, tag_size)
            mask: (batch, seq_len)
            tags: tensor  (batch, seq_len)
        output:
            score: sum of score for gold sequences within whole batch
        """
        # Gives the score of a provided tag sequence
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        ## convert tag value into a new format, recorded label bigram information to index
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        if self.gpu:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                ## start -> first score
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]

            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]

        ## transition for label to STOP_TAG
        end_transition = (
            self.transitions[:, self.STOP_TAG]
            .contiguous()
            .view(1, tag_size)
            .expand(batch_size, tag_size)
        )
        ## length for batch,  last word position = length - 1
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        ## index the label id of last word
        end_ids = torch.gather(tags, 1, length_mask - 1)

        ## index the transition score for end_id to STOP_TAG
        end_energy = torch.gather(end_transition, 1, end_ids)

        ## convert tag as (seq_len, batch_size, 1)
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        ### need convert tags id to search from 400 positions of scores
        tg_energy = torch.gather(
            scores.view(seq_len, batch_size, -1), 2, new_tags
        ).view(
            seq_len, batch_size
        )  # seq_len * bat_size
        ## mask transpose to (seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        # ## calculate the score from START_TAG to first label
        # start_transition = self.transitions[START_TAG,:].view(1, tag_size).expand(batch_size, tag_size)
        # start_energy = torch.gather(start_transition, 1, tags[0,:])

        ## add all score together
        # gold_score = start_energy.sum() + tg_energy.sum() + end_energy.sum()
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        # nonegative log likelihood
        forward_score, scores = self._calculate_PZ(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        return forward_score - gold_score


class CharCNN(nn.Module):
    def __init__(
        self,
        alphabet_size,
        embedding_dim,
        hidden_dim,
        dropout,
        # pretrain_char_embedding=None,
        gpu=False,
    ):
        self.gpu = gpu
        super(CharCNN, self).__init__()
        print("build char sequence feature extractor: CNN ...")
        self.hidden_dim = hidden_dim
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        # if pretrain_char_embedding is not None:
        #     self.char_embeddings.weight.data.copy_(
        #         torch.from_numpy(pretrain_char_embedding)
        #     )
        # else:
        #     self.char_embeddings.weight.data.copy_(
        #         torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim))
        #     )
        self.char_cnn = nn.Conv1d(
            embedding_dim, self.hidden_dim, kernel_size=3, padding=1
        )
        if self.gpu:
            self.char_drop = self.char_drop.cuda()
            self.char_embeddings = self.char_embeddings.cuda()
            self.char_cnn = self.char_cnn.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(
                -scale, scale, [1, embedding_dim]
            )
        return pretrain_emb

    def get_last_hiddens(self, input, seq_lengths):
        """
        input:
            input: Variable(batch_size, word_length)
            seq_lengths: numpy array (batch_size,  1)
        output:
            Variable(batch_size, char_hidden_dim)
        Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds)
        char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(
            batch_size, -1
        )
        return char_cnn_out

    def get_all_hiddens(self, input, seq_lengths):
        """
        input:
            input: Variable(batch_size,  word_length)
            seq_lengths: numpy array (batch_size,  1)
        output:
            Variable(batch_size, word_length, char_hidden_dim)
        Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds).transpose(2, 1).contiguous()
        return char_cnn_out

    def forward(self, input, seq_lengths):
        return self.get_all_hiddens(input, seq_lengths)


class WordRep(nn.Module):
    def __init__(
        self,
        word_alphabet_size: int,
        char_alphabet_size: int,
        word_emb_dim: int = 256,
        char_emb_dim: int = 32,
        char_hidden_dim: int = 64,
        dropout: float = 0.2,
        use_char: bool = True,
        # pretrain_char_embedding=None,
        feature_num: int = 0,
        feature_emb_dims: list = None,
        feature_alphabets: list = None,
        pretrain_feature_embeddings=None,
        pretrain_word_embedding=None,
        gpu: bool = False,
    ):
        super(WordRep, self).__init__()
        print("build word representation...")
        self.gpu = gpu
        self.use_char = use_char
        self.char_hidden_dim = 0
        self.char_all_feature = False
        if self.use_char:
            self.char_hidden_dim = char_hidden_dim
            self.char_embedding_dim = char_emb_dim
            self.dropout = dropout
            self.char_feature = CharCNN(
                alphabet_size=char_alphabet_size,
                embedding_dim=self.char_embedding_dim,
                hidden_dim=self.char_hidden_dim,
                dropout=self.dropout,
                gpu=self.gpu,
            )
            # elif data.char_feature_extractor == "LSTM":
            #     self.char_feature = CharBiLSTM(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            # elif data.char_feature_extractor == "GRU":
            #     self.char_feature = CharBiGRU(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            # elif data.char_feature_extractor == "ALL":
            #     self.char_all_feature = True
            #     self.char_feature = CharCNN(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            #     self.char_feature_extra = CharBiLSTM(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)

        self.embedding_dim = word_emb_dim
        self.drop = nn.Dropout(self.dropout)
        self.word_embedding = nn.Embedding(word_alphabet_size, self.embedding_dim)
        if pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(pretrain_word_embedding)
            )
        # else:
        #     self.word_embedding.weight.copy_(
        #         torch.from_numpy(
        #             self.random_embedding(word_alphabet_size, self.embedding_dim)
        #         )
        #     )

        self.feature_num = feature_num
        self.feature_embeddings = nn.ModuleList()
        if feature_emb_dims is not None and feature_num > 0:
            self.feature_embedding_dims = feature_emb_dims

            for idx in range(self.feature_num):
                self.feature_embeddings.append(
                    nn.Embedding(
                        feature_alphabets[idx].size(), self.feature_embedding_dims[idx]
                    )
                )
            for idx in range(self.feature_num):
                if pretrain_feature_embeddings[idx] is not None:
                    self.feature_embeddings[idx].weight.data.copy_(
                        torch.from_numpy(pretrain_feature_embeddings[idx])
                    )
                else:
                    self.feature_embeddings[idx].weight.data.copy_(
                        torch.from_numpy(
                            self.random_embedding(
                                feature_alphabets[idx].size(),
                                self.feature_embedding_dims[idx],
                            )
                        )
                    )

        if self.gpu:
            self.drop = self.drop.cuda()
            self.word_embedding = self.word_embedding.cuda()
            for idx in range(self.feature_num):
                self.feature_embeddings[idx] = self.feature_embeddings[idx].cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(
                -scale, scale, [1, embedding_dim]
            )
        return pretrain_emb

    def forward(
        self,
        word_inputs,
        feature_inputs,
        word_seq_lengths,
        char_inputs,
        char_seq_lengths,
        char_seq_recover,
    ):
        """
        input:
            word_inputs: (batch_size, sent_len)
            features: list [(batch_size, sent_len), (batch_len, sent_len),...]
            word_seq_lengths: list of batch_size, (batch_size,1)
            char_inputs: (batch_size*sent_len, word_length)
            char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
            char_seq_recover: variable which records the char order information, used to recover char order
        output:
            Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)

        word_embs = self.word_embedding(word_inputs)

        word_list = [word_embs]
        for idx in range(self.feature_num):
            word_list.append(self.feature_embeddings[idx](feature_inputs[idx]))
        if self.use_char:
            char_features = self.char_feature.get_last_hiddens(
                char_inputs, char_seq_lengths.cpu().numpy()
            )
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size, sent_len, -1)
            ## concat word and char together
            word_list.append(char_features)
            word_embs = torch.cat([word_embs, char_features], 2)
            if self.char_all_feature:
                char_features_extra = self.char_feature_extra.get_last_hiddens(
                    char_inputs, char_seq_lengths.cpu().numpy()
                )
                char_features_extra = char_features_extra[char_seq_recover]
                char_features_extra = char_features_extra.view(batch_size, sent_len, -1)
                ## concat word and char together
                word_list.append(char_features_extra)
        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)
        return word_represent


class LightweightConv1d(nn.Module):
    """Lightweight Convolution assuming the input is BxCxT
    This is just an example that explains LightConv clearer than the TBC version.
    We don't use this module in the model.
    Args:
        input_size: # of channels of the input and output
        kernel_size: convolution channels
        padding: padding
        num_heads: number of heads used. The weight is of shape
            `(num_heads, 1, kernel_size)`
        weight_softmax: normalize the weight with softmax before the convolution
    Shape:
        Input: BxCxT, i.e. (batch_size, input_size, timesteps)
        Output: BxCxT, i.e. (batch_size, input_size, timesteps)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias: the learnable bias of the module of shape `(input_size)`
    """

    def __init__(
        self,
        input_size,
        kernel_size=1,
        padding=0,
        num_heads=1,
        weight_softmax=False,
        bias=False,
        weight_dropout=0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None
        self.weight_dropout_module = nn.Dropout(weight_dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input):
        """
        input size: B x C x T
        output size: B x C x T
        """
        B, C, T = input.size()
        H = self.num_heads

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        weight = self.weight_dropout_module(weight)
        # Merge every C/H entries into the batch dimension (C = self.input_size)
        # B x C x T -> (B * C/H) x H x T
        # One can also expand the weight to C x 1 x K by a factor of C/H
        # and do not reshape the input instead, which is slow though
        input = input.view(-1, H, T)
        output = F.conv1d(input, weight, padding=self.padding, groups=self.num_heads)
        output = output.view(B, C, T)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)

        return output


class WordSequence(nn.Module):
    def __init__(
        self,
        word_alphabet_size: int,
        char_alphabet_size: int,
        label_alphabet_size: int,
        word_emb_dim: int = 256,
        word_hidden_dim: int = 512,
        char_emb_dim: int = 32,
        char_hidden_dim: int = 64,
        word_feature_extractor: str = "CNN",
        cnn_layer: int = 4,
        cnn_kernel: int = 5,
        lstm_layer: int = 3,
        dropout: float = 0.2,
        word_dropout: float = 0.05,
        use_char: bool = True,
        use_idcnn: bool = False,
        use_sepcnn: bool = False,
        use_sepcnn_rc: bool = False,
        use_bilstm: bool = False,
        use_batchnorm: bool = True,
        pretrain_word_embedding: Optional[np.array] = None,
        gpu: bool = False,
    ):
        super(WordSequence, self).__init__()
        print("build word sequence feature extractor: %s..." % (word_feature_extractor))
        self.use_char = use_char
        self.gpu = gpu
        self.wordrep = WordRep(
            word_alphabet_size=word_alphabet_size,
            char_alphabet_size=char_alphabet_size,
            word_emb_dim=word_emb_dim,
            char_emb_dim=char_emb_dim,
            char_hidden_dim=char_hidden_dim,
            dropout=dropout,
            use_char=use_char,
            # pretrain_char_embedding=None,
            # feature_num: int = 0,
            # feature_emb_dims: list = None,
            # feature_alphabets: list = None,
            # pretrain_feature_embeddings: list = None,
            pretrain_word_embedding=pretrain_word_embedding,
            gpu=gpu,
        )
        self.input_size = word_emb_dim
        if self.use_char:
            self.input_size += char_hidden_dim
        self.hidden_dim = word_hidden_dim
        self.word_feature_extractor = word_feature_extractor
        self.dropout_rate = dropout
        self.word_dropout_rate = word_dropout
        if self.word_feature_extractor in {"GRU", "LSTM"}:
            # The LSTM takes word embeddings as inputs, and outputs hidden states
            # with dimensionality hidden_dim.
            self.droplstm = nn.Dropout(self.dropout_rate)
            self.lstm_layer = lstm_layer
            self.bilstm_flag = use_bilstm
            if self.bilstm_flag:
                self.hidden_dim = word_hidden_dim // 2
            else:
                self.hidden_dim = word_hidden_dim

            if self.word_feature_extractor == "GRU":
                self.lstm = nn.GRU(
                    self.input_size,
                    self.hidden_dim,
                    num_layers=self.lstm_layer,
                    batch_first=True,
                    bidirectional=self.bilstm_flag,
                )
            elif self.word_feature_extractor == "LSTM":
                self.lstm = nn.LSTM(
                    self.input_size,
                    self.hidden_dim,
                    num_layers=self.lstm_layer,
                    batch_first=True,
                    bidirectional=self.bilstm_flag,
                )
            self.hidden2tag = nn.Linear(self.hidden_dim, label_alphabet_size)
        else:  # elif self.word_feature_extractor == "CNN":
            self.cnn_layer = cnn_layer
            print("CNN layer: ", self.cnn_layer)
            self.use_idcnn = use_idcnn
            self.use_sepcnn = use_sepcnn
            self.use_sepcnn_rc = use_sepcnn_rc
            self.use_bn = use_batchnorm
            self.cnn_kernel = cnn_kernel
            self.word_dropout = WordDropout(self.word_dropout_rate)
            if self.use_sepcnn_rc:
                # word2conv_list
                for idx in range(self.cnn_layer):
                    self.word2conv_list = nn.ModuleList()
                    self.depthwise_cnn_list = nn.ModuleList()
                    self.pointwise_cnn_list = nn.ModuleList()
                    self.conv2word_list = nn.ModuleList()
                    self.cnn_drop_list = nn.ModuleList()
                    self.depthwise_cnn_norm_list = nn.ModuleList()
                    self.pointwise_cnn_norm_list = nn.ModuleList()
                    pad_size = int((self.cnn_kernel - 1) / 2)
                    for idx in range(self.cnn_layer):
                        self.word2conv_list.append(
                            nn.Linear(self.input_size, self.hidden_dim)
                        )
                        self.depthwise_cnn_list.append(
                            LightweightConv1d(
                                self.hidden_dim,
                                self.cnn_kernel,
                                padding=pad_size,
                                num_heads=self.hidden_dim,
                                weight_softmax=False,
                                weight_dropout=self.dropout_rate,
                            )
                        )
                        self.pointwise_cnn_list.append(
                            nn.Conv1d(
                                self.hidden_dim,
                                self.hidden_dim,
                                kernel_size=1,
                            )
                        )
                        self.cnn_drop_list.append(nn.Dropout(self.dropout_rate))
                        self.depthwise_cnn_norm_list.append(
                            nn.BatchNorm1d(self.hidden_dim)
                            if self.use_bn
                            else nn.GroupNorm(1, self.hidden_dim)
                        )
                        self.pointwise_cnn_norm_list.append(
                            nn.BatchNorm1d(self.hidden_dim)
                            if self.use_bn
                            else nn.GroupNorm(1, self.hidden_dim)
                        )
                        self.conv2word_list.append(
                            nn.Linear(self.hidden_dim, self.input_size)
                        )
                    self.hidden2tag = nn.Linear(self.input_size, label_alphabet_size)
            else:
                self.word2cnn = nn.Linear(self.input_size, self.hidden_dim)
                if self.use_idcnn:
                    self.cnn_list = nn.ModuleList()
                    self.cnn_drop_list = nn.ModuleList()
                    self.cnn_norm_list = nn.ModuleList()
                    self.dcnn_drop_list = nn.ModuleList()
                    self.dcnn_norm_list = nn.ModuleList()
                    self.dilations = [1, 2, 1]
                    for idx in range(self.cnn_layer):
                        dcnn = nn.ModuleList()
                        dcnn_drop = nn.ModuleList()
                        dcnn_norm = nn.ModuleList()
                        for i, dilation in enumerate(self.dilations):
                            pad_size = self.cnn_kernel // 2 + dilation - 1
                            dcnn.append(
                                nn.Conv1d(
                                    in_channels=self.hidden_dim,
                                    out_channels=self.hidden_dim,
                                    kernel_size=self.cnn_kernel,
                                    dilation=dilation,
                                    padding=pad_size,
                                )
                            )
                            dcnn_drop.append(nn.Dropout(self.dropout_rate))
                            dcnn_norm.append(
                                nn.BatchNorm1d(self.hidden_dim)
                                if self.use_bn
                                else nn.GroupNorm(1, self.hidden_dim)
                            )
                        self.dcnn_drop_list.append(dcnn_drop)
                        self.dcnn_norm_list.append(dcnn_norm)
                        self.cnn_list.append(dcnn)
                        self.cnn_drop_list.append(nn.Dropout(self.dropout_rate))
                        self.cnn_norm_list.append(
                            nn.BatchNorm1d(self.hidden_dim)
                            if self.use_bn
                            else nn.GroupNorm(1, self.hidden_dim)
                        )
                elif self.use_sepcnn:
                    self.depthwise_cnn_list = nn.ModuleList()
                    self.pointwise_cnn_list = nn.ModuleList()
                    self.cnn_drop_list = nn.ModuleList()
                    self.depthwise_cnn_norm_list = nn.ModuleList()
                    self.pointwise_cnn_norm_list = nn.ModuleList()
                    pad_size = int((self.cnn_kernel - 1) / 2)
                    for idx in range(self.cnn_layer):
                        self.depthwise_cnn_list.append(
                            nn.Conv1d(
                                self.hidden_dim,
                                self.hidden_dim,
                                kernel_size=self.cnn_kernel,
                                padding=pad_size,
                                groups=self.hidden_dim,
                            )
                        )
                        self.pointwise_cnn_list.append(
                            nn.Conv1d(
                                self.hidden_dim,
                                self.hidden_dim,
                                kernel_size=1,
                            )
                        )
                        self.cnn_drop_list.append(nn.Dropout(self.dropout_rate))
                        self.depthwise_cnn_norm_list.append(
                            nn.BatchNorm1d(self.hidden_dim)
                            if self.use_bn
                            else nn.GroupNorm(1, self.hidden_dim)
                        )
                        self.pointwise_cnn_norm_list.append(
                            nn.BatchNorm1d(self.hidden_dim)
                            if self.use_bn
                            else nn.GroupNorm(1, self.hidden_dim)
                        )
                else:
                    # Sequentialでやるとloss発散
                    self.cnn_list = nn.ModuleList()
                    self.cnn_drop_list = nn.ModuleList()
                    self.cnn_norm_list = nn.ModuleList()
                    pad_size = int((self.cnn_kernel - 1) / 2)
                    for idx in range(self.cnn_layer):
                        self.cnn_list.append(
                            nn.Conv1d(
                                self.hidden_dim,
                                self.hidden_dim,
                                kernel_size=self.cnn_kernel,
                                padding=pad_size,
                            )
                        )
                        self.cnn_drop_list.append(nn.Dropout(self.dropout_rate))
                        self.cnn_norm_list.append(
                            nn.BatchNorm1d(self.hidden_dim)
                            if self.use_bn
                            else nn.GroupNorm(1, self.hidden_dim)
                        )

                self.hidden2tag = nn.Linear(self.hidden_dim, label_alphabet_size)

        if self.gpu:
            self.hidden2tag = self.hidden2tag.cuda()
            if self.word_feature_extractor == "CNN":
                if self.use_sepcnn_rc:
                    self.word_dropout = self.word_dropout.cuda()
                    # self.cnn = self.cnn.cuda()
                    for idx in range(self.cnn_layer):
                        self.word2conv_list[idx] = self.word2conv_list[idx].cuda()
                        self.depthwise_cnn_list[idx] = self.depthwise_cnn_list[
                            idx
                        ].cuda()
                        self.pointwise_cnn_list[idx] = self.pointwise_cnn_list[
                            idx
                        ].cuda()
                        self.conv2word_list[idx] = self.conv2word_list[idx].cuda()
                        self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                        self.depthwise_cnn_norm_list[
                            idx
                        ] = self.depthwise_cnn_norm_list[idx].cuda()
                        self.pointwise_cnn_norm_list[
                            idx
                        ] = self.pointwise_cnn_norm_list[idx].cuda()
                else:
                    self.word_dropout = self.word_dropout.cuda()
                    self.word2cnn = self.word2cnn.cuda()
                    for idx in range(self.cnn_layer):
                        if self.use_idcnn:
                            for i, dilation in enumerate(self.dilations):
                                self.cnn_list[idx][i] = self.cnn_list[idx][i].cuda()
                                self.dcnn_drop_list[idx][i] = self.dcnn_drop_list[idx][
                                    i
                                ].cuda()
                                self.dcnn_norm_list[idx][i] = self.dcnn_norm_list[idx][
                                    i
                                ].cuda()
                            self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                            self.cnn_norm_list[idx] = self.cnn_norm_list[idx].cuda()
                        elif self.use_sepcnn:
                            self.depthwise_cnn_list[idx] = self.depthwise_cnn_list[
                                idx
                            ].cuda()
                            self.pointwise_cnn_list[idx] = self.pointwise_cnn_list[
                                idx
                            ].cuda()
                            self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                            self.depthwise_cnn_norm_list[
                                idx
                            ] = self.depthwise_cnn_norm_list[idx].cuda()
                            self.pointwise_cnn_norm_list[
                                idx
                            ] = self.pointwise_cnn_norm_list[idx].cuda()
                        else:
                            self.cnn_list[idx] = self.cnn_list[idx].cuda()
                            self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                            self.cnn_norm_list[idx] = self.cnn_norm_list[idx].cuda()
            else:
                self.droplstm = self.droplstm.cuda()
                self.lstm = self.lstm.cuda()

    def forward(
        self,
        word_inputs,
        feature_inputs,
        word_seq_lengths,
        char_inputs,
        char_seq_lengths,
        char_seq_recover,
    ):
        """
        input:
            word_inputs: (batch_size, sent_len)
            feature_inputs: [(batch_size, sent_len), ...] list of variables
            word_seq_lengths: list of batch_size, (batch_size,1)
            char_inputs: (batch_size*sent_len, word_length)
            char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
            char_seq_recover: variable which records the char order information, used to recover char order
        output:
            Variable(batch_size, sent_len, hidden_dim)
        """

        word_represent = self.wordrep.forward(
            word_inputs,
            feature_inputs,
            word_seq_lengths,
            char_inputs,
            char_seq_lengths,
            char_seq_recover,
        )
        ## word_embs (batch_size, seq_len, embed_size)
        if self.word_feature_extractor == "CNN":
            word_represent = self.word_dropout.forward(word_represent)
            if self.use_sepcnn_rc:
                # BTC: word_represent
                cnn_feature = word_represent
                for idx in range(self.cnn_layer):
                    # BTC: for linear, residual
                    residual = cnn_feature
                    cnn_feature = F.relu(self.word2conv_list[idx](cnn_feature))
                    # BCT: for conv
                    cnn_feature = cnn_feature.transpose(2, 1).contiguous()
                    cnn_feature = F.relu(self.depthwise_cnn_list[idx](cnn_feature))
                    cnn_feature = self.depthwise_cnn_norm_list[idx](cnn_feature)
                    cnn_feature = F.relu(self.pointwise_cnn_list[idx](cnn_feature))
                    cnn_feature = self.pointwise_cnn_norm_list[idx](cnn_feature)
                    cnn_feature = self.cnn_drop_list[idx](cnn_feature)

                    # residual connection
                    # BTC for linear, residual
                    cnn_feature = cnn_feature.transpose(2, 1).contiguous()
                    cnn_feature = F.relu(self.conv2word_list[idx](cnn_feature))
                    cnn_feature = residual + cnn_feature
                    cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                # BTC for linear
                feature_out = cnn_feature
            else:
                # BTC: word_represent
                word_in = torch.tanh(self.word2cnn(word_represent))
                # BCT: for conv
                cnn_feature = word_in.transpose(2, 1).contiguous()
                for idx in range(self.cnn_layer):
                    if self.use_idcnn:
                        for i, dilation in enumerate(self.dilations):
                            cnn_feature = F.relu(self.cnn_list[idx][i](cnn_feature))
                            cnn_feature = self.dcnn_drop_list[idx][i](cnn_feature)
                            cnn_feature = self.dcnn_norm_list[idx][i](cnn_feature)
                    elif self.use_sepcnn:
                        cnn_feature = F.relu(self.depthwise_cnn_list[idx](cnn_feature))
                        cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                        cnn_feature = self.depthwise_cnn_norm_list[idx](cnn_feature)
                        cnn_feature = F.relu(self.pointwise_cnn_list[idx](cnn_feature))
                        cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                        cnn_feature = self.pointwise_cnn_norm_list[idx](cnn_feature)
                    else:
                        cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
                        cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                        cnn_feature = self.cnn_norm_list[idx](cnn_feature)
                # BTC for linear
                feature_out = cnn_feature.transpose(2, 1).contiguous()
        else:
            packed_words = pack_padded_sequence(
                word_represent, word_seq_lengths.cpu().numpy(), True
            )
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            ## lstm_out (seq_len, seq_len, hidden_size)
            feature_out = self.droplstm(lstm_out.transpose(1, 0))
        ## BTC: (batch_size, seq_len, hidden_size)
        outputs = self.hidden2tag(feature_out)
        return outputs


class TokenClassificationModel(nn.Module):
    def __init__(
        self,
        word_alphabet_size: int,
        char_alphabet_size: int,
        label_alphabet_size: int,
        word_emb_dim: int = 256,
        word_hidden_dim: int = 512,
        char_emb_dim: int = 32,
        char_hidden_dim: int = 64,
        word_feature_extractor: str = "CNN",
        cnn_layer: int = 4,
        cnn_kernel: int = 5,
        lstm_layers: int = 3,
        dropout: float = 0.2,
        word_dropout: float = 0.05,
        use_char: bool = True,
        use_idcnn: bool = False,
        use_sepcnn: bool = False,
        use_sepcnn_rc: bool = False,
        use_bilstm: bool = False,
        use_crf: bool = True,
        average_batch: bool = False,
        use_batchnorm: bool = True,
        pretrain_word_embedding: Optional[np.array] = None,
        gpu: bool = False,
    ):
        super(TokenClassificationModel, self).__init__()
        self.gpu = gpu
        ## add two more label for downlayer lstm, use original label size for CRF
        self.word_hidden = WordSequence(
            word_alphabet_size,
            char_alphabet_size,
            label_alphabet_size + 2,
            word_emb_dim,
            word_hidden_dim,
            char_emb_dim,
            char_hidden_dim,
            word_feature_extractor,
            cnn_layer,
            cnn_kernel,
            lstm_layers,
            dropout,
            word_dropout,
            use_char,
            use_idcnn,
            use_sepcnn,
            use_sepcnn_rc,
            use_bilstm,
            use_batchnorm,
            pretrain_word_embedding,
            gpu,
        )
        self.use_crf = use_crf
        if self.use_crf:
            self.crf = CRF(label_alphabet_size, self.gpu)
        self.average_batch = average_batch

    def calculate_loss(
        self,
        word_inputs,
        feature_inputs,
        word_seq_lengths,
        char_inputs,
        char_seq_lengths,
        char_seq_recover,
        batch_label,
        mask,
    ):
        outs = self.word_hidden.forward(
            word_inputs,
            feature_inputs,
            word_seq_lengths,
            char_inputs,
            char_seq_lengths,
            char_seq_recover,
        )
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        else:
            loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
        if self.average_batch:
            total_loss = total_loss / batch_size
        return total_loss

    def forward(
        self,
        word_inputs,
        feature_inputs,
        word_seq_lengths,
        char_inputs,
        char_seq_lengths,
        char_seq_recover,
        mask,
    ):
        outs = self.word_hidden.forward(
            word_inputs,
            feature_inputs,
            word_seq_lengths,
            char_inputs,
            char_seq_lengths,
            char_seq_recover,
        )
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            scores, tag_seq = self.crf.viterbi_decode(outs, mask)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            ## filter padded position with zero
            tag_seq = mask.long() * tag_seq
        return tag_seq

    def decode_nbest(
        self,
        word_inputs,
        feature_inputs,
        word_seq_lengths,
        char_inputs,
        char_seq_lengths,
        char_seq_recover,
        mask,
        nbest,
    ):
        if not self.use_crf:
            print("Nbest output is currently supported only for CRF! Exit...")
            exit(0)
        outs = self.word_hidden.forward(
            word_inputs,
            feature_inputs,
            word_seq_lengths,
            char_inputs,
            char_seq_lengths,
            char_seq_recover,
        )
        word_inputs.size(0)
        word_inputs.size(1)
        scores, tag_seq = self.crf.viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq


class Alphabet:
    def __init__(self, name, label=False, keep_growing=False):
        self.name = name
        self.UNKNOWN = "</unk>"
        self.PAD = "</pad>"
        self.label = label
        if name == "label":
            self.label = True
        self.keep_growing = keep_growing
        self.instance2index = {self.PAD: 0}
        self.index2instance = {0: self.PAD}
        self._next_index = 1
        if not self.label:
            self.add(self.UNKNOWN)
        else:
            self.add("O")

    def clear(self):
        self.instance2index = {self.PAD: 0}
        self.index2instance = {0: self.PAD}
        self._next_index = 1

    def add(self, instance):
        if instance not in self.instance2index:
            self.index2instance[self._next_index] = instance
            self.instance2index[instance] = self._next_index
            self._next_index += 1

    def get_index(self, instance):
        if instance in self.instance2index:
            return self.instance2index[instance]
        elif self.keep_growing:
            index = self._next_index
            self.add(instance)
            return index
        else:
            return self.instance2index["O" if self.label else self.UNKNOWN]

    def get_instance(self, index: int) -> str:
        if int(index) in self.index2instance:
            return self.index2instance[int(index)]
        else:
            print(
                f"WARNING:Alphabet index {index}({type(index)}) is out of {len(self.index2instance)}, return the UNK/O."
            )
            return self.UNKNOWN

    def size(self):
        return len(self.index2instance)

    def items(self):
        return self.instance2index.items()

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        return {
            "instance2index": self.instance2index,
            "index2instance": self.index2instance,
        }

    def from_json(self, data):
        self.index2instance = data["index2instance"]
        self.index2instance = {int(k): v for k, v in self.index2instance.items()}
        self.instance2index = data["instance2index"]
        self.instance2index = {k: int(v) for k, v in self.instance2index.items()}

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.name
        try:
            json.dump(
                self.get_content(),
                open(os.path.join(output_directory, saving_name + ".json"), "w"),
            )
        except Exception as e:
            print("Exception: Alphabet is not saved: " % repr(e))

    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        loading_name = name if name else self.name
        self.from_json(
            json.load(open(os.path.join(input_directory, loading_name + ".json")))
        )

    @staticmethod
    def normalize_word(word):
        new_word = ""
        for char in word:
            if char.isdigit():
                new_word += "0"
            else:
                new_word += char
        return new_word


class ExamplesBuilder:
    def __init__(
        self,
        data_dir: str,
        split: Union[Split, str],
        delimiter: str = "\t",
        is_bio: bool = False,
    ):
        self.examples = self.read_conll03_file(
            data_dir, split, delimiter=delimiter, is_bio=is_bio
        )
        print(f"0-th sentence length: {len(self.examples[0].words)}")
        print(self.examples[0].words[:10])
        print(self.examples[0].labels[:10])
        # exit(0)

    @staticmethod
    def is_boundary_line(line: str) -> bool:
        return line.startswith("-DOCSTART-") or line == "" or line == "\n"

    def bio2biolu(
        self, lines: ListStr, label_idx: int = -1, delimiter: str = "\t"
    ) -> ListStr:
        new_lines = []
        n_lines = len(lines)
        for i, line in enumerate(lines):
            if self.is_boundary_line(line):
                new_lines.append(line)
            else:
                next_iob = None
                if i < n_lines - 1:
                    next_line = lines[i + 1].strip()
                    if not self.is_boundary_line(next_line):
                        next_iob = next_line.split(delimiter)[label_idx][0]

                line = line.strip()
                current_line_content = line.split(delimiter)
                current_label = current_line_content[label_idx]
                word = current_line_content[0]
                tag_type = current_label[2:]
                iob = current_label[0]

                iob_transition = (iob, next_iob)
                current_iob = iob
                if iob_transition == ("B", "I"):
                    current_iob = "B"
                elif iob_transition == ("I", "I"):
                    current_iob = "I"
                elif iob_transition in {("B", "O"), ("B", "B"), ("B", None)}:
                    current_iob = "U"
                elif iob_transition in {("I", "B"), ("I", "O"), ("I", None)}:
                    current_iob = "L"
                elif iob == "O":
                    current_iob = "O"
                else:
                    # logger.warning(f"Invalid BIO transition: {iob_transition}")
                    if iob not in set("BIOLU"):
                        current_iob = "O"
                biolu = f"{current_iob}-{tag_type}" if current_iob != "O" else "O"
                new_line = f"{word}{delimiter}{biolu}"
                new_lines.append(new_line)
        return new_lines

    def read_conll03_file(
        self,
        data_dir: str,
        mode: Union[Split, str],
        label_idx: int = -1,
        delimiter: str = "\t",
        is_bio: bool = False,
    ) -> List[InputExample]:
        """
        Read token-wise data like CoNLL2003 from file
        """

        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}.txt")
        lines = []
        with open(file_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f]  # if line.strip()]

        if is_bio:
            lines = self.bio2biolu(lines)
        # guid_index = 1
        examples = []
        words = []
        labels = []
        for line in lines:
            if self.is_boundary_line(line):
                if words:
                    examples.append(InputExample(words, labels))
                    # guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.strip().split(delimiter)
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[label_idx])
                else:
                    # for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(words, labels))
        return examples


class Tokenizer:
    def __init__(self):
        self.wakati = MeCab.Tagger("-Owakati")
        self.wakati.parse("")

    def tokenize_texts(self, texts: ListStr) -> List[str]:
        return list(filter(None, map(self.tokenize, texts)))

    def tokenize(self, text: str) -> Optional[str]:
        """returns tokenized string splitted by space"""
        tokenized = self.wakati.parse(text)
        if tokenized is not None:
            return tokenized.strip()
        return None

    def make_dummy_examples(self, texts) -> List[InputExample]:
        tokens_list = [s.split(" ") for s in self.tokenize_texts(texts)]
        # fill dummy label on prediction
        return [InputExample(tokens, ["O" for _ in tokens]) for tokens in tokens_list]


class TokenClassificationDataset(Dataset):
    """
    Build feature dataset so that the model can load
    """

    def __init__(
        self,
        examples: List[InputExample],
        word_alphabet: Alphabet,
        char_alphabet: Alphabet,
        label_alphabet: Alphabet,
        tokens_per_batch: int = 256,
        window_stride: Optional[int] = None,
        number_normalized: bool = True,
        char_padding_size=-1,
        char_padding_symbol="</pad>",
    ):
        """tackle with long text with windowing and striding"""
        # self.examples: List[InputExample]
        self.fexamples: List[FeatureExample]
        self.features: List[InputFeature]

        self.examples = examples
        self.word_alphabet = word_alphabet
        self.char_alphabet = char_alphabet
        self.label_alphabet = label_alphabet

        self.number_normalized = number_normalized
        self.normalized_number = "0"

        self.tokens_per_batch = tokens_per_batch
        self.window_stride = tokens_per_batch
        if window_stride is not None:
            if window_stride > 0 and window_stride <= tokens_per_batch:
                self.window_stride = window_stride
        # segment input text into blocks of `tokens_per_batch` tokens
        token_blocks = [
            [
                InputExample(
                    example.words[st : st + self.tokens_per_batch],
                    self._fix_boundary_labels(example.labels[st : st + self.tokens_per_batch]),
                )
                for st in range(0, len(example.labels), self.window_stride)
            ]
            for example in self.examples
        ]
        segmented_examples = [s for block in token_blocks for s in block]  # flatten

        # convert to features (ids)
        instence_texts: List[FeatureExample] = []
        instence_Ids: List[InputFeature] = []
        for sentence in segmented_examples:
            words = []
            features = []
            chars = []
            labels = []
            word_Ids = []
            feature_Ids = []
            char_Ids = []
            label_Ids = []
            for word, label in zip(sentence.words, sentence.labels):
                words.append(word)
                labels.append(label)
                if self.number_normalized:
                    word = "".join(
                        [
                            self.normalized_number if char.isdigit() else char
                            for char in word
                        ]
                    )
                word_Ids.append(self.word_alphabet.get_index(word))
                label_Ids.append(self.label_alphabet.get_index(label))
                ## get features
                feat_list = []
                feat_Id = []
                # for idx in range(feature_num):
                #     feat_idx = pairs[idx + 1].split("]", 1)[-1]
                #     feat_list.append(feat_idx)
                #     feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
                features.append(feat_list)
                feature_Ids.append(feat_Id)
                ## get char
                char_list = list(word)
                if char_padding_size > 0:
                    char_number = len(char_list)
                    if char_number < char_padding_size:
                        char_list = char_list + [char_padding_symbol] * (
                            char_padding_size - char_number
                        )
                    assert len(char_list) == char_padding_size
                else:
                    ### not padding
                    pass
                char_Id = [self.char_alphabet.get_index(char) for char in char_list]
                chars.append(char_list)
                char_Ids.append(char_Id)
            # NOTE: no padding for words
            instence_texts.append(
                FeatureExample(
                    words,
                    features,
                    chars,
                    labels,
                )
            )
            instence_Ids.append(
                InputFeature(
                    word_Ids,
                    feature_Ids,
                    char_Ids,
                    label_Ids,
                )
            )
        self.fexamples = instence_texts
        self.features = instence_Ids

        self._n_features = len(self.features)

    def _fix_boundary_labels(self, labels) -> ListStr:
        """
        assert _fix_boundary_labels(['I-X', 'L-X', 'O']) == ['O', 'O', 'O']
        assert _fix_boundary_labels(['L-X', 'O']) == ['O', 'O']
        assert _fix_boundary_labels(['O', 'B-X', 'I-X']) == ['O', 'O', 'O']
        assert _fix_boundary_labels(['O', 'B-X']) == ['O', 'O']
        """
        if not labels:
            return labels
        else:
            # fix head boundary
            if labels[0].startswith('L-'):
                labels[0] = 'O'
            elif labels[0].startswith('I-'):
                _until = 0
                for i in range(1, len(labels)):
                    if labels[i].startswith('L-'):
                        _until = i + 1
                        break
                for i in range(0, _until):
                    labels[i] = 'O'
            # fix tail boundary
            if labels[-1].startswith('B-'):
                labels[-1] = 'O'
            elif labels[-1].startswith('I-'):
                _from = -1
                for i in range(2, len(labels)+1):
                    if labels[-i].startswith('B-'):
                        _from = -i
                        break
                for i in range(_from, 0, 1):
                    labels[i] = 'O'
            return labels


    def __len__(self):
        return self._n_features

    def __getitem__(self, idx) -> InputFeature:
        return self.features[idx]


class TokenClassificationBatch:
    def __init__(self, input_batch_list: List[InputFeature]):
        """from batchify_sequence_labeling_with_label in NCRFpp
        features: list of words, chars and labels, various length.
        [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)

        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
        """

        CHAR_PADDING_ID = 0

        self.word_seq_tensor: torch.Tensor
        self.feature_seq_tensors: List[torch.Tensor]
        self.word_seq_lengths: torch.Tensor
        self.word_seq_recover: torch.Tensor
        self.char_seq_tensor: torch.Tensor
        self.char_seq_lengths: torch.Tensor
        self.char_seq_recover: torch.Tensor
        self.label_seq_tensor: torch.Tensor
        self.mask: torch.Tensor

        self._n_features = len(input_batch_list)

        # if_train = False
        batch_size = len(input_batch_list)
        words = [sent.word_ids for sent in input_batch_list]
        features = [np.asarray(sent.feature_ids) for sent in input_batch_list]
        feature_num = len(features[0][0])
        chars = [sent.character_ids for sent in input_batch_list]
        labels = [sent.label_ids for sent in input_batch_list]

        ### deal with word

        word_seq_lengths = torch.LongTensor(list(map(len, words)))
        max_seq_len = word_seq_lengths.max().item()
        word_seq_tensor = torch.zeros(
            (batch_size, max_seq_len),  # requires_grad=if_train
        ).long()
        label_seq_tensor = torch.zeros(
            (batch_size, max_seq_len),  # requires_grad=if_train
        ).long()
        feature_seq_tensors = [
            torch.zeros(
                (batch_size, max_seq_len),  # requires_grad=if_train
            ).long()
            for idx in range(feature_num)
        ]
        mask = torch.zeros(
            (batch_size, max_seq_len),  # requires_grad=if_train
        ).bool()
        for idx, (seq, label, seqlen) in enumerate(
            zip(words, labels, word_seq_lengths)
        ):
            seqlen = seqlen.item()
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
            mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
            for idy in range(feature_num):
                feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(
                    features[idx][:, idy]
                )
        # sort by word_seq_lengths
        word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
        word_seq_tensor = word_seq_tensor[word_perm_idx]
        feature_seq_tensors = [
            feature_seq_tensors[idx][word_perm_idx] for idx in range(feature_num)
        ]
        label_seq_tensor = label_seq_tensor[word_perm_idx]
        mask = mask[word_perm_idx]

        ### deal with char
        # pad_chars (batch_size, max_seq_len)
        pad_chars = [
            chars[idx] + [[CHAR_PADDING_ID]] * (max_seq_len - len(chars[idx]))
            for idx in range(len(chars))
        ]
        length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
        max_word_len = max([max(ints) for ints in length_list])
        char_seq_tensor = torch.zeros(
            (batch_size, max_seq_len, max_word_len),  # requires_grad=if_train
        ).long()
        char_seq_lengths = torch.LongTensor(length_list)
        for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
            for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
                char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
        # (batch_size, max_seq_len, max_word_len) -> (batch_size * max_seq_len, max_word_len)
        char_seq_tensor = char_seq_tensor[word_perm_idx].view(
            batch_size * max_seq_len, -1
        )
        char_seq_lengths = char_seq_lengths[word_perm_idx].view(
            batch_size * max_seq_len,
        )
        # sort by char_seq_lengths
        char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
        char_seq_tensor = char_seq_tensor[char_perm_idx]
        _, char_seq_recover = char_perm_idx.sort(0, descending=False)
        _, word_seq_recover = word_perm_idx.sort(0, descending=False)

        self.word_seq_tensor = word_seq_tensor
        self.feature_seq_tensors = feature_seq_tensors
        self.word_seq_lengths = word_seq_lengths
        self.word_seq_recover = word_seq_recover
        self.char_seq_tensor = char_seq_tensor
        self.char_seq_lengths = char_seq_lengths
        self.char_seq_recover = char_seq_recover
        self.label_seq_tensor = label_seq_tensor
        self.mask = mask

    def __len__(self):
        return self._n_features

    def __getitem__(self, item):
        return getattr(self, item)


class TokenClassificationDataModule(pl.LightningDataModule):
    """
    Prepare dataset and build DataLoader
    """

    def __init__(self, hparams: argparse.Namespace):

        super().__init__()
        self.do_train = hparams.do_train
        self.do_predict = hparams.do_predict

        self.tokens_per_batch = hparams.tokens_per_batch
        self.window_stride = hparams.window_stride
        self.number_normalized = hparams.number_normalized

        self.data_dir = hparams.data_dir

        self.train_batch_size = hparams.train_batch_size
        self.eval_batch_size = hparams.eval_batch_size
        self.num_workers = hparams.num_workers
        self.num_samples = hparams.num_samples

        self.vocab_path = hparams.model_dir

        self.delimiter = hparams.delimiter
        self.is_bio = hparams.is_bio

    def prepare_data(self):
        """
        Downloads the data and prepare the tokenizer
        """

        if self.do_train:
            self.train_examples = ExamplesBuilder(
                self.data_dir, Split.train, self.delimiter, self.is_bio
            ).examples
            self.val_examples = ExamplesBuilder(
                self.data_dir, Split.dev, self.delimiter, self.is_bio
            ).examples
            self.test_examples = ExamplesBuilder(
                self.data_dir, Split.test, self.delimiter, self.is_bio
            ).examples
            if self.num_samples > 0:
                self.train_examples = self.train_examples[: self.num_samples]
                self.val_examples = self.val_examples[: self.num_samples]
                self.test_examples = self.test_examples[: self.num_samples]
            # build vocab from examples
            self.word_alphabet = Alphabet("word")
            self.char_alphabet = Alphabet("character")
            self.label_alphabet = Alphabet("label")
            all_examples = self.train_examples + self.val_examples + self.test_examples
            self.build_alphabet(all_examples)
            # save vocab
            os.makedirs(self.vocab_path, exist_ok=True)
            self.char_alphabet.save(self.vocab_path)
            self.word_alphabet.save(self.vocab_path)
            self.label_alphabet.save(self.vocab_path)

            self.train_dataset = self.create_dataset(self.train_examples)
            self.val_dataset = self.create_dataset(self.val_examples)
            self.test_dataset = self.create_dataset(self.test_examples)

        else:
            if self.vocab_path.endswith(".pkl"):
                with open(self.vocab_path, "rb") as fp:
                    model_dict = pickle.load(fp)
                    self.word_alphabet = model_dict["word_alphabet"]
                    self.char_alphabet = model_dict["char_alphabet"]
                    self.label_alphabet = model_dict["label_alphabet"]
            else:
                word_alphabet = Alphabet("word")
                word_alphabet.load(self.vocab_path)
                char_alphabet = Alphabet("character")
                char_alphabet.load(self.vocab_path)
                label_alphabet = Alphabet("label")
                label_alphabet.load(self.vocab_path)
                self.word_alphabet = word_alphabet
                self.char_alphabet = char_alphabet
                self.label_alphabet = label_alphabet

            self.tokenizer = Tokenizer()
            # constructed on demand
            self.examples = None
            self.dataset = None

        if any(
            label.startswith("L-") or label.startswith("U-")
            for label, _ in self.label_alphabet.items()
        ):
            self.tag_scheme = "BILOU"
        else:
            self.tag_scheme = "BIO"
        self.show_data_summary()

    def build_alphabet(self, examples: List[InputExample]):
        print("Building alphabet vocabulary...")
        time_start = time.time()
        for ex in examples:
            for word in ex.words:
                if self.number_normalized:
                    word = Alphabet.normalize_word(word)
                self.word_alphabet.add(word)
                for char in word:
                    self.char_alphabet.add(char)

            for label in ex.labels:
                self.label_alphabet.add(label)

            # ## build feature alphabet
            # for idx in range(self.feature_num):
            #     feat_idx = pairs[idx + 1].split("]", 1)[-1]
            #     self.feature_alphabets[idx].add(feat_idx)
        time_finish = time.time()
        timecost = time_finish - time_start
        print(self.label_alphabet.instance2index)
        print(f"End: {timecost / 60.} min.")

    def show_data_summary(self):

        print("++" * 50)
        print("DATA SUMMARY:")
        print("    Start   Sequence   Laebling   task...")
        print("    Tag          scheme: BILOU")
        # print("     Tag vocab: {}".format(self.label_alphabet.vocab()))
        print("    MAX SENTENCE LENGTH: {}".format(self.tokens_per_batch))
        print("    Word  alphabet size: {}".format(self.word_alphabet.size()))
        print("    Char  alphabet size: {}".format(self.char_alphabet.size()))
        print("    Label alphabet size: {}".format(self.label_alphabet.size()))
        if self.do_train:
            print("    Train instance number: {}".format(len(self.train_examples)))
            print("    Dev   instance number: {}".format(len(self.val_examples)))
            print("    Test  instance number: {}".format(len(self.test_examples)))
            print("    Train BatchSize: {}".format(self.train_batch_size))
        print("    Eval BatchSize: {}".format(self.eval_batch_size))
        print("    Workers: {}".format(self.num_workers))
        print("    Cores: {}".format(multiprocessing.cpu_count()))
        print("++" * 50)
        sys.stdout.flush()

    def setup(self, stage=None):
        """
        split the data into train, test, validation data
        :param stage: Stage - training or testing
        """
        # our dataset is splitted in prior

    def create_dataset(
        self, examples: List[InputExample]
    ) -> TokenClassificationDataset:
        return TokenClassificationDataset(
            examples,
            self.word_alphabet,
            self.char_alphabet,
            self.label_alphabet,
            self.tokens_per_batch,
            self.window_stride,
            self.number_normalized,
        )

    @staticmethod
    def create_dataloader(
        ds: TokenClassificationDataset,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = False,
    ) -> DataLoader:
        return DataLoader(
            ds,
            collate_fn=TokenClassificationBatch,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        if not self.do_train:
            self.train_examples = ExamplesBuilder(
                self.data_dir, Split.train, self.delimiter, self.is_bio
            ).examples
            self.train_dataset = self.create_dataset(self.train_examples)
        return self.create_dataloader(
            self.train_dataset,
            self.train_batch_size,
            self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        if not self.do_train:
            self.val_examples = ExamplesBuilder(
                self.data_dir, Split.dev, self.delimiter, self.is_bio
            ).examples
            self.val_dataset = self.create_dataset(self.val_examples)
        return self.create_dataloader(
            self.val_dataset, self.eval_batch_size, self.num_workers, shuffle=False
        )

    def test_dataloader(self):
        if not self.do_train:
            self.test_examples = ExamplesBuilder(
                self.data_dir, Split.test, self.delimiter, self.is_bio
            ).examples
            self.test_dataset = self.create_dataset(self.test_examples)
        return self.create_dataloader(
            self.test_dataset, self.eval_batch_size, self.num_workers, shuffle=False
        )

    def get_prediction_dataloader(self, texts: ListStr):
        self.examples = self.tokenizer.make_dummy_examples(texts)
        self.dataset = self.create_dataset(self.examples)
        return self.create_dataloader(
            self.dataset, self.eval_batch_size, self.num_workers, shuffle=False
        )

    def total_steps(self) -> int:
        """
        The number of total training steps that will be run. Used for lr scheduler purposes.
        """
        if not self.do_train:
            return -1  # TODO: logging
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = (
            self.hparams.train_batch_size
            * self.hparams.accumulate_grad_batches
            * num_devices
        )
        return (self.dataset_size / effective_batch_size) * self.hparams.max_epochs

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--train_batch_size",
            type=int,
            default=32,
            help="input batch size for training (default: 32)",
        )
        parser.add_argument(
            "--eval_batch_size",
            type=int,
            default=32,
            help="input batch size for validation/test (default: 32)",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            metavar="N",
            help="number of workers (default: 3)",
        )
        parser.add_argument(
            "--tokens_per_batch",
            default=250,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--window_stride",
            default=125,
            type=int,
            help="The stride of moving window over input sequence."
            "This must be shorter than tokens_per_batch.",
        )
        parser.add_argument(
            "--num_samples",
            type=int,
            default=-1,
            metavar="N",
            help="Number of samples to be used for training and evaluation steps",
        )
        return parser


class TokenClassificationModule(pl.LightningModule):
    """
    Initialize a model and config for token-classification
    """

    def __init__(
        self,
        hparams: Union[Dict, argparse.Namespace],
    ):

        # NOTE: internal code may pass hparams as dict **kwargs
        if isinstance(hparams, Dict):
            hparams = argparse.Namespace(**hparams)
        super().__init__()
        # Enable to access arguments via self.hparams
        self.save_hyperparameters(hparams)

        vocab_path = hparams.model_dir
        if hparams.do_train:
            self.gpu = hparams.gpu
            # NOTE: Fit and save vocab first in DataModule.
            self.model = self._load_model(
                vocab_path,
                hparams.config_path,
                pretrain_embed_path=hparams.pretrain_embed_path,
            )
        elif hparams.do_predict:
            # if self.gpu:
            #     loaded_state = torch.load(hparams.model_path)
            # else:
            self.gpu = False
            if hparams.model_path.endswith(".pkl"):
                # model と Alphabet(vocab) をまとめてロード
                self._load_pickle(hparams.model_path)
            else:
                # NOTE: Fit and save vocab first in DataModule.
                self.model = self._load_model(
                    vocab_path,
                    hparams.config_path,
                    model_path=hparams.model_path,
                )
            self.model.eval()  # plでは不要かも

        self.model_path: str = hparams.model_path
        self.nbest = hparams.nbest  # TODO: to be implemented
        # self.show_model_summary()
        self.train_loss_log = os.path.join(hparams.model_dir, 'train_loss.csv')
        self.dev_loss_log = os.path.join(hparams.model_dir, 'dev_loss.csv')
        self.test_loss_log = os.path.join(hparams.model_dir, 'test_loss.csv')
        self.loss_log_format = '{},{},{}'
        with open(self.train_loss_log, 'w') as fp:
            fp.write('PRECISION,RECALL,F1')
            fp.write('\n')
        with open(self.dev_loss_log, 'w') as fp:
            fp.write('PRECISION,RECALL,F1')
            fp.write('\n')
        with open(self.test_loss_log, 'w') as fp:
            fp.write('PRECISION,RECALL,F1')
            fp.write('\n')

    def _load_model(
        self,
        vocab_path: str,
        config_path: str,
        model_path: Optional[str] = None,
        pretrain_embed_path: Optional[str] = None,
    ) -> TokenClassificationModel:
        self._load_vocab(vocab_path)
        params = self._load_params(config_path)
        if pretrain_embed_path is not None:
            params["pretrain_word_embedding"] = self._build_pretrain_embedding(
                pretrain_embed_path, params["word_emb_dim"]
            )
        print(f"Loaded params from {config_path}: {params}")
        model = TokenClassificationModel(**params)
        if model_path is not None:
            loaded_state = torch.load(model_path, map_location=torch.device("cpu"))
            assert (
                loaded_state["word_hidden.wordrep.word_embedding.weight"].shape[0]
                == params["word_alphabet_size"]
            )
            assert (
                loaded_state[
                    "word_hidden.wordrep.char_feature.char_embeddings.weight"
                ].shape[0]
                == params["char_alphabet_size"]
            )
            assert (
                loaded_state["word_hidden.hidden2tag.weight"].shape[0]
                == params["label_alphabet_size"] + 2
            )
            model.load_state_dict(loaded_state)
        return model

    def save_pickle(self, save_path: Optional[Union[str, Path]] = None):
        # paramsに込めるAlphabet(vocab)データもまとめて保存
        if save_path is None:
            save_path = self.model_path + ".pkl"

        with open(save_path, "wb") as fp:
            model_dict = {
                "model": self.model,
                "word_alphabet": self.word_alphabet,
                "char_alphabet": self.char_alphabet,
                "label_alphabet": self.label_alphabet,
            }
            pickle.dump(model_dict, fp)

    def _load_pickle(self, model_path: str):
        with open(model_path, "rb") as fp:
            model_module = pickle.load(fp)
            self.model = model_module["model"]
            self.word_alphabet = model_module["word_alphabet"]
            self.char_alphabet = model_module["char_alphabet"]
            self.label_alphabet = model_module["label_alphabet"]

    def _load_vocab(self, vocab_path: str):
        word_alphabet = Alphabet("word")
        word_alphabet.load(vocab_path)
        char_alphabet = Alphabet("character")
        char_alphabet.load(vocab_path)
        label_alphabet = Alphabet("label")
        label_alphabet.load(vocab_path)
        self.word_alphabet = word_alphabet
        self.char_alphabet = char_alphabet
        self.label_alphabet = label_alphabet

    def _load_params(self, config: str) -> dict:
        params = dict(
            word_alphabet_size=0,
            char_alphabet_size=0,
            label_alphabet_size=0,
            word_emb_dim=256,
            word_hidden_dim=512,
            char_emb_dim=32,
            char_hidden_dim=64,
            word_feature_extractor="CNN",
            cnn_layer=4,
            cnn_kernel=5,
            dropout=0.2,
            word_dropout=0.05,
            use_char=True,
            use_idcnn=False,
            use_sepcnn=False,
            use_sepcnn_rc=False,
            use_bilstm=False,
            use_batchnorm=True,
            use_crf=True,
            gpu=False,
        )
        with open(config) as fp:
            items = [
                l.strip().split("=")
                for l in fp.readlines()
                if l.strip() and not l.startswith("#") and len(l.split("=")) == 2
            ]
        for k, v in items:
            if k in params:
                if k == "word_feature_extractor":
                    params[k] = v
                else:
                    params[k] = eval(v)
        params["word_alphabet_size"] = self.word_alphabet.size()  # </pad>,</unk>
        params["char_alphabet_size"] = self.char_alphabet.size()  # </pad>,</unk>
        params["label_alphabet_size"] = self.label_alphabet.size()  # </pad>

        params["gpu"] = torch.cuda.is_available()

        return params

    def _build_pretrain_embedding(
        self, embedding_path: str, embedd_dim: int
    ) -> np.array:
        embedd_dict = dict()
        if os.path.exists(embedding_path):
            # line := 'token dim1 dim2 ..\n'
            with open(embedding_path, "r", encoding="utf8") as fp:
                for line in fp:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    tokens = line.split()
                    if not embedd_dim + 1 == len(tokens):
                        continue
                    embedd = np.empty([1, embedd_dim])
                    embedd[:] = tokens[1:]
                    first_col = tokens[0]
                    embedd_dict[first_col] = embedd

        scale = np.sqrt(3.0 / embedd_dim)
        pretrain_emb = np.empty([self.word_alphabet.size(), embedd_dim])
        perfect_match = 0
        case_match = 0
        not_match = 0
        for word, index in self.word_alphabet.items():  # includes </pad>,</unk>
            if word in {self.word_alphabet.UNKNOWN, self.word_alphabet.PAD}:
                pretrain_emb[0, :] = np.zeros((1, embedd_dim))
            elif word in embedd_dict:
                pretrain_emb[index, :] = embedd_dict[word]
                perfect_match += 1
            elif word.lower() in embedd_dict:
                pretrain_emb[index, :] = embedd_dict[word.lower()]
                case_match += 1
            else:
                # init OOV vector
                pretrain_emb[index, :] = np.random.uniform(
                    -scale, scale, [1, embedd_dim]
                )
                not_match += 1

        pretrained_size = len(embedd_dict)
        print(
            "Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"
            % (
                pretrained_size,
                perfect_match,
                case_match,
                not_match,
                100 * (not_match + 0.0) / self.word_alphabet.size(),
            )
        )
        return pretrain_emb

    def recover_word(self, word_variable, mask_variable, word_recover) -> ListListStr:
        """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
        """
        word_variable = word_variable[word_recover]
        mask_variable = mask_variable[word_recover]
        batch_size = word_variable.size(0)
        seq_len = word_variable.size(1)

        mask = mask_variable.cpu().data.numpy()
        word_ids = word_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        return [
            [
                self.word_alphabet.get_instance(word_ids[idx][idy])
                for idy in range(seq_len)
                if mask[idx][idy] != 0
            ]
            for idx in range(batch_size)
        ]

    def recover_label(self, pred_variable, gold_variable, mask_variable, word_recover):
        """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
        """
        pred_variable = pred_variable[word_recover]
        gold_variable = gold_variable[word_recover]
        mask_variable = mask_variable[word_recover]
        batch_size = gold_variable.size(0)
        seq_len = gold_variable.size(1)

        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        gold_tag = gold_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        pred_label = []
        gold_label = []
        for idx in range(batch_size):
            pred = [
                self.label_alphabet.get_instance(pred_tag[idx][idy])
                for idy in range(seq_len)
                if mask[idx][idy] != 0
            ]
            gold = [
                self.label_alphabet.get_instance(gold_tag[idx][idy])
                for idy in range(seq_len)
                if mask[idx][idy] != 0
            ]
            assert len(pred) == len(gold)
            pred_label.append(pred)
            gold_label.append(gold)
        return pred_label, gold_label

    def recover_nbest_label(self, pred_variable, mask_variable, word_recover):
        """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
        """
        pred_variable = pred_variable[word_recover]
        mask_variable = mask_variable[word_recover]
        batch_size = pred_variable.size(0)
        seq_len = pred_variable.size(1)
        nbest = pred_variable.size(2)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        pred_label = []
        for idx in range(batch_size):
            pred = []
            for idz in range(nbest):
                each_pred = [
                    self.label_alphabet.get_instance(pred_tag[idx][idy][idz])
                    for idy in range(seq_len)
                    if mask[idx][idy] != 0
                ]
                pred.append(each_pred)
            pred_label.append(pred)
        return pred_label

    def forward(
        self,
        word_inputs,
        feature_inputs,
        word_seq_lengths,
        char_inputs,
        char_seq_lengths,
        char_seq_recover,
        mask,
    ):
        return self.model.forward(
            word_inputs,
            feature_inputs,
            word_seq_lengths,
            char_inputs,
            char_seq_lengths,
            char_seq_recover,
            mask,
        )

    def calculate_loss(self, batch: TokenClassificationBatch):
        word_inputs = batch.word_seq_tensor.to(self.device)
        word_seq_lengths = batch.word_seq_lengths.to(self.device)
        feature_inputs = [b.to(self.device) for b in batch.feature_seq_tensors]
        char_inputs = batch.char_seq_tensor.to(self.device)
        char_seq_lengths = batch.char_seq_lengths.to(self.device)
        char_seq_recover = batch.char_seq_recover.to(self.device)
        batch_label = batch.label_seq_tensor.to(self.device)
        mask = batch.mask.to(self.device)
        return self.model.calculate_loss(
            word_inputs,
            feature_inputs,
            word_seq_lengths,
            char_inputs,
            char_seq_lengths,
            char_seq_recover,
            batch_label,
            mask,
        )

    def predict(self, batch: TokenClassificationBatch):
        word_inputs = batch.word_seq_tensor.to(self.device)
        word_seq_lengths = batch.word_seq_lengths.to(self.device)
        word_seq_recover = batch.word_seq_recover.to(self.device)
        feature_inputs = [b.to(self.device) for b in batch.feature_seq_tensors]
        char_inputs = batch.char_seq_tensor.to(self.device)
        char_seq_lengths = batch.char_seq_lengths.to(self.device)
        char_seq_recover = batch.char_seq_recover.to(self.device)
        batch_label = batch.label_seq_tensor.to(self.device)
        mask = batch.mask.to(self.device)

        if self.nbest > 1:
            scores, nbest_tag_seq = self.model.decode_nbest(
                word_inputs,
                feature_inputs,
                word_seq_lengths,
                char_inputs,
                char_seq_lengths,
                char_seq_recover,
                mask,
                self.nbest,
            )
            # nbest_pred_result = self.recover_nbest_label(
            #     nbest_tag_seq, mask, word_seq_recover
            # )
            # pred_scores = scores[word_seq_recover].cpu().data.numpy().tolist()
            tag_seq = nbest_tag_seq[:, :, 0]
        else:
            tag_seq = self.model.forward(
                word_inputs,
                feature_inputs,
                word_seq_lengths,
                char_inputs,
                char_seq_lengths,
                char_seq_recover,
                # batch_label,
                mask,
            )
        words = self.recover_word(word_inputs, mask, word_seq_recover)
        pred_label, gold_label = self.recover_label(
            tag_seq, batch_label, mask, word_seq_recover
        )
        return words, pred_label, gold_label

    def eval_f1(self, outputs, output_file=None):
        preds_list = [p for x in outputs for p in x["prediction"]]
        target_list = [p for x in outputs for p in x["target"]]
        accuracy = accuracy_score(target_list, preds_list)
        p, r, f, s = precision_recall_fscore_support(
            target_list, preds_list, scheme=BILOU, average="micro"
        )

        if output_file is not None and os.path.exists(output_file):
            with open(output_file, 'a') as fp:
                fp.write(self.loss_log_format.format(p,r,f))
                fp.write('\n')

        return accuracy, p, r, f, s

    def training_step(
        self, train_batch: TokenClassificationBatch, batch_idx
    ) -> Dict[str, torch.Tensor]:
        loss = self.calculate_loss(train_batch)
        self.log("train_loss", loss, prog_bar=True)
        if self.hparams.monitor_training:
            words, pred_label, gold_label = self.predict(train_batch)
            return {
                "loss": loss,
                "target": gold_label,
                "prediction": pred_label,
            }
        else:
            return {"loss": loss}

    def training_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        if self.hparams.monitor_training:
            accuracy, precision, recall, f1, support = self.eval_f1(outputs, self.train_loss_log)
            self.log("train_accuracy", accuracy)
            self.log("train_precision", precision)
            self.log("train_recall", recall)
            self.log("train_f1", f1)
            self.log("train_support", support)

    def validation_step(
        self, val_batch: TokenClassificationBatch, batch_idx
    ) -> Dict[str, torch.Tensor]:
        if self.hparams.monitor == "loss":
            loss = self.calculate_loss(val_batch)
            return {"val_step_loss": loss}
        else:
            words, pred_label, gold_label = self.predict(val_batch)
            return {
                "target": gold_label,
                "prediction": pred_label,
            }

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        if self.hparams.monitor == "loss":
            avg_loss = torch.stack([x["val_step_loss"] for x in outputs]).mean()
            self.log("val_loss", avg_loss, sync_dist=True)
        else:
            accuracy, precision, recall, f1, support = self.eval_f1(outputs, self.dev_loss_log)
            self.log("val_accuracy", accuracy)
            self.log("val_precision", precision)
            self.log("val_recall", recall)
            self.log("val_f1", f1)
            self.log("val_support", support)

    def test_step(
        self, test_batch: TokenClassificationBatch, batch_idx
    ) -> Dict[str, torch.Tensor]:
        words, pred_label, gold_label = self.predict(test_batch)

        return {
            # "scores": pred_scores,
            # "input": words,
            "target": gold_label,
            "prediction": pred_label,
        }

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        accuracy, precision, recall, f1, support = self.eval_f1(outputs, self.test_loss_log)
        self.log("test_accuracy", accuracy)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)
        self.log("test_support", support)

    def predict_step(
        self, test_batch: TokenClassificationBatch, batch_idx
    ) -> Dict[str, torch.Tensor]:
        words, pred_label, gold_label = self.predict(test_batch)

        return {
            "input": words,
            "target": gold_label,
            "prediction": pred_label,
        }

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
            betas=(0.9, 0.999),
            weight_decay=self.hparams.weight_decay,
            # amsgrad=False,
        )
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min" if self.hparams.monitor == "loss" else "max",
            factor=self.hparams.anneal_factor,
            patience=self.hparams.patience,
            min_lr=1e-5,
            verbose=True,
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "monitor": "val_loss" if self.hparams.monitor == "loss" else "val_f1",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability.",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Weight decay if we apply some.",
        )
        parser.add_argument(
            "--learning_rate",
            default=0.015,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--adam_epsilon",
            default=1e-8,
            type=float,
            help="Epsilon for Adam optimizer.",
        )
        parser.add_argument(
            "--patience",
            default=3,
            type=int,
            help="Number of epochs with no improvement after which learning rate will be reduced.",
        )
        parser.add_argument(
            "--anneal_factor",
            default=0.5,
            type=float,
            help="Factor by which the learning rate will be reduced.",
        )
        parser.add_argument(
            "--pretrain_embed_path",
            default=None,
            help="Path for pretreined embedding.",
        )
        parser.add_argument(
            "--word_embed_dim",
            default=256,
            type=int,
            help="dimension of word embedding.",
        )
        parser.add_argument(
            "--word_hidden_dim",
            default=512,
            type=int,
            help="dimension of word hidden state.",
        )
        parser.add_argument(
            "--char_embed_dim",
            default=32,
            type=int,
            help="dimension of character embedding.",
        )
        parser.add_argument(
            "--char_hidden_dim",
            default=64,
            type=int,
            help="dimension of character hidden state.",
        )
        parser.add_argument(
            "--monitor",
            default="loss",
            type=str,
            help="what metrics to monitor(loss/f1)",
        )
        parser.add_argument(
            "--monitor_training", action="store_true", help="Whether to monitor train metrics."
        )        
        parser.add_argument("--use_crf", action="store_true")
        parser.add_argument("--use_char", action="store_true")
        parser.add_argument("--use_idcnn", action="store_true")
        return parser


def make_common_args():
    parser = argparse.ArgumentParser(description="CharCNN-NER module from NCRF++")
    parser.add_argument("--model_path", type=str)  #  default="model/cnn.0.model",
    parser.add_argument("--config_path", type=str)  #  default="model/train.config",
    parser.add_argument("--model_dir", type=str)  # , default="model/"
    parser.add_argument("--nbest", default=1, type=int, help="to be implemented")
    parser.add_argument("--number_normalized", default=True, type=bool)
    parser.add_argument(
        "--output_dir",
        # default="/app/workspace/models",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--data_dir",
        default="/app/workspace/data",
        type=str,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run predictions on the test set.",
    )
    parser.add_argument(
        "--download", action="store_true", help="Whether to download dataset."
    )
    return parser


def build_args(notebook=False):
    parser = make_common_args()
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = TokenClassificationModule.add_model_specific_args(parent_parser=parser)
    parser = TokenClassificationDataModule.add_model_specific_args(parent_parser=parser)
    if not notebook:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=[])
    args.delimiter = " "
    args.is_bio = False
    if args.download:
        download_dataset(args.data_dir)
        args.delimiter = "\t"
        args.is_bio = True
    return args


def save_pickle_in_module(
    save_path: str,
    model_path: str,
    vocab_path: str,
    config_path: str,
    notebook: bool = False,
):
    """Save pickle in the current module hierarchy from pytorch model.
    NOTE: アプリケーションコンテナ内でpickleを実行しないとpickle対象のクラスパスが解決できない。
    """
    args = build_args(notebook)
    args.model_path = model_path
    args.vocab_path = vocab_path
    args.config_path = config_path
    args.do_train = False
    args.do_predict = True
    args.gpu = False
    pl_module = TokenClassificationModule(args)
    pl_module.save_pickle(save_path)


def main_as_plmodule():
    """PyTorch-Lightning Moduleとして訓練・予測を行う"""

    def make_trainer(argparse_args: argparse.Namespace):
        """
        Prepare pl.Trainer with callbacks and args
        """

        # early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=argparse_args.output_dir,
            filename="checkpoint-{epoch}-{val_f1:.2f}",
            save_top_k=10,
            verbose=True,
            monitor="val_loss" if argparse_args.monitor == "loss" else "val_f1",
            mode="min" if argparse_args.monitor == "loss" else "max",
        )
        lr_logger = LearningRateMonitor(logging_interval="step")

        trainer = pl.Trainer.from_argparse_args(
            argparse_args,
            callbacks=[lr_logger, checkpoint_callback],
            deterministic=True,
            accumulate_grad_batches=argparse_args.accumulate_grad_batches,
        )
        return trainer, checkpoint_callback

    args = build_args()
    args.gpu = torch.cuda.is_available()

    pl.seed_everything(args.seed)
    Path(args.output_dir).mkdir(exist_ok=True)

    dm = TokenClassificationDataModule(args)
    dm.prepare_data()

    if args.do_train:
        dm.setup(stage="fit")
        # mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        # mlflow.set_experiment("cnn-ner")
        # mlflow.pytorch.autolog(log_every_n_epoch=1)
        model = TokenClassificationModule(args)
        trainer, checkpoint_callback = make_trainer(args)
        trainer.fit(model, dm)

        trainer.test(ckpt_path=checkpoint_callback.best_model_path)
        # save best model
        best_model = TokenClassificationModule.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        save_path = Path(checkpoint_callback.best_model_path).parent / "best_model.pt"
        torch.save(best_model.model.state_dict(), save_path)
        # save_path = Path(checkpoint_callback.best_model_path).parent / "best_model.pkl"
        # best_model.save_pickle(save_path)
    elif args.do_predict:
        if args.model_path.endswith(".ckpt"):
            ## NOTE: the path structure on training is pickled in .ckpt
            print(args.model_path)
            model = TokenClassificationModule.load_from_checkpoint(args.model_path)
            save_path = Path(args.model_path).parent / "prediction_model.pt"
            torch.save(model.model.state_dict(), save_path)

            trainer, _ = make_trainer(args)
            trainer.test(
                model=model,
                ckpt_path=args.model_path,
                test_dataloaders=dm.test_dataloader(),
            )

            args.model_path = str(save_path)
        else:
            model = TokenClassificationModule(args)

        save_path = Path(args.model_path).parent / "prediction_model.pkl"
        model.save_pickle(save_path)
        datadir = Path(args.data_dir)
        if datadir.exists():
            if (datadir / "test.txt").exists():
                dl = dm.test_dataloader()
                fexamples = dm.test_dataset.fexamples
            else:
                texts = []
                for txt_path in datadir.glob("*.txt"):
                    with open(txt_path) as fp:
                        text = fp.read()
                        texts.append(text)
                dl = dm.get_prediction_dataloader(texts)
                fexamples = dm.dataset.fexamples

            print(f"Start Prediction...")
            time_start = time.time()

            prediction_batch = [
                model.predict_step(batch, i) for i, batch in enumerate(dl)
            ]
            content_list = [w for d in prediction_batch for w in d["input"]]
            decode_results = [l for d in prediction_batch for l in d["prediction"]]
            # print(len(content_list), len(decode_results))
            # print(len(dm.dataset), len(dm.dataset.fexamples))

            time_finish = time.time()
            timecost = time_finish - time_start
            print(f"End: {timecost} sec.")
            # TODO: do alignment with original tokens
            outpath = Path(args.output_dir) / "result.txt"
            print(content_list[0])
            print(decode_results[0])
            # model.write_decoded_results(content_list, decode_results, outpath)
            sent_num = len(decode_results)
            assert sent_num == len(content_list)
            with open(outpath, "w") as fout:
                for idx in range(sent_num):
                    sent_length = len(decode_results[idx])
                    for idy in range(sent_length):
                        fout.write(
                            "{} {} {}\n".format(
                                fexamples[idx].words[idy],
                                content_list[idx][idy],
                                decode_results[idx][idy],
                            )
                        )
                    fout.write("\n")
        else:
            print(f"no input file given: {datadir}")
            exit(0)


if __name__ == "__main__":
    main_as_plmodule()
