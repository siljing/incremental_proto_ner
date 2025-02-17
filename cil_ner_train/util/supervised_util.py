# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
import json

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification.一个单独的训练/测试句子，用于标记分类任务"""

    def __init__(self, guid, text, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example句子的唯一标识符.
            words: list. The words of the sequence句子中的单词序列，以列表形式存储.
            labels: (Optional) list. The labels for each word of the sequence（可选）序列中每个单词的标签. This should be specified for
            train and dev examples, but not for test examples.对于训练和开发句子，需要提供标签列表；对于测试句子，不需要提供标签列表。
        """
        self.guid = guid
        self.text = text
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def read_examples_from_file(data_dir, mode):

    if mode == "rehearsal":
        mode1 = "train"
        mode2 = "memory"
        train_examples = read_example_from_file(data_dir, mode1)
        memory_examples = read_example_from_file(data_dir, mode2)
        examples = train_examples + memory_examples

    # elif mode == "support":
    #     mode1 = "memory"
    #     mode2 = "memory_o"
    #     memory_examples = read_example_from_file(data_dir, mode1)
    #     memory_o_examples = read_example_from_file(data_dir, mode2)
    #     return memory_examples, memory_o_examples

    else:
        examples = read_example_from_file(data_dir, mode)

    return examples


def read_example_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.json".format(mode))  # 构造文件路径
    examples = []
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)
        for item in data:
            guid = item['sample_id']
            text = item['text']
            entities = item['entities']
            labels = ['O'] * len(text)  # Initialize labels as 'O' for Outside
            for entity in entities:
                start = entity['start']
                end = entity['end']
                entity_type = entity['type']
                for i in range(start, end):
                    labels[i] = f'{entity_type}'  # Inside entity
            examples.append(InputExample(guid=f"{mode}-{guid}", text=text, labels=labels))
    return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                 pad_token_label_id=-1,):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}  # 构建标签到索引的映射 label_map，用于将标签转换为对应的整数索引。

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = []
        label_ids = []
        # 使用分词器 tokenizer 对单词进行分词，并在label_ids列表中添加它们的标签编号。
        for word, label in zip(example.text, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if word_tokens:
                tokens.extend(word_tokens)
                # 一个单词可能有多个token，第一个token使用真实标签，后面使用PAD标签。
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
            label_ids = label_ids[:(max_seq_length - 2)]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += ([0] * padding_length)
        input_mask += ([0] * padding_length)
        segment_ids += ([0] * padding_length)
        label_ids += ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length, print(len(label_ids), max_seq_length)

        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,
                                        segment_ids=segment_ids, label_ids=label_ids))
    return features


def get_labels_dy(path, per_types, step_id):
    with open(path, "r", encoding='utf-8') as f:
        types_list = f.read().splitlines()
    if "O" in types_list:
        types_list.remove("O")
    labels = types_list[:(step_id + 1) * per_types]  # 提取当前任务及之前的所有标签集
    # 只添加一个“O”
    if "O" not in labels:
        labels = ["O"] + labels
    print(len(labels), labels)
    return labels