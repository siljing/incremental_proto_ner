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
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert). """

from __future__ import absolute_import, division, print_function

import argparse
# python标准模块，是命令行选项、参数和子命令解析器。
import glob
import logging
import os
import random
import json
import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
# from tensorboardX import SummaryWriter
import sys
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from util.supervised_util import convert_examples_to_features, read_examples_from_file, get_labels_dy

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer,BertModel
from datasets import load_metric
from util.metric import compute_metrics
from model.supcon_net import MySftBertModel
from util.ncm_classifier import NcmClassification, NNClassification
from util.gather import select_label_token

logger = logging.getLogger(__name__)
# python标准库logging：记录日志

# bert_create is used for load previous model, which classifier's weight and bias are different with current.
# bert is default setting
MODEL_CLASSES = {
    "bert": (BertConfig, MySftBertModel, BertTokenizer)
}
# 定义了一个字典 MODEL_CLASSES，用于将模型类别名称映射到相应的配置、模型和分词器。在这里，只定义了一个模型类别 "bert"。
# (BertConfig, MySftBertModel, BertTokenizer)：是一个元组，包含了三个类：BERT 模型的配置类，自定义的基于 BERT 的模型类，BERT 模型的分词器。
# 通过这种设置，可以根据需要轻松地切换不同的模型类别，例如，如果需要使用其他模型，只需在 MODEL_CLASSES 中添加相应的配置、模型和分词器即可。

def set_seed(args):
    # args.seed: random seed for initialization
    # 设置随机数种子，使得后续的随机操作和每次的程序运行基于相同的种子产生相同的随机结果。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:  # 如果存在GPU
        torch.cuda.manual_seed_all(args.seed)  # torch.cuda.manual_seed_all：是为所有的GPU设置种子


def train(args, train_dataset, train_dataloader, model, tokenizer, labels, pad_token_label_id, data_dir, output_dir, t_logits, out_new_labels):
    """ Train the model """

    # 设置训练总步数 t_total
    # 如果设置了训练最大步数max_steps，则t_total = args.max_steps，并计算num_train_epochs，否则根据num_train_epochs计算t_total。
    if args.max_steps > 0:
        # max_steps If > 0: set total number of training steps to perform. Override num_train_epochs.训练的最大步数，即一共执行多少次参数更新操作,决定num_train_epochs,default = -1.
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        # num_train_epochs: Total number of training epochs to perform, 训练的总轮数（default = 3）.
        # gradient_accumulation_steps: Number of updates steps to accumulate before performing a backward/update passt梯度累计步数(defalut=1).
        # len(train_dataloader) 表示总批次数。
        # len(train_dataloader) // args.gradient_accumulation_steps 表示在一个 epoch 中，模型实际上会执行多少次参数更新操作。
        # 如果args.max_steps不能完全整除每个 epoch 中的步数，就需要进行额外的一轮训练来达到指定的步数。
    else:  # 未设置最大步数
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        #  总的训练步数 = 一个 epoch 中的步数（模型参数更新的总次数） 乘以 训练的总 epoch 数

    # 配置优化器（筛选出需要被优化的参数以及设置参数更新系数，使用 AdamW 优化器）
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    # no_decay：定义了不需要衰减的参数，比如偏置参数和 LayerNormalization 层的权重参数。
    optimizer_grouped_parameters = [
        # 用列表将模型的参数按照是否需要衰减进行分组，并设置对应的权重衰减，用于配置优化器。
        # 在这个列表中，每个元素都是一个字典，表示了一个参数组的设置。
        # 通过对模型参数进行分组和设置不同的权重衰减系数，可以在优化过程中针对不同类型的参数采用不同的优化策略，从而提高模型的训练效果和泛化能力。
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         # model.named_parameters()是一个模型对象的方法，用于迭代模型的所有参数。每次迭代返回一个元组 (name, param)，其中 name 是参数的名称，param 是参数的张量值。
         # 该表达式用于过滤出不包含在 no_decay 列表中的参数。
         # any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False，则返回 False，如果有一个为 True，则返回 True。
         # any(nd in n for nd in no_decay) 的作用是检查参数名称 n 是否全都不包含在 no_decay 列表中，如果全都不包含则返回 False，否则返回 True
         # "params"对应该参数组中需要优化的，即需要被权重衰减的模型参数的张量值p列表。
         "weight_decay": args.weight_decay},
        # weight_decay：Weight decay if we apply some，指定了该参数组的权重衰减系数，用于控制参数的正则化程度。
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        # 第二个字典元素中，对所有不需要被优化参数设置了零的权重衰减值，这样在优化器更新参数时就不会应用权重衰减。
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # 使用 AdamW 优化器，其中包含了分组参数和学习率等超参数设置。
    # learning_rate：The initial learning rate for Adam.
    # num_warmup_steps：初始预热步数，当设置为0时，learning rate没有预热的上升过程，只有从初始设定的learning rate 逐渐衰减到0的过程。
    # num_training_steps：整个训练过程的总步数。
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        # 如果启用了混合精度训练 (args.fp16)，则会初始化 Apex 库，并将模型和优化器切换为混合精度模式。
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        # 如果使用多个 GPU 进行训练 (args.n_gpu > 1)，则使用 torch.nn.DataParallel 对模型进行并行化处理。
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        # 如果启用了分布式训练 (args.local_rank != -1)，则使用 torch.nn.parallel.DistributedDataParallel 对模型进行分布式并行化处理。
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    # 记录训练过程中的一些重要信息
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))  # 训练集中的样本数量
    logger.info("  Num Epochs = %d", args.num_train_epochs) # 训练的总轮数
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)  # 每个GPU上的实时批处理大小
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",  # 考虑到并行训练、分布式训练和累积梯度更新后的总训练批处理大小。
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps) # 梯度累积的步数
    logger.info("  Total optimization steps = %d", t_total)  # 总优化步数，即训练的总迭代次数

    # 设置参数
    global_step = 0  # 全局步数，用于记录当前的训练步数。
    tr_loss, logging_loss = 0.0, 0.0  # 训练损失和记录损失，用于记录损失值。
    model.zero_grad()  # 清除模型的梯度，以便进行下一轮的梯度更新。
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    # 用来显示进度条以及展示每一轮（iteration)所耗费的时间。
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3) 设置随机种子，以确保实验的可重复性。
    num_labels = len(labels)
    top_emissions = None  # 用于存储排名前几的预测概率。
    negative_top_emissions = None  # 用于存储负样本的排名前几的预测概率。
    loss_name = args.loss_name1  # Name of entity-oriented loss function

    for epoch in train_iterator:
        # 计算类别相似度
        if epoch >= args.start_train_o_epoch:
            # start_train_o_epoch（dafault=3）:The number of training type 'O' epoch to perform.表示模型在训练的第几个epoch后开始对"O"类别进行训练。
            # get_rehearsal_prototype 函数获取用于rehearsal的类别相似度。
            prototype_dists = get_rehearsal_prototype(args, model, tokenizer, labels,
                                       pad_token_label_id = pad_token_label_id, mode="rehearsal",
                                       data_dir=data_dir)
            print(prototype_dists)
        # disable=args.local_rank not in [-1, 0]表示如果参数 args.local_rank 不在 [-1, 0] 中，表示当前不是单 GPU 训练，此时进度条会被禁用。
        epoch_iterator = tqdm(train_dataloader, desc="Iterator", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()  # 将模型切换到训练模式
            # tuple() 函数将列表转换为元组，这里将张量组合成一个元组，以便在后续的代码中更方便地处理和传递数据。
            batch = tuple(t.to(args.device) for t in batch)
            if num_labels-1 > args.per_types:  # 如果不是第一个任务
                # t_logits  是一个列表，包含了老师模型的最后一个全连接层的输出。
                t_logits_step = t_logits[step]
                #print(t_logits_step.size())
                t_logits_step.to(args.device)
                # new_labels将从out_new_labels中选取的与当前批次匹配的新标签。
                new_labels = out_new_labels[step * args.train_batch_size:step * args.train_batch_size + len(batch[3])]
                # new_labels是一个包含了当前批次的标签的列表。为了在模型中进行计算，需要将这些标签数据转换为PyTorch张量，并将其移动到指定的设备上。
                new_labels = torch.tensor(new_labels).to(args.device)
            else:  # 如果是第一个任务
                t_logits_step = None
                new_labels = batch[3]
            if epoch >= args.start_train_o_epoch:
                loss_name = args.loss_name2  # loss_name2：Name of entity-aware with 'O' loss function
                cls = NNClassification()
                encodings, encoding_labels = get_token_features_and_labels(args, model, batch)
                # auto-selected positive samples for 'O'
                top_emissions_step, _ = cls.get_top_emissions_with_th(encodings,
                                                                           encoding_labels,
                                                                           th_dists=torch.median(
                                                                               prototype_dists).item())
            else:
                # 如果当前训练轮次小于 args.start_train_o_epoch，则直接将 top_emissions_step 设置为之前的 top_emissions，即None。
                top_emissions_step = top_emissions

            # 构建模型的输入字典 inputs，以供模型进行训练。
            inputs = {"input_ids": batch[0],  # 输入的token IDs，即tokenized句子的编码。
                      "attention_mask": batch[1],  # 输入句子的attention mask，用于指示哪些token是真实的、需要被注意的，哪些是填充的、可以被忽略的。
                      # XLM and RoBERTa don"t use segment_ids，仅对BERT和XLNet模型有用，表示句子中不同部分的标识。
                      "token_type_ids": batch[2] if args.model_type in ["bert", "xlnet"] else None,
                      "labels": new_labels,
                      "t_logits": t_logits_step,  # 上一步预测的逻辑回归层输出。
                      "mode": "train",  # 指定模型当前是处于训练模式。
                      "loss_name": loss_name,  # 指定使用的损失函数的名称。
                      "top_emissions": top_emissions_step,  # 'O'类别的自动选择的正样本，用于某些损失函数中。
                      "topk_th": True  # 表示是否使用 topk 样本的阈值，具体使用与否由模型定义的损失函数决定。
                      }
            outputs = model(**inputs)  # 将构建好的输入 inputs 传递给模型 model 进行前向传播，并获取模型的输出。
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            # outputs 的第一个元素是损失值，用于后续的损失计算和反向传播。
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            # 如果使用了多个GPU进行训练，那么在多个GPU上计算的损失值需要进行平均处理
            # 如果设置了梯度累积步数大于1，则需要对损失值进行除以梯度累积步数，以得到平均损失值
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:  # 如果使用了混合精度训练，则使用 amp.scale_loss 函数对损失值进行缩放，并在 with 语句块中进行反向传播。
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # 这个过程会将损失值缩放到一个合适的范围，以避免梯度溢出或者梯度消失的问题，并且提高计算效率。
            else:  # 如果没有使用混合精度训练，则直接对损失值进行反向传播。
                loss.backward()

            # loss.item() 返回当前批次的损失值，它是一个标量值。
            # 这行代码用于累加当前批次的损失值到总的训练损失值 tr_loss 中。
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
            # 检查是否已经完成了一定数量的梯度累积步骤。梯度累积是一种训练技巧，它允许在更新模型参数之前收集多个小批量的梯度。
                if args.fp16:  # 如果启用了混合精度训练
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                # 则使用 torch.nn.utils.clip_grad_norm_ 函数裁剪 amp.master_params(optimizer) 中的梯度。
                # 该函数会计算所有梯度的范数，并将其裁剪为指定的最大范数 args.max_grad_norm。
                else:  # 如果没有启用混合精度训练
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 则对model.parameters()中的梯度进行裁剪。
            # 对模型的梯度进行裁剪，以防止梯度爆炸的问题。裁剪梯度是一种常见的训练技巧，特别是在使用深度神经网络时。
            # 它通过限制梯度的范数来减少梯度更新的幅度，从而稳定训练过程。
                optimizer.step()  # 优化器执行一步参数更新的操作。
                scheduler.step()  # Update learning rate schedule 更新学习率调度器状态
                # model.zero_grad()
                optimizer.zero_grad()  # 清除优化器中所有参数的梯度
                global_step += 1  # 更新全局步数

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # args.local_rank in [-1, 0] 表示当前进程要么不在分布式训练中（args.local_rank 为 -1），要么在分布式训练中但是是主进程（args.local_rank 为 0）
                # args.logging_steps > 0: 设置了记录日志的步数间隔,args.logging_steps == 0: 全局步数可以被记录日志的步数间隔整除。
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        _, results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev",
                                                 data_dir=data_dir)
                        # evaluate 函数用于在开发集上评估模型性能。
                    logging_loss = tr_loss  # 记录训练过程中的损失。

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # args.save_steps > 0：确保保存步骤数大于零，即要求在训练中进行保存模型的操作。
                    # global_step % args.save_steps == 0：确保当前全局步数是保存步骤数的倍数，即当前步数是保存模型的步骤。
                    # Save model checkpoint
                    output_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))  # 设置保存模型的检查点的文件路径。
                    if not os.path.exists(output_dir):  # 检查路径 output_dir 是否存在
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                    # hasattr(object, name)： 用于判断对象是否包含对应的属性。
                    # 这行代码用于处理分布式或并行训练时的模型保存。
                    # 如果模型被封装在 torch.nn.DataParallel 中，即具有 .module 属性，那么 model_to_save 将指向 model.module，
                    # 这样可以确保保存的模型参数不包含 DataParallel 的外壳，而只包含实际的模型参数。
                    # 如果模型没有被封装在 DataParallel 中，那么 model_to_save 将指向 model，即保存整个模型对象。
                    model_to_save.save_pretrained(output_dir)
                    # save_pretrained(output_dir) 是一个 Transformers 模型的方法，用于将模型的权重、配置文件和词汇表保存到指定的目录中，以便后续可以加载和重用该模型。
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))  # 将训练过程中的参数保存到文件中
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:  # 训练在达到最大步数后停止
                epoch_iterator.close()  # 关闭当前的 epoch 迭代器
                break

    return global_step, tr_loss / global_step  # 返回训练过程中的全局步数和平均损失


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, data_dir, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode, data_dir=data_dir)
    support_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="memory",
                                              data_dir=data_dir)
    support_o_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="memory_o",
                                                data_dir=data_dir)
    train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train",
                                            data_dir=data_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    support_sampler = SequentialSampler(support_dataset) \
        if args.local_rank == -1 else DistributedSampler(support_dataset)
    support_o_sampler = SequentialSampler(support_o_dataset) \
        if args.local_rank == -1 else DistributedSampler(support_o_dataset)
    train_sampler = SequentialSampler(train_dataset) \
        if args.local_rank == -1 else DistributedSampler(train_dataset)
    support_dataloader = DataLoader(support_dataset, sampler=support_sampler, batch_size=args.eval_batch_size)
    support_o_dataloader = DataLoader(support_o_dataset, sampler=support_o_sampler, batch_size=args.eval_batch_size)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.eval_batch_size)
    # eval_batch_size：
    # sampler：定义从数据集中抽取样本的策略，barch_size：一个batch加载多少样本
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    eval_iterator = tqdm(eval_dataloader, desc="Evaluating")
    if args.cls_name == "ncm_dot":  # 本文使用NCM分类器
        support_encodings, support_labels = get_support_encodings_and_labels_total(
            args, model, support_dataloader, support_o_dataloader, train_dataloader, pad_token_label_id)
    else:
        support_encodings, support_labels = get_support_encodings_and_labels(
            args, model, support_dataloader, support_o_dataloader, pad_token_label_id)
    
    exemplar_means = get_exemplar_means(args, support_encodings, support_labels)


    for _, batch in enumerate(eval_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        encodings, encoding_labels = get_token_encodings_and_labels(args, model, batch)

        # 如果是rehearsal模式，则去除掉当前task样本的support_encodings和support_labels
        if mode=="rehearsal":
            cls = NNClassification()
            # 剔除当前任务标签的样本
            support_encodings = support_encodings[support_labels < len(labels) - args.per_types]
            support_labels = support_labels[support_labels < len(labels) - args.per_types]

            # nn_preds(batch_size, sent_len) 包含每个样本中原型相似度最大值所在的类别索引
            # nn_emissions(batch_size, sent_len, ndim) 包含每个样本与每个类别的原型相似度的最大值
            # prototype_dists 每个旧类别的原型重新标记阈值列表（还未乘βi）
            nn_preds, nn_emissions, prototype_dists = cls.nn_classifier_dot_prototype(
                encodings, support_encodings, support_labels, exemplar_means)
        if args.cls_name == "ncm_dot":  # args.cls_name：Name of classifier.
            cls = NcmClassification()
            nn_preds = cls.ncm_classifier_dot(encodings, support_encodings, support_labels, exemplar_means)
        elif args.cls_name == "linear":
            nn_preds, encoding_labels = get_token_logits_and_labels(args, model, batch)

        # 第一次预测
        if preds is None:
            preds = nn_preds.detach().cpu().numpy()
            out_label_ids = encoding_labels.detach().cpu().numpy()
            if mode == "rehearsal":
                emissions = nn_emissions.detach().cpu().numpy()

        else:
            preds = np.append(preds, nn_preds.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, encoding_labels.detach().cpu().numpy(), axis=0)
            if mode == "rehearsal":
                emissions = np.append(emissions, nn_emissions.detach().cpu().numpy(), axis=0)
        # memory management
        del nn_preds
        torch.cuda.empty_cache()
    # eval_loss = eval_loss / nb_eval_steps
    if mode == "rehearsal":
        return preds, emissions, out_label_ids, prototype_dists
    
    if args.cls_name == "linear":
        preds = np.argmax(preds, axis=2)
    # 将预测的标签和真实标签从索引形式转换为字符串形式，并存储在 preds_list 和 out_label_list 中
    label_map = {i: "I-"+label for i, label in enumerate(labels)}
    label_map[0] = "O"
    # print(label_map)
    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
                # print(label_map[preds[i][j]])
    # 使用 seqeval 库计算序列标注任务的评价指标，如 Precision、Recall 和 F1-score。
    metric = load_metric("../seqeval")
    metric.add_batch(
        predictions=preds_list,
        references=out_label_list,
    )

    macro_results, micro_results, _ = compute_metrics(metric)

    logger.info("***** Eval macro results %s *****", prefix)
    for key in sorted(macro_results.keys()):
        logger.info("  %s = %s", key, str(macro_results[key]))

    logger.info("***** Eval micro results %s *****", prefix)
    for key in sorted(micro_results.keys()):
        logger.info("  %s = %s", key, str(micro_results[key]))


    return macro_results, micro_results, preds_list

def get_rehearsal_prototype(args, model, tokenizer, labels, pad_token_label_id, mode, data_dir):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    support_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="memory",
                                              data_dir=data_dir)
    support_o_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="memory_o",
                                                data_dir=data_dir)
    support_sampler = SequentialSampler(support_dataset) \
        if args.local_rank == -1 else DistributedSampler(support_dataset)
    support_o_sampler = SequentialSampler(support_o_dataset) \
        if args.local_rank == -1 else DistributedSampler(support_o_dataset)
    support_dataloader = DataLoader(support_dataset, sampler=support_sampler, batch_size=args.eval_batch_size)
    support_o_dataloader = DataLoader(support_o_dataset, sampler=support_o_sampler, batch_size=args.eval_batch_size)
    support_encodings, support_labels = get_support_features_and_labels(
        args, model, support_dataloader, support_o_dataloader, pad_token_label_id)

    prototype_dists = []

    # for j in range(support_encodings.size(0)):  # Normalize
    #     support_encodings.data[j] = support_encodings.data[j] / support_encodings.data[j].norm()
    from torch.nn import functional as F
    support_encodings = F.normalize(support_encodings)
    for i in range(1, len(labels)):  # 迭代每个非"O"标签的类别
        # 计算每个类别的样本之间的余弦相似度
        support_reps_dists = torch.matmul(support_encodings[support_labels == i],
                                          support_encodings[support_labels == i].T)
        # 将对角线上的元素（样本与自身的相似度）设置为0，以避免将自身视为原型。
        support_reps_dists = torch.scatter(support_reps_dists, 1,
                                           torch.arange(support_reps_dists.shape[0]).view(-1, 1).to(args.device),
                                           0.)
        # 添加类别i的类别相似度
        prototype_dists.append(support_reps_dists[support_reps_dists > 0].view(-1).mean(-1))
    # print(prototype_dists)
    prototype_dists = torch.stack(prototype_dists).to(args.device)
    return prototype_dists

def teacher_evaluate(args, train_dataloader, model, tokenizer, labels, pad_token_label_id, mode, data_dir, prefix=""):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if mode == "train":
        eval_dataloader = train_dataloader
    elif mode == "dev":
        # 使用 load_and_cache_examples 函数加载开发数据集。
        eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode,
                                               data_dir=data_dir)
        # Note that DistributedSampler samples randomly
        # 顺序采样
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        # 加载数据
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running teacher model evaluation %s *****", prefix)
    # logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    logits_list = []
    # 将模型设置为评估模式
    model.eval()
    # 迭代数据加载器中的批次，并通过模型获取预测的分数 logits 和输出标签 out_labels。
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # 将批次中的每个张量（t）都移动到指定的设备上（args.device），并将它们组合成一个元组
        batch = tuple(t.to(args.device) for t in batch)
        # 使用 get_token_logits_and_labels 函数获取 logits 和 out_labels。
        logits, out_labels = get_token_logits_and_labels(args, model, batch)
            # eval_loss += tmp_eval_loss.item()
        # 对评估步骤计数，以便跟踪已评估的批次数量
        nb_eval_steps += 1
        # 将每个批次的 logits 分数添加到 logits_list 列表中
        logits_list.append(logits.detach().cpu())  # 使用 detach().cpu() 方法将其从计算图中分离并移动到 CPU 上
        # 在每次迭代结束时，使用 torch.cuda.empty_cache() 方法清空 CUDA 缓存，以释放显存空间。
        torch.cuda.empty_cache()
        
    # Relabeling Old Entity Classes with prototype relabeling threshold
    # 计算原型重新标记阈值
    # 根据旧模型的示例表示计算每个类别的示例与其原型的最低相似度prototype_dists
    # preds(num_batch, batch_size, sent_len) 包含每个批次样本中原型相似度最大值所在的类别索引
    # emissions(num_batch, batch_size, sent_len, ndim) 包含每个批次样本与每个类别的原型相似度的最大值
    # prototype_dists 每个旧类别的原型重新标记阈值列表（还未乘βi）
    preds, emissions, out_label_ids, prototype_dists = evaluate(args, model, tokenizer, labels, pad_token_label_id,
                                                                mode="rehearsal", data_dir=data_dir)

    current_task_id = (len(labels) - 1) // args.per_types  # 计算当前任务的ID
    th_para = args.relabel_th  # args.relabel_th：default=1.0
    th_reduction = args.relabels_th_reduction  # args.relabel_th_reduction ：default=0.05 超参数，用于控制重新标记的程度，即论文里的β

    # 遍历每个旧任务old task
    for i in range(current_task_id):
        """论文原解释：“由于在增量学习过程中，区分旧实体类的能力不断下降，因此旧任务需要较低的阈值来重新标记足够的样本。
        因此，对于每个旧任务i，我们设βi = 0.98−0.05∗(t−i) (即βi = th_para−(t−i)∗th_reduction)，其中t为当前步骤。”
        args.change_th：是否根据不同步骤调整βi，即task_para。
        """
        if args.change_th:
            task_para = th_para - (current_task_id - i - 1)*th_reduction  # βi
        else:
            task_para = th_para
        # 将旧任务i的每个类别的示例与其原型的最低相似度乘βi获得原型重新标记阈值prototype_dists
        prototype_dists[i*args.per_types+1:(i+1)*args.per_types+1] *= task_para

    # 重新标记旧实体类
    out_label_new_list = [[] for _ in range(out_label_ids.shape[0])]
    for i in range(out_label_ids.shape[0]):  # 迭代每个batch
        for j in range(out_label_ids.shape[1]):  # 迭代每个样本
            idx = preds[i][j]  # 根据原型相似度预测的类别索引
            # 如果原型的相似度大于重新标记阈值并且预测的标签是旧实体类的标签
            if emissions[i][j] > prototype_dists[idx].item() and out_label_ids[i][j] < len(labels) - args.per_types:
                out_label_new_list[i].append(preds[i][j])  # 则将该样本预测为这个旧实体类
            else:
                out_label_new_list[i].append(out_label_ids[i][j])  # 否则，保持原始的标签不变
    return logits_list, out_label_new_list

def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode, data_dir):

    # 当前进程不是分布式训练中的主进程并且不是评估模式
    if args.local_rank not in [-1, 0] and not evaluate:
        # 通过torch.distributed.barrier()，创建一个同步栅栏，等待所有进程到达栅栏处（包括主进程数据处理完毕）
        # 确保只有分布式训练的第一个进程处理数据集，其他进程将使用缓存。
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file 从缓存文件或数据集文件中加载特征
    cached_features_file = os.path.join(data_dir, "cached_{}_{}_{}".format(mode,
        # model_name_or_path: Path to pre-trained model or shortcut name
        # filter(function, iterable) 函数用于过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象，如果要转换为列表，可以使用 list() 来转换。
        # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        # max_seq_length: The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        # overwrite_cache: Overwrite the cached training and evaluation sets 覆盖缓存的训练集和评估集.
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        examples = read_examples_from_file(data_dir, mode)
        features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                                                # model_type：Model type selected in the list
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ["roberta"]),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ["xlnet"]),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                pad_token_label_id=pad_token_label_id
                                                )
        if args.local_rank in [-1, 0]:  # 当前进程不是非分布式训练或是分布式训练中的主进程，则将特征保存到缓存文件中。
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:  # 当前进程是分布式训练中的主进程且不是评估模式
        # 当主进程读取数据并处理之后所有进程都到达了当前的栅栏处，这样所有进程就达到了同步，并同时得到释放。
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset 转换为张量并构建数据集
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset
def get_token_features_and_labels(args, model, batch):
    """
    Get token encoding using pretrained BERT-NER model as well as groundtruth label
    """
    batch = tuple(t.to(args.device) for t in batch)
    label_batch = batch[3]
    with torch.no_grad():
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "output_hidden_states": True,
                  "mode": "dev"}
        if model.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if model.config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use token_type_ids
        outputs = model(**inputs)
        features = outputs[2]
    return features.view(label_batch.shape[0],label_batch.shape[1], -1), label_batch
def get_token_encodings_and_labels(args, model, batch):
    """
    Get token encoding using pretrained BERT-NER model as well as groundtruth label
    """
    batch = tuple(t.to(args.device) for t in batch)
    label_batch = batch[3]
    with torch.no_grad():
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "output_hidden_states": True,
                  "mode": "dev"}
        if model.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if model.config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use token_type_ids
        outputs = model(**inputs)
        hidden_states = outputs[1] # last layer representations
    return hidden_states, label_batch

def get_token_logits_and_labels(args, model, batch):
    """
    Get token encoding using pretrained BERT-NER model as well as groundtruth label
    使用预训练的BERT-NER模型获得token encoding作为参考标注
    """
    batch = tuple(t.to(args.device) for t in batch)
    label_batch = batch[3]
    with torch.no_grad():
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "output_hidden_states": True,
                  "mode": "dev"}
        if model.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if model.config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use token_type_ids
        outputs = model(**inputs)
        logits = outputs[-1]  # last layer representations
    return logits, label_batch

def get_support_encodings_and_labels(args, model, support_loader, support_o_loader, pad_token_label_id):
    """
    Get token encodings and labels for all tokens in the support set
    """
    support_encodings, support_labels = [], []
    support_iterator = tqdm(support_loader, desc="Support data representations")
    for index, batch in enumerate(support_iterator):
        encodings, labels = get_token_encodings_and_labels(args, model, batch)
        encodings = encodings.view(-1, encodings.shape[-1])
        labels = labels.flatten()
        idx = torch.where(labels*(labels - pad_token_label_id) != 0)[0]
        support_encodings.append(encodings[idx])
        support_labels.append(labels[idx])
    support_o_iterator = tqdm(support_o_loader, desc="Support data representations")
    for _, batch in enumerate(support_o_iterator):
        encodings, labels = get_token_encodings_and_labels(args, model, batch)
        encodings = encodings.view(-1, encodings.shape[-1])
        labels = labels.flatten()
        # filter out PAD tokens
        idx = torch.where((labels-pad_token_label_id) != 0)[0]
        labels = labels[idx]
        encodings = encodings[idx]
        support_encodings.append(encodings[labels == 0])
        support_labels.append(labels[labels == 0])
    return torch.cat(support_encodings), torch.cat(support_labels)

def get_support_features_and_labels(args, model, support_loader, support_o_loader, pad_token_label_id):
    """
    Get token encodings and labels for all tokens in the support set
    """
    support_encodings, support_labels = [], []
    support_iterator = tqdm(support_loader, desc="Support data representations")
    for index, batch in enumerate(support_iterator):
        encodings, labels = get_token_features_and_labels(args, model, batch)
        encodings = encodings.view(-1, encodings.shape[-1])
        #print(labels.size())
        #print(encodings.size())
        labels = labels.flatten()
        # filter out PAD tokens
        idx = torch.where(labels*(labels - pad_token_label_id) != 0)[0]
        support_encodings.append(encodings[idx])
        support_labels.append(labels[idx])
    support_o_iterator = tqdm(support_o_loader, desc="Support data representations")
    for _, batch in enumerate(support_o_iterator):
        encodings, labels = get_token_features_and_labels(args, model, batch)
        encodings = encodings.view(-1, encodings.shape[-1])
        labels = labels.flatten()
        # filter out PAD tokens
        idx = torch.where((labels-pad_token_label_id) != 0)[0]
        labels = labels[idx]
        encodings = encodings[idx]
        support_encodings.append(encodings[labels == 0])
        support_labels.append(labels[labels == 0])
    return torch.cat(support_encodings), torch.cat(support_labels)

def get_support_encodings_and_labels_total(args, model, support_loader, support_o_loader, train_loader, pad_token_label_id):
    """
    Get token encodings and labels for all tokens in the support set
    """
    support_encodings, support_labels = [], []
    train_iterator = tqdm(train_loader, desc="Support data representations")
    for index, batch in enumerate(train_iterator):
        encodings, labels = get_token_encodings_and_labels(args, model, batch)
        encodings = encodings.view(-1, encodings.shape[-1])
        #print(labels.size())
        #print(encodings.size())
        labels = labels.flatten()
        # filter out PAD tokens 过滤掉标签为填充标记的部分
        idx = torch.where((labels - pad_token_label_id) != 0)[0]
        support_encodings.append(encodings[idx])
        support_labels.append(labels[idx])

    support_iterator = tqdm(support_loader, desc="Support data representations")
    for index, batch in enumerate(support_iterator):
        encodings, labels = get_token_encodings_and_labels(args, model, batch)
        encodings = encodings.view(-1, encodings.shape[-1])
        #print(labels.size())
        #print(encodings.size())
        labels = labels.flatten()
        # filter out PAD tokens 过滤掉标签为“O”和填充标记的部分
        idx = torch.where(labels * (labels - pad_token_label_id) != 0)[0] # lable = 0 代表“O”
        support_encodings.append(encodings[idx])
        support_labels.append(labels[idx])
    support_o_iterator = tqdm(support_o_loader, desc="Support data representations")
    for _, batch in enumerate(support_o_iterator):
        encodings, labels = get_token_encodings_and_labels(args, model, batch)
        encodings = encodings.view(-1, encodings.shape[-1])
        labels = labels.flatten()
        # 过滤掉标签为填充标记的部分
        idx = torch.where((labels - pad_token_label_id) != 0)[0]
        labels = labels[idx]
        encodings = encodings[idx]
        support_encodings.append(encodings[labels == 0])
        support_labels.append(labels[labels == 0])
    return torch.cat(support_encodings), torch.cat(support_labels)

def get_exemplar_means(args, support_reps, support_labels):
    exemplar_means = {}
    n_tags = torch.max(support_labels) + 1  # n_tags 的值是支持集中类别数量的上限,即labels的最大值
    cls_exemplar = {cls: [] for cls in range(n_tags)}  # 创建了一个字典，包含了所有可能的类别索引，并且每个类别对应一个空列表。
    # 将每个样本按照标签分类存储在cls_exemplar字典中。（包含了“O”标签）
    # zip(support_reps, support_labels)：将 support_reps 和 support_labels 中对应位置的元素一一配对，生成一个可迭代的元组序列
    # for x, y in ... 遍历了这个序列中的每个元组，其中 x 是reps，y 是labels。
    for x, y in zip(support_reps, support_labels):
        # 将当前样本reps x 加入到与其labels y 相关联的列表中。
        #  y.item() 用于获取标签值，因为标签可能是张量，.item() 方法用于将其转换为 Python 的标量值。
        cls_exemplar[y.item()].append(x)
    for cls, exemplar in cls_exemplar.items():
        features = []
        # Extract feature for each exemplar in p_y
        # 将每个样本的 feature 归一化，并将归一化后的特征添加到 features 列表中。
        # 归一化的目的是将数据缩放到相似的范围，以便模型在训练过程中更容易学习到特征之间的关系。
        # 在这个场景中，归一化被用来确保样本的每个特征的尺度是一致的，例如某些特征的取值范围可能是0到1，而另一些特征的取值范围可能是几千或几百万。
        # 消除尺度差异对模型训练的影响，使得模型更加公平地对待各个特征。这样可以防止某些特征对模型的训练产生过度影响。通过归一化，可以使得模型更加稳定和收敛更快。
        for feature in exemplar:
            feature.data = feature.data / feature.data.norm()  # Normalize
            features.append(feature)
        # 如果当前类别下没有样本，则随机初始化一个与样本表示reps大小相同的张量作为该类别的样本相似度
        if len(features) == 0:
            mu_y = torch.normal(0, 1, size=tuple(x.size())).to(args.device)
            mu_y = mu_y.squeeze()  # 将张量中尺寸为 1 的维度去除
        else:  # 如果有，则计算当前类别的所有样本的reps的均值作为样本相似度
            features = torch.stack(features)  # 向量化操作：将特征张量将按照默认的维度0进行堆叠，生成一个新的张量，方便计算这些特征张量的均值。
            mu_y = features.mean(0).squeeze()
        mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
        exemplar_means[cls] = mu_y
    return exemplar_means

def train_and_eval(args, labels, num_labels, pad_token_label_id, model_name_or_path, output_dir, data_dir, step_id):
    # Load pretrained model and tokenizer

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    # obtain features from teacher model

    # Training
    # 加载上一轮模型的参数，分词器，模型
    # args.model_type：Model type selected in the MODEL_CLASSES.keys()'s list
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # args.config_name：Pretrained config name or path if not the same as model_name
    config = config_class.from_pretrained(args.config_name if args.config_name else model_name_or_path,
                                          num_labels=num_labels)
    # args.do_lower_case: 设置标志表示正在使用一个大小写不敏感的模型
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    # from_tf=bool(".ckpt" in model_name_or_path) 用于指定是否从 TensorFlow 检查点加载模型
    model = model_class.from_pretrained(model_name_or_path, from_tf=bool(".ckpt" in model_name_or_path),
                                        config=config)

    model.to(args.device)

    # args.do_train：Whether to run training.
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="rehearsal",
                                                data_dir=data_dir)
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        # 顺序采样
        train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        if step_id > 0:  # 如果不是第一个任务
            # get teacher model's features
            t_logits, out_new_labels = teacher_evaluate(args, train_dataloader, model, tokenizer, labels, pad_token_label_id,
                                          mode="train", data_dir=data_dir)
            # t_features_eval = teacher_evaluate(args, train_dataloader, t_model, tokenizer, labels, pad_token_label_id,
            #                                    mode="dev", data_dir=data_dir)
            model.new_classifier()  # 创建一个新的分类器
            model.to(args.device)
        else:
            t_logits = None
            out_new_labels = None

        global_step, tr_loss = train(args, train_dataset, train_dataloader, model, tokenizer, labels,
                                     pad_token_label_id, data_dir=data_dir, output_dir=output_dir,
                                     t_logits=t_logits, out_new_labels=out_new_labels)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(output_dir)

        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                "module") else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:

        tokenizer = tokenizer_class.from_pretrained(output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [output_dir]
        # args.eval_all_checkpoints:Evaluate all checkpoints starting with the same prefix as model_name
        #                           ending and ending with step number
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, mode="dev")
            model.to(args.device)
            train_dataloader=None
            _, result,  _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev",
                                    data_dir=data_dir, prefix=global_step)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(output_dir, mode="test")
        model.to(args.device)

        # data_dir = args.data_dir
        macro_results, micro_results, predictions = evaluate(args, model, tokenizer, labels,
                                                                          pad_token_label_id,
                                                                          mode="test", data_dir=data_dir)

        # Save results
        output_test_results_file = args.log_dir

        with open(output_test_results_file, "a") as writer:
            writer.write("{}\n".format(step_id))
            for key in sorted(macro_results.keys()):
                writer.write("macro_{} = {}\n".format(key, str(macro_results[key])))
            writer.write("\n")
            for key in sorted(micro_results.keys()):
                writer.write("micro_{} = {}\n".format(key, str(micro_results[key])))

        # Save predictions
        output_test_predictions_file = os.path.join(output_dir, "test_pred_gold.txt")
        
        with open(output_test_predictions_file, "w", encoding="utf-8") as writer:
            with open(os.path.join(data_dir, "test.txt"), "r", encoding="utf-8") as f:
                example_id = 0
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        writer.write(line)
                        if not predictions[example_id]:
                            example_id += 1
                    elif predictions[example_id]:
                        output_line = line.split()[0] + " " + predictions[example_id].pop(0)[2:] \
                                      + " " + line.split()[-1] + "\n"
                        writer.write(output_line)
                    else:
                        logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])
    return results

def main():
    # 创建解释器，即用来装载参数的容器。
    parser = argparse.ArgumentParser()

    # 给这个解析对象添加命令行参数
    # "..."是参数名，type是要传入的参数的数据类型，help是该参数的提示信息，可以使用 arg.参数名 来提取这个参数
    # 若改变参数输入的顺序或在输入参数同时携带参数名，可以使用选择型参数，即在添加参数时参数名前加两个"-"
    # required属性要求该所有参数必须被赋值，否则报错
    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_type_create", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_type_eval", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    ## Traning or evaluating parameters
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--change_th", action="store_true")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--memory_update", action="store_true",
                        help="Whether to update memory data.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    
    ## Epoch
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--start_train_o_epoch", default=3.0, type=float,
                        help="The number of training type 'O' epoch to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--scale", type=int, default=25)
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    ## Relabling parameters
    parser.add_argument("--relabel_th", type=float, default=1.0,required=True)
    parser.add_argument("--relabels_th_reduction", type=float, default=0.05, required=True)
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--loss_name1", type=str, default="", help="Name of entity-oriented loss function.")
    parser.add_argument("--loss_name2", type=str, default="", help="Name of entity-aware with 'O' loss function.")
    parser.add_argument("--cls_name", type=str, default="", help="Name of classifier.")
    ## Task parameters
    parser.add_argument("--nb_tasks", type=int, default=1, help="The number of tasks.")
    parser.add_argument("--start_step", type=int, default=0, help="The index of start step.")
    parser.add_argument("--log_dir", type=str, default="",
                        help="The logging directory where the test results will be written.")
    parser.add_argument("--per_types", type=int, default=0,
                        help="The number of each task.")
    parser.add_argument("--feat_dim", type=int, default=128,
                        help="The dimension of features.")
    parser.add_argument("--train_temp", type=int, default=2,
                        help="The distilling temperature in training parse.")
    parser.add_argument("--eval_temp", type=int, default=1,
                        help="The distilling temperature in inference parse.")
    args = parser.parse_args()

    # output_dir:The output directory where the model predictions and checkpoints will be written.
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", # 指定日志记录的格式和内容
                        datefmt="%m/%d/%Y %H:%M:%S",  # 指定时间格式
                        # 设置打印日志的级别，level级别以上的日志会打印出(level=logging.DEBUG 、INFO 、WARNING、ERROR、CRITICAL)
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)  # INFO或WARN
    # 记录了一条 warning 级别的日志消息，包括进程排名、设备信息、GPU 数量、是否进行了分布式训练以及是否进行了 16 位训练。
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    per_types = args.per_types
    # nb_tasks = 11

    # incremental learning setting
    output_test_results_file = args.log_dir  # args.log_dir：the logging directory where the test results will be written.
    with open(output_test_results_file, "a") as writer:
        writer.write("num_train_epochs={} start_train_o_epoch={}\n"
                     .format(args.num_train_epochs, args.start_train_o_epoch))
        writer.write("config_name={}\n".format(args.config_name))
    for step_id in range(args.start_step, args.nb_tasks):
        # args.labels：Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.
        labels = get_labels_dy(args.labels, per_types, step_id=step_id)
        num_labels = len(labels)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = CrossEntropyLoss().ignore_index
        if step_id == 0: # 如果是第一个任务，则加载 bert-base-uncased 模型
            model_name_or_path = "../bert-base-uncased"
        else:  # 否则加载上一轮模型
            # args.output_dir：Overwrite the content of the output directory
            model_name_or_path = os.path.join(args.output_dir, "task_" + str(step_id - 1))

        output_dir = os.path.join(args.output_dir, "task_{}".format(step_id))
        data_dir = os.path.join(args.data_dir, "task_{}".format(step_id))
        # args.data_dir：The input data dir. Should contain the training files for the CoNLL-2003 NER task.
        train_and_eval(args, labels, num_labels, pad_token_label_id, model_name_or_path,
                       output_dir, data_dir, step_id)

if __name__ == "__main__":
    main()
