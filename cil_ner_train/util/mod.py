import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss
from util.loss_extendner import SupConLoss, KdLoss, ExtendNerLoss, \
    SupConLoss_o, BceLoss, lwf_criterion, NLLLoss, ce_bft_criterion, BceLossNoKd
from util.gather import gather_rh, gather_rh_ce
import tqdm


class MySftBertModel(BertPreTrainedModel):

    def __init__(self, config, head="mlp", feat_dim=128, per_types=6, mode="train", requires_grad=True):
        super().__init__(config)
        """backbone + projection head"""
        self.per_types = per_types  # 设置每轮任务的类型数量。
        self.feat_dim = feat_dim  # 设置特征维度。
        self.hidden_size = config.hidden_size  # 设置隐藏状态的大小。
        self.num_labels = config.num_labels  # 设置标签数量。

        self.bert = BertModel(config, add_pooling_layer=False)

        classifier_dropout = (  # 设置分类器的dropout概率
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        if mode == "train":  # 根据模式选择性地设置分类器的输出层
            if self.num_labels - 1 > self.per_types:  # 不是第一个任务
                self.classifier = nn.Linear(config.hidden_size, config.num_labels - self.per_types)  # 分类到之前的标签
            else:
                self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        # self.post_init()
        self.init_weights()
        # 根据不同的head选择不同的方式来处理模型的输出。
        if head == 'linear':
            self.head = nn.Linear(self.hidden_size, self.hidden_size)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.feat_dim)
            )
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

        if requires_grad is False:  # 冻结参数，不再进行梯度更新。
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = False

    def new_classifier(self):
        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        new_cls = nn.Linear(self.hidden_size, self.num_labels)
        new_cls.weight.data[:self.num_labels - self.per_types] = weight
        new_cls.bias.data[:self.num_labels - self.per_types] = bias
        self.classifier = new_cls

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            t_logits=None,
            top_emissions=None,
            negative_top_emissions=None,
            mode=None,
            loss_name=None,
            pseudo_labels=None,
            entity_top_emissions=None,
            topk_th=False,
            o_weight=None
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,  # (batch_size, sequence_length)词汇表中输入序列tokens的索引
            attention_mask=attention_mask,
            # (batch_size, sequence_length)对padding token进行mask。1表示未屏蔽的tokens，0表示已屏蔽的tokens。
            token_type_ids=token_type_ids,  # (batch_size, sequence_length)段嵌入。0表示句子A，1表示句子B。
            position_ids=position_ids,  # (batch_size, sequence_length)位置嵌入。取值为[0, config.max_position_embeddings - 1]。
            head_mask=head_mask,  # (num_layers, num_heads)掩蔽使self-attention模块的选定头无效。1表示该head未被掩蔽，0表示该head被掩蔽。
            inputs_embeds=inputs_embeds,  # (batch_size, sequence_length, hidden_size)输入序列的嵌入表示
            output_attentions=output_attentions,  # 是否返回所有attention层的attention张量
            output_hidden_states=output_hidden_states,  # 是否返回所有层的隐藏状态
            return_dict=return_dict,  # 返回一个ModelOutput(True)或者一个的元组(False)。
        )

        features_enc = outputs[0]  # [batch_size, seq_length, embedding_size]获取特征表示
        # 通过self.head对特征进行线性变换和激活函数操作，然后归一化。
        features = F.normalize(self.head(features_enc.view(-1, self.hidden_size)), dim=1)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if mode != "train":
            return loss, features_enc, features, logits

        if labels is not None:
            supcon_loss_fct = SupConLoss(temperature=0.1, topk_th=topk_th)
            supcon_o_loss_fct = SupConLoss_o(temperature=0.1, topk_th=topk_th)
            kd_loss_fct = KdLoss()  # 计算知识蒸馏损失
            ce_loss_fct = CrossEntropyLoss()  # 交叉熵损失
            bce_loss_fct = BceLoss(o_weight=o_weight)  # 二元交叉熵损失

            if pseudo_labels is None:
                pseudo_labels = labels
            features = features.unsqueeze(1)  # [batch_size*seq_length, 1, embedding_size]
            labels = labels.view(-1)  # [batch_size*seq_length]
            pseudo_labels = pseudo_labels.view(-1)
            logits = logits.view(-1, self.num_labels)

            if loss_name == "supcon_ce":
                supcon_loss = supcon_loss_fct(features, pseudo_labels,
                                              entity_topk=entity_top_emissions)
                if supcon_loss is None:
                    print("supcon_loss was not computed. Check conditions and inputs.")
            if loss_name == "supcon_o_bce":
                supcon_o_loss = supcon_o_loss_fct(features, pseudo_labels, top_emissions,
                                                  negative_top_emissions,
                                                  entity_topk=entity_top_emissions)

            if self.num_labels - 1 == self.per_types:  # 第一个任务
                ce_loss = ce_loss_fct(logits, labels)
                if ce_loss is None:
                    print("ce_loss was not computed. Check conditions and inputs.")

                bce_loss = bce_loss_fct(logits, labels, self.num_labels)
                if loss_name == "supcon_o_bce":
                    loss = supcon_o_loss + bce_loss
                elif loss_name == "supcon_ce":
                    loss = supcon_loss + ce_loss
                    if loss is None:
                        print("loss was not computed. Check conditions and inputs.")

            elif self.num_labels > self.per_types:
                # print(t_logits.size())
                if t_logits is not None:
                    t_logits = t_logits.view(-1, t_logits.shape[-1])

                    # 整理新类别标签labels_new，新类别样本的logits：student_new，旧类别样本的logits：s_logits，teacher模型的logits：old_logits
                    labels_new, student_new, s_logits, old_logits = gather_rh_ce(
                        labels, t_logits, logits, self.num_labels - self.per_types)

                    if labels.shape[0] != 0:
                        ce_loss = ce_loss_fct(student_new, labels_new)
                    else:
                        ce_loss = 0.
                    kd_loss = kd_loss_fct(s_logits, old_logits, t=2)

                if loss_name == "supcon_o_bce":
                    bce_loss = bce_loss_fct(logits, labels, self.num_labels, t_logits)
                    loss = supcon_o_loss + bce_loss
                elif loss_name == "supcon_ce":
                    # kd_loss = kd_loss_fct(s_logits, t_logits, t=2)
                    loss = supcon_loss + ce_loss + kd_loss

        return loss, features_enc, features, logits
