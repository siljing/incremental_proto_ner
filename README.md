# 类增量命名实体识别

一个面向中文细粒度实体识别场景的类增量学习项目。围绕“**在持续学习新实体类别时，如何减少旧类别遗忘，并处理被错误标成 `O` 的隐藏实体**”这一问题展开，实现了一个结合**实体感知对比学习、基于原型的重标注、知识蒸馏和原型分类器**的增量 NER 框架。

## 项目要解决什么问题

在类增量命名实体识别（Class-Incremental NER）里，模型会分阶段学习新类别。问题在于：

- 训练第 `t` 个任务时，通常只能看到当前新类别的数据；
- 旧类别实体在当前数据里往往不会被继续标注；
- 这些旧实体，甚至未来潜在的新实体，都会被统一写成 `O`。

这会带来两个典型问题：

1. **灾难性遗忘**：模型学习新类后，旧类识别能力明显下降。
2. **`O` 类混淆**：模型会把“真正的非实体”与“被漏标的实体”混在一起学。

这个项目的核心出发点正是：

> “更好地学习 `O` 的表示，有助于学习更多实体。”

## 仓库结构

```text
incremental_proto_ner/
├─ cil_ner_train/
│  ├─ proto_chinese.py          # 中文 JSON 数据的主训练/评估脚本
│  ├─ proto_uie.py              # 平行实验脚本
│  ├─ run_incremental_proto.py  # 早期/通用版本入口
│  ├─ run_incremental_proto.sh  # 参数示例
│  ├─ model/
│  │  └─ supcon_net.py          # 模型定义：BERT + 投影头 + 分类头
│  └─ util/
│     ├─ loss_extendner.py      # 对比学习、蒸馏、BCE 等损失
│     ├─ ncm_classifier.py      # NCM / NN 分类与原型计算
│     ├─ supervised_util.py     # 数据读取与特征转换
│     ├─ metric.py              # 评估指标
│     └─ gather.py              # 新旧类别 logits 对齐
└─ data/
   ├─ labels.txt               # 全部标签
   ├─ *_ner_train/dev.json     # 原始/中间数据文件
   └─ tasks/
      ├─ task_0/
      ├─ task_1/
      ├─ task_2/
      ├─ task_3/
      ├─ task_4/
      └─ task_5/
```

## 核心思路

整个方法由四部分组成：

### 1. 实体感知对比学习

先用 BERT 提取 token 表示，再通过一个 `Linear -> ReLU -> Linear` 的投影头映射到低维特征空间。这样做的目标不是只为了降维，而是为了让“相似实体更近、不同实体更远”的结构更清晰。

在训练早期，模型主要在**明确标注过的实体 token**上做监督式对比学习，先把实体空间学稳。

### 2. 对 `O` 的动态挖掘

在若干个 epoch 之后，项目开始显式处理 `O` 类：

- 根据支持样本的相似度动态计算阈值；
- 在 `O` token 中自动筛出“相似度高的一簇样本”；
- 将这些可能隐藏实体的样本对作为正样本参与对比学习。

这样，`O` 不再被当作一个完全同质的类别，而是被进一步拆分成“真正的 O”与“可能隐藏实体”的不同结构。

### 3. 基于原型的重标注

为了把当前任务中被错误标成 `O` 的旧实体重新找出来，项目使用了**prototype relabeling**：

- 用旧模型提取旧类别支持样本的特征；
- 计算每个旧类别的原型向量；
- 计算当前 `O` token 与各个旧类原型的相似度；
- 如果最高相似度超过阈值，就把该 token 重标成对应旧类别。

这一步相当于给当前任务补回一部分旧类别监督信号，是缓解遗忘的关键。

### 4. 知识蒸馏 + 原型分类

增量训练时，旧模型充当 teacher，新模型充当 student：

- teacher 先对当前训练集跑一遍，输出旧类别 logits；
- student 在学习新类别的同时，用 KL 散度对齐旧类别输出分布；
- 推理阶段支持使用 **NCM（Nearest Class Mean）** 分类器，以降低线性分类头在增量场景下的偏置。

## 方法流程


1. 加载第 `t-1` 步模型，或在第一步加载预训练 BERT。
2. 用旧模型对当前任务数据进行一次前向推理，获取 teacher logits。
3. 基于记忆样本构建旧类别原型，并对当前数据里的 `O` 做重标注。
4. 扩展分类头，使模型可以输出“旧类别 + 新类别 + `O`”。
5. 训练初期使用实体对比学习，随后引入 `O` 感知对比学习。
6. 将对比损失、分类损失和蒸馏损失联合优化。
7. 评估时用记忆集构建原型，并通过 NCM 或线性头进行预测。

## 代码实现

- **BERT + 投影头 + 可扩展分类器**：`cil_ner_train/model/supcon_net.py`
- **监督对比学习、`O` 感知对比学习、蒸馏损失、BCE 损失**：`cil_ner_train/util/loss_extendner.py`
- **NCM / 最近邻分类与原型相似度计算**：`cil_ner_train/util/ncm_classifier.py`
- **JSON 数据读取、标签动态扩展、特征构建**：`cil_ner_train/util/supervised_util.py`
- **中文增量训练主入口**：`cil_ner_train/proto_chinese.py`

其中几个特别值得一提的工程细节：

- 分类头会随着任务推进动态扩展，而不是一次性固定全部类别。
- 训练集会和 `memory.json` 一起组成 rehearsal 数据。
- 仓库内部直接使用“类别名”做 token 标签，评估前再映射成 `I-<type>` 形式计算 seqeval 指标。
- 默认评估方式更偏向增量学习友好的原型分类，而不是完全依赖线性分类头。


## 训练数据


- `task_0` 到 `task_5`，共 **6 个增量任务**
- 默认配置下每步引入 **6 个新类别**，对应参数 `per_types=6`
- 每个任务目录包含：
  - `train.json`：当前任务训练集
  - `test.json`：测试集
  - `memory.json`：记忆回放样本

数据样式如下：

```json
[
  {
    "text": "俄罗斯正在开发一种新型空降防空系统……",
    "entities": [
      {
        "start": 72,
        "end": 78,
        "text": "BMD-4M",
        "type": "装甲车辆"
      }
    ],
    "sample_id": 0
  }
]
```

说明：

- `start` / `end` 使用字符级区间；
- `type` 是实体类别；
- 读取时会被转换成 token 级标签序列。

## 运行环境

- Python 3.8+
- PyTorch
- transformers
- datasets
- seqeval
- tqdm
- numpy

```bash
pip install torch transformers datasets seqeval tqdm numpy
```

## 如何运行

### 1. 准备预训练模型

对于当前中文版本脚本，`proto_chinese.py` 在第一个任务会默认从 `../bert-base-chinese` 加载底座模型。


**建议在 `cil_ner_train` 目录下执行脚本**。否则需要自行修改脚本中第一个任务的 `model_name_or_path` 逻辑。

### 2. 运行中文增量训练

```bash
cd cil_ner_train

python proto_chinese.py \
  --data_dir ../data/tasks \
  --model_type bert \
  --model_type_create bert \
  --model_type_eval bert \
  --labels ../data/labels.txt \
  --model_name_or_path ../output_nerd \
  --output_dir ../output_nerd \
  --log_dir ../log/results.txt \
  --overwrite_output_dir \
  --max_seq_length 500 \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 4 \
  --num_train_epochs 3 \
  --start_train_o_epoch 2 \
  --seed 1 \
  --start_step 0 \
  --nb_tasks 6 \
  --per_types 6 \
  --cls_name ncm_dot \
  --feat_dim 128 \
  --relabel_th 0.98 \
  --relabels_th_reduction 0.05 \
  --loss_name1 supcon_ce \
  --loss_name2 supcon_o_bce \
  --change_th \
  --do_train \
  --do_predict
```


### 3. 输出结果

训练完成后，结果会写到：

- `output_nerd/task_x/`：各阶段模型输出
- `output_nerd/task_x/eval_results.txt`：评估结果
- `output_nerd/task_x/test_pred_gold.txt`：预测结果
- `log/results.txt`：增量任务整体日志
