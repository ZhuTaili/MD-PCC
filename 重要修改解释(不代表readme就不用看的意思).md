# MD-PCC 项目优化指南

## 一、项目概述

本项目是一个基于 mT5 和 COMET 的中文常识增强虚假新闻检测系统。原始代码存在以下问题：

- 训练时间过长（单数据集处理需十几小时）
- 进度条设计不合理，导致用户误以为程序卡死
- 参数硬编码，缺乏灵活性

---

## 二、卡死原因分析

### 2.1 根因定位

原始代码的进度条设计存在严重问题：

```python
# 原始代码：外层进度条只在处理完一条完整数据后更新
for string in tqdm(data):  # 13条数据
    for rel in extract_prompts_cn.keys():  # 18个关系
        # 每个关系需要：
        # - 2次 mT5-large generate（抽取实体1、实体2）
        # - 1次 COMET generate（常识推理）
```

**问题**：第一条数据需要完成约 **90 次 mT5-large 前向传播**，进度条才会从 `0/13` 变成 `1/13`。

### 2.2 性能瓶颈

| 步骤 | 调用次数 | 耗时来源 |
|------|---------|---------|
| 实体1抽取 | 18次/数据 | mT5-large beam search |
| 实体2抽取 | 18次/数据 | mT5-large beam search |
| COMET推理 | 18次/数据 | 另一个mT5模型 |

---

## 三、代码优化改动

### 3.1 generate_cn.py 改动

#### 3.1.1 参数优化

```python
# 改动前
parser.add_argument('--device', type=str, default='cuda:0')

# 改动后
parser.add_argument('--device', type=str, default='cpu')  # 默认CPU
parser.add_argument('--num_beams', type=int, default=1)   # 新增：贪心解码
parser.add_argument('--comet_num_beams', type=int, default=1)  # 新增：贪心解码
```

#### 3.1.2 解码策略统一

```python
# 改动前：硬编码设备判断
if args.device == "cpu":
    answer_ids = triplet_model.generate(..., num_beams=1, ...)
else:
    answer_ids = triplet_model.generate(..., num_beams=10, ...)

# 改动后：统一参数控制
answer_ids = triplet_model.generate(..., num_beams=args.num_beams, ...)
```

#### 3.1.3 多层级进度条

```python
# 改动后：三层进度条
pbar_data = tqdm(total=len(data), desc='Data', position=0)     # 数据条数
pbar_rel = tqdm(total=len(extract_prompts_cn), desc='Relations', position=1)  # 关系数
pbar_step = tqdm(total=3, desc='Steps', position=2)            # 每个关系的步骤
```

#### 3.1.4 数据集选择优化

```python
# 改动后：支持单独处理单个数据集
parser.add_argument('--split', type=str, default='all', choices=['train', 'val', 'test', 'all'])

splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
for split in splits:
    # 处理逻辑
```

### 3.2 comet_cn.py 改动

```python
# 改动前：硬编码设备判断
class Comet:
    def __init__(self, model_path, device="cuda"):
        self.device = device
    
    def generate(self, queries, ...):
        if self.device == "cpu":
            results = self.model.generate(..., num_beams=1, ...)
        else:
            results = self.model.generate(..., num_beams=5, ...)

# 改动后：参数化控制
class Comet:
    def __init__(self, model_path, device="cuda", num_beams=1):
        self.device = device
        self.num_beams = num_beams
    
    def generate(self, queries, ...):
        results = self.model.generate(..., num_beams=self.num_beams, ...)
```

---

## 四、参数说明

### 4.1 参数对照表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `cpu` | 运行设备（cpu/cuda） |
| `--num_beams` | `1` | mT5模型束搜索宽度 |
| `--comet_num_beams` | `1` | COMET模型束搜索宽度 |
| `--icl_num` | `0` | in-context示例数量 |
| `--split` | `all` | 处理的数据集（train/val/test/all） |

### 4.2 参数效果对比

| 配置 | 速度 | 质量 | 适用场景 |
|------|------|------|---------|
| `--num_beams 1` | 最快 | 较低 | 快速测试 |
| `--num_beams 5` | 中等 | 中等 | 日常训练 |
| `--num_beams 10` | 最慢 | 最高 | 最终实验 |

---

## 五、运行方式

### 5.1 快速测试（默认配置）

```bash
python generate_cn.py --dataset ours
```

等效于：
```bash
python generate_cn.py --dataset ours --device cpu --num_beams 1 --comet_num_beams 1 --split all
```

### 5.2 平衡配置

```bash
python generate_cn.py --dataset ours --device cuda --num_beams 5 --comet_num_beams 3
```

### 5.3 完整复现（原始论文配置）

```bash
python generate_cn.py --dataset ours --device cuda --num_beams 10 --comet_num_beams 5
```

### 5.4 分布式处理

多个同学可以并行处理不同数据集：

```bash
# 同学A
python generate_cn.py --dataset ours --split train --device cuda

# 同学B
python generate_cn.py --dataset ours --split val --device cuda

# 同学C
python generate_cn.py --dataset ours --split test --device cuda
```

---

## 六、输出文件说明

### 6.1 数据增强阶段

| 输入 | 输出 |
|------|------|
| `train.json` | `train{时间戳}.json` |
| `val.json` | `val{时间戳}.json` |
| `test.json` | `test{时间戳}.json` |

### 6.2 模型训练阶段

| 文件类型 | 路径 |
|----------|------|
| 模型权重 | `./param_model/{model_name}/parameter_bert-{dataset}.pkl` |
| 训练日志 | `./logs/param/{model_name}_param{时间戳}.txt` |
| 测试结果 | `./logs/json/{model_name}{aug_prob}-{时间戳}.json` |

---

## 七、数据增强效果对比

### 原始数据

```json
{
    "content": "【司机朋友请注意！！！】：如果晚上驾驶汽车经过漆黑路段时，受到鸡蛋攻击...",
    "label": "fake",
    "time": "",
    "entity_list": []
}
```

### 增强后数据

```json
{
    "content": "【司机朋友请注意！！！】：如果晚上驾驶汽车经过漆黑路段时，受到鸡蛋攻击... 但是，实体1和实体2。实体 包含 X想表达他/她的想法 而不是 ",
    "label": "fake",
    "time": "",
    "entity_list": []
}
```

---

## 八、训练流程总结
