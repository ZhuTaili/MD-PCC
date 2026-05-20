# MD-PCC
Source code of the paper titled "Robust Misinformation Detection by Visiting Potential Commonsense
Conflict"

## 模型下载（记得改一下generate_cn.py的三个文件的路径）
### 第一个 ： https://www.modelscope.cn/models/google/mt5-large/files
### 第二个 ： https://huggingface.co/svjack/comet-atomic-zh/tree/main
### 第三个 ： https://huggingface.co/hfl/chinese-bert-wwm-ext/tree/main，目前使用的是bert而已
### 微博数据集，英文数据集的位置：
```
You can download _Weibo_ and _GossipCop_ from [ENDEF, SIGIR 2023](https://github.com/ICTMCG/ENDEF-SIGIR2022), and place them to the folder `./data`;
Our constructed dataset is located in `./data/ours`.
```
## 修改预览总结
## generate_cn.py 改动

- `--device` 默认值：原 `cuda:0` → 新 `cpu`
- `--icl_num` 默认值：保持 `0`
- 新增 `--num_beams`：默认 `1`（贪心解码）
- 新增 `--comet_num_beams`：默认 `1`
- 删除 `if args.device == "cpu"` 分支，统一用 `args.num_beams`
- 添加内层进度显示：每 6 个关系打印一次进度

## comet_cn.py 改动

- `__init__`：原硬编码 → 新接收 `num_beams` 参数
- `generate`：原设备判断分支 → 新统一使用 `self.num_beams`
### 接受以上所有 diff 后，直接运行：python generate_cn.py --dataset ours --device cuda --num_beams 5 --comet_num_beams 3
### 注意修改一下generate三个文件的路径，ours数据集就是comet数据集



### Requirements

```
torch==1.12.1
cudatoolkit==11.3.1
transformers==4.27.4
```

### Prepare Datasets



### Run

1. Generate augmented samples

- for English datasets, you can run
```shell
python generate.py --dataset gossip --icl_num 5
```
- for Chinese datasets, you can run
```shell
python generate_cn.py --dataset weibo --icl_num 5
```

2. Train misinformation detectors
```shell
python main.py --model_name bert --dataset gossip 
```
where `--dataset` includes gossip, weibo, ours, politifact, snopes; `--model_name` contains bert, bertemo, eann, mdfend.

### Citation
```

```
