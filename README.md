# MD-PCC
Source code of the paper titled "Robust Misinformation Detection by Visiting Potential Commonsense
Conflict"
## 目前的工作：

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

| 改动项 | 原值 | 新值 |
|--------|------|------|
| `--device` 默认值 | `cuda:0` | `cpu` |
| `--icl_num` 默认值 | `0` | 保持 `0` |
| 新增 `--num_beams` | 无 | 默认 `1`（贪心解码） |
| 新增 `--comet_num_beams` | 无 | 默认 `1` |
| 新增 `--split` | 无 | 默认 `'all'`，可选 `train/val/test/all` |
| 删除 `if args.device == "cpu"` 分支 | - | 统一用 `args.num_beams` |
| 添加内层进度显示 | 无 | 三层进度条（Data/Relations/Steps） |

## comet_cn.py 改动

| 改动项 | 原逻辑 | 新逻辑 |
|--------|--------|--------|
| `__init__` | 硬编码 `num_beams` | 接收 `num_beams` 参数 |
| `generate` | 设备判断分支 | 统一用 `self.num_beams` |

### 运行命令示例

```bash
# 完整复现原始行为（GPU + beam search）
python generate_cn.py --dataset ours --device cuda --num_beams 10 --comet_num_beams 5

# 快速测试（CPU + 贪心解码）
python generate_cn.py --dataset ours --device cpu

# 单独处理某个数据集（三个数据集可并行运行）
python generate_cn.py --dataset ours --split train --device cuda --num_beams 10 --comet_num_beams 5
python generate_cn.py --dataset ours --split val --device cuda --num_beams 10 --comet_num_beams 5
python generate_cn.py --dataset ours --split test --device cuda --num_beams 10 --comet_num_beams 5
```

### 注意修改一下generate三个文件的路径，ours数据集就是comet数据集
##  做完这些工作才能正式训练模型
python main.py --model_name bert --dataset gossip 

