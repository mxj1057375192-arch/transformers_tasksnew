# transformers_tasks 本机跑通报告

日期：2026-09-28
目标：在无 GPU 的 Windows 笔记本上，以「最小可跑通」配置验证项目 9 大任务的全部代码路径。

---

## 一、结论

**12 个训练/推理任务全部跑通（exit 0）**，loss 均正常下降、评测指标正常产出、checkpoint 与训练曲线图全部落盘。

跳过 3 项（均为约定跳过）：
- `tools/tokenizer_viewer` —— 按你的要求跳过。**补充说明：它其实是可行的**，只是一个 streamlit 页面，本机 2 分钟就能起来，随时可以补做。
- `LLM/chatglm_finetune` —— 6B 模型 fp16 需 ~13GB 显存，本机 16GB 内存无独显，不可行。
- `LLM/LLMsTrainer` —— 依赖 DeepSpeed，官方只支持 Linux，Windows 不可行。

---

## 二、结果明细

| # | 任务 | 状态 | 耗时 | 关键指标 |
|---|---|---|---|---|
| 1 | text_classification | ✅ | ~17 min | F1 0.83（3 epoch） |
| 2 | prompt_tasks/p-tuning | ✅ | ~5 min | loss 2.77 → 0.27 |
| 3 | prompt_tasks/PET | ✅ | 17 min | F1 0.73 |
| 4 | text_matching/PointWise | ✅ | 28 min | **F1 0.8716** |
| 5 | text_matching/DSSM | ✅ | 22 min | F1 0.6774 |
| 6 | text_matching/SentenceTransformer | ✅ | 22 min | F1 0.7089 |
| 7 | UIE 信息抽取 | ✅ | 30 min | **F1 0.8764** |
| 8 | RLHF 奖励模型 | ✅ | 30 min | 排序 acc 0.37 |
| 9 | SimCSE 无监督 | ✅ | 66 min | F1 0.7136（含 spearman） |
| 10 | answer_generation (T5) | ✅ | 12 min | BLEU-4 0（见问题 6） |
| 11 | filling_model (T5) | ✅ | 21 min | BLEU-4 0（见问题 6） |
| 12 | RLHF PPO | ✅ | 27 min | reward 0.730 → 0.808 → 0.756 |

机器训练时间合计约 **5 小时**，加上环境搭建、模型下载、失败重试，全程约 **7 小时**（无人值守）。

训练曲线图产出在各自目录的 `logs/` 下，共 10 张 PNG；checkpoint 保存在各自 `checkpoints/`。

---

## 三、环境隔离（你要求不动的部分）

新建独立 conda 环境 `plm`（从 py310 克隆，复用其 torch 2.1.0+cpu，未重新下载）。

**原有 3 个环境经核验完全未改动：**

| 环境 | Python | 状态 |
|---|---|---|
| base | 3.9.12 | numpy 1.21.5，无 transformers/torch —— 原样 |
| py310 | 3.10.10 | torch 2.1.0+cpu、numpy 2.2.6 —— 原样 |
| py312 | 3.12.14 | transformers 5.17.0、torch 2.14.0、datasets 5.0.1 —— 原样 |
| **plm（新建）** | 3.10.10 | transformers 4.22.1、torch 2.1.0+cpu、datasets 2.4.0、numpy 1.26.4 |

仅对 `plm` 设置了环境变量 `HF_ENDPOINT`（conda env config vars），`conda activate plm` 时自动生效。

---

## 四、模型与数据准备

**模型**（全部落到本地目录，运行时完全离线，不再依赖网络）：

| 模型 | 位置 | 用途 |
|---|---|---|
| bert-base-chinese | `_models/bert-base-chinese/` | 替代 ernie，覆盖 6 个任务的 backbone |
| nghuyong/ernie-3.0-base-zh | `_models/ernie-3.0-base-zh/` | 备好未启用 |
| uer/t5-base-chinese-cluecorpussmall | `_models/t5-base-chinese-cluecorpussmall/` | T5 问答/填空 |
| uer/gpt2-chinese-cluecorpussmall | `_models/gpt2-chinese-cluecorpussmall/` | PPO |
| uer/roberta-base-finetuned-jd-binary-chinese | `_models/roberta-.../` | PPO 的 reward 来源 |
| Pky/uie-base-zh | `UIE/uie-base-zh/` | UIE（代码硬编码 HF 地址，需手动放） |

**最小数据集**（`_minimal_data/`，只新建不覆盖原文件）：RLHF 1000 条、SimCSE 1 万条、T5 问答 1000 条、T5 填空 2000 条。

**执行脚本**（可重复运行）：`run_all.sh` / `run_final.sh` / `run_cleanup.sh`，日志在 `_minimal_data/runlogs/`。

---

## 五、为跑通而做的代码修改（共 7 处）

| 文件 | 问题 | 修法 |
|---|---|---|
| `UIE/train.py` ×2 处 | 官方权重在 GPU 上保存，`torch.load` 无 `map_location`，纯 CPU 机器直接崩 | 加 `map_location='cpu'` |
| `RLHF/trl/core.py` | `collections.Mapping` 在 Python 3.10 已移除 | 改 `collections.abc.Mapping` |
| `text_classification/train.py` | 评测器从不 `reset()`，指标在历史结果上累加，越训越失真 | `evaluate_model` 开头加 `metric.reset()` |
| `answer_generation/inference.py` | 形参写 `qustion`、函数体用 `question`，一调用就 NameError | 统一为 `question` |
| `RLHF/ppo_sentiment_example.py` | 模型名硬编码为 HF 仓库 ID；157 轮跑不完 | 改指向本地目录；steps 20000 → 640（5 轮） |
| `requirements`（plm 环境） | `sentencepiece 0.2.2` 与 py3.10 冲突，`import datasets` 后再导入会**段错误** | 锁 `0.1.96` |
| `requirements`（plm 环境） | `matplotlib 3.8+` 移除 `seaborn-darkgrid`，而 `iTrainingLogger.py` 硬依赖 | 锁 `3.6.0` |

---

## 六、遗留问题（未修，属项目原有缺陷）

1. **BLEU 恒为 0**（T5 两个任务）：`train.py` 把含 `-100` padding 的完整 labels 当作 reference 做 token-id 级 n-gram 匹配，参考长度被 padding 撑满，brevity penalty 失效。指标不可用，但训练本身正常。
2. **`--learning_rate` 参数被吞**：text_matching 三个脚本、answer_generation、filling_model 均硬编码 `lr=5e-5`，命令行传的值无效。
3. **FocalLoss 只支持二分类**：`alpha` 硬编码为 2 元素张量，配 `--num_labels 8` 会越界。
4. **`--use_class_weights` 的 `type=bool` 陷阱**：传 `False` 仍是 True。
5. **RLHF 奖励模型极慢**：0.04 step/s（逐句 forward、未批量化），3000 条就要 78 分钟。
6. **readme 笔误 3 处**：`get_embeddings.py`（实际是 `get_embedding.py`）、filling_model 与 simcse 的 `pip install -r ../requirements.txt` 路径少一层。

---

## 七、与事前估算的对比（重要）

| 项目 | 事前估算 | 实际 | 偏差 |
|---|---|---|---|
| 全部任务机器训练时间 | 1.5~2 小时 | **5 小时** | 偏乐观 2.7 倍 |
| 单任务最快 | 3 分钟 | 5 分钟 | 接近 |
| SimCSE | 15 分钟 | 66 分钟 | 偏乐观 4.4 倍 |
| 模型下载 | 0.5~1.5 小时 | ~1.5 小时 | 符合 |

**偏差主因**：事前按「236 GFLOPS 峰值 × 有效利用率」推算，得到 BERT-base 约 2 step/s；实测只有 **0.16 step/s**，相差 12 倍。原因是 CPU 上大量小算子、内存搬运使有效算力远低于峰值。后续估时应直接以 0.16 step/s 为基准。

---

## 八、基础设施发现（对下次最有用）

1. **网络**：`huggingface.co` 与 `hf-mirror.com` 均不可达（DNS 能解析，连接被丢弃）。**`aifasthub.com` 是完整可用的 HF 镜像**，设 `HF_ENDPOINT=https://aifasthub.com` 即可让所有原版模型 ID 正常工作，**无需改一行代码**。
2. **镜像的坑**：它 302 重定向到 `us.aws.cdn.hf.co`，那条连接**会被提前掐断且不报错**，导致下载到截断文件（首次下载 ernie 只拿到 20MB／实际 474MB）。必须用 curl + 断点续传，并在下载后校验文件大小。
3. **务必过滤冗余权重**：`tf_model.h5` / `flax_model.msgpack` 每个 400~850MB，PyTorch 用不到。
4. **`evaluate` 库需要联网**：它会下载 accuracy/f1 等指标脚本，设 `HF_HUB_OFFLINE=1` 会导致 `FileNotFoundError`。用本地模型路径 + 联网找指标即可。
5. **下载速度约 0.7MB/s**，400MB 模型约 9 分钟，是最大瓶颈。

---

## 九、未授权操作说明

执行过程中发现一个**脱管进程**（PID 17660，本会话 run_all.sh 队列的 `train_reward_model.py`，因 TaskStop 终止父进程时 timeout 包装器被一并杀掉而失去管控）。我尝试定向终止它，但被权限策略拦截——理由是**你不在场，无法确认该进程归属**。这个拦截是合理的，我没有绕过，让它自行跑完（约 1 小时后自然结束）。它对本机无损害，只是短暂占用了一些 CPU。
