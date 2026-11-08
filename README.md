# rft-rllm：Base 模型的 RFT–GRPO 对比框架

从同一个冻结的 Base checkpoint 出发，比较 **初始模型、离线 RFT、GRPO** 的 pass@k、回答长度和截断率。训练与评估共用三条手工 few-shot 示例、raw-completion 提示和官方数学判分器，重点观察整体及 Level 4–5 难题上的差异。

RFT 使用官方 rLLM 生成、正确轨迹筛选和 `AgentSFTTrainer(backend="verl")`；GRPO 使用 `AgentTrainer(backend="verl")`。本项目负责数据、Base 模型输入适配、训练编排和汇总，不自定义训练损失。实验口径见 [docs/PROTOCOL.md](docs/PROTOCOL.md)。

Base 适配包含统一 raw 提示、SFT 的 prompt/completion mask，以及 GRPO 中 EOS 与 padding 共用 token ID 时的长度 mask 修复。后者带固定上游源码校验，并安装到实际训练的 Ray 工作进程。

## 模型与预设

```bash
python -m rft_rllm models
```

| `--model` | 初始 checkpoint |
| --- | --- |
| `qwen3_1p7b_base` | `Qwen/Qwen3-1.7B-Base` |
| `qwen3_4b_base` | `Qwen/Qwen3-4B-Base` |
| `qwen35_2b_base` | `Qwen/Qwen3.5-2B-Base` |
| `qwen35_4b_base` | `Qwen/Qwen3.5-4B-Base` |

默认选择 Qwen3-1.7B-Base。`--model` 同时调整实验名；`--name` 可显式指定名称。四个模型共用协议，各自运行，不会由一次 `run` 自动启动全部模型。

| 配置 | 训练题池 | 测试采样 |
| --- | --- | --- |
| `baseline.json` | MATH 分层 pilot2048 | 每题 N=256 |
| `formal.json` | MATH full，包含 pilot，排除同一 dev256 | 每题 N=256 |
| `high_precision.json` | 同 formal | 每题 N=1024 |
| `iterative.json` | pilot2048，三轮离线 RFT | 每题 N=256 |
| `gsm8k_debug.json` | 仅 GSM8K 固定 debug32，生成和判分 | 每题 N=8，不训练 |
| `deepmath_8k.json` / `deepmath_16k.json` | DeepMath 固定 8192 / 16384 题 | 复用 MATH dev256 和 MATH-500 |

默认测试为完整 MATH-500，报告 `k=1,2,4,8,16,32,64,128,256`；高精度预设只增加 N，保留这些 k。DeepMath 需要手动选择配置。`formal` 和 `high_precision` 使用独立实验名，执行 `run` 时会重新训练和评估。

共享训练设置为 temperature=1.0、top_p=1.0，prompt 和回答各最多 4096 token。RFT 每题采样 8 次，全部正确轨迹进入官方 SFT，1 epoch、lr=1e-6、batch=32；GRPO 为每批 32 题×8 回答、1 epoch、lr=1e-6、KL=0。两组均走全参数路径，SFT 长度上限为 8193，为 EOS 额外留一个位置。默认 1 GPU；实际显存和硬件适配须在目标机器验证。

## 安装

推理和训练面向 Linux/CUDA。安装脚本需要 `uv`，默认创建 Python 3.11 环境：

```bash
bash scripts/setup_linux.sh
source .venv/bin/activate
```

脚本固定 rLLM 提交 `4e828c81680e5a85b638ceedeaf09d1e90109ca8`，使用其 verl 0.8.0 依赖，保存实际依赖 freeze。默认 CUDA wheel 后端为 `cu128`，可通过 `TORCH_BACKEND` 调整。安装不会下载模型或启动训练。

仅查看计划和运行轻量检查，可在 Python 3.11/3.12 中执行 `python -m pip install -e .`；本地包本身不安装 GPU 栈。

## 运行

先查看计划，再用 GSM8K 检查真实模型的生成和判分：

```bash
# 只打印计划，不下载数据/模型，不启动 GPU 任务
python -m rft_rllm plan --config configs/baseline.json --model qwen3_1p7b_base

# 固定 32 题 × 8 次，仅生成、判分；--stage all 也只执行 debug
python -m rft_rllm run --config configs/gsm8k_debug.json --stage debug

# Pilot：准备 → Base 评估 → RFT → GRPO → 终点评估 → 汇总
python -m rft_rllm run --config configs/baseline.json

# Full：与 pilot 共用 dev，训练题池扩大到全部剩余 MATH train
python -m rft_rllm run --config configs/formal.json
```

四个模型可用同一组 CLI 参数查看计划；将 `plan` 改成 `run` 才会依次执行实验：

```bash
for model in qwen3_1p7b_base qwen3_4b_base qwen35_2b_base qwen35_4b_base; do
  python -m rft_rllm plan --config configs/formal.json \
    --model "$model" --gpus 1 --tensor-parallel-size 1
done
```

`--gpus` 控制训练 GPU 数，`--tensor-parallel-size` 控制推理张量并行度，物理设备用 `CUDA_VISIBLE_DEVICES` 选择。`--revision <HF提交或标签>` 指定远端版本，首次运行记录确定提交；`--local-model /path/to/checkpoint` 使用已有的不可变完整 HF checkpoint。可追加 `--name <实验名>` 区分预算或硬件设置。

MATH 和 DeepMath 支持分阶段执行，例如：

```bash
python -m rft_rllm run --config configs/baseline.json --stage prepare
python -m rft_rllm run --config configs/baseline.json --stage rft
python -m rft_rllm run --config configs/baseline.json --stage grpo
python -m rft_rllm run --config configs/baseline.json --stage base
python -m rft_rllm run --config configs/baseline.json --stage evaluate
python -m rft_rllm run --config configs/baseline.json --stage report
```

生成和评估各自启动并关闭本地 vLLM 服务，默认地址为 `127.0.0.1:8010`，训练前释放该服务。状态文件跳过已完成阶段；中断且产物不完整的训练仍需检查和归档，不能据此认定支持任意 step 恢复。

## 数据与产物

`data/math/` 一次生成 `pilot.jsonl`、`full.jsonl`、`dev.jsonl`、`test.jsonl` 及 manifest。MATH 按 subject×level 确定性分层，先留 dev256，再从剩余题目选 pilot2048；切换 pilot/full 不改变 dev。真实缓存回归得到 full7243，正式题数以本次 manifest 为准。其他协议分别写入 `data/gsm8k_debug/`、`data/deepmath8k/`、`data/deepmath16k/`。每个物理 split 使用不同的 rLLM registry 名称，防止上游 materialization 缓存混用训练集和测试集。

每次实验位于 `runs/<experiment.name>/`：

```text
config.json、environment.json    解析配置与实际安装包版本
initial_model.json、initial_model/  冻结来源与本次运行的 tokenizer 适配目录
state.json                       阶段状态
base_eval/、rft_eval/、grpo_eval/  results、episodes、protocol、summary
rft/round_01/collect/、train/、hf/  候选轨迹、SFT checkpoint、导出模型
grpo/train/                      GRPO checkpoint、有效配置、预算计划与 rollout 日志
summary.json、summary.csv         整体、Level 4–5、各 level、各 subject 指标
comparison.json、comparison.csv   三组之间的 pass@k 差值
pass_at_k.svg                    整体/难题曲线及 GRPO−RFT 差值，共四个面板
budget.json、commands.jsonl、logs/  可得预算、命令与耗时
```

GSM8K 仅生成 `debug_eval/` 及通用运行记录。SFT 分片 checkpoint 通过官方 merger 导出；GRPO 优先使用其完整 HF 导出，必要时再合并分片。不能将只有配置和 tokenizer 的目录当作完整权重。

变更源码、提示、模型或训练/评估设置时使用新实验名；变更数据源、划分 seed 或 pilot 大小时使用新的 `data.name`。旧版根目录 `data/manifest.json` 和已有运行不会自动迁移或覆盖。远端模型缓存保持原样，当前运行通过权重软链接及独立 tokenizer 文件适配 raw 输入；原始权重必须保持可用且不可变。

## 解读与验证

pass@k 按题计算再平均，只报告 k≤N；错误 attempt 按失败计，缺失、重复或任务指纹不匹配会使汇总报错。缺失 token/截断遥测保留为 `null`，同时报告覆盖数。差值图使用百分点，不包含置信区间或显著性声明。

RFT 与 GRPO 共享题池和测试协议，但生成 token、优化器更新及训练成本并不相等。`budget.json` 分开记录计划和观察值，未计量的反向传播 token 保留为空；三轮 RFT 也不会自动扩大 GRPO 预算。

已完成 CPU 回归及真实缓存的 MATH/GSM8K 数据检查。尚未运行 GPU 推理或训练，也未下载和验证真实 DeepMath。Qwen3.5 的输入和 checkpoint 适配已纳入代码；这不表示其硬件、显存和训练依赖已经实测兼容。Qwen3.5 保留完整多模态 checkpoint，以文本题目输入，未配置视觉模块冻结。

```bash
python scripts/check_light.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

当前检查范围见 `CHECKS.json`。上游依据：[rLLM 固定版本](https://github.com/rllm-org/rllm/tree/4e828c81680e5a85b638ceedeaf09d1e90109ca8)、[数学 cookbook](https://github.com/rllm-org/rllm/tree/4e828c81680e5a85b638ceedeaf09d1e90109ca8/cookbooks/math)、[verl 0.8.0](https://github.com/volcengine/verl/tree/v0.8.0)。
