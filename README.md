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
| `training_smoke.json` | 独立 MATH64，RFT / GRPO 各最多 2 步 | 按阶段运行，仅验证训练与保存 |
| `deepmath_8k.json` / `deepmath_16k.json` | DeepMath 固定 8192 / 16384 题 | 复用 MATH dev256 和 MATH-500 |

默认测试为完整 MATH-500，报告 `k=1,2,4,8,16,32,64,128,256`；高精度预设只增加 N，保留这些 k。DeepMath 需要手动选择配置。`formal` 和 `high_precision` 使用独立实验名，执行 `run` 时会重新训练和评估。

共享训练设置为 temperature=1.0、top_p=1.0，prompt 和回答各最多 4096 token。RFT 每题采样 8 次，全部正确轨迹进入官方 SFT，1 epoch、lr=1e-6、batch=32；GRPO 为每批 32 题×8 回答、1 epoch、lr=1e-6、KL=0。两组均走全参数路径，SFT 长度上限为 8193，为 EOS 额外留一个位置，默认只保留最近两份 checkpoint。默认 1 GPU；实际显存和硬件适配须在目标机器验证。

当前共享生成协议为 `raw_math_fewshot_v2`：生成出现 `\nProblem:` 时停止，避免答完后继续编写下一道题。Base、RFT 收集、GRPO 训练/验证和评测使用同一规则；停止条件记录于评测协议并由报告校验。原 `qwen3_1p7b_base_training_smoke` 的 v1 报告保留，原源码快照位于 `environment/source_snapshots/`。新协议运行需新实验名。

## 安装

推理和训练面向 Linux/CUDA。Python、虚拟环境、下载/编译缓存和临时文件统一放在当前仓库的数据盘目录中。请在新终端中执行；首次先安装本项目自己的 `uv`，不会修改 shell 启动文件：

```bash
cd /mnt/public2/wangzixu/repos/rft-rllm
source scripts/env.sh
curl -LsSf --retry 3 https://astral.sh/uv/install.sh -o "$RFT_RUNTIME_DIR/uv-install.sh"
UV_UNMANAGED_INSTALL="$RFT_RUNTIME_DIR/bin" sh "$RFT_RUNTIME_DIR/uv-install.sh"
TORCH_BACKEND=cu130 bash scripts/setup_linux.sh
source scripts/activate.sh
```

脚本固定 rLLM 提交 `4e828c81680e5a85b638ceedeaf09d1e90109ca8`，使用其 verl 0.8.0、vLLM 0.22.1 与 PyTorch 2.11.0 依赖，匹配默认 CUDA 13 wheel。原 `cu128` 选项不适用于这一组合。FlashAttention 可能需要源码编译，因此先在 `.venv` 中安装版本一致的 CUDA 13.0.2 编译工具，避免调用本机系统 CUDA 12.2。默认编译并行数为 `MAX_JOBS=4`，可在安装时自行调整。

安装脚本通过 `scripts/prepare_cuda.sh` 补齐项目内的 `libcudart.so → libcudart.so.13` 链接，并先运行一个小型 C++ 链接检查。NVIDIA 的运行库 wheel 在本机只有带版本号的文件，而 `-lcudart` 需要未带版本号的链接名称；缺少它会导致 FlashAttention 在完成 CUDA 编译后链接失败。[GNU 链接器说明](https://sourceware.org/binutils/docs/ld/Options.html)

`constraints/runtime.txt` 将 NumPy 固定为 2.3.5，满足当前 Mistral 依赖的 `<2.4` 要求。固定 rLLM 自身覆盖了 verl 的 `numpy<2` 声明，而当前 vLLM 要求的 OpenCV 使用 NumPy 2；因此 `uv pip check` 仍会报告这一项上游元数据冲突。导入和 CPU 测试通过不等于训练已经兼容，需要后续 GPU 实验验证；本项目没有改写这些包的元数据来隐藏冲突。

本机 A800 / 570 驱动使用项目内的 CUDA 13 前向兼容库；安装脚本自动下载、校验并解压到 `vendor/cuda-compat-13-0/`，不安装系统驱动，不需要 `sudo`。这是 NVIDIA 支持的数据中心 GPU 部署方式，仍需安装后进行实际 GPU 验证。[兼容矩阵](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)

安装日志保存在 `environment/install.log`，完成后保存实际依赖 freeze。安装不会下载模型或启动训练。安装成功以脚本正常退出并输出 `Installed.` 为准；依赖安装完成后，仍需单独验证真实 GPU 推理和训练。

```text
.venv/                     本实验 Python 虚拟环境及 CUDA 编译工具
.runtime/bin/、python/      uv 工具与受管 Python 3.11
.runtime/cache/            uv/pip/PyTorch/vLLM/Triton 等缓存
.runtime/huggingface/       模型与 Hugging Face 数据缓存
.runtime/tmp/              下载、解压与编译临时文件
ray/                       Ray 会话、日志和临时文件
vendor/                    固定 rLLM 源码与 CUDA 兼容库
environment/               安装日志、版本与依赖清单
data/、runs/、.rllm/        实验数据、结果与 registry
```

每次新开终端或 tmux 窗口都执行 `source scripts/activate.sh`，以同时恢复虚拟环境、缓存路径和兼容库；仅激活 `.venv/bin/activate` 不会恢复这些路径。现有系统驱动和基础命令继续使用系统提供的版本，Python 实验依赖不安装到系统目录。

激活脚本同时把本机主机名和所有本机 IP 加入 `NO_PROXY` / `no_proxy`，并保留已有排除项。GRPO 的 gateway 和 vLLM worker 使用内网 IP 通信；只排除 localhost 会使这些请求误走下载代理，出现 `Gateway did not become healthy within 30.0s`。需在新启动 Ray 之前重新激活环境，单纯延长超时不能修复这一问题。

安装完成后先做轻量检查和计划查看：

```bash
python scripts/check_light.py
python -m rft_rllm plan --config configs/baseline.json
```

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

GSM8K 调试完成后，先用 `training_smoke.json` 验证两种训练路径。该配置使用独立的 `data/math_training_smoke/` 和运行目录，采样 64 题 × 8 条 RFT 候选，SFT batch=4、最多 2 步；GRPO 每批 4 题 × 4 条、共 2 步，关闭开发集验证。SFT 上限会按实际可用完整批次数收紧，仍保存最后一步。它不用于效果比较，正式训练继续使用原有 pilot/full 配置。

首次实际 GSM8K 调试暴露了响应角色问题：vLLM 在 `add_generation_prompt=False` 时把返回角色继承为 `user`，gateway 保存后无法直接用于 SFT。当前 flow 设置该标志为 `True`；共享 identity template 不读取该标志，因此提示 token 不变，返回角色改为 `assistant`。训练 smoke 会重新采样，不复用旧 debug 的训练消息。

下面假设本地模型已放在 `models/Qwen3-1.7B-Base/`，运行前选择一张空闲 GPU；两个阶段均从同一个初始模型出发。先完成 RFT，再执行 GRPO。显式指定阶段可避免额外启动 MATH-500 评测：

```bash
source scripts/activate.sh
CUDA_VISIBLE_DEVICES=0 python -m rft_rllm run \
  --config configs/training_smoke.json \
  --local-model models/Qwen3-1.7B-Base --stage rft

CUDA_VISIBLE_DEVICES=0 python -m rft_rllm run \
  --config configs/training_smoke.json \
  --local-model models/Qwen3-1.7B-Base --stage grpo
```

两条训练路径完成后，用同一配置执行 `--stage evaluate`，再执行 `--stage report`。前者依次加载 Base、RFT、GRPO，在完整 MATH-500 上每题采样 2 次，共 3,000 条，检查导出模型推理；后者生成表格、曲线和预算记录。这是端到端流程验收，两步训练和每题 2 次采样不能支持正式的 pass@256 比较。验收结束前不要改该运行的源码或配置，否则身份校验会拒绝复用已有结果。

当前 Qwen3-1.7B-Base 的 smoke 已完整通过。v2 生成检查通过后，先运行 2,048 题 pilot 的两条训练路径，再单独启动完整评估。以下命令使用新的实验名，GPU 编号需按当时空闲情况选择：

```bash
for stage in rft grpo; do
  CUDA_VISIBLE_DEVICES=0 python -m rft_rllm run \
    --config configs/baseline.json \
    --local-model models/Qwen3-1.7B-Base \
    --name qwen3_1p7b_base_pilot_v2 \
    --stage "$stage" || break
done
```

这一阶段 RFT 计划生成 16,384 条候选，GRPO 为 64 个训练 step、每步 256 条轨迹；SFT 更新数由正确轨迹数量决定，二者更新及 token 预算并不相等。GRPO 默认每 10 步及末步使用独立 dev256 验证。训练成功后用该运行保存的 `config.json` 执行 `evaluate` 和 `report`；正式评测每组 500×256，三组共 384,000 条回答。

首次准备 MATH 数据仍需访问 Hugging Face；使用本地模型只省去权重下载。若下载依赖 SSH 代理，数据准备期间仍需保持该连接。

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

已完成 CPU 回归及真实缓存的 MATH/GSM8K 数据检查；恢复的 FlashAttention 扩展已通过 A800 上小型 FP16 前向/反向计算与 CPU FP32 参考的对照。v1 本地 Qwen3-1.7B-Base 的 GSM8K debug32 × 8 已完成，256 次生成、0 次请求错误，pass@1=41.41%、pass@8=96.88%，2 条回答达到长度上限。结果位于 `runs/qwen3_1p7b_base_gsm8k_debug_local/debug_eval/`。该轮发现部分回答会续题并影响最终答案提取，v2 已统一增加停止规则。这些数字只用于流程检查。

v2 的真实 GPU 生成检查也已通过：`runs/qwen3_1p7b_base_gsm8k_stop_v2/` 中 256 条回答全部非空、assistant 角色正确，错误数和截断数均为 0，可见回答中下一题边界 `\nProblem:` 的出现次数为 0。CPU 回归共 55 项通过。`qwen3_1p7b_base_pilot_v2` 已准备 train2048/dev256/test500，dev/test 与 smoke 的文件哈希相同，尚未启动 pilot 训练。

MATH64 的 v1 smoke 已完成：512 条 RFT 采样中接受 127 条，SFT 更新 2 步并导出完整 HF 模型；修复网关代理配置后，GRPO 完成 2 个训练 step，其中第二步具有非零梯度，并保存完整 HF 模型。三组各完成 1,000 条 MATH-500 评测且错误数均为 0；Base/RFT/GRPO 的 pass@1 分别为 18.0%/18.0%/16.9%，pass@2 分别为 29.4%/29.6%/27.2%。报告位于 `runs/qwen3_1p7b_base_training_smoke/summary.csv`，不足以得出训练方法优劣结论。真实 DeepMath 尚未下载和验证。Qwen3.5 的输入和 checkpoint 适配已纳入代码；这不表示其硬件、显存和训练依赖已经实测兼容。Qwen3.5 保留完整多模态 checkpoint，以文本题目输入，未配置视觉模块冻结。

```bash
python scripts/check_light.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

当前检查范围见 `CHECKS.json`。上游依据：[rLLM 固定版本](https://github.com/rllm-org/rllm/tree/4e828c81680e5a85b638ceedeaf09d1e90109ca8)、[数学 cookbook](https://github.com/rllm-org/rllm/tree/4e828c81680e5a85b638ceedeaf09d1e90109ca8/cookbooks/math)、[verl 0.8.0](https://github.com/volcengine/verl/tree/v0.8.0)。
