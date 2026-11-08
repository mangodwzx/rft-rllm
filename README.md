# rft-rllm：官方 RFT–GRPO 基线

从同一个初始 checkpoint 比较 **初始模型、RFT、GRPO** 的 pass@k、回答长度和截断率。默认先做一轮离线 RFT：用初始模型在训练集生成回答，保留正确轨迹，再交给官方 SFT 训练。`configs/iterative.json` 提供三轮迭代式 RFT。

本项目负责数据划分、配置、进程与 checkpoint 衔接、日志及指标汇总。RFT 使用官方 `rllm eval → rllm dataset from-eval → rllm sft --backend verl`；GRPO 使用官方数学 cookbook 的 `AgentTrainer(backend="verl")` 路径。没有自定义 RFT/GRPO 损失。实验定义见 [docs/PROTOCOL.md](docs/PROTOCOL.md)。

## 起步配置

| 项目 | `configs/baseline.json` |
| --- | --- |
| 初始模型 | `Qwen/Qwen3-1.7B`，首次运行冻结 HF revision |
| 训练 / dev / 测试 | MATH 2048 / 256 题；完整 MATH-500 |
| 提示与判分 | 同一数学 chat flow、同一官方数学判分器 |
| 采样 | temperature=1.0、top_p=1.0、最多 4096 新 token |
| RFT | 训练题每题 8 次；全部正确轨迹；SFT 1 epoch、lr=1e-6、batch=32 |
| GRPO | 每批 32 题、每题 8 次；1 epoch、lr=1e-6；关闭 KL |
| 训练方式 | 两组均全参数微调；默认 1 GPU，显存是否足够需在目标机器确认 |
| 评估 | 每题 64 次，k=1、2、4、8、16、32、64 |

这里的模型使用正常 chat template。它与旧目录中的 raw-completion Base 模型实验属于不同协议。`formal.json` 只把评估提高到每题 1024 次、k≤256，训练配置仍继承 baseline；`iterative.json` 只把 RFT 改成三轮，不自动扩大 GRPO 预算。

## 安装

训练和推理面向 **Linux / CUDA GPU** 环境，Python 3.11；当前 Windows 工作区用于编辑与轻量检查。先把整个目录放到训练机器，再执行：

```bash
cd rft-rllm
bash scripts/setup_linux.sh
source .venv/bin/activate
```

安装脚本固定 rLLM 提交 `4e828c81680e5a85b638ceedeaf09d1e90109ca8`，使用其声明的 verl 0.8.0 GPU 依赖。CUDA/PyTorch 安装需要与目标机器匹配；运行记录保存实际依赖 freeze，不能把一个源码提交当成全部依赖都已锁定。安装不会自动下载模型或启动训练。

只查看计划时可在可用的 Python 3.11+ 环境执行 `python -m pip install -e .`；本地包不自动安装 GPU 栈。

## 运行

先修改 JSON 中的模型、GPU 数、并行度和预算；正式运行前保留明确的 `experiment.name`。

```bash
# 查看计划，不下载模型、不启动 GPU 任务
python -m rft_rllm plan --config configs/baseline.json

# 完整执行：准备 → 初始模型评估 → RFT → GRPO → 终点评估 → 汇总
python -m rft_rllm run --config configs/baseline.json --stage all
```

也可按阶段执行，已完成阶段通过状态文件跳过；这不等于任意中断的 GPU 训练都能从中间 step 恢复：

```bash
python -m rft_rllm run --config configs/baseline.json --stage prepare
python -m rft_rllm run --config configs/baseline.json --stage base
python -m rft_rllm run --config configs/baseline.json --stage rft
python -m rft_rllm run --config configs/baseline.json --stage grpo
python -m rft_rllm run --config configs/baseline.json --stage evaluate
python -m rft_rllm run --config configs/baseline.json --stage report
```

生成和评估时，程序启动自己的本地 vLLM 服务，结束后关闭；训练前释放该服务占用的显存。默认端口为 `127.0.0.1:8010`，可在配置中修改。RFT 首轮和 GRPO 都读取冻结的初始模型；迭代 RFT 后续轮从上一轮导出的模型重新生成并微调，优化器按轮重新初始化。

```bash
# 三轮迭代式 RFT
python -m rft_rllm run --config configs/iterative.json

# 较大评估预算；默认仍会完整执行三组基线
python -m rft_rllm run --config configs/formal.json
```

变更已有运行的训练/评估配置时，使用新的 `experiment.name`，避免把不同协议的产物混在一起。数据由整个项目共用 `data/` 和 `.rllm/`；改变数据源、划分或题数时需使用新的项目目录或单独保留原 `data/` 后建立新数据，程序不会自动覆盖已有 manifest。只重新汇总已有结果可运行 `--stage report`。

## 产物

数据 JSONL 与 manifest 位于项目根目录的 `data/`，官方 registry 位于 `.rllm/`。每组实验输出位于 `runs/<experiment.name>/`：

```text
config.json                 解析后的实验配置
environment.json            本次运行的 Python 与实际安装包版本
initial_model.json          初始模型来源与冻结记录
state.json                  阶段完成状态
base_eval/                  初始模型的官方 results.json 与 episodes/
rft/round_01/collect/        训练集生成、判分和完整轨迹
rft/round_01/train/          官方 SFT checkpoint 与配置
rft/round_01/hf/             导出的 HF 模型
grpo/train/                 官方 GRPO checkpoint、配置、预算和 rollout 日志
grpo/train/global_step_N/actor/huggingface/  GRPO 原生导出的完整 HF 模型
grpo/hf/                    原生权重缺失时由 merger 导出的备用位置
rft_eval/、grpo_eval/        两组终点的官方评估结果
summary.json、summary.csv   三组可比较指标
budget.json                 各轮接受率、生成预算、可得 token 统计和命令耗时
commands.jsonl、logs/       命令、耗时、执行日志
```

SFT 的分片 checkpoint 需要调用官方 `verl.model_merger` 导出，不能把其中只有配置/tokenizer 的 `huggingface/` 目录直接当作完整模型。编排层负责导出和下一阶段路径衔接。

## 如何解读结果

- pass@k 由每题固定 N 次采样估计，再按题平均；仅报告 k≤N。缺失或重复的题目/attempt 会使汇总报错，运行错误保留并按失败处理。
- `acceptance_rate` 是正确轨迹数/全部生成次数；实际 SFT 样本还要求轨迹具有可用 messages。`observed_task_coverage` 是 N 次采样内至少成功一次的题目比例。
- 回答长度和正确回答长度使用服务报告的生成 token；缺失时为 `null`，并列出统计覆盖数。截断率依据实际 `finish_reason="length"`，不把缺失遥测记成零。
- 两组共享题池、提示、判分和测试采样协议，但离线 RFT 与在线 GRPO 的生成次数、训练 token 和更新次数不相等。用日志和预算记录说明差异，不能据此宣称等算力或隔离了优化目标的影响。
- `budget.json` 区分计划值与观察值；checkpoint 的 `trainer_steps` 不保证等于优化器更新数。未直接计量的反向传播 token 和更新数保留为 `null`，GRPO token 汇总附有日志覆盖数。
- 固定数据划分与部分训练/服务 seed 不保证 GPU 或并发请求逐位可复现；不要给同题的每次采样设置相同请求 seed。

当前交付仅搭建框架并做轻量检查，没有运行 GPU 训练、模型推理或大规模测试；兼容性和显存需求仍须由目标机器上的首次小规模运行确认。

轻量检查记录位于 `CHECKS.json`；需要时运行 `python scripts/check_light.py`，不会加载 GPU 框架。

## 官方实现依据

- [rLLM 固定版本](https://github.com/rllm-org/rllm/tree/4e828c81680e5a85b638ceedeaf09d1e90109ca8)
- [评估入口](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/cli/eval.py)、[正确轨迹筛选](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/cli/dataset.py)、[SFT 入口](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/cli/sft.py)
- [官方数学训练 cookbook](https://github.com/rllm-org/rllm/tree/4e828c81680e5a85b638ceedeaf09d1e90109ca8/cookbooks/math)、[verl 模型导出](https://github.com/volcengine/verl/blob/v0.8.0/verl/model_merger/__main__.py)
