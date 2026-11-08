# 实验协议

## 问题与三组基线

首先观察同一起点下，官方 RFT 与 GRPO 完整训练流程的 pass@k、正确率、回答长度有何差异。初始模型提供训练前参照。此版本不实现旧框架的按题归一化 RFT、不要求先复现某篇论文，也不提前加入归一化或梯度机制消融。

默认初始模型为 `Qwen/Qwen3-1.7B`，使用正常 chat template。模型远端 revision 首次解析为确定提交，RFT 首轮与 GRPO 复用该快照。替换成其他模型时须确认其 chat template 与官方 parser 支持情况；不能将结果标成旧 raw-completion Base 协议。

## 数据与共享设置

MATH 原始 train 按问题文本做 NFKC/空白规范化、精确去重并排除测试重叠。固定 seed 的 hash 排序先留出 dev256，再选择 train2048；它不是按学科/难度分层抽样。测试使用完整 MATH-500，最终题数以 manifest 为准。精确去重不保证排除所有改写或语义相似题。

训练、RFT 收集和最终评估共用同一 flow、数学提示及官方数学判分器。参考解只用于提供答案与判分，RFT 监督文本来自模型生成。测试集不参与 RFT 筛选或 GRPO 训练。dev 用于过程观察；最终比较使用预定训练终点，不用测试成绩挑 checkpoint。

默认统一 temperature=1.0、top_p=1.0、回答上限 4096 token。prompt 预算 2048，SFT 总长度上限 8192。具体过长输入处理由各官方入口执行，需结合日志确认数据未因长度约束产生不同筛选。

## RFT 的准确含义

1. 使用本轮起始模型，在固定训练题池上执行官方 `rllm eval --attempts 8`，保存完整成功和失败轨迹。
2. 执行 `rllm dataset from-eval --filter solved --select correct --no-dedup`，保留全部正确轨迹。无每题数量上限、不选择最短答案、不按成功数重加权。
3. 官方 `rllm sft --backend verl --lora-rank 0` 对这些轨迹的 assistant token 做交叉熵微调。默认 1 epoch、lr=1e-6、batch=32。

默认一轮为**离线 RFT**：采样数据只生成一次，SFT 期间不刷新。三轮配置为**迭代式 RFT**：后续轮从上一轮 HF 导出模型重新生成本轮正确轨迹，再继续训练模型权重；每轮重新建立优化器，不混入之前轮的轨迹。

此路径使用成功样本混池及 batch token 平均，成功多的题目贡献更多轨迹；全错题不会进入 SFT。它并不等价于每题成功回答平均、零成功题保留在分母的在线 RFT。依据：[官方 curation](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/eval/curation.py)、[SFT backend](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/trainer/sft/verl_backend.py)、[verl SFT loss](https://github.com/volcengine/verl/blob/v0.8.0/verl/workers/utils/losses.py)。

该固定版本的 episode 文件名回退可能包含 rollout 编号，导致 curation 的“任务数”显示与真实题数不同；因此不使用其 task-level 留出或 cap 功能。当前 solved/correct 筛选仍保留全部正确轨迹。真实题目覆盖从 `results.json` 的 `idx` 分组计算。

## GRPO 与预算

通过官方 unified config 与 `AgentTrainer(backend="verl")` 启动，使用组内标准化优势、官方 GRPO 更新及 sequence-mean/token-mean 归约。默认 32 题×8 回答、1 epoch、lr=1e-6、KL=0；不启用拒绝重采样或 stale-policy 异步训练。vLLM 的 async rollout 后端不代表异步策略训练。依据：[数学 cookbook](https://github.com/rllm-org/rllm/tree/4e828c81680e5a85b638ceedeaf09d1e90109ca8/cookbooks/math)。

RFT 只反向传播筛选后的成功轨迹，GRPO 在线刷新并使用组内全部回答。因此统一题池、采样上限和学习率只是共享协议，不构成严格等 token、等更新或等 FLOP 比较。记录每轮候选数、正确率、实际可训练轨迹数、训练 token、checkpoint step、耗时及上游日志；有缺失时明确留空。三轮 RFT 配置是额外实验，不能视作与一轮 GRPO 等预算。

## 评估与汇总

三组在同一测试集上按相同顺序、相同提示和解码设置独立生成。pilot 每题 N=64；formal 每题 N=1024。第 i 题有 c_i 次正确时，使用：

```text
pass@k_i = 1 - C(N - c_i, k) / C(N, k),  1 <= k <= N
pass@k = 各题 pass@k_i 的平均
```

官方已提供这一统计；本地汇总根据官方逐条记录重新计算，并更严格地拒绝缺失/重复的 attempt 和 k>N。基础输出包括 accuracy、pass@k、平均生成 token、正确回答平均 token、接受率、观察题目覆盖率、截断率、错误数及遥测覆盖数。token 数据缺失不重新分词冒充服务的原始 token。

错误 attempt 计为失败，不能重试到成功后替换原记录。观察题目覆盖只表示 N 次采样至少成功一次，不代表已确定模型能力边界。这个首轮框架不输出显著性声明或跨种子置信区间；出现稳定差异后再增加独立训练种子和配对统计。

## 可追踪性与当前验证范围

每次运行保留解析配置、模型快照来源、数据源 revision/文件哈希、实际环境 freeze、命令与日志。默认的 seed 控制范围有限：固定数据 hash 划分、服务初始化和部分 GRPO 随机性；官方 SFT CLI 没有完整 seed 接口。并发 GPU 采样不能保证逐位复现，也不为各 attempt 复用同一请求 seed。

完成状态属于阶段级恢复。已有阶段可跳过；中断且产物不完整的训练不能自动认定成功。变更训练/评估配置应使用新 experiment 名称；项目共用 `data/` 与 `.rllm/`，改变数据配置还需独立保留原数据并使用新的数据目录/项目，不能仅换 experiment 名称后覆盖。formal preset 使用独立名称重新执行完整训练与评估，会额外消耗训练预算。SFT checkpoint 根目录和 GRPO 的 actor 子目录经官方 merger 转成 HF 权重后，才供下一轮或评估使用。

当前仅完成代码框架与必要轻量检查。未安装并验证完整 GPU 栈，未执行训练或真实模型评估；这些不应表述为已复现实验结果。
