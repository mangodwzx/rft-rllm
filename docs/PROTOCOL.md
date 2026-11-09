# RFT–GRPO 实验协议

## 比较对象与模型

每次运行比较同一初始 checkpoint 的 Base、RFT 和 GRPO。目标是观察两种完整训练流程在 pass@k、回答长度及截断率上的差异，重点分析整体和 Level 4–5 题目。最终比较使用预定训练终点；dev 用于过程观察，测试集不参与轨迹筛选、训练或 checkpoint 选择。

支持 Qwen3-1.7B-Base、Qwen3-4B-Base、Qwen3.5-2B-Base、Qwen3.5-4B-Base。远端 revision 首次解析为确定提交，各训练分支复用本次冻结的初始权重。运行目录保存权重软链接和独立 tokenizer 配置，不修改源 checkpoint 或 HF 缓存。使用本地模型时，调用者须保持源目录不可变。

两组均使用全参数训练路径。Qwen3.5 保留完整多模态 checkpoint，实验输入为文本；配置没有显式冻结视觉模块。四个模型使用相同文本协议，仍需分别确认目标硬件和上游训练栈兼容性。

## Raw-completion 提示与判分

[提示模块](../src/rft_rllm/prompts.py) 内置原实验的三条手工 few-shot 示例，依次为一元一次方程、矩形面积和骰子概率。示例与目标问题按 `Problem: … / Solution: …` 拼接，目标末尾停在 `Solution:`。RFT 收集、GRPO rollout 和终点评估均使用这一提示，参考解只用于获取答案和判分。

OpenAI chat endpoint 仅作为 rLLM gateway 的传输接口：identity 模板原样输出单条 user 消息中的 raw 提示，不添加 ChatML、角色边界或 assistant 前缀。模型输入不是普通对话模板。提示版本和哈希写入产物；轨迹保留 prompt 与 assistant completion，供筛选和训练使用。

`raw_math_fewshot_v2` 保留原提示文本，统一在生成出现 `\nProblem:` 时停止，防止模型继续编写下一道题并改变“最后一个 boxed”答案。该规则通过共享 flow 的 OpenAI `stop` 参数应用于 Base 评测、RFT 收集、GRPO 训练和验证；不在第一个 boxed 处停止，允许同一道题修正中间结论。每份 `protocol.json` 记录停止字符串，报告拒绝混用不同版本或停止条件。

vLLM 返回的可见文本不含停止标记，但生成至停止边界的 token IDs 和 logprobs 保留原样供 GRPO 使用，不手动裁切或重新编码；RFT 仍对可见回答编码并补 EOS。两种训练对终止 token 的处理因此存在差异。原 v1 smoke 的完整报告及源码快照保留，新 v2 实验使用独立名称，不能把两套停止协议的分数混为同一实验。

判分委托固定版本的官方数学 evaluator。GSM8K 数据准备从 `####` 后提取数值参考答案，模型回答仍通过共享的数学提示和判分流程处理。

默认 temperature=1.0、top_p=1.0，prompt/response 预算分别为 4096 token。SFT 总长度上限为 8193，额外预留 EOS；raw SFT adapter 拒绝超长样本，不静默截断。生成侧的超长请求或运行错误保留在评估结果中。各 attempt 不复用同一请求 seed，固定服务/训练 seed 也不保证并发 GPU 采样逐位一致。

## 数据协议

MATH 使用原始 train 和完整 MATH-500。问题文本经过 NFKC/空白规范化后做精确去重，并从训练候选中排除测试题和固定示例的精确重叠。该检查不声称排除改写或语义相似题。

按 **subject×level** 分层，以各层占比和最大余数法分配准确题数，层内和最终顺序由固定 seed 的问题哈希决定。先留出 dev256，再从剩余 full 题池选 pilot2048；full 包含 pilot，并排除同一 dev。`baseline.json` 使用 pilot，`formal.json` 使用 full，两者共享 `data/math/` 的物理文件。当前真实缓存回归为 pilot2048/full7243/dev256/test500，正式运行以 manifest 为准。

原始 MATH 的六条异常按明确规则处理：两条 `\boxed 2` / `\boxed 9` 参考答案仅允许单个数字修复；两条空框通过[审计表](../src/rft_rllm/reference_corrections.json)同时匹配问题与原解 SHA-256，修正为 0；两条 `Level ?` 保留为未知难度 `None`。修正只用于参考数据，不放宽模型判分。manifest 记录修正、未知难度、删除原因、分层数量和文件哈希。

GSM8K 从官方 train 中固定选取 debug32，仅用于生成和判分检查，每题 8 次。该配置的 `--stage all` 也只执行 debug；训练阶段会被拒绝。

DeepMath 为手动启用的扩展：按相同 seed 的稳定哈希顺序取前 8192 或 16384 题，两规模嵌套；排除 MATH dev、MATH-500 和固定示例的精确重叠，沿用同一 dev/test。它不要求与 MATH 训练题池互斥。DeepMath 首次运行可建立所依赖的 MATH 数据；重用时验证完整协议，恢复时重新核对保护集哈希。真实 DeepMath 数据尚未下载验证。

数据保存在 `data/<data.name>/`。registry 名包含有效数据配置指纹及物理 split，因此 train/dev/test/debug 互相隔离，pilot/full 也有不同注册名，避免固定 rLLM 版本按 benchmark 名缓存而忽略后续 split 的问题。评估逐条记录中的任务指纹进一步与 manifest 对应题目比对。

## RFT 与 GRPO

RFT 每轮先从本轮起始模型在固定训练池上采样 8 次，再通过官方 `dataset from-eval --filter solved --select correct --no-dedup` 保留全部正确轨迹。不按题限制数量，不选择最短答案，不按成功数重加权；可用监督行数必须与正确轨迹数一致。

筛选结果交给官方 `AgentSFTTrainer(backend="verl")`。本地 [raw SFT 数据适配器](../src/rft_rllm/raw_sft_dataset.py)仅负责 prompt/completion 分开分词、拼接及监督 mask：prompt 不参与损失，completion 和 EOS 参与损失。交叉熵、归约和优化仍使用官方 verl 实现；未复现旧版按题归一化的在线 RFT，也未使用固定 4096 分母的损失。

默认一轮为离线 RFT，采样数据在 SFT 期间不刷新。`iterative.json` 执行三轮：每轮从上一轮导出的 HF 模型重新收集并微调，不混入此前轮的轨迹，每轮重新建立优化器。默认每轮 SFT 为 1 epoch、lr=1e-6、batch=32、LoRA rank=0。

GRPO 走官方 unified config 和 `AgentTrainer(backend="verl")`，使用组内标准化优势、官方 GRPO 更新和 `seq-mean-token-mean` 归约。默认每批 32 题×8 回答、1 epoch、lr=1e-6、KL=0，不启用拒绝重采样或 stale-policy 异步训练；vLLM 的 async rollout 仅指服务后端。

Base tokenizer 可能将 EOS 同时用作 padding token。固定 rLLM 版本按 token 值构造 attention mask，会遮掉真实 EOS；本地 [兼容适配器](../src/rft_rllm/verl_compat.py)依据捕获的真实序列长度恢复 attention mask 和 position IDs，保留官方 token、response mask、奖励、优势和损失。适配器通过 Ray worker hook 在实际构造训练 batch 的进程中安装，并校验 verl 版本及固定上游函数源码哈希。上游实现变化时会停止训练，要求重新核对适配。

成功样本混池让 RFT 中成功多的题目贡献更多轨迹，全错题不进入 SFT。GRPO 在线刷新轨迹，训练组织与 RFT 不同。统一题池、采样上限和学习率不构成等 token、等更新或等 FLOP 比较；后端还可能丢弃不满 batch 的尾部样本。预算记录区分候选数、接受数、可得 token、trainer step 和耗时，未直接计量的反向传播 token/优化器更新保留为 `null`。三轮 RFT 不自动增加 GRPO 预算。

## 评估、分组与图表

三组对同一 MATH-500 按相同顺序和采样协议独立生成。默认 N=256；`high_precision.json` 使用 full 训练并将 N 提高到 1024。两者均报告 `k=1,2,4,8,16,32,64,128,256`，且严格要求 k≤N。第 i 题在 N 次中有 cᵢ 次正确：

```text
pass@k_i = 1 - C(N - c_i, k) / C(N, k)
pass@k   = 各题 pass@k_i 的平均
```

缺失、重复 attempt、任务指纹不一致或分组元数据错误会使汇总失败。运行错误按失败计入分母，不能通过重试到成功替换原记录。回答长度和正确回答长度使用服务返回的 completion token；截断率依据 `finish_reason="length"`。缺失遥测保留为 `null` 并报告覆盖数，不用重新分词代替原始服务计量。

输出包括整体、Level 4–5、各 level 和各 subject 分组；未知 level 单列，空组指标为空。`summary.csv` 列出各模型分支和分组，`comparison.json/csv` 给出分支间差值。`pass_at_k.svg` 的四个面板分别显示整体曲线、Level 4–5 曲线及对应的 GRPO−RFT 差值，横轴为 log₂(k)，差值以百分点显示。

k=N 时 pass@k 等于观察到的题目覆盖率，不能据此确定能力边界。单次训练种子的曲线和差值不提供显著性结论；当前不输出置信区间。

## 追踪、恢复与验证范围

运行保存解析配置、源码和提示指纹、模型 revision、数据源 revision/文件哈希、实际环境版本、命令与日志。配置或源码变化需要新实验名；数据源或划分变化需要新 `data.name`。仅切换 MATH pilot/full 可复用数据。旧版数据和运行不会自动迁移、覆盖或继续混用。

恢复以完成阶段为单位；中断且非空的阶段目录会要求检查和归档。SFT 分片 checkpoint 经官方 merger 导出后再用于下一轮或评估；GRPO 使用其完整 HF 权重或合并后的导出。`report` 校验已保存数据、任务顺序、提示和评估协议后重新汇总。

当前已完成 Qwen3-1.7B-Base 的真实 GPU 生成、MATH64 上两步 SFT/GRPO、完整权重导出、三组各 500×2 的独立推理、判分和报告检查。v1 smoke 三组均无请求错误，结果用于流程验收，不能据此判断 RFT/GRPO 优劣。CPU 检查覆盖数据规范化、分层划分、合成 DeepMath 的保护集与嵌套规模，以及 v2 停止规则透传和 token 对齐。完整 pilot/full 训练、pass@256 和真实 DeepMath 准备仍待执行；这些检查不代表四个模型均已验证兼容。

上游依据：[rLLM 固定版本](https://github.com/rllm-org/rllm/tree/4e828c81680e5a85b638ceedeaf09d1e90109ca8)、[官方 curation](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/eval/curation.py)、[SFT backend](https://github.com/rllm-org/rllm/blob/4e828c81680e5a85b638ceedeaf09d1e90109ca8/rllm/trainer/sft/verl_backend.py)、[数学 cookbook](https://github.com/rllm-org/rllm/tree/4e828c81680e5a85b638ceedeaf09d1e90109ca8/cookbooks/math)、[verl SFT loss](https://github.com/volcengine/verl/blob/v0.8.0/verl/workers/utils/losses.py)。
