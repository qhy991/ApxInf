# 夏令营题目发布就绪审计：Qwen3.8-27B × RTX 4090

日期：2026-08-18  
结论：**公开自测合同和精确推理接口已达到可校准状态，但尚不能直接发布正式中期榜。**

当前最合适的主任务是单请求推理优化，不是多请求 serving。Marlin-M64 与
ApxInf M8 的现有对比、当前 Rust 服务、vLLM 参考配置都运行在
`parallel_requests=1` / `max_num_seqs=1` 下。若把并发请求放入同一个主分，
题目会同时考查 admission、scheduler、continuous batching、KV 生命周期、
公平性和尾延迟，学生将无法判断分数来自 kernel/runtime 还是调度策略。

因此 v1 主榜冻结为：单张 RTX 4090、单个 text 请求、预分词输入、greedy
decode。多请求和多模态应在接口成熟后作为独立 bonus track，不与 v1 主分
相加。

## 1. 发布边界

学生公开获得：

- `INTERFACE.md` 中的预分词推理接口；
- `contract-v1.json`、submission schema、公开数据生成器和 scorer；
- 不含有效实现的 ApxInf extension points；
- 公开正确性用例、负控制和 scorer 单元测试。

教师侧保留：

- 当前 M8/Marlin-M64 实现、vLLM 参考运行和权重部署细节；
- 隐藏正确性与长上下文数据；
- leaderboard runner、环境镜像和最终评分服务；
- 原始教师校准证据。

学生通过 PR 提交。教师评分必须检出 PR SHA，在干净环境中构建并运行；不得
直接执行学生自报的 JSON 作为最终榜分。

## 2. 已冻结的最小合同

| 项目 | v1 决定 |
|---|---|
| 模型 | `cyankiwi/Qwen3.8-27B-AWQ-INT4` |
| revision | `63768c10df38c0395e12ef49edac1bd539eaeeea` |
| 模式 | W4A16，单卡，单请求，text-only，greedy |
| 输入 | 教师 tokenizer 生成的精确 `input_ids` |
| 输出 | 每 token 一个 token-ID event；性能单元必须生成精确预算 |
| 计时 | client send → first token 为 TTFT；first → last token / (N-1) 为 TPOT |
| 性能单元 | TTFT：1K/2K/4K/8K/16K；TPOT：1K/8K；均输出 128 token |
| 容量 | 1K 到 32K prompt 的对数得分；至少完整输出 16 token 并验证失败后恢复 |
| 并发 | 1；不计 queueing、continuous batching 和 multi-request goodput |

唯一评分权威是 `contract-v1.json`。runner、课程手册、README 和 leaderboard
页面只能引用或派生其中的值，不能各自保存另一份阈值。

## 3. 当前实测校准

精确接口解决了两个旧 evaluator 问题：

1. ApxInf 的 chat-template 路径比教师 tokenizer 多报告 40 个 token，导致
   1064/8232 被错误拿来代表 1024/8192 性能单元。新接口直接接收相同的
   `input_ids`，严格运行得到 1024/8192。
2. NIAH 输出的答案正确，但 token-ID 接口包含终止特殊 token
   `<|im_end|>`。旧验证把它拼入可见文本后判错；教师 tokenizer 使用
   `skip_special_tokens=true` 解码后，1K 三个公开用例为 3/3。

当前严格单次 ApxInf 测量：

| Prompt | TTFT | TPOT | 输出 | 轨迹对齐 |
|---:|---:|---:|---:|---:|
| 1,024 | 5.6176 s | 22.582 ms | 128/128 | 128/128 token IDs |
| 8,192 | 46.7490 s | 26.969 ms | 128/128 | 128/128 token IDs |

公开校准分：

| 实现 | Correctness | TTFT | TPOT | Context | Reliability | 总分 |
|---|---:|---:|---:|---:|---:|---:|
| ApxInf Marlin-M64 | 25.000 | 1.005 | 20.000 | 9.000 | 10.000 | **65.005** |
| vLLM | 25.000 | 30.000 | 6.312 | 15.000 | 10.000 | **86.312** |

ApxInf 的数字是 public calibration：1K/8K 各一次测量，无 warm-up，不能当作
正式中期榜分。vLLM 行来自冻结的每单元一次 warm-up、三次测量中位数。
中期 profile 已要求每个 latency cell 至少一次 warm-up、三次测量。

这个分解是合理的：ApxInf decode 已在两个主单元满分，但 prefill 只得到
1.005/30；严格验证的最大 prompt 为 8K，得到 9/15。教师当前实现无需继续
优化才能发题，它适合作为 scorer 的非平凡校准点和 decode anchor。

## 4. MFU 与 BWU 的口径

W4A16 端到端路径混合量化权重加载/反量化、BF16 Tensor Core MMA、attention、
recurrent 和 elementwise 工作，不能把 RTX 4090 的 INT4 TOPS 直接当作整个
模型的 MFU 分母。

公开脚本输出三类明确命名的诊断值：

- `estimated_mfu_bf16_equivalent_pct`：冻结的 dense-equivalent FLOP proxy，
  除以 wall time 和 165.2 TFLOP/s dense BF16 Tensor peak；
- `minimum_model_bwu_pct`：冻结的最低模型/checkpoint bytes，除以 wall time
  和 1008 GB/s；它是 lower-bound model-byte proxy，不是实测 HBM 流量；
- `profiled_bwu_pct`：仅在提供 phase-scoped Nsight Compute DRAM counters 时，
  用 `(read_bytes + write_bytes) / kernel_elapsed / peak_bandwidth` 计算。

当前 proxy：

| 实现 | 1K prefill MFU | 8K prefill MFU | 1K decode BWU | 8K decode BWU |
|---|---:|---:|---:|---:|
| ApxInf | 5.98% | 5.90% | 92.33% | 77.31% |
| vLLM | 77.93% | 86.62% | 24.87% | 24.65% |

MFU/BWU 只用于定位和解释，不进入主榜分。proxy 可因模型计数假设不完整而失真；
脚本不截断超过 100% 的值，超过 100% 应触发计数/边界审计。

## 5. 评分与提交

自动 leaderboard 为 100 分：correctness 25、TTFT 30、TPOT 20、最大上下文
15、reliability 10。课程总评由自动榜 80% 与 PR review 20% 组成。

PR review 的 20 分为：

- tests 与 negative controls：8；
- public interface 与错误处理：4；
- 一条命令 clean replay、环境和 provenance：4；
- 瓶颈分析、负结果和最终 decision：4。

正确性和 reliability 是 eligibility gate。未过 gate 的运行仍返回
`diagnostic_score`，但 `leaderboard_score=null`。缺少性能单元直接失去对应
分数，不能用估算补值。

## 6. 发布优先级

### P0：开题前必须完成

- **已完成**：精确 `input_ids` 输入和 token-ID 输出接口；health 暴露 contract、
  model revision、capacity、并发和 capabilities。
- **已完成**：严格 token/output/success 检查、公开/中期两种 profile、MFU/BWU
  脚本和 5 个 scorer 回归测试。
- **待完成**：一个 canonical runner 直接生成
  `leaderboard_submission.v1`，禁止教师或学生手填聚合 JSON。
- **待完成**：1K→32K staircase、边界加一、失败后 health + 小请求恢复的自动
  capacity runner；明确 capacity stall 与 CUDA OOM 都是有效失败边界。
- **待完成**：公开 skeleton 的 clean checkout 构建；删除/隔离教师实现、权重
  路径、私有主机名、历史结果和 dirty overlay。
- **待完成**：CI 正负控制：空 IDs、非法 token、非零 temperature、超长预算、
  不支持 modality、丢失 token index、silent fallback 与服务恢复。
- **待完成**：contract 标记 `released`，记录 starter commit、evaluator commit、
  config/tokenizer/weights/data hashes。

### P1：中期 leaderboard 前完成

- 冻结 hidden split，只允许换 case 内容，不允许换接口、阈值和计时边界；
- 在至少两张独立 4090 上重跑 vLLM 与教师 ApxInf，确定机器/热漂移容差；
- 每轮榜单开始和结束运行 vLLM control，超出容差则整轮作废或重跑；
- leaderboard 服务只接受 PR SHA 和教师 runner artifact，不接受学生自报分数；
- 保存 driver/CUDA/GPU UUID/power limit/clocks/temperature、binary hash、原始样本；
- 用 targeted Nsight Compute 验证 profiler counter parser，但 profiler 运行不参与
  官方 latency。

### P2：主榜稳定后

- 独立 multi-request bonus：arrival process、并发、goodput、p95/p99、fairness、
  cancellation 和 KV 回收必须另立合同；
- 独立 multimodal bonus：冻结 image preprocessing、vision token budget 和任务集；
- 32K 以上容量、CUDA Graph、batching 和 speculative decode 作为挑战方向。

## 7. 与课程手册 v4 的一致性

手册仍明确标为 release candidate，并把 `evaluation-contract.json` 作为 SSOT；
其中模型/revision/starter/BF16 包仍有 `INSTRUCTOR_MUST_FREEZE`。它当前还定义了
另一套 100 分 rubric（正确性 30、性能 25、24GB 15、Agent 15 等）、W0–W3
以及 110 次正式配对样本。这不能与本目录的 leaderboard 合同同时作为权威。

按当前真实资源，建议修订手册时采用以下映射：

- 本目录自动 leaderboard 作为机器评分 SSOT；
- 手册中的证据链、负结果、replay 和 Agent 审计收敛到 20 分 PR review；
- G2/G3 的 BF16 layer/logit 深度审计由教师隐藏校验或 Challenge 路线承担，
  Standard 学生主榜使用公开功能集、token trajectory 和隐藏功能集；
- W0–W3/110 次 AB/BA 不再作为所有学生硬门；学生本地用 public calibration，
  教师榜每单元至少 3 次，重要候选再做 paired confirmation；
- 手册中的模型身份更新为真实 AWQ checkpoint revision，并声明 text-only、
  single-request 与 32K score cap。

在手册和 machine contract 完成这次 SSOT 合并前，作业仍应标为
release candidate。

## 8. 复现命令

```bash
python3 benchmarks/qwen38_4090/evaluation/score_submission.py \
  --submission benchmarks/qwen38_4090/evaluation/fixtures/apxinf-marlin-current.json

python3 benchmarks/qwen38_4090/evaluation/score_submission.py \
  --submission benchmarks/qwen38_4090/evaluation/fixtures/vllm-reference.json

python3 benchmarks/qwen38_4090/evaluation/compute_efficiency.py \
  --submission benchmarks/qwen38_4090/evaluation/fixtures/apxinf-marlin-current.json

python3 -m unittest discover \
  -s benchmarks/qwen38_4090/evaluation -p 'test_*.py' -v
```
