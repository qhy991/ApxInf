# 夏令营题目发布就绪报告：Qwen3.8-27B × RTX 4090

日期：2026-08-19
结论：**题目、接口、公开/隐藏评测、动态计分、MFU/BWU、上下文探边、多请求
bonus、独立多模态能力徽章、PR cohort 编排和发布哈希均已形成可执行闭环，可以发布
v1 题目。**

这里的“已发布”表示 workload contract 与 release artifacts 已冻结；当前没有执行
GitHub push，也没有把 `infinigence/ApxInf` 改成公开仓库。开源动作仍由课程负责人在
上课时完成。

## 1. 唯一评分权威

机器权威为 `contract-v1.json`，状态为 `released`：

```text
100 分主榜
  correctness 30 + TTFT 35 + TPOT 25 + reliability 10

20 分 bonus
  32K 以上可验证上下文 10 + 多请求正确 goodput 10

最大展示分 120；自动课程分 min(80, 0.8 × eligible_score)
PR review 20
```

vLLM 是每轮必须运行的 control，不是固定满分。TTFT、TPOT 和 goodput 的每个 cell
均以同轮所有 eligible PR 与 vLLM 中的最好有效值为满分参考。公开 calibration 是
bring-up 结果；正式中期榜要求每格 warm-up 1 次、measure 5 次、中位数和 CV ≤ 10%。

Correctness 已冻结为公开与隐藏两部分：

- protocol 5 分，必须通过；
- 公开功能 6 题 5 分，必须 6/6；
- 隐藏功能 12 题 15 分，至少 11/12；
- 公开/隐藏 trajectory 各 256 token，分别 2/3 分；
- trajectory 使用离散 token ID 的 Levenshtein 编辑相似率按比例赋分，证据必须
  完整，但 INT4 题目不以逐 token 等价作为性能资格的一票否决。

## 2. 发布冻结物

### Starter 与 evaluator

- 最新 upstream starter commit：
  `b85f6def8e7b64b30752d9fc1ee56796cf66a2c3`；
- starter tree：`4733f0486ce7e781ff2c4945fc824db485fd3885`；
- contract SHA-256：
  `16005ce97ee85c4d2ffd580b08a382077d40d6de83a9c1be00e21dc25c2367d6`；
- multimodal overlay SHA-256：
  `3e9844540e3db6c36ec5c6ac214f33e6e419c0ca3423cfdf8d3fa7dfff30d72f`；
- canonical runner SHA-256：
  `f482648abe3bf3a893851d23ad727b5d8869f3bacd71b009525052a321413467`；
- cohort scorer SHA-256：
  `e062f79c8658f465167b984ede1eb8e0acdea4167431aa1fdfe9a00894e2e4e7`；
- teacher evaluator bundle SHA-256：
  `6b29c1c06cdfdd108d3a6208d938c76df0a1a9bc9fd416c2a3205241359913ce`；
- public release manifest SHA-256：
  `1d7457de333f1acefc58284ea303b92731c54093eb6b5992946e90dd19a3ed84`。

### 模型与数据

- 模型：`cyankiwi/Qwen3.8-27B-AWQ-INT4`，revision
  `63768c10df38c0395e12ef49edac1bd539eaeeea`；
- 模型文件 bundle SHA-256：
  `d0e5af982e5023701d5743b89d9786e7bbfc6fd47aec480fd4b9cf43aabfffd0`；
- tokenizer/config combined SHA-256：
  `c8fea6a53676e3793408ee0492f4fae1f0920b1cc0c1eb3908541e6a09175b8e`；
- 公开数据 manifest：
  `e21831e6732544033b693b346acd6859a0ec7af3a697a4446bde5ffe8de26e70`；
- 隐藏数据 manifest：
  `53c6a2eed290c5495d06173e804197ba5df79011e48cdbd93e42f98616b817a0`；
- 上下文数据 manifest：
  `1eb5fad13f9ed0a3e3de7d1160fd790517dfa4544ff2645e212e55c51317e257`；
- 不计分文学分析数据 manifest：
  `ba8d55e54b1402d75060c8a6789834963f691b1568ed760485b832cb0f36371e`；
- 多模态公开数据 manifest：
  `91dfae157b555e87055d40b837a78f0ed68f5526ee3580afecec77b29cc45cbc`；
- 多模态隐藏数据 manifest：
  `9bd51fcc84fcd7be69f8d42c6cba41ca062e5bbeca8fbfaaa22b51e1cd18519a`；
- vLLM 公开+隐藏 trajectory reference SHA-256：
  `48e5388b7dd2e0224891907f5100afa64963141adbac88b5e4418908acd44904`；
- teacher release manifest SHA-256：
  `a0e40a0133a5d54afb34bf24d4456556ce71dcf4866a93b5b24dfdd3e4af695e`。

隐藏 seed 文件权限为 `0600`。学生发布目录只包含隐藏 manifest hash、公开形状与规则，
不包含 seed、隐藏生成器、case IDs、expected outputs 或 reference token IDs。

## 3. 当前 ApxInf 的 4090 实测

当前候选二进制：

```text
/root/apxinf-target-sm89-715a0ed/release/apxinf-course-native-mm-fa2-final-20260819
SHA-256 4b1f1231c051e67522f66ab942d164ee784d2934e20004a44b5c89937889d007
```

完整 public calibration + hidden + context 结果：

| 项目 | 当前 ApxInf |
|---|---:|
| protocol | 9/9 live checks |
| 公开功能 | 6/6 |
| 隐藏功能 | 12/12 |
| 公开 trajectory | 179/256（69.92%） |
| 隐藏 trajectory | 51/256（19.92%） |
| 基础请求成功率 | 100% |
| reliability booleans | 5/5 |
| 最大已验证 prompt | 32,640 + 128 output |
| 首个失败 prompt | 32,768 + 128 output，HTTP 400 admission failure |
| 失败后恢复 | `/health` 与 1K 请求均通过 |
| 多请求 | `parallel_requests=1`，合法获得 0 bonus |

单次测量性能：

| Prompt/output | TTFT | TPOT | Peak VRAM |
|---:|---:|---:|---:|
| 1K/128 | 4.512 s | 22.133 ms | 18,373 MiB |
| 2K/128 | 9.038 s | 22.825 ms | 18,373 MiB |
| 4K/128 | 18.453 s | 24.083 ms | 18,373 MiB |
| 8K/128 | 38.478 s | 26.416 ms | 18,373 MiB |
| 16K/128 | 82.734 s | 31.240 ms | 18,373 MiB |

当前 ApxInf 满足主榜资格门槛。trajectory 的偏差会减少 correctness 分，但不会把
INT4 实现错误地排除在 TTFT/TPOT cohort 之外。

表中的完整文字 calibration 来自同一 Marlin-M64 文字路径的前一版二进制。多模态候选
将 text/KV position 与三轴 mRoPE 拆开，但文字请求仍把三轴设置为同一线性位置。部署后
重新通过了 1K/1-token smoke 和 8K 多跳功能题 `MH-521240`；没有为图片能力接线重跑
耗时的全部文字性能 cell。

## 4. vLLM control 与动态分

vLLM v0.27.1、同一 AWQ checkpoint、BF16 compute、FP8 KV 的单请求 control：

| Prompt/output | TTFT | TPOT | Peak VRAM |
|---:|---:|---:|---:|
| 1K/128 | 0.422 s | 82.789 ms | 22,789 MiB |
| 2K/128 | 0.814 s | 83.109 ms | 22,789 MiB |
| 4K/128 | 1.584 s | 83.261 ms | 22,789 MiB |
| 8K/128 | 3.171 s | 83.133 ms | 22,789 MiB |
| 16K/128 | 6.504 s | 81.788 ms | 22,789 MiB |

它通过公开 6/6、隐藏 12/12、公开/隐藏 trajectory 512/512。当前双实现 public
calibration snapshot（一次测量，非正式中期分）：

| 实现 | Correctness | TTFT | TPOT | Reliability | Context | Multi | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| ApxInf | 28.496 | 2.965 | 25.000 | 10.000 | 0 | 0 | **66.461** |
| vLLM control | 30.000 | 35.000 | 7.415 | 10.000 | 0 | 5.000 | **87.415** |

这个结果说明架构取舍清楚：当前 ApxInf 的 decode 明显优于 vLLM control，但 prefill
明显更慢；vLLM 不是固定满分，因为 ApxInf 为两个 TPOT cell 提供了更好的动态参考。

## 5. 多模态能力实测

固定图片集已经在同一 RTX 4090、同一 INT4 权重上完成真实验证：

| 实现 | 公开 | 隐藏 | 中位 E2E | 结果 |
|---|---:|---:|---:|---|
| vLLM 0.27.1 control | 4/4 | 8/8 | 公开 303.7 ms；隐藏 299.7 ms | `multimodal-ready` |
| 当前 ApxInf native FA2 | 4/4 | 7/8 | 公开 8.403 s；隐藏 8.256 s | `multimodal-public-pass` |

当前 ApxInf 明确报告 `multimodal=true`、`fallback_active=false`。真实请求经过 pinned
`Qwen3VLProcessor`、ApxInf 原生 27 层视觉塔、primary merger、image-token embedding
注入、三轴 interleaved mRoPE、64 层 GDN/全注意力 INT4 文字主干和 greedy decode。
公开 OCR、空间颜色、柱状图算术和目标计数 4/4，隐藏 8 题中 7 题通过；唯一失败是
`hidden-mm-bar-arithmetic-02`，期望 `2`、稳定输出 `3`。因此当前支持真实图片推理，
但还没有达到 `multimodal-ready` 的隐藏 8/8 promotion gate。

实现没有把 Qwen3-VL 文字 runtime 冒充 Qwen3.8：只复用了配置驱动的视觉 encoder
原语，并为 Qwen3.8 增加了 head-dim 72、零 deepstack、image embedding 注入，以及
独立的 cache position 和 T/H/W mRoPE。视觉 attention 使用仓库 vendored FA2；相对原
单-warp SDPA，失败图片的 `vision.primary` cosine 从 0.999192 提升到 0.999389。官方
视觉 feature 注入同一 ApxInf 文字主干时失败题会输出正确 `2`，说明剩余差异来自视觉
BF16 数值累计，而不是 processor、mRoPE 或文字接线。事后 feature 缩放和 reference
override 均被拒绝，最终服务不包含任何调试 fallback。

## 6. 多请求 pilot

vLLM `max_num_seqs=8` 的 closed-loop 结果：

| Cell | Goodput | Jain fairness | p95 TTFT | p95 TPOT | 判断 |
|---|---:|---:|---:|---:|---|
| C4，32×1K/128 | 40.73 tok/s | 0.9989 | 1.503 s | 94.59 ms | 有效，5/5 |
| C8，32×1K/128 | 62.23 tok/s | 0.9723 | 12.276 s | 100.28 ms | TTFT tail guard 失败，0/5 |

C8 吞吐更高但尾延迟过大，因此不计 bonus。该负结果验证了 bonus 不会仅凭 aggregate
throughput 掩盖严重排队。

## 7. MFU/BWU 诊断

这些数值是合同冻结的 proxy，不直接计分：

| 实现/cell | Prefill BF16-eq MFU | Decode BF16-eq MFU | Decode minimum-model BWU |
|---|---:|---:|---:|
| ApxInf 1K | 7.45% | 1.49% | 94.21% |
| ApxInf 8K | 7.17% | 1.31% | 78.93% |
| vLLM 1K | 79.69% | 0.40% | 25.19% |
| vLLM 8K | 86.98% | 0.42% | 25.08% |

这与端到端现象一致：当前 ApxInf decode 接近权重带宽受限，vLLM prefill 的计算利用率
高得多。`nvidia-smi utilization.memory` 没有被冒充为 BWU；只有 Nsight Compute 的
phase-scoped DRAM bytes 才会产生 `profiled_bwu_pct`。

## 8. P0 完成情况

- [x] canonical runner 直接生成 submission 与 raw/environment evidence；
- [x] 公开 6 题、隐藏 12 题和 4 条 trajectory 冻结；
- [x] hidden seed 只存在教师侧，权限 0600；
- [x] 32,640 诊断、32K→原生上限 staircase、失败与恢复自动化；
- [x] C4/C8 goodput、tail、fairness、stream isolation 与 health 自动化；
- [x] 动态 cohort scorer 与强制 vLLM control；
- [x] MFU/BWU proxy 与 optional profiler counter 口径；
- [x] detached clean-worktree PR 编排与 snapshot provenance；
- [x] 公开 CI 负控制；教师 17/17、学生 bundle 16/16；
- [x] 多模态公开 4 题/隐藏 8 题、确定性 PNG 与 fail-closed 负控制；
- [x] 最新 upstream clean checkout `cargo check --workspace`；
- [x] student bundle 敏感信息扫描；
- [x] contract/model/tokenizer/data/reference/evaluator 哈希冻结。
- [x] 学生实验要求、教师介绍指南与逐页 slide 大纲纳入公开 release manifest。
- [x] 一周日程、中期 leaderboard、workflow 创新加分与最终报告政策冻结。

## 9. 发布边界与后续项

学生 bundle 位于本地 clean worktree：

```text
/Users/haiyan-infiniai/rusin-dev-course-release
```

它只新增 `.github/workflows/qwen38-evaluation.yml` 与
`benchmarks/qwen38_4090/evaluation/` 的公开文件；不包含教师实现、teacher 目录、
fixtures/results、隐藏生成器、私有 host/path、Marlin/M8 或当前 dirty overlay。

开课时仍需执行的不是题目合同修改，而是外部发布动作：

1. 将 clean student bundle 提交到届时公开的 `infinigence/ApxInf`；
2. 在中期轮开始前冻结实际 image/driver/CUDA/power/clock 元数据；
3. 在第二张 RTX 4090 做漂移 pilot，并执行正式 1 warm-up + 5 measurements；
4. 生成命名 cohort snapshot 并按申诉规则整轮重跑无效测量。

多模态仍不进入 v1 主榜，但已经作为独立的
`multimodal-contract-v1.json` 能力赛道冻结：公开 4 题、隐藏 8 题、确定性 448×448 PNG、
图片/JSONL/manifest 哈希、OpenAI image content-part 接口、exact validator、失败关闭和
`multimodal-ready` 徽章都有机器规则。当前 ApxInf
`/health.capabilities.multimodal=true`，公开 4/4、隐藏 7/8，因此获得
`multimodal-public-pass`；它支持真实图片推理，但不会虚报为隐藏集全过。
未来若要为图片能力加分，必须发布新的 leaderboard contract 并重新冻结 cohort，不能在
v1 活跃轮次中追溯改分。

课程手册 v4 仍包含另一套 rubric/W0–W3/AB-BA 规则，不能与本 contract 同时作为 SSOT。
发布时应以 `ASSIGNMENT.md` 与 `contract-v1.json` 为准；旧手册只能在完成 rubric 合并后
再发给学生。
