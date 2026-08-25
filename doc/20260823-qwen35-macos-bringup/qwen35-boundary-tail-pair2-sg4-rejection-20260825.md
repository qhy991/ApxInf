# Tail head pair2/SG4 诊断：停止并撤回

结论：`w8_rows_topk4_pair2_sg4` 在 teacher128 和六次 free128 中保持完全精确，但被直接改动的 tail transaction 反而变慢，未通过预先设定的“tail 中位数至少提升 10% 且端到端至少提升 2%”双门槛。候选源码以诊断提交保存，随后从最终运行时代码中撤回；原始回执和机器可读拒绝摘要继续保留。

## 假设与边界

基线 `w8_rows_topk4` 使用每 SIMDgroup 一行、每 threadgroup 八个 SIMDgroup，共 256 threads；候选让每个 SIMDgroup 独立处理连续两行、每 threadgroup 四个 SIMDgroup，共 128 threads。两者都保持每 threadgroup 八行，partial count、partial buffer、final merge、dispatch 数和 ledger 不变。候选尝试复用 `hidden[index]` 的加载，并降低首阶段 row kernel 的调度开销。

这不是 llama.cpp-equivalent。llama.cpp 的 `NR0=2/NSG=4` 是四个 SIMDgroup 合作处理两行；本候选是每个 SIMDgroup 独立处理两行、每个 threadgroup 仍处理八行，两者执行布局和归约关系不同。

## 正确性

- legacy teacher128 与 pair2 teacher128 全部通过，top-4 mismatch 为 0。
- 六次交错 free128 全部通过，mismatch 为 0。
- 所有 free128 token 序列相同；compact JSON 的 SHA-256 为 `2042d5522ed5e768b938ac3fd9e19d3936dfc6e685fc157d41a8e8132c5b42fe`。
- 奇数和偶数 candidate row 都被 teacher top-4 覆盖，pair2 的第二行路径不是死代码。
- 独立审阅未发现 Metal 正确性、旧 ABI 兼容、selector 传播、receipt 绑定或 Apple M4 目标语义问题。

## Tail 直接计时

固定 teacher128、每个版本 128 个 tail transaction。偶数样本的中位数按排序后两个中央值的算术平均计算。

| 指标 | legacy | pair2/SG4 | 候选改善 |
|---|---:|---:|---:|
| tail transaction 中位数 | 3.367500 ms | 3.400021 ms | -0.9657% |
| 128 次 tail 总耗时 | 439.794082 ms | 443.994294 ms | -0.9550% |
| 逐 token 配对改善中位数 | — | — | -1.3364% |

候选只有 42/128 次更快，86/128 次更慢。因此它距离预设的 tail `≥10%` 保留线很远，并且方向相反。

## 端到端诊断

固定工作负载是 Qwen3.5-0.8B、raw13 `Hello`、greedy free128。同一个 release binary 通过显式 selector 切换 kernel，顺序固定为 `A1 B1 A2 B2 A3 B3`；A 是 legacy，B 是 pair2/SG4。

| 配对 | A legacy TPS | B pair2 TPS | TPS 变化 | A 总延迟 | B 总延迟 | 总延迟改善 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 65.970275 | 66.132653 | +0.2461% | 2050.379 ms | 2069.105 ms | -0.9133% |
| 2 | 64.593605 | 66.080712 | +2.3023% | 2093.312 ms | 2094.941 ms | -0.0778% |
| 3 | 63.226042 | 64.966754 | +2.7532% | 2142.539 ms | 2128.833 ms | +0.6397% |

配对 TPS 变化中位数是 +2.3023%，配对 TPOT 改善中位数是 +2.2504%；但 ratio-of-mean TPS 只有 +1.7494%，配对变化均值是 +1.7672%，完整 generation latency 的配对改善中位数是 -0.0778%。原先“端到端至少 2%”没有进一步预声明 reduction，不能事后只选择恰好越线的统计口径。

而且本轮始终是 A 后接 B，没有反转顺序；A 的 TPS 从 65.970275 连续降到 63.226042，显示明显时序漂移。宿主观测到约 2822 MB swap，后台活动未静默。这个 screen 只能用于拒绝明显未达标的假设，不能用于性能 promotion。

## 决定与可复现性

双门槛是 AND 关系。无论怎样解释噪声较大的端到端信号，tail 门槛都已明确失败，所以结论稳定为 `STOP_REVERTED`。

- 基线父提交：`3fde933ca78d1537bca9757f0363c13cb12f22f3`
- 精确候选源码提交：`4e22799f55c147947ee9b51efbb5a57fa836e6c0`
- 测量 binary SHA-256：`59f0554f49a678d73f5ad1e0cc6b1cfc6d33f8d7732cce78627a1ccf62589933`
- binary 大小：9,576,824 bytes
- build：`cargo build --release -p apxinf-model --features accelerate,metal-w8 --example qwen35_metal_w8_boundary_tail_head_v1_gate --target-dir target/pair2-sg4-v1`
- 验证：Metal tail 测试 13 passed；gate example 23 passed、1 ignored；release build、格式和 whitespace 检查通过。

候选源码先独立提交，再由后续提交撤回，因此 Git 历史能重建实际被测实现，而最终主分支不携带被拒绝的 selector、kernel 或 API。十份原始回执及每份 SHA/大小列在机器可读摘要 `crates/apxinf-metal/evidence/next-hotspot/qwen35-boundary-tail-pair2-sg4-v1-rejected-diagnostic-summary-v1-20260825.json`。

## 对 llama.cpp 对比的影响

无影响。本次是未静默宿主上的 ApxInf 内部拒绝筛选，不会替换已发布的诊断对比。发布值仍为 ApxInf Metal W8 66.336728 TPS、llama.cpp Metal Q8_0 70.663943 TPS；它们仍受量化、KV 路径、线程和宿主噪声差异限制，不能宣称正式等价。

后续优化应先重新定位 tail 内部真正占优的阶段，再设计不同的 kernel 映射；不能把本次 pair2 独立行布局描述为 llama.cpp 的合作式布局。
