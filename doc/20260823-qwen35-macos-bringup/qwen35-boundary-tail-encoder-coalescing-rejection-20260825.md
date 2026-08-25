# Encoder 合并诊断：停止并撤回

结论：将 boundary-tail 解码路径的 Metal compute encoder 从每 token 24 个合并到 7 个，正确性完全保持，但端到端收益没有越过预先设定的 1% 保留线。因此候选代码已撤回，原始回执与拒绝结论保留。

## 结果

固定工作负载为 Qwen3.5-0.8B、raw13 `Hello`、greedy free128。三组交错顺序为 `A1 B1 A2 B2 A3 B3`，A 是提交 `3994dced2b3f4d9b17b6f8e17b513720654b734b`，B 是未提交的 encoder 合并候选。

| 配对 | A 基线 TPS | B 候选 TPS | B 相对变化 |
|---|---:|---:|---:|
| 1 | 65.650340 | 66.091656 | +0.6722% |
| 2 | 66.773431 | 66.831140 | +0.0864% |
| 3 | 66.321477 | 66.126067 | -0.2946% |

TPS 均值从 66.248416 到 66.349621，仅 +0.1528%；独立中位数从 66.321477 到 66.126067，反而 -0.2946%；配对变化的中位数是 +0.0864%。initial stack 加五段 boundary 的 Metal body 时间，配对改善中位数也只有 +0.1292%。

结构计数确实按预期变化：compute encoder 为 24→7；command buffer、commit、wait 仍为 7，kernel dispatch 仍为 267，H2D/D2H 仍为 28,672/28,688 bytes。也就是说，实验准确隔离了 encoder 生命周期，但它不是当前主要瓶颈。

## 正确性与证据等级

teacher128、free128 和六次交错运行全部通过，mismatch 为 0；128-token compact JSON 的 SHA-256 均为 `2042d5522ed5e768b938ac3fd9e19d3936dfc6e685fc157d41a8e8132c5b42fe`。

本次只能标为诊断：宿主存在约 2.8 GB swap 和后台进程活动，只跑了三对样本，没有满足六个 ABBA/BAAB block、每版本至少 12 个样本的正式协议。完整机器可读摘要和全部原始回执位于 `crates/apxinf-metal/evidence/next-hotspot/`，摘要文件为 `qwen35-boundary-tail-encoder-coalesce-v1-rejected-diagnostic-summary-v1-20260825.json`。

## 对 llama.cpp 对比的影响

无影响。这是一个被拒绝、宿主不受控的 ApxInf 内部筛选，不会替换已发布的诊断对比。当前发布值仍为 ApxInf Metal W8 66.336728 TPS、llama.cpp Metal Q8_0 70.663943 TPS；二者本身仍受量化、KV 路径、线程与宿主噪声差异限制，不能宣称正式等价。

下一项实验转向 tail-only `w8_rows_topk4_r2_sg4`：每 SIMDgroup 处理两行、每 threadgroup 四个 SIMDgroup，并保持旧 kernel 可回退。它只在端到端至少提升 2%、tail 中位数至少提升 10% 且 teacher/free 全精确时保留。
