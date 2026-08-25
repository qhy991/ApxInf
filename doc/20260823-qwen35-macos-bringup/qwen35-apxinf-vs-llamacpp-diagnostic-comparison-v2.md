# Qwen3.5-0.8B：ApxInf 与 llama.cpp 可比诊断 v2

日期：2026-08-25

## 结论

在冻结的 raw13 / greedy / free128 单提示词工作负载上，四条轨迹逐 token 完全一致：ApxInf CPU F32、ApxInf Metal W8 候选、llama.cpp CPU F32 和 llama.cpp Metal Q8_0 均生成同一组 128 个 token。compact JSON token array 的 SHA-256 为 `2042d5522ed5e768b938ac3fd9e19d3936dfc6e685fc157d41a8e8132c5b42fe`。

在当前单次、噪声主机诊断中：

- F32 泳道的 llama.cpp decode TPS 比 ApxInf 高 7.77%，总延迟低 1.93%，但 TTFT 高 216.79%。
- 非等价 8-bit 存储泳道的 llama.cpp decode TPS 比 ApxInf 高 6.52%，总延迟低 10.89%，TTFT 低 75.59%。

这些数字不是 formal benchmark，也不是发布或晋级依据。它们不证明跨 runtime teacher-forced exactness、通用质量等价、量化机制等价或内存优势。ApxInf Metal W8 路径仍是未默认可达的 candidate-only 路径。

机器可校验证据见 [diagnostic summary](../../crates/apxinf-metal/evidence/llama-cpp/qwen35-0.8b-apxinf-vs-llamacpp-raw13-free128-diagnostic-summary-v2-20260825.json)，其 canonical content SHA-256 为 `a70a8a3b46dd9efd37d0cd5aac906a5ee10f4a61eaf1960e92ec9ebc690bf884`，文件 SHA-256 为 `edb86c4e245bb0e5db561e19c7648253bfa485ac245950510c89d68602e770a6`。

## 冻结工作负载

- 模型：`Qwen/Qwen3.5-0.8B`，revision `2fc06364715b967f1860aea9cf38778875588b17`。
- 输入：13 个 raw token ID，不经过各 runtime 的模板或 tokenizer。
- 生成：greedy argmax，固定 128 token，忽略 EOS。
- 计时起点：紧邻 raw-token prefill 之前。
- TTFT：第一个 token ready；总延迟：第 128 个 token ready。
- TPOT：`(total - TTFT) / 127`；TPS：`127000 / (total - TTFT)`。
- 模型加载不计入；llama.cpp 的第 128 个 token 后执行证明 decode 也不计入。
- 两边 requested context 均为 142；固定 llama.cpp 版本实际报告 effective context 256，因此不声称 effective allocation 相同。

完整合同见 [comparison contract](../../configs/qwen35-0.8b-llamacpp-comparison-v1.json)。合同 canonical content SHA-256 为 `23f46184dce0882ab15c6e7e0b87832d143194b80bf3929d5b5c13f5f2173d89`，原始文件 SHA-256 为 `f3b2057fc9b5be3211c6aa4ba73965c4554de8c9b882fbf70bc3a606b6ae3973`。

## 单次诊断结果

| 泳道 | Runtime | TTFT (ms) | TPOT (ms) | Decode TPS | Total (ms) |
|---|---|---:|---:|---:|---:|
| F32 | ApxInf CPU F32 | 152.304916 | 49.667261 | 20.133987 | 6460.047083 |
| F32 | llama.cpp CPU F32 / F32 KV | 482.482292 | 46.088152 | 21.697551 | 6335.677542 |
| 非等价 8-bit | ApxInf hybrid W8 | 140.958833 | 15.074605 | 66.336728 | 2055.433708 |
| 非等价 8-bit | llama.cpp pure Q8_0 Metal / F16 KV | 34.406750 | 14.151489 | 70.663943 | 1831.645833 |

所有相对值均以 ApxInf 为分母：

| 泳道 | llama.cpp TPS 变化 | llama.cpp TPOT 变化 | llama.cpp total 变化 | llama.cpp TTFT 变化 |
|---|---:|---:|---:|---:|
| F32 | +7.7658% | -7.2062% | -1.9252% | +216.7871% |
| 非等价 8-bit | +6.5231% | -6.1237% | -10.8876% | -75.5909% |

F32 中 llama.cpp 的 steady-state decode 更快，但较高 TTFT 抵消了大部分 total 优势。8-bit 行只能说明两个明确命名、机制不同的实现对该冻结工作负载的诊断观测，不能解释为同一种 W8 实现之间的横评。

## 质量边界

ApxInf 自身的同进程门禁包含：

- Metal W8 候选对 ApxInf CPU F32 的 teacher-forced 128/128 exact。
- Metal W8 候选对 ApxInf CPU F32 的 free-run 128/128 exact。

跨 runtime 已验证的是同一冻结提示词上的 free-run 轨迹：四组 128 token 逐位置相等，match count 和 exact prefix 均为 128，first mismatch 为 `null`。

llama.cpp 尚未采集 teacher-forced receipt。因此不能把 ApxInf 内部的 teacher exactness 延伸为 llama.cpp 的跨 runtime teacher exactness，也不能从一个提示词推出一般质量等价。

## 精度与执行披露

F32 泳道是“同一 HF revision 的 F32 runtime comparison”，不是同一模型文件：

- ApxInf：原始 safetensors，CPU F32 权重与运行路径。
- llama.cpp：从同 revision 转换的 F32 GGUF，CPU placement，F32 KV，4 threads。

8-bit 泳道刻意标记为 non-equivalent：

- ApxInf：自定义 G32/G64 对称 W8 混合 CPU/Metal 路径；attention、KV 和部分余项仍在 CPU F32，并执行 exact top-4 rerank。
- llama.cpp：pure Q8_0 GGUF，F16 KV；24 个 transformer layer endpoint 与 output head 在 `MTL0`，input embedding 保留在 CPU。

两边的 quantization、weight regime、KV precision 和 execution mechanism 均不相同。ApxInf 的线程策略也尚未建立与 llama.cpp 4-thread 设置的等价控制，所以当前数据只能归类为 diagnostic。

llama.cpp 运行器在测量结束后，用同一 context 对第 128 个 sampled token 额外 decode 一次。调度器完成回调证明：

- CPU 泳道：26/26 sentinel 全部在 CPU 完成。
- Metal 泳道：input embedding 1 个 sentinel 在 CPU，24 个 layer endpoint 加 output 共 25 个 sentinel 在 `MTL0` 完成。

该 proof decode 位于全部 token timing 和 perf snapshot 之后，单独计时，不进入本文性能值。

## 内存与资源不可直接横比

本文不计算任何跨 runtime 内存 ratio，也不下“更省内存”的结论：

- ApxInf 的 `799,543,312` bytes 是 resident MTLBuffer-only ledger。
- llama.cpp Q8_0 的内部 ledger 是 GPU `837,255,488` bytes 加 CPU `270,285,952` bytes；CPU 部分包含 input embedding fallback。
- `/usr/bin/time -l` 的 maximum RSS 是进程高水位，和上述内部 allocation ledger 不是同一口径。
- ApxInf W8 free-run 的 `5,905,612,800` bytes RSS 包含同进程 CPU oracle 的高水位，不是部署态 W8 footprint。

llama.cpp 的外部资源日志没有被原子写入 JSON receipt，也未作为可移植文件入库；summary 只登记其哈希和观测值。它们只用于诊断背景。

## 托管与可复核性

- ApxInf 测量代码提交：`820ee4ed98f66feaec0324e1a8870a7eb0967531`。
- ApxInf 门禁证据提交：`cb976735bbd373ed09f8e593af75c13236096f24`。
- llama.cpp 运行器与合同提交：[`27ab4e670b5a523af3f56540eb9c3369fd0e778a`](https://github.com/qhy991/ApxInf/commit/27ab4e670b5a523af3f56540eb9c3369fd0e778a)。
- llama.cpp 固定源码：[`f280b26983ad0fdb705a0d9ebf0503e76f2899b0`](https://github.com/ggml-org/llama.cpp/commit/f280b26983ad0fdb705a0d9ebf0503e76f2899b0)，tree `21045aed8b426d7a5e25a98e646054cbd9487e81`。
- [raw-token runner](../../benchmarks/llama_cpp/raw_token_runner.cpp) SHA-256：`76a5a354f729d22659387557ef368b75e83910e28a09d52876ddb366106c66e4`。
- runner binary SHA-256：`ccfa5ecd78119d4f8cdd8721e7faae360cb94b8334f9d61ed47e2e00290f2716`；一次 clean rebuild byte-identical。
- runner receipt 不自报 executable SHA-256；本文由外层 summary 绑定预期 binary 与 receipt 文件哈希，因此这不是同一 receipt 内的原子托管证明。
- runner 禁止动态 backend 扫描，llama/ggml/Metal backend 静态链接；可执行文件仍含系统实现带入的 `dlopen`/`dlsym` 符号，因此不声称这些符号不存在。
- GGUF SHA-256 由外层合同在运行后核验；runner receipt 本身绑定 pinned FD 的设备、inode、size、nlink 和 ctime，但未从该 FD 计算 SHA-256。
- 系统动态库/框架只做了路径闭包检查，没有逐个封存 loaded-image hash。

两份 llama.cpp receipt：

- CPU F32：[JSON](../../crates/apxinf-metal/evidence/llama-cpp/qwen35-0.8b-f280b269-f32-cpu-f32kv-raw13-free128-diagnostic-v2-20260825.json)，SHA-256 `4dda7e5c6364cedb349e0e611dd944abc5848691bcb34f3433946e2d129fee12`。
- Q8_0 Metal：[JSON](../../crates/apxinf-metal/evidence/llama-cpp/qwen35-0.8b-f280b269-q8_0-metal-raw13-free128-diagnostic-v2-20260825.json)，SHA-256 `5c78e89225f62afd1448e9d0ec92d7241b131877bc75ae2c3643bdc25debe31c`。

离线验证：

```sh
/usr/bin/python3 -I -B scripts/validate_qwen35_llamacpp_comparison_contract.py \
  --contract configs/qwen35-0.8b-llamacpp-comparison-v1.json
/usr/bin/python3 -I -B scripts/validate_qwen35_llamacpp_diagnostic_evidence.py \
  --summary crates/apxinf-metal/evidence/llama-cpp/qwen35-0.8b-apxinf-vs-llamacpp-raw13-free128-diagnostic-summary-v2-20260825.json
```

## Formal campaign 的剩余门槛

当前主机不安静，运行后仍有非 allowlisted 进程超过 5% CPU，system swap 非零；每个实现/泳道只有一次观测，也没有 warmup 或 ABBA/BAAB 调度。

正式比较需要先完成：线程策略显式对齐、quiet-host 与 zero/invariant-swap gate、每实现 3 次不计时 warmup、6 个 ABBA/BAAB block、每实现每泳道 12 个 fresh-process timed samples、runner/model SHA-256 原子绑定、loaded system-library image hashes，以及 llama.cpp teacher-forced quality receipt。任一条件不满足都必须 fail closed，不发布部分 formal 结论。
