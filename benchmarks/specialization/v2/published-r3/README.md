# r3 公开测量记录

这些文件来自本地实验 `model-specialization-v2-20261010-r3`，经过路径替换和结构整理。
它们不是原始报告的逐字节副本，也不是最终 PR commit 的重新测量。

原测量使用以 `df5c55a` 为基线的未提交源码快照，其中含同期 serving 文件及锁文件变化。
PR 排除了 serving 内容，只给基线锁文件增加本地 `apxinf-qwen35` 包。
在 macOS 上，PR 最小构建的 94 项依赖包名与版本和原实验完全一致。
PR 准备还把 Accelerate 限定在 macOS，并修订了复现脚本的模型路径和锁接口。
这些差异没有重新计入历史性能数据。
独立 PR 检出的最小 release 产物与 r3 二进制 SHA-256 一致，见[本次验证](../PR-VALIDATION.md)。

| 文件 | 内容 |
|---|---|
| [summary.json](summary.json) | 构建与运行统计，以及输出对照值 |
| [builds.json](builds.json) | 六次构建命令、时间、产物身份、特性及对象清单 |
| [requests.jsonl](requests.jsonl) | 全部 80 个驻留请求，保留输入、输出、计时和执行路径 |
| [processes.json](processes.json) | 四个驻留进程的启动及系统状态，四次通用 CLI、单次 JSON 和文本检查 |
| [hardware.json](hardware.json) | 原测量的硬件与系统信息 |
| [provenance.json](provenance.json) | 原报告哈希、输入绑定、替换方法与未公开范围 |
| `*-model.d`、`*-metal.d`、`*-loader.d`、`*-tokenizer.d` | 每种选择的代表性源码依赖记录 |
| `*-dependency-tree.txt`、`*-head-symbols.txt` | 代表性依赖树与 head 符号记录 |

`$RUN`、`$SNAPSHOT`、`$MODEL`、`$SOURCE`、`$CARGO_CACHE`、`$HOME` 代替本机路径前缀。
报告内原有 `sha256` 字段仍指向本地原件；路径替换后的公开文件不具有相同哈希。
原件哈希可以核对持有的归档，不能让未公开文件自动变得可下载。
模型、可执行文件、target 目录、完整源码归档和完整编译日志未随 PR 上传。

无需 GPU 或模型即可核对公开数据：

```sh
python3 benchmarks/specialization/v2/published-r3/verify.py
```

核对覆盖六次构建统计、80 个请求的完整网格、48 个测量值、输出一致性、执行路径和系统换页记录。
它检查公开记录内部是否一致，不能替代独立实机复测。
长回答本身含模型生成的事实错误，保留原文只为核对输出一致性；本实验没有证明模型回答质量。

`accepted=true` 只对应输入身份、编译范围、样本网格和固定语料的输出一致性。
`formal_performance_accepted=false`：存在全系统 swap-in，独立进程样本很少，不能据此认定稳定推理提速。
