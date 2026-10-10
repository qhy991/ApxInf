# PR 检出验证：2026-10-10

本次在基于 `df5c55a` 的独立 worktree 中验证，仅纳入模型专用化改动。
原工作区的 serving 文件、根 README 改动及 serving 锁文件变化没有进入此 PR。
最终 PR 的 CPU 检查与历史 r3 性能实验分别记录。

## 本次通过的检查

以下 Cargo 命令均使用 `--locked --offline`，macOS SDK 为已安装的 `MacOSX15.4.sdk`。

| 检查 | 结果 |
|---|---|
| `cargo test -p apxinf-qwen35` | 15 项单元测试和 2 项入口测试通过 |
| `cargo test -p apxinf-model --lib --no-default-features --features model-qwen35,accelerate,metal-w8-head-mlp -- --skip metal` | 23 项 CPU 测试通过，5 项 Metal 测试按命令排除 |
| `cargo test -p apxinf-tokenizer --lib --no-default-features` | 12 项测试通过 |
| `cargo test -p apxinf-model --test model_features` | 3 项注册与错误行为测试通过 |
| `cargo check -p apxinf --features accelerate,metal-w8` | 通用 CLI 与完整 Metal features 编译检查通过 |
| `cargo check -p apxinf-metal --no-default-features` | 空 Metal feature 组合通过 |
| `cargo build --release --jobs 4 -p apxinf-qwen35 --bin apxinf-qwen35-08b` | 专用 release 构建通过 |
| `python3 -m unittest discover -s benchmarks/specialization/v2 -p 'test_*.py'` | 6 项模型输入与共享锁检查通过 |
| `python3 benchmarks/specialization/v2/published-r3/verify.py` | 历史公开记录的统计、样本网格、输出和路径核对通过 |
| 新版 `compare_native.py prepare`，显式指定真实模型目录 | 六项固定资产 SHA-256 校验、源码快照及输入绑定通过 |

五个改动 crate 的 `cargo fmt -- --check` 和改动内容的空白检查通过。
入口 CONTRACT、入口 README、v2 README 的 STE 结构扫描为 0 项问题。
该扫描不包含完整词典审查，也不构成认证。

## 平台与依赖检查

代码审阅发现，专用 crate 无条件启用 Accelerate 会影响 Linux workspace 的链接。
最终 manifest 只在 macOS 启用 core/model 的 Accelerate features。
`cargo tree` 分别检查 Darwin 与 Linux 目标：Darwin 保留 Accelerate，Linux 不启用。
这里没有执行 Linux 交叉编译或运行测试。

PR 的 Cargo.lock 只增加本地 `apxinf-qwen35` 包，未增加第三方包。
最小构建的 94 项依赖包名与版本和历史 r3 对应依赖树完全一致。
上述 manifest 限定和新版复现脚本均在历史测量之后完成。

本次 release 二进制为 7,267,376 bytes，SHA-256 为
`80a76be2aac2677a4da3af993f18723412da5cdd70ddde3bfd5e8ba19d3178e1`。
该摘要与历史 r3 的 `3-minimal` 实测产物完全一致。
这说明本机重建得到相同的二进制内容；不表示重新完成了性能测量或证明任意环境都能逐字节重建。

## 本次没有完成的 GPU 检查

准备使用最终 release 二进制，对中文首都与英文乘法问题执行两个 JSONL 请求。
共享 Metal 测量锁在 60 秒内未释放，锁封装退出，模型进程没有启动。
没有绕过锁或干扰其他测量；这项检查记录为未执行，不能计为通过。

本 PR 的 GPU 行为证据来自[历史 r3 记录](published-r3/README.md)：80 个驻留请求、四次通用 CLI 对照及单次 JSON/文本检查。
没有用修改后的公开复现脚本重跑完整性能实验，也没有用本次 release 重新测量生成速度。
合并或发布前，可按 [README](README.md) 在测量锁可用时复测最终构建。
