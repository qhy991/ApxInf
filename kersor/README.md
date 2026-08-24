# KerSor：Hugging Face -> macOS onboarding

这里提供的是模型接入的 KerSor 控制合同，不是新的推理 backend。

## 当前可运行入口

`scripts/resolve_hf_source.py` 先把 Hugging Face URL 解析为固定 commit 的
metadata-only source lock；`scripts/prepare_hf_macos_intake.py` 再把该 lock
编译为只读 `kersor-mission-v1`、Session v2 和 KerSor runtime hash lock。

```bash
python3 scripts/resolve_hf_source.py \
  https://huggingface.co/Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --output .apxinf/onboarding/qwen35-0.8b/source-lock.json

python3 scripts/prepare_hf_macos_intake.py \
  https://huggingface.co/Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --source-lock .apxinf/onboarding/qwen35-0.8b/source-lock.json \
  --kersor-root /Users/haiyan-mini/Agent4Kernel/kersor
```

脚本会打印：

- `MISSION` 与合同 SHA-256；
- 固定的 runtime config SHA-256；
- `KERSOR_RUNTIME_LOCK` 及当前已绑定 KerSor 主闭包的自校验 hash；
- 只做重验、不启动 Agent 的 `RUN_DRY` 命令；
- 新 run 的 `FORMAL_ADMIT` 与既有 frozen admission 的 `FORMAL_RESUME`；
- 运行结束后的 `VERIFY` 命令。

不要把 HF token 作为参数传给脚本或写入 Mission。Intake 默认只读元数据，不下载完整权重，也不执行仓库 remote code。

runtime lock v2 是正式入口的 fail-closed 边界。它用 `O_NOFOLLOW` 打开并通过
fd 哈希所有绑定文件，记录 path/hash/size/mode/dev/ino/uid/gid/nlink；绑定内容
包括：

- ApxInf 的 lock、launcher、prepare；Mission、严格只读 runtime config；
- source-lock 的 raw bytes 与 semantic content identity；Session config/state；
- 当前启动 launcher 的 Host Python、其全部非系统 Mach-O dylib、每个 evaluator
  的完整 request/argv、本地传递 Python scripts，以及 argv 中已存在的 ApxInf
  直接普通输入文件（包括固定 deployment profile）；
- KerSor admission 时始终在线的代码，以及新 admission 会复制的 executor source；
- Node 可执行文件及全部非系统 Mach-O dylib、Codex native payload，以及路由所需
  的外部命令。

重验只做 fd hash/stat 和纯解析，不会重新运行 `codex --version`、`otool`、Git、
Host evaluator 或 Agent。旧 v1 lock 无论自校验是否匹配都不能正式运行。

Mission 的 `workspace` 必须逐字等于 canonical ApxInf root；Session
`session-config.json.task_dir` 必须等于该 workspace，`state.json.session_id`
必须等于 Mission `mission_id`。Host evaluator 只接受精确
`command-v1/read-only/denied/sealed` request，`materialize` 必须缺省或为空，且
`cwd` 必须缺省或为 `.`，execution 必须不可重试；其 Python argv 固定为
`[当前解释器, "-S", "-B", script, ...]`。所有 Agent capability 均不得声明
`side_effect=write`、`transaction_artifacts`、`commit_failed_outputs` 或
`candidate_verifier`。这些约束把本阶段保持为真正的 read-only Intake，而不是
预先授权未来的实现事务。

Host Python lock 覆盖解释器文件和 `otool -L` 可见的全部非系统 Mach-O 依赖；
macOS 系统 dylib 和 Python 标准库树仍属于显式 platform TCB，并没有被描述为
完整的 Python 文件系统闭包。`-S -B` 禁止加载 site 初始化并禁止生成 bytecode，
但不能把 platform TCB 变成零。named profile 内允许 Agent 调用的最小系统工具
同样属于该 TCB；lock 逐文件绑定 launcher 路由所需工具，但不声称枚举了 Agent
可能选择的每个系统可执行文件。

新任务第一步必须显式使用 `--admit-only`。launcher 只创建并核对
`binding.json`、Mission/controller/runtime/Session snapshot 和完整
`executor-runtime`，回显 admission receipt 后停止，绝不在同一次调用中启动
Agent。用户检查 frozen admission 后，再以独立的 `--resume` 调用启动；resume
仍要求 admission pristine。绝不根据目录存在与否自动猜测，旧 `--fresh` 参数会
直接拒绝。

Codex 凭据只做 metadata 检查，launcher 和 runtime lock 都不会读取或 hash
`auth.json` 内容。`auth.json` 必须由当前 uid 拥有、mode `0600`、nlink=1，
auth home 必须是当前 uid 的直接 mode `0700` 目录，父目录不能是软链接或可被
group/other 写入。比如当前 home 尚不是 0700 时，先由用户自行修正：

```bash
chmod 700 "$HOME/.codex"
chmod 600 "$HOME/.codex/auth.json"
```

KerSor broker 还必须暴露
`codex-named-permissions-auth-read-deny-v2` 标记，并用 Codex named permission
profile 同时拒绝 generated command 读取物理凭据路径和 activation link；缺少
标记时 v2 lock 创建即失败。auth custody 下 runtime `extra_args` 必须严格为空，
worker 的固定空继承环境由外层 launcher 提供。
同一 broker 还必须精确导出
`CODEX_COMMAND_READ_SCOPE_MECHANISM=codex-minimal-project-read-v1`；lock 和 dry-run
receipt 都会回显该机制。该 named profile 只保留最小系统工具和 ApxInf project
读取，不把整个用户目录、相邻 checkout 或模型 cache 暴露给 generated command。

frozen admission 的顶层和 `executor-runtime` inventory 必须精确匹配合同；run、
executor 和后续 execution-evidence 目录必须是直接目录，绑定 JSON/脚本必须是
直接普通文件，任意软链接、路径别名、额外文件或额外目录都会拒绝 resume。
这些 hash/stat 是每个验证点的 fail-closed 快照；它们不宣称消除了同一 uid 在
验证完成后、实际执行前并发替换可变路径的所有 TOCTOU。正式运行仍要求没有同一
uid 的并发 writer，并在 admission、resume 前后重复验证 lock 与 frozen closure。

## 已通过的固定 URL 部署入口

Qwen3.5-0.8B 已可用 Rust 入口完成来源、可恢复 staging、完整文件哈希、
Accelerate build 身份、固定 token 轨迹、受限执行和内存验收。Python
controller 路径仍保留用于开发和审计；面向用户的命令是：

```bash
target/release/apxinf onboard \
  https://huggingface.co/Qwen/Qwen3.5-0.8B \
  --controller /Users/haiyan-mini/Agent4Kernel/ApxInf/scripts/onboard_hf_macos.py \
  --python /opt/homebrew/Cellar/python@3.14/3.14.3/Frameworks/Python.framework/Versions/3.14/bin/python3.14 \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --source-lock .apxinf/onboarding/qwen35-0.8b/source-lock.json \
  --model-dir /Users/haiyan-mini/Agent4Kernel/.apxinf-models/Qwen3.5-0.8B-2fc063647-staged \
  --oracle-dir /Users/haiyan-mini/Agent4Kernel/.apxinf-oracles/qwen35-0.8b \
  --binary target/release/apxinf \
  --receipt-output .apxinf/deployments/qwen35-0.8b-macos-cpu/generation-receipt-reused-offline-v4.json \
  --lock-output .apxinf/deployments/qwen35-0.8b-macos-cpu/deployment-lock-reused-offline-v4.json \
  --offline
```

若目标 bundle 不存在，去掉 `--offline` 并显式加入 `--download-missing`。stager
只接受上述 canonical URL、固定 commit、SafeTensors allowlist 与总字节上限；
支持严格 HTTP Range 续传、坏断点受控重试和 macOS 原子 no-replace 发布。
已有 bundle 一律经内部 `--existing-only` 路径全文件复验，目标在竞态中消失时
只会失败，不会转成联网。这个 tracer 不启动 Agent；未知模型的架构分析和端口
开发仍走后续两阶段 KerSor 流程。

## 为什么 Intake 与实现分开

KerSor 的候选事务必须在 worker 启动前声明精确 `transaction_artifacts`。任意 HF 模型所需的文件只有在完成架构分析后才能确定，所以安全流程是：

1. read-only Intake Mission 产出 `port_manifest`；
2. 确定性 controller 校验 manifest 中的每条路径；
3. controller 生成绑定 write runtime 的实现 Mission；
4. 每个 mutation capability 绑定不可重试的 Host candidate verifier；
5. build/oracle/memory/offline smoke 未全部通过时回滚候选。

现有目录不能直接作为 transaction artifact，必须逐文件列出；一个尚不存在的新目录路径可以整体声明，并在失败时由 Host 移除。`.git`、`.kersor`、cache、权重、凭据以及 Host run directory 永远不能进入候选路径。

Qwen3.5-0.8B 已走通确定性的 source lock、真实 checkpoint/oracle、Accelerate
CLI smoke 和 deployment lock。未知模型的 Agent Intake 仍停在第 1 步，因为
实现 Mission 的精确路径必须来自真实 Intake 结果，不能用一个预先写死的
宽权限模板代替。

## KerSor 根目录规则

一次 job 只能选择一个 KerSor root。生成器按以下顺序解析：

1. `--kersor-root`；
2. ApxInf 同级的 `../kersor`。

`KERSOR_ROOT` 及其他 ambient `KERSOR_*` 不会成为隐藏输入。正式 launcher 从
空 allowlist 构造环境，固定 `KERSOR_NODE_BIN`、`KERSOR_CODEX_COMMAND`、Host
Python 和 autonomous runner，并为每次运行创建空的私有
`PYTHONPYCACHEPREFIX`；`NODE_OPTIONS`、`PYTHONSTARTUP`、`BASH_ENV`、`DYLD_*`、
token 和云凭据等 ambient 注入不会传给 KerSor。

Mission 和 runtime config 都会绑定 SHA-256。不要在同一次 run 中混用源码 checkout、Codex plugin cache 或 Claude plugin cache；它们可能具有不同的 runtime contract。

## 当前完成边界

Intake Mission 完成只表示：

- 来源、架构、安全和资源分析产物齐全；
- 提出了下一阶段的精确路径和验证门；
- 未下载权重、未执行 remote code、未修改 ApxInf。

它不表示模型已经部署，也不产生 `SUPPORTED` 标记。完整验收合同和 macOS 路线见 `doc/20260823-macos-hf-agent-onboarding/design.md`。
