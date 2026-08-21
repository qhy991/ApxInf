# RTX 4090 GPU queue rules

- Route every command that creates a CUDA context through the machine-wide
  `gpu-run` client. CPU-only source inspection, compilation, and documentation
  do not require the broker.
- Always provide `--owner codex-apxinf`, a short `--label`,
  `--mode shared|exclusive`, `--gpu-count 1`, realistic queue/run timeouts, and
  an estimate. When submitting from the remote root account, also pass
  `--cwd` under `/mnt/user_dir/hanjinchen` or `/var/lib/agent-gpu-broker`;
  broker jobs run as the unprivileged `gpuq` user and cannot enter `/root`.
- Use `shared` only for bounded correctness checks that are known to fit beside
  the resident service. Use `exclusive` for model loading, OOM/capacity probes,
  benchmarks, NCU/Nsight, and deployment acceptance.
- Never set `CUDA_VISIBLE_DEVICES`; the broker owns physical assignment.
- Never bypass an unavailable or queued broker. Inspect with `gpuq status` and
  keep the original `gpu-run` client alive while queued.
- The Qwen3.8 reference service is owned by
  `apxinf-qwen38-broker.service`. Stop it only through systemd when an authorized
  exclusive Omni phase needs the card, and restore it after a failed or
  non-promoted Omni run.
