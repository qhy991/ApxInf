import fs from 'node:fs/promises'
import crypto from 'node:crypto'
import path from 'node:path'
import process from 'node:process'
import {fileURLToPath, pathToFileURL} from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const projectRoot = path.resolve(here, '../../../..')
const workflowPath = path.resolve(here, '../apxinf-metal-w8-head-optimization.js')
const kersorRoot = path.resolve(process.argv[2])
const missingModel = process.argv.includes('--missing-model')
const relativeModel = process.argv.includes('--relative-model')
const sameBytesRetry = process.argv.includes('--same-bytes-retry')
const replacementRequired = process.argv.includes('--replacement-required')
const tamperGate = process.argv.includes('--tamper-gate')
const tamperCommand = process.argv.includes('--tamper-command')
const tamperArtifact = process.argv.includes('--tamper-artifact')
const tamperBuild = process.argv.includes('--tamper-build')
const tamperBlock = process.argv.includes('--tamper-block')
const missingField = process.argv.includes('--missing-field')
const tamperSourceEnd = process.argv.includes('--tamper-source-end')
const tamperModelEnd = process.argv.includes('--tamper-model-end')
const tamperToolchainEnd = process.argv.includes('--tamper-toolchain-end')
const nonCommandV1 = process.argv.includes('--non-command-v1')
const evaluatorTimeout = process.argv.includes('--evaluator-timeout')
const streamViolation = process.argv.includes('--stream-violation')
const tamperCandidateHash = process.argv.includes('--tamper-candidate-hash')
const {runWorkflow} = await import(
  pathToFileURL(path.join(kersorRoot, 'runtime/workflow-host.mjs')).href
)

const candidateSource = [
  '#include <metal_stdlib>',
  'using namespace metal;',
  'kernel void w8_rows_topk4() {}',
  'kernel void w8_final_topk4() {}',
  '',
].join('\n')
const candidateSourceSha256 = crypto.createHash('sha256').update(candidateSource).digest('hex')
const sha256 = (value) => crypto.createHash('sha256').update(value).digest('hex')
const evaluatorSha256 = sha256(await fs.readFile(path.join(projectRoot, 'kersor/workflows/ApxMetal/host_evaluator.py')))
const contractSha256 = sha256(await fs.readFile(path.join(projectRoot, 'kersor/workflows/ApxMetal/host_contract.json')))
const commandEvidence = (label) => {
  const argv = ['/usr/bin/true', label]
  return {
    argv,
    argv_sha256: sha256(JSON.stringify(argv)),
    exit_code: 0,
    timed_out: false,
    overall_deadline_exhausted: false,
    stdout_size_bytes: 0,
    stdout_sha256: sha256(''),
    stdout_tail: '',
    stderr_size_bytes: 0,
    stderr_sha256: sha256(''),
    stderr_tail: '',
  }
}
const baselineShaderSha256 = sha256('baseline shader')
const baselineTreeSha256 = sha256('baseline tree')
const candidateTreeSha256 = sha256('candidate tree')
const trajectorySha256 = sha256('trajectory')
const cargoSha256 = sha256('cargo')
const rustcSha256 = sha256('rustc')
const sourcePair = {
  baseline: {file_count: 100, manifest_sha256: baselineTreeSha256},
  candidate: {file_count: 100, manifest_sha256: candidateTreeSha256},
}
const modelManifest = {
  identity: {
    architectures: ['Qwen3_5ForConditionalGeneration'],
    full_attention_interval: 4,
    head_dim: 256,
    hidden_size: 1024,
    intermediate_size: 3584,
    model_type: 'qwen3_5',
    num_attention_heads: 8,
    num_hidden_layers: 24,
    num_key_value_heads: 2,
    text_model_type: 'qwen3_5_text',
    tie_word_embeddings: true,
    vocab_size: 248320,
  },
  file_count: 3,
  files: [
    {name: 'config.json', size_bytes: 100, sha256: sha256('config')},
    {name: 'model.safetensors', size_bytes: 1000, sha256: sha256('weights')},
    {name: 'tokenizer.json', size_bytes: 200, sha256: sha256('tokenizer')},
  ],
  manifest_sha256: sha256('model manifest'),
}
const gateNames = [
  'metal_adversarial_tests',
  'qwen35_tests',
  'teacher_forced_native_f32_128',
  'trajectory_exact_100',
  'execution_path_hit_and_negative_control',
]
const gates = gateNames.map((name) => ({
  name,
  passed: true,
  ...commandEvidence(name),
}))
gates[2].teacher_receipt_sha256 = sha256('teacher receipt')
gates[2].native_f32_rerank_matches = 128
gates[2].production_prefill_decode_tokens = 10
gates[3] = {
  name: gateNames[3],
  passed: true,
  generated_ids_sha256: trajectorySha256,
  tokens: 100,
  baseline_command: commandEvidence('baseline-trajectory'),
  candidate_command: commandEvidence('candidate-trajectory'),
}
const pathEvidence = {
  candidate_binary_sha256: sha256('candidate binary'),
  candidate_shader_sha256: candidateSourceSha256,
  exact_shader_bytes_in_binary: true,
  negative_control_build_flag: false,
  positive_build_flag: true,
  one_token_id: 9419,
  negative_command: commandEvidence('negative-control'),
  positive_command: commandEvidence('positive-control'),
}
gates[4] = {name: gateNames[4], passed: true, ...pathEvidence}
const orders = ['ABBA', 'BAAB', 'ABBA', 'BAAB', 'ABBA', 'BAAB']
const blocks = orders.map((order, index) => ({
  index,
  order,
  quiet_host: {
    passed: true,
    logical_cpus: 10,
    load_1m: 1.0,
    maximum_load_1m: 5.0,
    maximum_external_process_cpu_percent: 25.0,
    offenders: [],
  },
  system_swap_used_before_bytes: 1024,
  system_swap_used_after_bytes: 1024,
  system_swap_growth_bytes: 0,
  samples: [...order].map((variant, sampleIndex) => ({
    variant,
    generation_tps: variant === 'A' ? 100.0 : 112.0,
    ttft_ms: 20.0,
    max_rss_bytes: 1_000_000_000,
    process_swaps: 0,
    generated_ids_sha256: trajectorySha256,
    command: commandEvidence(`block-${index}-sample-${sampleIndex}`),
  })),
}))

class FixtureBroker {
  constructor() {
    this.requests = []
  }

  async execute(request) {
    this.requests.push(request)
    return {
      output: {
        strategy_id: 'fixture',
        hypothesis: 'fixture only; no real Agent or benchmark',
        candidate_source: candidateSource,
      },
      thread_id: 'fixture-thread',
      usage: {input_tokens: 1, output_tokens: 1, total_tokens: 2},
    }
  }
}

const broker = new FixtureBroker()
const evaluatorRequests = []
const deterministicEvaluator = async (request) => {
  evaluatorRequests.push(request)
  const body = JSON.parse(request.argv[5])
  const receipt = {
    format: 'apxinf-kersor-metal-w8-host-evaluation-v1',
    schema_version: 1,
    status: replacementRequired ? 'replacement_required' : 'accepted',
    accepted: !replacementRequired,
    strategy_id: body.strategy_id,
    candidate_shader_sha256: candidateSourceSha256,
    candidate_scope: ['crates/apxinf-metal/src/metal_w8.metal'],
    platform: {os: 'macos', arch: 'arm64', soc: 'Apple M4', python: '3.9.6'},
    snapshot: {
      tree_differences: ['crates/apxinf-metal/src/metal_w8.metal'],
      baseline_tree_sha256: baselineTreeSha256,
      candidate_tree_sha256: candidateTreeSha256,
      baseline_shader_sha256: baselineShaderSha256,
      candidate_shader_sha256: candidateSourceSha256,
    },
    toolchain: {cargo_sha256: cargoSha256, rustc_sha256: rustcSha256, offline: true},
    custody: {
      protocol: 'command-v1',
      overall_deadline_seconds: 1650,
      workflow_artifacts: {
        host_contract_sha256: contractSha256,
        host_evaluator_sha256: evaluatorSha256,
      },
      model: {start: modelManifest, end: modelManifest, unchanged: true},
      sources: {
        start: sourcePair,
        after_gates: sourcePair,
        end: sourcePair,
        unchanged: true,
      },
      toolchain: {
        start: {cargo_sha256: cargoSha256, rustc_sha256: rustcSha256},
        end: {cargo_sha256: cargoSha256, rustc_sha256: rustcSha256},
        unchanged: true,
      },
    },
    builds: [
      {
        variant: 'A',
        binary_sha256: sha256('baseline binary'),
        shader_sha256: baselineShaderSha256,
        command: commandEvidence('release-build-A'),
      },
      {
        variant: 'B',
        binary_sha256: pathEvidence.candidate_binary_sha256,
        shader_sha256: candidateSourceSha256,
        command: commandEvidence('release-build-B'),
      },
    ],
    problems: replacementRequired ? ['formal schedule observed system swap growth'] : [],
    correctness: {executed: true, passed: true, gates},
    execution_path: {passed: true, evidence: pathEvidence},
    formal_benchmark: {
      executed: true,
      accepted: !replacementRequired,
      sample_count: 24,
      block_orders: orders,
      minimum_speedup: 1.10,
      same_direction_blocks_required: 6,
      replacement_required: replacementRequired,
      preserved_blocks: blocks,
      generation_tps_speedup: 1.12,
      ttft_ratio: 1.0,
      rss_delta_bytes: 0,
      system_swap_used_start_bytes: 1024,
      system_swap_used_end_bytes: replacementRequired ? 5120 : 1024,
      system_swap_growth_bytes: replacementRequired ? 4096 : 0,
      problems: [],
      contamination: replacementRequired ? ['formal schedule observed system swap growth'] : [],
      same_direction_blocks: 6,
    },
    quality_claim: 'native_f32_only',
    claims_hf_bf16_parity: false,
  }
  if (tamperGate) receipt.correctness.gates[2].passed = false
  if (tamperArtifact) receipt.custody.workflow_artifacts.host_evaluator_sha256 = '0'.repeat(64)
  if (tamperBuild) receipt.builds[1].shader_sha256 = '0'.repeat(64)
  if (tamperBlock) receipt.formal_benchmark.preserved_blocks[0].order = 'BAAB'
  if (missingField) delete receipt.custody.model
  if (tamperSourceEnd) {
    receipt.custody.sources.end = {
      baseline: sourcePair.baseline,
      candidate: {file_count: 100, manifest_sha256: '0'.repeat(64)},
    }
  }
  if (tamperModelEnd) {
    receipt.custody.model.end = {...modelManifest, manifest_sha256: '0'.repeat(64)}
  }
  if (tamperToolchainEnd) receipt.custody.toolchain.end.rustc_sha256 = '0'.repeat(64)
  if (streamViolation) receipt.correctness.gates[0].stdout_size_bytes = 2097153
  if (tamperCandidateHash) receipt.candidate_shader_sha256 = '0'.repeat(64)
  return {
    protocol: nonCommandV1 ? 'agent-shell' : 'command-v1',
    passed: !replacementRequired,
    exit_code: replacementRequired ? 1 : 0,
    timed_out: evaluatorTimeout,
    filesystem_enforcement: {
      policy: tamperCommand ? 'workspace-write' : 'read-only',
      network: 'denied',
      temporary_writes: 'private_ephemeral',
    },
    stdout_json: receipt,
  }
}

const source = await fs.readFile(workflowPath, 'utf8')
const args = {
  kernel_path: path.join(projectRoot, 'crates/apxinf-metal/src/metal_w8.metal'),
  model_path: missingModel ? '' : (relativeModel ? 'fixture/qwen35' : '/private/fixture/qwen35'),
  turn_timeout_min: 30,
  retry_candidate_source: sameBytesRetry ? candidateSource : '',
  retry_candidate_sha256: sameBytesRetry ? candidateSourceSha256 : '',
  retry_strategy_id: sameBytesRetry ? 'fixture' : '',
}
const run = await runWorkflow({
  source,
  scriptPath: workflowPath,
  args,
  broker,
  projectRoot,
  runDir: projectRoot,
  totalTokens: 100,
  turnTimeoutSeconds: 1800,
  deterministicEvaluator,
})
const request = evaluatorRequests[0] ?? null
process.stdout.write(JSON.stringify({
  run_status: run.status,
  workflow_status: run.result.status,
  workflow_ok: run.result.ok,
  agent_calls: broker.requests.length,
  agent_transaction: broker.requests[0]?.options?.transaction ?? null,
  evaluator_calls: evaluatorRequests.length,
  filesystem_policy: request?.filesystem_policy ?? null,
  network_policy: request?.network_policy ?? null,
  evaluator_argv_prefix: request?.argv?.slice(0, 5) ?? null,
  evaluator_cwd: request?.cwd ?? null,
  retry_used: run.result.retry_used ?? false,
  request_candidate_source_sha256: request ? JSON.parse(request.argv[5]).candidate_source_sha256 ?? null : null,
  expected_candidate_source_sha256: candidateSourceSha256,
  retry_candidate_source: run.result.retry_candidate_source ?? null,
  retry_candidate_sha256: run.result.retry_candidate_sha256 ?? null,
  best_kernel_code: run.result.best_kernel_code ?? null,
  expected_source: candidateSource,
}) + '\n')
