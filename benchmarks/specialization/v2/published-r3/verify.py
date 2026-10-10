#!/usr/bin/env python3
"""Check the published measurement subset without a model or GPU."""

import json
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9), (actual, expected)


def stats(values, expected):
    assert len(values) == expected['count']
    close(statistics.median(values), expected['median'])
    close(min(values), expected['min'])
    close(max(values), expected['max'])


def path_identity(receipt):
    assert receipt['metal_w8_lm_head'] is True
    assert receipt['metal_w8_mlp_block'] is True
    assert [layer['layer_index'] for layer in receipt['mlp_block_layers']] == list(range(24))


def main():
    summary = read('summary.json')
    builds = read('builds.json')['builds']
    processes = read('processes.json')
    requests = [json.loads(line) for line in (ROOT / 'requests.jsonl').read_text().splitlines()]
    assert summary['accepted'] is True
    assert summary['formal_performance_accepted'] is False
    assert [b['variant'] for b in builds] == ['full', 'reference', 'minimal', 'minimal', 'reference', 'full']
    for variant, expected in summary['build'].items():
        rows = [b for b in builds if b['variant'] == variant]
        for phase, field in [('clean', 'clean_seconds'), ('noop', 'noop_seconds'), ('incremental', 'incremental_seconds')]:
            assert all(row[phase]['returncode'] == 0 for row in rows)
            stats([row[phase]['wall_seconds'] for row in rows], expected[field])
        stats([row['binary']['bytes'] for row in rows], expected['binary_bytes'])
    assert len(requests) == summary['resident_requests'] == 80
    assert sum(not r['warmup'] for r in requests) == summary['measured_requests'] == 48
    resident = processes['resident']
    assert [p['variant'] for p in resident] == ['reference', 'minimal', 'minimal', 'reference']
    assert set(r['process_id'] for r in requests) == set(p['process_id'] for p in resident)
    cases = list(summary['references'])
    swap_pages = 0
    for process in resident:
        assert process['returncode'] == 0
        rows = [r for r in requests if r['process_id'] == process['process_id']]
        assert [(r['round'], r['case']) for r in rows] == [(i, c) for i in range(5) for c in cases]
        assert all(r['warmup'] == (r['round'] < 2) for r in rows)
        decode_calls = 0
        for index, row in enumerate(rows, 1):
            assert row['variant'] == process['variant']
            result = row['result']
            reference = summary['references'][row['case']]
            for key in ['prompt_token_ids', 'generated_token_ids', 'text', 'stop_reason']:
                assert result[key] == reference[key], (process['process_id'], row['case'], key)
            receipt = result['generation_path_receipt']
            path_identity(receipt)
            decode_count = len(result['generated_token_ids']) - 1
            decode_calls += decode_count
            assert receipt['lm_head']['prefill_calls'] == index
            assert receipt['lm_head']['decode_calls'] == decode_calls
            assert all(layer['decode_calls'] == decode_calls for layer in receipt['mlp_block_layers'])
            timing = result['timing']
            close(timing['decode_tps'], decode_count * 1000 / timing['decode_ms'])
        before = process['environment_before']['vm']['stdout']
        after = process['environment_after']['vm']['stdout']
        assert 'page size of 16384 bytes' in before
        swap_pages += int(re.search(r'Swapins:\s+(\d+)', after)[1]) - int(re.search(r'Swapins:\s+(\d+)', before)[1])
    assert swap_pages == 5333
    for variant in ['reference', 'minimal']:
        for case in cases:
            selected = [r['result']['timing'] for r in requests if r['variant'] == variant and r['case'] == case and not r['warmup']]
            for metric, expected in summary['runtime'][variant]['cases'][case].items():
                stats([r[metric] for r in selected], expected)
        for metric, expected in summary['runtime'][variant]['startup'].items():
            stats([p['ready']['startup'][metric] for p in resident if p['variant'] == variant], expected)
    assert [p['case'] for p in processes['cli']] == cases
    for process in processes['cli']:
        result = process['receipt']
        reference = summary['references'][process['case']]
        assert process['returncode'] == 0
        assert result['generated_token_ids'] == reference['generated_token_ids']
        assert result['prompt_token_count'] == len(reference['prompt_token_ids'])
        path_identity(result['generation_path'])
    plain = processes['plain']
    single = processes['single_json']
    assert plain['returncode'] == single['returncode'] == 0
    assert plain['text'] == summary['references'][plain['case']]['text'] + '\n'
    assert single['result']['text'] == summary['references'][single['case']]['text']
    for build in builds:
        scope = build['scope']
        expected_bridges = 2 if build['variant'] == 'minimal' else 10
        assert len(scope['native']['.a']) == len(scope['native']['.o']) == expected_bridges
        if build['variant'] == 'minimal':
            assert scope['model_features'] == ['accelerate', 'metal-w8-head-mlp', 'model-qwen35']
            assert not any(scope['registry_modules'].values())
            assert scope['experimental_metal_modules'] == []
            assert not scope['gguf']
            assert not any(scope['dependency_presence'].values())
    print('PASS: 6 builds, 80 resident requests, 48 measured requests, 4 CLI checks, JSON/plain parity, scope, and 5333 swap-in pages')
    print('Historical evidence is internally consistent. Stable inference speedup remains unproven.')


if __name__ == '__main__':
    main()
