#!/usr/bin/env python3
"""Fetch only SafeTensors headers for the frozen Qwen3.8-Flash-Next source."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import struct
import sys
import urllib.parse
import urllib.request


DEFAULT_REPO = "Qwen/Qwen3.8-Flash-Next"
DEFAULT_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
MAX_HEADER_BYTES = 16 * 1024 * 1024
FLOAT_DTYPES = frozenset({"F32", "F16", "BF16"})
DERIVED_SUFFIXES = (
    ".ple_embedding.layer_multipliers",
    ".ple_embedding.ngram_heads_offsets",
    ".ple_embedding.ngram_heads_vocab_sizes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read Qwen4-Exp tensor shapes/dtypes with HTTP Range requests; "
            "no tensor payload is downloaded."
        )
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def resolve_url(repo: str, revision: str, path: str) -> str:
    return (
        "https://huggingface.co/"
        + urllib.parse.quote(repo, safe="/")
        + "/resolve/"
        + urllib.parse.quote(revision, safe="")
        + "/"
        + urllib.parse.quote(path, safe="")
    )


def request_bytes(url: str, byte_range: tuple[int, int] | None = None) -> tuple[bytes, object]:
    headers = {"User-Agent": "apxinf-qwen4-exp-header-audit/1"}
    if byte_range is not None:
        start, end = byte_range
        headers["Range"] = f"bytes={start}-{end}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
        if byte_range is not None:
            expected = byte_range[1] - byte_range[0] + 1
            if response.status != 206 or len(payload) != expected:
                raise RuntimeError(
                    f"server did not honor exact Range {byte_range}: "
                    f"status={response.status}, bytes={len(payload)}"
                )
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {byte_range[0]}-{byte_range[1]}/"):
                raise RuntimeError(f"invalid Content-Range: {content_range!r}")
        return payload, response.headers


def is_text_tensor(name: str) -> bool:
    return (name == "lm_head.weight" or name.startswith("model.language_model.")) and not name.endswith(
        DERIVED_SUFFIXES
    )


def fetch_shard_header(repo: str, revision: str, shard: str) -> dict[str, object]:
    url = resolve_url(repo, revision, shard)
    prefix, _ = request_bytes(url, (0, 7))
    header_size = struct.unpack("<Q", prefix)[0]
    if header_size == 0 or header_size > MAX_HEADER_BYTES:
        raise RuntimeError(f"{shard}: unsafe SafeTensors header size {header_size}")
    raw, _ = request_bytes(url, (8, 8 + header_size - 1))
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise RuntimeError(f"{shard}: SafeTensors header is not an object")
    return header


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.workers > 32:
        raise SystemExit("--workers must be in [1, 32]")
    index_url = resolve_url(args.repo, args.revision, "model.safetensors.index.json")
    index_raw, index_headers = request_bytes(index_url)
    index = json.loads(index_raw)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("SafeTensors index has no non-empty weight_map")
    observed_revision = index_headers.get("X-Repo-Commit")
    if observed_revision and observed_revision != args.revision:
        raise RuntimeError(
            f"resolved revision {observed_revision} differs from requested {args.revision}"
        )

    selected = {name: shard for name, shard in weight_map.items() if is_text_tensor(name)}
    shards = sorted(set(selected.values()))
    print(
        f"fetching {len(shards)} SafeTensors headers for {len(selected)} runtime tensors",
        file=sys.stderr,
    )
    headers_by_shard: dict[str, dict[str, object]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_shard_header, args.repo, args.revision, shard): shard
            for shard in shards
        }
        for future in concurrent.futures.as_completed(futures):
            shard = futures[future]
            headers_by_shard[shard] = future.result()

    metadata: dict[str, dict[str, object]] = {}
    for name, shard in selected.items():
        entry = headers_by_shard[shard].get(name)
        if not isinstance(entry, dict):
            raise RuntimeError(f"{name}: missing from assigned shard {shard}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        if dtype not in FLOAT_DTYPES:
            raise RuntimeError(f"{name}: unexpected runtime dtype {dtype!r}")
        if not isinstance(shape, list) or not all(type(value) is int and value >= 0 for value in shape):
            raise RuntimeError(f"{name}: invalid shape {shape!r}")
        metadata[name] = {"dtype": dtype, "shape": shape}

    json.dump(
        {
            "format": "apxinf-qwen4-exp-safetensors-header-manifest-v1",
            "repo": args.repo,
            "revision": args.revision,
            "index_tensor_count": len(weight_map),
            "runtime_tensor_count": len(metadata),
            "shard_count": len(shards),
            "metadata": metadata,
        },
        sys.stdout,
        sort_keys=True,
        separators=(",", ":"),
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
