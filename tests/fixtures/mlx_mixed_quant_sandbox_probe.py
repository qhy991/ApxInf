#!/usr/bin/env python3
"""Tiny subprocess probe for the MLX mixed-quant Seatbelt launcher tests."""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import socket
import sys
import time


def _attempt(label: str, operation) -> dict[str, object]:
    try:
        value = operation()
    except OSError as error:
        return {
            "label": label,
            "allowed": False,
            "errno": error.errno,
            "error": error.strerror,
        }
    return {"label": label, "allowed": True, "value": value}


def _connect() -> str:
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        connection.settimeout(0.25)
        connection.connect(("127.0.0.1", 9))
        return "connected"
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("probe", "sleep", "stdout", "stderr"))
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--policy-file", required=True)
    parser.add_argument("--scratch-file", required=True)
    parser.add_argument("--outside-read", required=True)
    parser.add_argument("--outside-write", required=True)
    arguments = parser.parse_args()

    if arguments.mode == "sleep":
        time.sleep(30)
        return 0
    if arguments.mode in {"stdout", "stderr"}:
        stream = sys.stdout.buffer if arguments.mode == "stdout" else sys.stderr.buffer
        stream.write(b"x" * (2 * 1024 * 1024))
        stream.flush()
        return 0

    result = {
        "format": "apxinf-mlx-mixed-sandbox-probe-v1",
        "source_read": _attempt(
            "source-read", lambda: Path(arguments.source_file).read_text()
        ),
        "policy_read": _attempt(
            "policy-read", lambda: Path(arguments.policy_file).read_text()
        ),
        "scratch_write": _attempt(
            "scratch-write",
            lambda: Path(arguments.scratch_file).write_text("scratch-ok"),
        ),
        "outside_read": _attempt(
            "outside-read", lambda: Path(arguments.outside_read).read_text()
        ),
        "outside_write": _attempt(
            "outside-write",
            lambda: Path(arguments.outside_write).write_text("forbidden"),
        ),
        "network": _attempt("network", _connect),
        "environment": dict(sorted(os.environ.items())),
        "denied_errnos": sorted(
            value for value in {errno.EPERM, errno.EACCES} if value is not None
        ),
    }
    sys.stdout.write(
        json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
