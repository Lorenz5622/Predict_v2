#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launcher for running Qwen stage1 then stage2 with two separate torchrun invocations.

Example:
  python launch_two_stage_torchrun_qwen.py --nproc_per_node 2 -- --config configs/qw_xxx.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


TORCHRUN_KNOWN_ARGS = {
    "nproc_per_node",
    "nnodes",
    "node_rank",
    "master_addr",
    "master_port",
    "rdzv_backend",
    "rdzv_endpoint",
    "rdzv_id",
    "max_restarts",
    "monitor_interval",
    "role",
    "tee",
    "log_dir",
    "standalone",
}


def _build_torchrun_args(ns: argparse.Namespace) -> List[str]:
    args: List[str] = []
    for key, value in vars(ns).items():
        if key not in TORCHRUN_KNOWN_ARGS:
            continue
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend([flag, str(value)])
    return args


def _extract_config_path(forwarded: List[str]) -> Path:
    for i, token in enumerate(forwarded):
        if token == "--config":
            if i + 1 >= len(forwarded):
                raise SystemExit("--config was provided without a path")
            return Path(forwarded[i + 1]).expanduser()
    raise SystemExit("Missing required forwarded arg: --config <path>")


def _mapping_to_argv(mapping: Dict[str, Any], *, output_dir: str) -> List[str]:
    argv: List[str] = []
    for key, value in mapping.items():
        if isinstance(value, str):
            value = value.format(output_dir=output_dir)
        flag = key if key.startswith("--") else f"--{key}"
        argv.extend([flag, str(value)])
    return argv


def _load_launch_args(config_path: Path) -> tuple[List[str], List[str], str]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = str(cfg["run"]["output_dir"])
    common_args = _mapping_to_argv(cfg.get("common_args", {}), output_dir=output_dir)
    stage1_args = _mapping_to_argv(cfg.get("stage1_args", {}), output_dir=output_dir)
    stage2_args = _mapping_to_argv(cfg.get("stage2_args", {}), output_dir=output_dir)
    return common_args + stage1_args, common_args + stage2_args, output_dir


def _run(cmd: List[str], env: dict) -> None:
    print("\n[launcher] running:\n  " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument("--nproc_per_node", type=int, default=None)
    parser.add_argument("--nnodes", type=int, default=None)
    parser.add_argument("--node_rank", type=int, default=None)
    parser.add_argument("--master_addr", type=str, default=None)
    parser.add_argument("--master_port", type=int, default=None)
    parser.add_argument("--rdzv_backend", type=str, default=None)
    parser.add_argument("--rdzv_endpoint", type=str, default=None)
    parser.add_argument("--rdzv_id", type=str, default=None)
    parser.add_argument("--standalone", action="store_true", default=False)
    parser.add_argument("--max_restarts", type=int, default=None)
    parser.add_argument("--monitor_interval", type=float, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--tee", type=str, default=None)
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Args after `--` will be forwarded to this launcher; must include --config <path>.",
    )

    ns = parser.parse_args()

    if not ns.train_args or ns.train_args[0] != "--":
        print(
            "ERROR: You must pass launcher args after `--`.\n"
            "Example:\n"
            "  python launch_two_stage_torchrun_qwen.py --nproc_per_node 2 -- --config configs/qw_xxx.json\n",
            file=sys.stderr,
        )
        raise SystemExit(2)

    forwarded = ns.train_args[1:]
    config_path = _extract_config_path(forwarded)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    stage1_args, stage2_args, output_dir = _load_launch_args(config_path)
    torchrun_args = _build_torchrun_args(ns)

    env = os.environ.copy()
    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8",
    )

    repo_root = Path(__file__).resolve().parent
    torchrun = [sys.executable, "-m", "torch.distributed.run"]

    cmd1 = torchrun + torchrun_args + [str(repo_root / "run_stage1_qwen_dynamic_moe.py")] + stage1_args
    _run(cmd1, env)

    if not any(arg == "--stage2_init_path" for arg in stage2_args):
        stage2_args = stage2_args + ["--stage2_init_path", f"{output_dir}/ckpt_after_stage1"]

    cmd2 = torchrun + torchrun_args + [str(repo_root / "run_stage2_qwen_dynamic_moe.py")] + stage2_args
    _run(cmd2, env)

    print("\n[launcher] done.", flush=True)


if __name__ == "__main__":
    main()
