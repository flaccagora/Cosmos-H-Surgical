#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Download the Hugging Face artifacts required by predict/examples/inference.py.

Run this on a machine with internet access, then copy the Hugging Face cache to
the offline machine and set HF_HUB_OFFLINE=1 or COSMOS_HF_NO_DOWNLOAD=1 before
running inference.

Usage:
    # Download the complete default inference dependency set:
    python hf_download.py --groups inference

    # Download everything listed by this script:
    python hf_download.py --all

    # List available groups:
    python hf_download.py --list

    # Custom cache root (default: $HF_HOME or ~/.cache/huggingface):
    python hf_download.py --groups inference --cache-dir /scratch/hf_cache

    # Dry-run (show what would be downloaded without downloading):
    python hf_download.py --groups inference --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import traceback
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SnapshotEntry:
    """Download a repository snapshot via snapshot_download()."""

    repo_id: str
    revision: str = "main"
    ignore_patterns: list[str] = field(default_factory=list)
    allow_patterns: list[str] = field(default_factory=list)
    description: str = ""
    gated: bool = False


@dataclass(frozen=True)
class FileEntry:
    """Download one file via hf_hub_download()."""

    repo_id: str
    filename: str
    revision: str = "main"
    description: str = ""
    gated: bool = False


GROUPS: dict[str, dict[str, object]] = {
    "inference": {
        "label": "Default predict/examples/inference.py dependencies",
        "entries": [
            FileEntry(
                repo_id="nvidia/Cosmos-H-Surgical",
                filename="config.json",
                revision="main",
                description="Cosmos-H-Surgical config loaded by checkpoint_db before checkpoint loads",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-H-Surgical",
                filename="predict/cosmos-h-surgical-predict_model_ema_bf16.pt",
                revision="ebbb7c6daf64f06c2dfe1b01654911789c5c9fdc",
                description="Default Cosmos-H-Surgical Predict checkpoint",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-Predict2.5-2B",
                filename="tokenizer.pth",
                revision="f176dc95b4a70f53ce01c4b302851595e7322b00",
                description="Wan2.1 VAE tokenizer checkpoint",
            ),
            SnapshotEntry(
                repo_id="nvidia/Cosmos-Reason1-7B",
                revision="3210bec0495fdc7a8d3dbb8d58da5711eab4b423",
                description="Reason1 text encoder checkpoint used for online prompt embeddings",
            ),
            SnapshotEntry(
                repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
                description="Qwen2.5-VL tokenizer/processor used by the Reason1 text encoder",
                gated=True,
            ),
            SnapshotEntry(
                repo_id="nvidia/Cosmos-Guardrail1",
                revision="d6d4bfa899a71454a700907664f3e88f503950cf",
                description="Cosmos-Guardrail1 video safety model, used when guardrails are enabled",
            ),
            SnapshotEntry(
                repo_id="Qwen/Qwen3Guard-Gen-0.6B",
                description="Qwen3Guard text safety model, used when guardrails are enabled",
                gated=True,
            ),
            SnapshotEntry(
                repo_id="google/siglip-so400m-patch14-384",
                description="SigLIP vision encoder used by the video safety filter",
            ),
        ],
    },
    "guardrails": {
        "label": "Optional guardrail models",
        "entries": [
            SnapshotEntry(
                repo_id="nvidia/Cosmos-Guardrail1",
                revision="d6d4bfa899a71454a700907664f3e88f503950cf",
                description="Cosmos-Guardrail1 video safety model",
            ),
            SnapshotEntry(
                repo_id="Qwen/Qwen3Guard-Gen-0.6B",
                description="Qwen3Guard text safety model",
                gated=True,
            ),
            SnapshotEntry(
                repo_id="google/siglip-so400m-patch14-384",
                description="SigLIP vision encoder used by the video safety filter",
            ),
        ],
    },
    "predict25_2b": {
        "label": "Additional upstream Cosmos-Predict2.5-2B checkpoints",
        "entries": [
            FileEntry(
                repo_id="nvidia/Cosmos-Predict2.5-2B",
                filename="base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt",
                revision="15a82a2ec231bc318692aa0456a36537c806e7d4",
                description="Upstream pre-trained base checkpoint",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-Predict2.5-2B",
                filename="base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
                revision="15a82a2ec231bc318692aa0456a36537c806e7d4",
                description="Upstream post-trained base checkpoint",
            ),
        ],
    },
}

ALL_GROUPS = list(GROUPS.keys())


def _print_separator(char: str = "-", width: int = 72) -> None:
    print(char * width)


def _download_snapshot(entry: SnapshotEntry, dry_run: bool) -> bool:
    from huggingface_hub import snapshot_download

    kwargs: dict[str, object] = {
        "repo_id": entry.repo_id,
        "repo_type": "model",
        "revision": entry.revision,
    }
    if entry.ignore_patterns:
        kwargs["ignore_patterns"] = entry.ignore_patterns
    if entry.allow_patterns:
        kwargs["allow_patterns"] = entry.allow_patterns

    print(f"  snapshot_download({entry.repo_id!r}, revision={entry.revision!r})")
    if entry.ignore_patterns:
        print(f"    ignore_patterns={entry.ignore_patterns}")
    if entry.allow_patterns:
        print(f"    allow_patterns={entry.allow_patterns}")
    if dry_run:
        return True
    try:
        path = snapshot_download(**kwargs)
        print(f"  -> {path}")
        return True
    except Exception:
        print(f"  ERROR downloading {entry.repo_id}:")
        traceback.print_exc()
        return False


def _download_file(entry: FileEntry, dry_run: bool) -> bool:
    from huggingface_hub import hf_hub_download

    print(f"  hf_hub_download({entry.repo_id!r}, {entry.filename!r}, revision={entry.revision!r})")
    if dry_run:
        return True
    try:
        path = hf_hub_download(
            repo_id=entry.repo_id,
            repo_type="model",
            filename=entry.filename,
            revision=entry.revision,
        )
        print(f"  -> {path}")
        return True
    except Exception:
        print(f"  ERROR downloading {entry.repo_id}/{entry.filename}:")
        traceback.print_exc()
        return False


def _download_entry(entry: SnapshotEntry | FileEntry, dry_run: bool) -> bool:
    if isinstance(entry, SnapshotEntry):
        return _download_snapshot(entry, dry_run)
    return _download_file(entry, dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Download all model groups.")
    group.add_argument(
        "--groups",
        nargs="+",
        metavar="GROUP",
        choices=ALL_GROUPS,
        help=f"Download specific groups. Choices: {ALL_GROUPS}",
    )
    group.add_argument("--list", action="store_true", help="List available groups and exit.")
    parser.add_argument(
        "--cache-dir",
        metavar="PATH",
        default=None,
        help=(
            "Root directory for the Hugging Face cache. Sets HF_HOME for this "
            "process. Defaults to $HF_HOME or ~/.cache/huggingface."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be downloaded.")
    parser.add_argument(
        "--skip-gated",
        action="store_true",
        help="Skip models marked as gated, which require Hugging Face account access.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        print("Available groups:")
        for key, meta in GROUPS.items():
            entries = meta["entries"]
            assert isinstance(entries, list)
            print(f"  {key:<16} {meta['label']}  [{len(entries)} artifact(s)]")
        return 0

    if args.cache_dir:
        cache_dir = os.path.abspath(args.cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        print(f"Cache directory set to: {cache_dir}")
    else:
        cache_dir = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
        print(f"Using cache directory: {cache_dir}")

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub")
        return 1

    selected_groups = ALL_GROUPS if args.all else args.groups

    _print_separator("=")
    print("Cosmos-H-Surgical predict inference downloader")
    print(f"Groups    : {', '.join(selected_groups)}")
    print(f"Cache     : {cache_dir}")
    print(f"Dry-run   : {args.dry_run}")
    print(f"Skip gated: {args.skip_gated}")
    _print_separator("=")

    total = 0
    succeeded = 0
    failed: list[str] = []

    for group_key in selected_groups:
        meta = GROUPS[group_key]
        entries = meta["entries"]
        assert isinstance(entries, list)

        _print_separator()
        print(f"GROUP: {group_key} - {meta['label']}")
        _print_separator()

        for entry in entries:
            if args.skip_gated and entry.gated:
                print(f"  [SKIP gated] {entry.repo_id}")
                continue

            label = f"{entry.repo_id}/{entry.filename}" if isinstance(entry, FileEntry) else entry.repo_id
            gated_tag = "  [gated]" if entry.gated else ""
            print(f"\n  {entry.description}{gated_tag}")

            total += 1
            if _download_entry(entry, dry_run=args.dry_run):
                succeeded += 1
            else:
                failed.append(label)

    _print_separator("=")
    print(f"Done. {succeeded}/{total} artifacts {'listed' if args.dry_run else 'downloaded'} successfully.")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for failure in failed:
            print(f"  {failure}")
    _print_separator("=")

    print(
        textwrap.dedent(
            f"""
            Offline usage:
              export HF_HOME={cache_dir}
              export HF_HUB_OFFLINE=1
              export COSMOS_HF_NO_DOWNLOAD=1

            Then run predict/examples/inference.py with the same HF_HOME on the
            offline node.
            """
        ).strip()
    )

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
