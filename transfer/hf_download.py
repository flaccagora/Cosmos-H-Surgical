#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Download all models required by Cosmos-Transfer2.5 into the local HuggingFace cache.

Run on a machine with internet access, then rsync the cache directory to your
HPC cluster and set HF_HUB_OFFLINE=1 on the compute nodes.

Usage:
    # Download everything (all groups):
    python hf_download.py --all

    # Download specific groups:
    python hf_download.py --groups text_encoders guardrails vae

    # List available groups:
    python hf_download.py --list

    # Custom cache root (default: $HF_HOME or ~/.cache/huggingface):
    python hf_download.py --all --cache-dir /scratch/hf_cache

    # Dry-run (show what would be downloaded without downloading):
    python hf_download.py --all --dry-run
"""

import argparse
import os
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

@dataclass
class SnapshotEntry:
    """Download an entire repo snapshot via snapshot_download()."""
    repo_id: str
    revision: str = "main"
    ignore_patterns: list[str] = field(default_factory=list)
    allow_patterns: list[str] = field(default_factory=list)
    description: str = ""
    gated: bool = False


@dataclass
class FileEntry:
    """Download a single file via hf_hub_download()."""
    repo_id: str
    filename: str
    revision: str = "main"
    description: str = ""
    gated: bool = False


# Each group maps to the list of entries the user wants to materialise.
GROUPS: dict[str, dict] = {
    # ------------------------------------------------------------------
    "text_encoders": {
        "label": "Text encoders (T5-11B, UMT5-XXL)",
        "entries": [
            SnapshotEntry(
                repo_id="google-t5/t5-11b",
                revision="90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3",
                ignore_patterns=["tf_model.h5"],
                description="T5-11B text encoder (PyTorch weights only)",
            ),
            SnapshotEntry(
                repo_id="google/umt5-xxl",
                description="UMT5-XXL tokenizer + encoder (multilingual)",
            ),
        ],
    },
    # ------------------------------------------------------------------
    "guardrails": {
        "label": "Guardrail / safety models",
        "entries": [
            SnapshotEntry(
                repo_id="meta-llama/Llama-Guard-3-8B",
                revision="7327bd9f6efbbe6101dc6cc4736302b3cbb6e425",
                ignore_patterns=["original/*"],
                description="LlamaGuard-3-8B content safety classifier",
                gated=True,
            ),
            SnapshotEntry(
                repo_id="nvidia/Cosmos-Guardrail1",
                revision="d6d4bfa899a71454a700907664f3e88f503950cf",
                description="Cosmos-Guardrail1 safety model",
            ),
            SnapshotEntry(
                repo_id="google/siglip-so400m-patch14-384",
                description="SigLIP vision encoder used by the safety filter",
            ),
            SnapshotEntry(
                repo_id="Qwen/Qwen3Guard-Gen-0.6B",
                description="Qwen3Guard safety guard model",
                gated=True,
            ),
        ],
    },
    # ------------------------------------------------------------------
    "reason1": {
        "label": "Cosmos-Reason1 / Qwen2.5-VL backbone",
        "entries": [
            SnapshotEntry(
                repo_id="nvidia/Cosmos-Reason1-7B",
                revision="3210bec0495fdc7a8d3dbb8d58da5711eab4b423",
                description="Cosmos-Reason1-7B (includes Qwen2.5-VL-7B tokenizer/processor)",
            ),
            SnapshotEntry(
                repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
                description="Qwen2.5-VL-7B-Instruct processor (used directly by some inference paths)",
                gated=True,
            ),
        ],
    },
    # ------------------------------------------------------------------
    "vae": {
        "label": "VAE / tokenizer weights",
        "entries": [
            FileEntry(
                repo_id="nvidia/Cosmos-Predict2.5-2B",
                filename="tokenizer.pth",
                revision="6787e176dce74a101d922174a95dba29fa5f0c55",
                description="Cosmos Wan2.1 VAE tokenizer",
            ),
            SnapshotEntry(
                repo_id="stabilityai/sd-vae-ft-mse",
                description="Stability AI SD-VAE-ft-mse (diffusers fallback)",
            ),
        ],
    },
    # ------------------------------------------------------------------
    "predict25_2b": {
        "label": "Cosmos-Predict2.5-2B model checkpoints",
        "entries": [
            FileEntry(
                repo_id="nvidia/Cosmos-Predict2.5-2B",
                filename="base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt",
                revision="15a82a2ec231bc318692aa0456a36537c806e7d4",
                description="Pre-trained base (720p, 16 fps)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-Predict2.5-2B",
                filename="base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
                revision="15a82a2ec231bc318692aa0456a36537c806e7d4",
                description="Post-trained base (720p, 16 fps)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-Predict2.5-2B",
                filename="auto/multiview/524af350-2e43-496c-8590-3646ae1325da_ema_bf16.pt",
                revision="865baf084d4c9e850eac59a021277d5a9b9e8b63",
                description="Auto multiview (720p, 30 fps, 7 views, 29 frames)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-Predict2.5-2B",
                filename="auto/multiview/6b9d7548-33bb-4517-b5e8-60caf47edba7_ema_bf16.pt",
                revision="15a82a2ec231bc318692aa0456a36537c806e7d4",
                description="Auto multiview alternate (720p, 30 fps, 7 views)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-Predict2.5-2B",
                filename="robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt",
                revision="main",
                description="Robot action-conditioned (360p, 4 fps)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-Predict2.5-2B",
                filename="robot/multiview-agibot/f740321e-2cd6-4370-bbfe-545f4eca2065/model_ema_bf16.pt",
                revision="main",
                description="Robot multiview-agibot (720p, 16 fps)",
            ),
        ],
    },
    # ------------------------------------------------------------------
    "predict25_14b": {
        "label": "Cosmos-Predict2.5-14B model checkpoints",
        "entries": [
            FileEntry(
                repo_id="nvidia/Cosmos-Predict2.5-14B",
                filename="base/pre-trained/54937b8c-29de-4f04-862c-e67b04ec41e8_ema_bf16.pt",
                revision="03eb354f35eae0d6e0c1be3c9f94d8551e125570",
                description="Pre-trained base 14B (720p, 16 fps)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-Predict2.5-14B",
                filename="base/post-trained/e21d2a49-4747-44c8-ba44-9f6f9243715f_ema_bf16.pt",
                revision="2bc4ca5ba5a20b9858a7ddb856bc82d70b030fbe",
                description="Post-trained base 14B (720p, 16 fps)",
            ),
        ],
    },
    # ------------------------------------------------------------------
    "transfer25_2b": {
        "label": "Cosmos-Transfer2.5-2B model checkpoints",
        "entries": [
            # uniform sampling variants
            FileEntry(
                repo_id="nvidia/Cosmos-Transfer2.5-2B",
                filename="general/edge/61f5694b-0ad5-4ecd-8ad7-c8545627d125_ema_bf16.pt",
                revision="b67b64abda3801a9aceddbff2bdb86126c06db74",
                description="General edge (uniform, 720p, 16 fps)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-Transfer2.5-2B",
                filename="general/depth/626e6618-bfcd-4d9a-a077-1409e2ce353f_ema_bf16.pt",
                revision="dea7737ca29dd8d9086413c6dc5724b8250a0bb4",
                description="General depth (uniform, 720p, 16 fps)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-Transfer2.5-2B",
                filename="general/blur/ba2f44f2-c726-4fe7-949f-597069d9b91c_ema_bf16.pt",
                revision="eb5325b77d358944da58a690157dd2b8071bbf85",
                description="General blur (uniform, 720p, 16 fps)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-Transfer2.5-2B",
                filename="general/seg/5136ef49-6d8d-42e8-8abf-7dac722a304a_ema_bf16.pt",
                revision="23057a4167b89de89a4a397fdbf3887994d115eb",
                description="General segmentation (uniform, 720p, 16 fps)",
            ),
            # non-uniform sampling variants (all on same revision)
            FileEntry(
                repo_id="nvidia/Cosmos-Transfer2.5-2B",
                filename="general/edge/ecd0ba00-d598-4f94-aa09-e8627899c431_ema_bf16.pt",
                revision="bd963eabcfc2d61dc4ea365cacf41d45ac480aa5",
                description="General edge (non-uniform, 720p, 16 fps)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-Transfer2.5-2B",
                filename="general/seg/fcab44fe-6fe7-492e-b9c6-67ef8c1a52ab_ema_bf16.pt",
                revision="bd963eabcfc2d61dc4ea365cacf41d45ac480aa5",
                description="General segmentation (non-uniform, 720p, 16 fps)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-Transfer2.5-2B",
                filename="general/blur/20d9fd0b-af4c-4cca-ad0b-f9b45f0805f1_ema_bf16.pt",
                revision="bd963eabcfc2d61dc4ea365cacf41d45ac480aa5",
                description="General blur (non-uniform, 720p, 16 fps)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-Transfer2.5-2B",
                filename="general/depth/0f214f66-ae98-43cf-ab25-d65d09a7e68f_ema_bf16.pt",
                revision="bd963eabcfc2d61dc4ea365cacf41d45ac480aa5",
                description="General depth (non-uniform, 720p, 16 fps)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-Transfer2.5-2B",
                filename="auto/multiview/b5ab002d-a120-4fbf-a7f9-04af8615710b_ema_bf16.pt",
                revision="bd963eabcfc2d61dc4ea365cacf41d45ac480aa5",
                description="Auto multiview alternate (720p, 16 fps, 7 views, 29 frames)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-Transfer2.5-2B",
                filename="auto/multiview/4ecc66e9-df19-4aed-9802-0d11e057287a_ema_bf16.pt",
                revision="00c591edab119e8a6ca06e6e091351a04ce0ecc9",
                description="Auto multiview (720p, 10 fps, 7 views, 29 frames)",
            ),
        ],
    },
    # ------------------------------------------------------------------
    "cosmos_h_surgical_transfer": {
        "label": "Cosmos-H-Surgical Transfer checkpoints + runtime dependencies",
        "entries": [
            FileEntry(
                repo_id="nvidia/Cosmos-H-Surgical",
                filename="config.json",
                revision="main",
                description="Cosmos-H-Surgical config (resolved by checkpoint_db before checkpoint loads)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-H-Surgical",
                filename="transfer/edge/cosmos-h-surgical-transfer-edge_model_ema_bf16.pt",
                revision="ebbb7c6daf64f06c2dfe1b01654911789c5c9fdc",
                description="Cosmos-H-Surgical Transfer edge control (720p, 16 fps)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-H-Surgical",
                filename="transfer/depth/cosmos-h-surgical-transfer-depth_model_ema_bf16.pt",
                revision="ebbb7c6daf64f06c2dfe1b01654911789c5c9fdc",
                description="Cosmos-H-Surgical Transfer depth control (720p, 16 fps)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-H-Surgical",
                filename="transfer/vis/cosmos-h-surgical-transfer-vis_model_ema_bf16.pt",
                revision="ebbb7c6daf64f06c2dfe1b01654911789c5c9fdc",
                description="Cosmos-H-Surgical Transfer vis/blur control (720p, 16 fps)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-H-Surgical",
                filename="transfer/seg/cosmos-h-surgical-transfer-seg_model_ema_bf16.pt",
                revision="ebbb7c6daf64f06c2dfe1b01654911789c5c9fdc",
                description="Cosmos-H-Surgical Transfer segmentation control (720p, 16 fps)",
            ),
            FileEntry(
                repo_id="nvidia/Cosmos-Predict2.5-2B",
                filename="tokenizer.pth",
                # revision="6787e176dce74a101d922174a95dba29fa5f0c55",
                revision="f176dc95b4a70f53ce01c4b302851595e7322b00",
                description="Wan2.1 VAE tokenizer checkpoint (registered as Wan2.1/vae)",
            ),
            SnapshotEntry(
                repo_id="google/siglip2-so400m-patch16-naflex",
                description="SigLIP2 vision encoder required by Transfer2 image context network",
            ),
            SnapshotEntry(
                repo_id="google-t5/t5-11b",
                revision="90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3",
                ignore_patterns=["tf_model.h5"],
                description="T5-11B text encoder used for prompt embeddings",
            ),
            SnapshotEntry(
                repo_id="nvidia/Cosmos-Guardrail1",
                revision="d6d4bfa899a71454a700907664f3e88f503950cf",
                description="Cosmos-Guardrail1 safety model (video safety filter)",
            ),
            SnapshotEntry(
                repo_id="Qwen/Qwen3Guard-Gen-0.6B",
                description="Qwen3Guard text safety model",
                gated=True,
            ),
            SnapshotEntry(
                repo_id="google/siglip-so400m-patch14-384",
                description="SigLIP vision encoder for video safety filter",
            ),
            SnapshotEntry(
                repo_id="facebook/sam2-hiera-large",
                description="SAM2 Hiera-Large video segmentation (auto seg controls)",
            ),
            SnapshotEntry(
                repo_id="IDEA-Research/grounding-dino-base",
                description="GroundingDINO-Base text-conditioned object detection (SAM2 prompts)",
            ),
            FileEntry(
                repo_id="depth-anything/Video-Depth-Anything-Small",
                filename="video_depth_anything_vits.pth",
                description="Video-Depth-Anything Small (ViT-S) weights for auto depth controls",
            ),
            FileEntry(
                repo_id="depth-anything/Video-Depth-Anything-Large",
                filename="video_depth_anything_vitl.pth",
                description="Video-Depth-Anything Large (ViT-L) weights for auto depth controls",
            ),
            SnapshotEntry(
                repo_id="nvidia/Cosmos-Reason1-7B",
                revision="3210bec0495fdc7a8d3dbb8d58da5711eab4b423",
                description="Cosmos-Reason1-7B (includes Qwen2.5-VL-7B tokenizer/processor)",
            ),
            SnapshotEntry(
                repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
                description="Qwen2.5-VL-7B-Instruct processor (used directly by some inference paths)",
                gated=True,
            ),

        ],
    },
    # ------------------------------------------------------------------
    "auxiliary": {
        "label": "Auxiliary vision models (SAM2, GroundingDINO, SigLIP2, CLIP, Depth-Anything)",
        "entries": [
            SnapshotEntry(
                repo_id="facebook/sam2-hiera-large",
                description="SAM2 Hiera-Large video segmentation",
            ),
            SnapshotEntry(
                repo_id="IDEA-Research/grounding-dino-base",
                description="GroundingDINO-Base zero-shot object detection",
            ),
            SnapshotEntry(
                repo_id="google/siglip2-so400m-patch16-naflex",
                description="SigLIP2 vision encoder (Transfer2.5 network)",
            ),
            SnapshotEntry(
                repo_id="openai/clip-vit-base-patch32",
                description="CLIP ViT-B/32 vision encoder (default config)",
            ),
            FileEntry(
                repo_id="depth-anything/Video-Depth-Anything-Small",
                filename="video_depth_anything_vits.pth",
                description="Video-Depth-Anything Small (ViT-S) weights",
            ),
            FileEntry(
                repo_id="depth-anything/Video-Depth-Anything-Large",
                filename="video_depth_anything_vitl.pth",
                description="Video-Depth-Anything Large (ViT-L) weights",
            ),
        ],
    },
}

ALL_GROUPS = list(GROUPS.keys())


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _print_separator(char: str = "-", width: int = 72) -> None:
    print(char * width)


def _download_snapshot(entry: SnapshotEntry, dry_run: bool) -> bool:
    from huggingface_hub import snapshot_download

    kwargs: dict = dict(repo_id=entry.repo_id, repo_type="model", revision=entry.revision)
    if entry.ignore_patterns:
        kwargs["ignore_patterns"] = entry.ignore_patterns
    if entry.allow_patterns:
        kwargs["allow_patterns"] = entry.allow_patterns

    print(f"  snapshot_download({entry.repo_id!r}, revision={entry.revision!r})")
    if entry.ignore_patterns:
        print(f"    ignore_patterns={entry.ignore_patterns}")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Download all model groups.",
    )
    group.add_argument(
        "--groups",
        nargs="+",
        metavar="GROUP",
        choices=ALL_GROUPS,
        help=f"Download specific groups. Choices: {ALL_GROUPS}",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List all available groups and exit.",
    )
    parser.add_argument(
        "--cache-dir",
        metavar="PATH",
        default=None,
        help=(
            "Root directory for the HuggingFace cache "
            "(sets HF_HOME for this process). "
            "Defaults to $HF_HOME or ~/.cache/huggingface."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without downloading.",
    )
    parser.add_argument(
        "--skip-gated",
        action="store_true",
        help=(
            "Skip models marked as gated (require HF account access). "
            "Useful for CI or when you have not accepted the relevant licenses."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # --list
    if args.list:
        print("Available groups:")
        for key, meta in GROUPS.items():
            n = len(meta["entries"])
            print(f"  {key:<20} {meta['label']}  [{n} artifact(s)]")
        return 0

    # Override HF_HOME before importing huggingface_hub so the cache path
    # is respected for all subsequent calls.
    if args.cache_dir:
        cache_dir = os.path.abspath(args.cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        print(f"Cache directory set to: {cache_dir}")
    else:
        cache_dir = os.environ.get(
            "HF_HOME",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
        )
        print(f"Using cache directory: {cache_dir}")

    # Ensure huggingface_hub is available
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub")
        return 1

    selected_groups = ALL_GROUPS if args.all else args.groups

    _print_separator("=")
    print(f"Cosmos-Transfer2.5 model downloader")
    print(f"Groups   : {', '.join(selected_groups)}")
    print(f"Cache    : {cache_dir}")
    print(f"Dry-run  : {args.dry_run}")
    print(f"Skip gated: {args.skip_gated}")
    _print_separator("=")

    total = 0
    succeeded = 0
    failed: list[str] = []

    for group_key in selected_groups:
        meta = GROUPS[group_key]
        _print_separator()
        print(f"GROUP: {group_key}  —  {meta['label']}")
        _print_separator()

        for entry in meta["entries"]:
            if args.skip_gated and entry.gated:
                print(f"  [SKIP gated] {entry.repo_id}")
                continue

            label = (
                f"{entry.repo_id}/{entry.filename}"
                if isinstance(entry, FileEntry)
                else entry.repo_id
            )
            gated_tag = "  [gated]" if entry.gated else ""
            print(f"\n  {entry.description}{gated_tag}")

            total += 1
            ok = _download_entry(entry, dry_run=args.dry_run)
            if ok:
                succeeded += 1
            else:
                failed.append(label)

    # Summary
    _print_separator("=")
    print(f"Done.  {succeeded}/{total} artifacts {'listed' if args.dry_run else 'downloaded'} successfully.")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")
    _print_separator("=")

    # Print cluster env-var instructions
    hub_cache = os.path.join(cache_dir, "hub")
    print(
        textwrap.dedent(f"""
        ╔══════════════════════════════════════════════════════════════════════╗
        ║  To use the downloaded models on an offline HPC cluster:             ║
        ╠══════════════════════════════════════════════════════════════════════╣
        ║ files downloaded to                                                  ║
        ║               {cache_dir}                                            ║
        ║                                                                      ║
        ╚══════════════════════════════════════════════════════════════════════╝
        """)
    )

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

