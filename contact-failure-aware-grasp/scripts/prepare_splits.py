#!/usr/bin/env python3
"""Prepare object-level splits from a real asset inventory; never generate assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfgrasp.construction import audit_manifest, load_manifest, partition_objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object_ids", type=Path, help="JSON list of unique canonical base object IDs")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON with train/val/test lists")
    parser.add_argument("--seed", type=int, default=0, help="Implementation default; original paper seed unavailable")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.1, 0.2), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--audit-manifest", type=Path, help="Optionally verify an existing JSONL manifest against these splits")
    parser.add_argument("--check-paths", action="store_true", help="Also require manifest feature/label files to exist")
    args = parser.parse_args()
    objects = json.loads(args.object_ids.read_text(encoding="utf-8-sig"))
    if not isinstance(objects, list):
        parser.error("object_ids must contain a JSON list, not a metadata object")
    partitions = partition_objects(objects, seed=args.seed, ratios=args.ratios)
    if args.audit_manifest is not None:
        counts = audit_manifest(load_manifest(args.audit_manifest), partitions,
                                require_paths=args.check_paths, base_dir=args.audit_manifest.parent)
        print("Manifest state counts:", counts)
    elif args.check_paths:
        parser.error("--check-paths requires --audit-manifest")
    # Creating a split file may change downstream membership: never overwrite it.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(partitions, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print("Object counts:", {name: len(ids) for name, ids in partitions.items()})
    print(f"Saved {args.output}; seed={args.seed}. This does not reproduce original paper split IDs.")


if __name__ == "__main__":
    main()
