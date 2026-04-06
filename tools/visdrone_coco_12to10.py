#!/usr/bin/env python3
"""
Convert VisDrone COCO-style annotations from 12 classes (category_id 0..11)
to 10 classes by dropping specified category ids and remapping the remaining
category ids to a contiguous 0..(K-1) range.

This repo's evaluator uses prediction `labels` directly as COCO `category_id`,
so keeping category ids contiguous and aligned is important.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set


def _parse_int_list(values: List[str]) -> List[int]:
    out: List[int] = []
    for v in values:
        v = v.strip()
        if v == "":
            continue
        out.append(int(v))
    return out


def convert_coco(
    data: dict,
    drop_ids: Set[int],
    keep_images_without_anns: bool,
    crowd_from_ids: Set[int],
) -> dict:
    categories = data.get("categories", [])
    annotations = data.get("annotations", [])
    images = data.get("images", [])

    # Determine kept category ids from categories if possible, else from annotations
    cat_ids_in_cats = {int(c["id"]) for c in categories if "id" in c}
    if cat_ids_in_cats:
        keep_ids = sorted(cat_ids_in_cats - drop_ids)
    else:
        ann_ids = {int(a["category_id"]) for a in annotations if "category_id" in a}
        keep_ids = sorted(ann_ids - drop_ids)

    if not keep_ids:
        raise ValueError("No categories left after dropping ids.")

    old_to_new: Dict[int, int] = {old: new for new, old in enumerate(keep_ids)}

    new_categories = []
    if categories:
        for c in categories:
            if int(c.get("id")) in drop_ids:
                continue
            c2 = dict(c)
            c2["id"] = old_to_new[int(c2["id"])]
            new_categories.append(c2)
    else:
        new_categories = [{"id": old_to_new[old], "name": str(old)} for old in keep_ids]

    new_annotations = []
    # VisDrone (in this repo) uses string image ids; do not coerce types.
    kept_image_ids: Set[object] = set()
    for a in annotations:
        if "category_id" not in a:
            continue
        old = int(a["category_id"])
        if old in drop_ids:
            if old in crowd_from_ids:
                for new_cat_id in old_to_new.values():
                    a2 = dict(a)
                    a2["category_id"] = new_cat_id
                    a2["iscrowd"] = 1
                    a2["ignore"] = 1
                    new_annotations.append(a2)
                    if "image_id" in a2:
                        kept_image_ids.add(a2["image_id"])
            continue
        if old not in old_to_new:
            continue
        a2 = dict(a)
        a2["category_id"] = old_to_new[old]
        new_annotations.append(a2)
        if "image_id" in a2:
            kept_image_ids.add(a2["image_id"])

    if keep_images_without_anns:
        new_images = list(images)
    else:
        new_images = []
        for im in images:
            if "id" not in im:
                continue
            if im["id"] in kept_image_ids:
                new_images.append(im)

    out = dict(data)
    out["categories"] = new_categories
    out["annotations"] = new_annotations
    out["images"] = new_images
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input COCO json path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output COCO json path")
    ap.add_argument(
        "--drop",
        nargs="+",
        default=["0", "11"],
        help="Category ids to drop. Default: 0 11 (drop first & last).",
    )
    ap.add_argument(
        "--keep-images-without-anns",
        action="store_true",
        help="Keep images even if they have zero remaining annotations after filtering.",
    )
    ap.add_argument(
        "--crowd-from",
        nargs="+",
        default=[],
        help=(
            "Category ids to convert into crowd/ignore regions and duplicate across all kept "
            "categories (e.g. VisDrone ignored regions)."
        ),
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    drop_ids = set(_parse_int_list(args.drop))
    crowd_from_ids = set(_parse_int_list(args.crowd_from))

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    converted = convert_coco(
        data=data,
        drop_ids=drop_ids,
        keep_images_without_anns=args.keep_images_without_anns,
        crowd_from_ids=crowd_from_ids,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f)

    cats = converted.get("categories", [])
    anns = converted.get("annotations", [])
    cat_ids = sorted({int(c["id"]) for c in cats if "id" in c})
    ann_cat_ids = sorted({int(a["category_id"]) for a in anns if "category_id" in a})
    print(f"Wrote: {out_path}")
    print(f"categories: {len(cats)} ids={cat_ids}")
    print(f"annotations: {len(anns)} ann_category_ids={ann_cat_ids}")


if __name__ == "__main__":
    main()

