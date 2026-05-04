#!/usr/bin/env python3
"""
Calibration tool for the local analyzer.

Point this at a real plant photo (or folder of photos) to see every CV signal
the local analyzer extracts. Use the output to tune the thresholds in
src/analyzers/local.py (_synthesise function) against YOUR plants and YOUR
lighting conditions.

Workflow:
    1. Take a photo of a healthy plant you know is healthy. Run:
         python calibrate.py photo.jpg
       Note the signal ratios. These are your 'healthy baseline'.
    2. Take a photo of a plant with a known problem (yellow leaves, crispy
       edges, etc). Run the same command. Compare ratios.
    3. Adjust thresholds in local.py _synthesise() so your healthy baseline
       stays in 'optimal' and your problem cases trigger the correct issues.

Usage:
    python calibrate.py photo.jpg                     # analyse one image
    python calibrate.py photo.jpg --save-masks out/   # save debug masks
    python calibrate.py folder/                       # batch mode
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from PIL import Image

from analyzers.local import (
    LocalAnalyzer, _segment_plant, _plant_silhouette,
    _color_analysis, _droop_score, _bounding_box,
)


def save_debug_mask(mask: np.ndarray, path: Path):
    """Save a boolean mask as a black-and-white image."""
    img = (mask * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def calibrate_one(image_path: Path, save_masks_dir: Path = None):
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > 1600:
        img.thumbnail((1600, 1600), Image.LANCZOS)
    arr = np.asarray(img)

    plant_mask = _segment_plant(arr)
    silhouette = _plant_silhouette(plant_mask, close_px=12)
    color = _color_analysis(arr, plant_mask)
    droop = _droop_score(plant_mask)
    bbox = _bounding_box(plant_mask)

    print(f"\n{'='*60}")
    print(f"File: {image_path}")
    print(f"Size: {img.size[0]}x{img.size[1]}")
    print(f"{'='*60}")
    print(f"SEGMENTATION")
    print(f"  Plant coverage:     {color['plant_coverage']:.3%}   ({plant_mask.sum():>7} px)")
    print(f"  Silhouette coverage:{silhouette.sum() / silhouette.size:.3%}   ({silhouette.sum():>7} px)")
    print(f"  Bounding box:       x={bbox['x']:.2f} y={bbox['y']:.2f} w={bbox['w']:.2f} h={bbox['h']:.2f}")
    print(f"\nCOLOR RATIOS (within plant / silhouette)")
    print(f"  Healthy green:      {color['healthy_ratio']:.3%}")
    print(f"  Yellow (chlorosis): {color['yellow_ratio']:.3%}    → triggers overwater at 4%, 8% w/ droop")
    print(f"  Brown (necrosis):   {color['brown_ratio']:.3%}    → triggers underwater at 5%, 12% w/ droop")
    print(f"  Scorched/bleached:  {color['scorch_ratio']:.3%}    → triggers excessive light at 4%")
    print(f"\nOTHER SIGNALS")
    print(f"  Average value:      {color['avg_value']:.3f}    (brightness, 0-1)")
    print(f"  Average saturation: {color['avg_saturation']:.3f}    (<0.25 w/ bright scene = possible 'insufficient')")
    print(f"  Droop score:        {droop:.3f}    (0=balanced, 1=severe; used as modifier)")

    # Run the full analyzer to see final verdict
    analyzer = LocalAnalyzer()
    result = analyzer.analyse(image_path)
    print(f"\nFINAL VERDICT")
    print(f"  Overall: {result['overall_health']} ({result['health_score']}/100)")
    print(f"  Water:   {result['watering_status']} ({result['watering_confidence']}%)")
    print(f"           {result['watering_reasoning']}")
    print(f"  Light:   {result['sunlight_status']} ({result['sunlight_confidence']}%)")
    print(f"           {result['sunlight_reasoning']}")
    if result['issues_detected']:
        print(f"  Issues:  {result['issues_detected']}")

    if save_masks_dir:
        save_masks_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        save_debug_mask(plant_mask, save_masks_dir / f"{stem}_plant_mask.png")
        save_debug_mask(silhouette, save_masks_dir / f"{stem}_silhouette.png")
        save_debug_mask(silhouette & ~plant_mask, save_masks_dir / f"{stem}_silhouette_fill.png")
        print(f"\n  Debug masks saved to {save_masks_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="Image file or directory of images")
    p.add_argument("--save-masks", type=Path, metavar="DIR",
                   help="Save debug masks (plant, silhouette, fill) to this directory")
    args = p.parse_args()

    if args.path.is_file():
        calibrate_one(args.path, args.save_masks)
    elif args.path.is_dir():
        images = sorted(
            f for f in args.path.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not images:
            print(f"No images found in {args.path}")
            sys.exit(1)
        for img in images:
            calibrate_one(img, args.save_masks)
    else:
        print(f"Path not found: {args.path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
