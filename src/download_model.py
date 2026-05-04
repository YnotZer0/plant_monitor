#!/usr/bin/env python3
"""
Set up the TFLite plant-disease model for offline use.

This is OPTIONAL. The local analyzer works without it (using classical CV only)
and will handle any plant. Adding a trained model only helps if you grow one of
the 14 crop species in the PlantVillage dataset (apple, blueberry, cherry, corn,
grape, orange, peach, bell pepper, potato, raspberry, soybean, squash,
strawberry, tomato).

There is no single canonical, freely-hosted URL for this model — most public
implementations are in notebooks/repos that you need to run yourself. Your
options, in order of effort:

OPTION 1 — Train it yourself (1-2 hours on Colab, best accuracy):
  Use this notebook (well-maintained, produces a TFLite file):
  https://github.com/obeshor/Plant-Diseases-Detector

  Steps:
    1. Open Plant_Diseases_Detection_with_TF2_V4.ipynb in Google Colab
    2. Run all cells — it downloads the PlantVillage dataset, trains a
       MobileNetV2, and exports a .tflite file
    3. Download the .tflite file and drop it at:
       plant_monitor/models/plant_disease.tflite

OPTION 2 — Find a pretrained checkpoint on Kaggle:
  https://www.kaggle.com/search?q=plantvillage+tflite
  Several users have uploaded trained .tflite files. Check the class list
  matches the one in src/analyzers/local.py (PLANTVILLAGE_CLASSES).

OPTION 3 — Skip the model entirely:
  Just use the classical CV analyzer. It'll still detect overwatering,
  underwatering, light problems, and general health from color/shape — it
  just won't name specific diseases.

If you have a URL for a known-good TFLite model, pass it to this script:
    python download_model.py --url <URL>
"""

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError


def download(url: str, dest: Path) -> bool:
    print(f"Downloading from: {url}")
    print(f"To: {dest}")
    try:
        def progress(block, block_size, total):
            if total > 0:
                pct = min(100, block * block_size * 100 // total)
                sys.stdout.write(f"\r  Progress: {pct}%")
                sys.stdout.flush()
        urlretrieve(url, dest, progress)
        size_kb = dest.stat().st_size // 1024
        print(f"\n✅ Downloaded {size_kb} KB")
        if size_kb < 100:
            print("⚠️  File is suspiciously small — likely an error page, not a model.")
            return False
        return True
    except URLError as e:
        print(f"\n❌ Download failed: {e}")
        return False


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--url", help="TFLite model URL (if you have a verified one)")
    p.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent.parent / "models" / "plant_disease.tflite",
        help="Where to save the model",
    )
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not args.url:
        print(__doc__)
        print(f"\nWhen you have a model file, place it at:\n  {args.output}")
        return

    if args.output.exists():
        print(f"Model already exists at {args.output}")
        resp = input("Overwrite? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    if not download(args.url, args.output):
        print("\nFall back to one of the manual options described above.")
        sys.exit(1)

    print("\n✅ Model installed. The local analyzer will now include disease detection.")
    print("   Verify with: python cli.py status")


if __name__ == "__main__":
    main()
