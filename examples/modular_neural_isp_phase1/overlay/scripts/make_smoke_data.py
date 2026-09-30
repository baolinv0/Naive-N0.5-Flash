"""Create explicitly synthetic data to exercise the original training entrypoints.

These images are an engineering fixture, not a camera dataset or IQ benchmark.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="smoke_data")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    repo = Path(__file__).resolve().parents[1]
    yy, xx = np.mgrid[0:args.size, 0:args.size].astype(np.float32) / args.size
    sections = {}
    for split, count in (("train", 2), ("validation", 1), ("test", 1)):
        base = root / split
        for folder in ("denoised_raw_images", "srgb_images_style_0", "data"):
            (base / folder).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            rgb = np.stack((0.03 + 0.7 * xx, 0.04 + 0.6 * yy,
                            0.05 + 0.45 * (1 - xx) * yy), axis=-1)
            rgb = np.clip(rgb + i * 0.025, 0, 1)
            target = rgb ** (1 / 2.2)
            name = f"synthetic_{split}_{i:03d}"
            cv2.imwrite(str(base / "denoised_raw_images" / f"{name}.png"),
                        np.round(rgb[..., ::-1] * 65535).astype(np.uint16))
            cv2.imwrite(str(base / "srgb_images_style_0" / f"{name}.jpg"),
                        np.round(target[..., ::-1] * 255).astype(np.uint8))
            (base / "data" / f"{name}.json").write_text(json.dumps({
                "cam_illum": [1.0, 1.0, 1.0],
                "ccm": np.eye(3).tolist(), "orientation": 1,
            }), encoding="utf-8")
        sections[split] = {
            "input_dir": str(base / "denoised_raw_images"),
            "gt_dir": str(base / "srgb_images_style_0"),
            "metadata_dir": str(base / "data"),
        }
    config = {
        "repo_dir": str(repo), "python": sys.executable,
        "runs_dir": str(root / "runs"), **sections,
        "epochs": 2, "batch_size": 1, "in_size": args.size,
        "validation_frequency": 1, "num_workers": 0, "seed": 7,
        "eval_size": args.size,
        "init_checkpoint": str(repo / "photofinishing/models/photofinishing_s24-style-0.pth"),
        "init_config_dir": str(repo / "photofinishing/config"),
    }
    (root / "baseline.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (root / "README.txt").write_text(
        "SYNTHETIC ENGINEERING FIXTURE ONLY. Not real camera data or a PSNR benchmark.\n"
        "Original model/loss/optimizer are used; image size, batch and epochs are reduced for CPU plumbing verification.\n",
        encoding="utf-8",
    )
    print(root / "baseline.yaml")


if __name__ == "__main__":
    main()
