#!/usr/bin/env python3
"""Build a candidate-independent TRAIN/DEV profile with official ISP transforms.

Example: python scripts/build_dev_profile.py --config configs/pilot.yaml
  --rules frozen-rules.json --output-dir profiles/p001
Rules: {"threshold_source":"explicit", "codevalue_thresholds":[0.2,0.8],
        "gradient_thresholds":[0.01,0.05], "profile_revision":"p001", "roi":true}
Or use threshold_source TRAIN to calibrate two global TRAIN pixel quantiles.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tm_research.diagnostics import build_dev_profile
from tm_research.runner import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--rules', required=True, help='Frozen profile rules JSON; TRAIN or explicit thresholds')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    rules = json.loads(Path(args.rules).read_text(encoding='utf-8'))
    result = build_dev_profile(load_config(args.config), rules, args.output_dir)
    print(json.dumps({key: result[key] for key in ('profile_ref', 'profile_revision', 'roi_profile_ref') if key in result}))


if __name__ == '__main__':
    main()
