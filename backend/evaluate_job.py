"""Offline, read-only quality summary for any completed or failed room job."""

import argparse
import json
from pathlib import Path

from reconstruction.quality_report import assess_job


def main():
    parser = argparse.ArgumentParser(description='Review reconstruction evidence without provider calls.')
    parser.add_argument('metadata', nargs='+', type=Path, help='One or more job metadata.json files')
    args = parser.parse_args()
    reports = [assess_job(json.loads(path.read_text(encoding='utf-8'))) for path in args.metadata]
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
