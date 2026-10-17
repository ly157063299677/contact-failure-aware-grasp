"""Check schema and leakage; optionally verify the manuscript's dataset counts."""
import argparse
import json
from cfgrasp.data import audit_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--paper-counts", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit_dataset(args.data, args.paper_counts), indent=2))
