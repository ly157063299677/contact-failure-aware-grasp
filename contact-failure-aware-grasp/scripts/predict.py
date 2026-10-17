import argparse
import csv
from pathlib import Path
from cfgrasp.data import load_features
from cfgrasp.inference import Predictor
from cfgrasp.physics import Deadline
from cfgrasp.subsets import MASKS, utilities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score all 31 subsets of one prepared state")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    predictor = Predictor(args.checkpoint, args.device)
    probability, deviation = predictor(load_features(args.features), Deadline())
    utility = utilities(probability, deviation, **predictor.config.get("utility", {}))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subset_id", "thumb", "index", "middle", "ring", "little", "success_probability", "deviation", "utility"])
        for index in range(31):
            writer.writerow([index + 1, *MASKS[index].astype(int), probability[index], deviation[index], utility[index]])
