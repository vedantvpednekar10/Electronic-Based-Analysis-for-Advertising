"""Evaluate clip-level looking labels using a held-out CSV manifest."""

import argparse
import csv
from pathlib import Path

from pipeline import FacePose, analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "models" /
                        "shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--yaw", type=float, default=0)
    parser.add_argument("--pitch", type=float, default=0)
    parser.add_argument("--tolerance", type=float, default=20)
    parser.add_argument("--dwell", type=float, default=0.5)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not {"path", "looking"}.issubset(reader.fieldnames or []):
            parser.error("Manifest must have path and looking columns.")
        clips = list(reader)
    if not clips or any(not row["path"] or row["looking"] not in {"0", "1"} for row in clips):
        parser.error("Provide one or more clips with a path and a looking label of 0 or 1.")
    if not args.model.is_file():
        parser.error(f"Missing landmark predictor at {args.model}")

    model = FacePose(args.model)
    counts = {(actual, predicted): 0 for actual in (0, 1) for predicted in (0, 1)}
    for row in clips:
        source = args.manifest.parent / row["path"]
        for _, result in analyze(str(source), model, yaw=args.yaw, pitch=args.pitch,
                                 tolerance=args.tolerance, dwell=args.dwell):
            pass
        counts[int(row["looking"]), int(result["estimated_viewers"] > 0)] += 1

    tn, fp, fn, tp = (counts[0, 0], counts[0, 1], counts[1, 0], counts[1, 1])
    print(f"Clips: {len(clips)} | TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    for label, numerator, denominator in (
        ("Accuracy", tp + tn, len(clips)), ("Precision", tp, tp + fp),
        ("Recall", tp, tp + fn), ("F1", 2 * tp, 2 * tp + fp + fn),
    ):
        score = f"{numerator / denominator:.2%}" if denominator else "undefined"
        print(f"{label}: {score}")


if __name__ == "__main__":
    main()
