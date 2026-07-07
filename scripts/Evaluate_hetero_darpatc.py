"""
evaluate_darpatc_hetero.py

Same output format as ThreaTrace's evaluate_darpatc.py (tp fp fn line, then
Precision / Recall / F-Score lines) so numbers are directly comparable to
your Phase 0 baseline reproduction.

Ground truth path matches ThreaTrace's own convention, confirmed from
train_darpatc.py: ../groundtruth/<scene>.txt -- one malicious node uuid per
line, matching the raw uuid strings stored in data[nt].node_id by
graph_construction.py.

Usage:
    python evaluate_darpatc_hetero.py --scene cadets
"""

import argparse
import pickle


def load_ground_truth(path: str) -> set:
    gt = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                gt.add(line)
    return gt


def evaluate(fp_ids: set, ground_truth: set):
    # fp_ids is a set of (node_type, raw_uuid) tuples; ground truth is keyed
    # by raw uuid string, so compare on the uuid component only.
    predicted = {str(nid) for (_, nid) in fp_ids}

    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(tp, fp, fn)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-Score: {f1}")
    return precision, recall, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="cadets")
    parser.add_argument(
        "--groundtruth_dir",
        default="../groundtruth",
        help="matches ThreaTrace's own convention: ../groundtruth/<scene>.txt",
    )
    args = parser.parse_args()

    with open(f"results/{args.scene}_hetero_fp_ids.pkl", "rb") as f:
        fp_ids = pickle.load(f)

    ground_truth = load_ground_truth(f"{args.groundtruth_dir}/{args.scene}.txt")
    print(f"Loaded {len(ground_truth)} ground-truth malicious node ids")
    evaluate(fp_ids, ground_truth)