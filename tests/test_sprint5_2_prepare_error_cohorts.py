from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sprint5_2_prepare_error_cohorts as preparation


class ThresholdSelectionTests(unittest.TestCase):
    def test_threshold_maximizes_fraud_f1(self):
        labels = np.asarray([1, 0, 1, 0], dtype=np.int64)
        scores = np.asarray([0.9, 0.8, 0.7, 0.1], dtype=np.float64)

        selected = preparation.select_threshold(
            preparation.build_threshold_curve(labels, scores)
        )

        self.assertAlmostEqual(selected["threshold"], 0.7)
        self.assertAlmostEqual(selected["precision"], 2 / 3)
        self.assertAlmostEqual(selected["recall"], 1.0)
        self.assertAlmostEqual(selected["f1"], 0.8)

    def test_equal_scores_form_one_candidate_threshold(self):
        labels = np.asarray([1, 0, 1], dtype=np.int64)
        scores = np.asarray([0.5, 0.5, 0.2], dtype=np.float64)

        curve = preparation.build_threshold_curve(labels, scores)

        self.assertEqual(curve["threshold"].tolist(), [0.5, 0.2])
        self.assertEqual(int(curve.iloc[0]["true_positive"]), 1)
        self.assertEqual(int(curve.iloc[0]["false_positive"]), 1)


class TargetSelectionTests(unittest.TestCase):
    @staticmethod
    def example_frame() -> pd.DataFrame:
        rows = []
        node_id = 0
        for cohort, label, predicted, scores in (
            ("TP", 1, 1, [0.91, 0.81, 0.71]),
            ("FP", 0, 1, [0.89, 0.79]),
            ("FN", 1, 0, [0.31, 0.21]),
            ("TN", 0, 0, [0.39, 0.29, 0.19]),
        ):
            for score in scores:
                rows.append(
                    {
                        "split": "validation",
                        "node_id": node_id,
                        "label": label,
                        "predicted_class": predicted,
                        "cohort": cohort,
                        "fraud_logit": float(np.log(score / (1 - score))),
                        "fraud_score": score,
                        "distance_to_threshold": abs(score - 0.5),
                        "in_degree": node_id + 1,
                        "log1p_degree": np.log1p(node_id + 1),
                        "sampled_event_count": min(node_id + 1, 15),
                        "score_bin": min(int(score * 10), 9),
                    }
                )
                node_id += 1
        return pd.DataFrame(rows)

    def test_selection_is_deterministic_paired_and_unique(self):
        frame = self.example_frame()

        first = preparation.select_targets(frame, "validation")
        second = preparation.select_targets(frame, "validation")

        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(first[["split", "node_id"]].duplicated().any())
        self.assertTrue((first.groupby("pair_id").size() == 2).all())
        self.assertEqual(set(first["target_role"]), {"error", "control"})
        self.assertEqual(
            set(first[first["target_role"] == "error"]["cohort"]), {"FP", "FN"}
        )


if __name__ == "__main__":
    unittest.main()
