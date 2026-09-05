from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sprint5_2_error_attribution as attribution
import sprint5_2_tgat_ablation as ablation


def test_prediction_push_follows_operating_prediction_side():
    assert attribution.prediction_push(2.0, 1.25, predicted_class=1) == 0.75
    assert attribution.prediction_push(-2.0, -1.25, predicted_class=0) == 0.75
    assert attribution.prediction_push(1.0, 1.5, predicted_class=1) == -0.5
    assert attribution.prediction_push(-1.0, -1.5, predicted_class=0) == -0.5


def test_error_control_summary_uses_paired_signed_difference():
    records = []
    for role, cohort, push in (
        ('error', 'FP', 0.75), ('control', 'TP', 0.25)
    ):
        direct = np.zeros(len(attribution.FEATURE_NAMES), dtype=float)
        direct[0] = push
        direct[1] = -0.25 if role == 'error' else -0.5
        records.append({
            'split': 'validation',
            'pair_id': 'validation_fp_00',
            'target_role': role,
            'error_cohort': 'FP',
            'cohort': cohort,
            'locked_predicted_class': 1,
            'feature_direct_prediction_push': direct.tolist(),
        })

    summary = attribution.summarize_error_control_pairs(records)[0]
    feature = summary['feature_contrasts'][0]

    assert summary['control_cohort'] == 'TP'
    assert summary['n_pairs'] == 1
    assert feature['paired_median_prediction_push_difference'] == 0.5
    assert feature['paired_difference_positive_rate'] == 1.0
    assert summary['positive_error_specific_candidates'] == ['F01']


def test_ablation_uses_only_directly_interpretable_variants():
    assert ablation.VARIANTS == ("FULL", "T-CONST", "N-BASE", "MP-OFF")


def test_ablation_summary_contains_only_ranking_metrics():
    arrays = {
        "label": np.asarray([0, 0, 1, 1]),
        **{
            f"{variant}_logit": np.asarray([-2.0, -1.0, 1.0, 2.0])
            for variant in ablation.VARIANTS
        },
    }
    rows = ablation.summarize_split(arrays, "validation")
    assert len(rows) == 4
    assert all(
        set(row) == {"split", "variant", "n", "average_precision", "roc_auc"}
        for row in rows
    )
