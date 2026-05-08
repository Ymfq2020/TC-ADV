from tc_adv.experiments.error_analysis import classify_error_types


def test_classify_error_types_groups_by_dominant_channel(tmp_path):
    diagnostics = [
        {"subject": "A", "relation": "rel", "timestamp": 1, "gold": "B",
         "top1_prediction": "X", "p_trm": 0.9, "p_ecm": 0.2, "p_fake": 0.8,
         "classification": "TRM-only"},
        {"subject": "A", "relation": "rel", "timestamp": 2, "gold": "B",
         "top1_prediction": "Y", "p_trm": 0.2, "p_ecm": 0.9, "p_fake": 0.8,
         "classification": "ECM-only"},
        {"subject": "A", "relation": "rel", "timestamp": 3, "gold": "B",
         "top1_prediction": "Z", "p_trm": 0.6, "p_ecm": 0.6, "p_fake": 0.7,
         "classification": "both"},
        {"subject": "A", "relation": "rel", "timestamp": 4, "gold": "B",
         "top1_prediction": "B", "p_trm": 0.95, "p_ecm": 0.95, "p_fake": 0.95,
         "classification": "both"},  # correct prediction → ignored
    ]
    diag_path = tmp_path / "diag.json"
    diag_path.write_text(__import__("json").dumps(diagnostics), encoding="utf-8")
    payload = classify_error_types(diag_path, violation_threshold=0.7)
    assert payload["counts"]["lifecycle_out_of_bounds"] == 1
    assert payload["counts"]["evolution_mutation"] == 1
    assert payload["counts"]["static_semantic_conflict"] == 1
    assert payload["total_high_confidence_errors"] == 3
