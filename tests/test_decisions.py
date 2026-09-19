import json
import pandas as pd
import pytest
from annotation_qc.decisions import ReviewDecision, apply_decisions_to_dataset, decisions_csv, decisions_json, upsert_decision

def test_decision_validation_and_upsert():
    with pytest.raises(ValueError): ReviewDecision(1, "approve")
    with pytest.raises(ValueError): ReviewDecision(2, "maybe")
    items = upsert_decision([], ReviewDecision(2, "needs_fix", " crop box "))
    items = upsert_decision(items, ReviewDecision(2, "approve", "fixed"))
    assert len(items) == 1 and items[0]["decision"] == "approve"

def test_exports_and_dataset_join():
    items = upsert_decision([], ReviewDecision(3, "reject", "wrong label", "sunny"))
    assert json.loads(decisions_json(items))["schema_version"] == 1
    assert "wrong label" in decisions_csv(items)
    df = pd.DataFrame([{"label":"car"},{"label":"person"}])
    out = apply_decisions_to_dataset(df, items)
    assert out.iloc[0]["review_decision"] == ""
    assert out.iloc[1]["review_decision"] == "reject"
