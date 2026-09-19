from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import StringIO

VALID_DECISIONS = {"approve", "reject", "needs_fix"}

@dataclass(frozen=True)
class ReviewDecision:
    row: int
    decision: str
    comment: str = ""
    reviewer: str = "reviewer"
    reviewed_at: str = ""

    def __post_init__(self):
        if self.row < 2:
            raise ValueError("row must be a CSV data row (>= 2)")
        if self.decision not in VALID_DECISIONS:
            raise ValueError(f"decision must be one of {sorted(VALID_DECISIONS)}")

    def normalized(self):
        return ReviewDecision(self.row, self.decision, self.comment.strip(), self.reviewer.strip() or "reviewer",
                              self.reviewed_at or datetime.now(timezone.utc).isoformat())

def upsert_decision(items: list[dict], decision: ReviewDecision) -> list[dict]:
    current = {int(item["row"]): item for item in items}
    current[decision.row] = asdict(decision.normalized())
    return [current[row] for row in sorted(current)]

def decisions_json(items: list[dict]) -> str:
    return json.dumps({"schema_version": 1, "decisions": items}, indent=2)

def decisions_csv(items: list[dict]) -> str:
    out = StringIO()
    fields = ["row", "decision", "comment", "reviewer", "reviewed_at"]
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: item.get(key, "") for key in fields} for item in items)
    return out.getvalue()

def apply_decisions_to_dataset(df, items: list[dict]):
    result = df.copy()
    result["review_decision"] = ""
    result["review_comment"] = ""
    by_row = {int(item["row"]): item for item in items}
    for index in range(len(result)):
        item = by_row.get(index + 2)
        if item:
            result.at[index, "review_decision"] = item.get("decision", "")
            result.at[index, "review_comment"] = item.get("comment", "")
    return result
