"""The model-facing, read-only mission preflight tool."""
import json

import preflight


def mission_preflight() -> str:
    """Return one orientation snapshot and its healthy/deep-path decision."""
    try:
        snapshot = preflight.build_snapshot()
    except Exception as exc:  # tools return evidence; they never kill the loop
        snapshot = {
            "fast_path_eligible": False,
            "attention_required": [
                f"mission preflight failed: {type(exc).__name__}: {exc}"],
        }
    return json.dumps(snapshot, indent=2, ensure_ascii=False)
