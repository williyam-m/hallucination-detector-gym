"""Quick script to check schema and extracted fields."""
from hallucination_detector_gym.models import HallucinationAction
import json
import sys
sys.path.insert(0, ".")

schema = HallucinationAction.model_json_schema()

# Simulate what _extract_action_fields does
from openenv.core.env_server.web_interface import _extract_action_fields
fields = _extract_action_fields(HallucinationAction)

print("=== FIELD SUMMARY ===")
for f in fields:
    print(f"  {f['name']:25s} → widget: {f['type']:10s}  choices: {f.get('choices')}")

# Assertions
field_map = {f["name"]: f for f in fields}

assert field_map["action_type"]["type"] == "select", "action_type must be select (dropdown)"
assert field_map["hallucination_detected"]["type"] == "checkbox", "hallucination_detected must be checkbox"
assert field_map["hallucination_type"]["type"] == "select", "hallucination_type must be select (dropdown)"
assert field_map["hallucinated_span"]["type"] == "textarea", "hallucinated_span should be textarea"
assert field_map["corrected_text"]["type"] == "textarea", "corrected_text should be textarea"
assert field_map["reasoning"]["type"] == "textarea", "reasoning should be textarea"

# Field order
field_names = [f["name"] for f in fields]
assert field_names == [
    "action_type",
    "hallucination_detected",
    "hallucination_type",
    "hallucinated_span",
    "corrected_text",
    "reasoning",
], f"Field order mismatch: {field_names}"

print("\n✅ All UI widget assertions passed!")
