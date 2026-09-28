import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from backend.routing.query_analyzer import requires_external_validation

tests = [
    ("What is gradient descent?", False),
    ("Is this information correct?", True),
    ("Verify this claim.", True),
    ("What is the latest version?", True)
]

for query, expected in tests:
    res = requires_external_validation(query, False)
    assert res == expected, f"Failed on '{query}': expected {expected}, got {res}"

res = requires_external_validation("what is this?", True)
assert res == True, "Failed explicit validate_externally=true"
res = requires_external_validation("verify this fact", False)
assert res == True, "Failed explicit trigger with validate_externally=false"
print("Query routing tests passed.")
