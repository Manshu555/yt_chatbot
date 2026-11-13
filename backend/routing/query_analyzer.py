import re

# Whole-word patterns: plain substring matching made "concurrent" match
# "current", "researcher" match "research", and "incorrect?" match "correct?".
_VALIDATION_TRIGGER = re.compile(
    r"\b(?:"
    r"verify|verified|verifying|validate|fact[ -]check|is this correct|is this true|"
    r"correct\?|accurate\?|confirm|latest|current|recent|recently|"
    r"according to other sources|cross-check|external|additional context|"
    r"evidence|corroborat\w*|research"
    r")(?!\w)",
    re.IGNORECASE,
)


def requires_external_validation(query: str, user_toggle: bool) -> bool:
    if _VALIDATION_TRIGGER.search(query):
        return True

    # Search every enabled query, including summaries. The explicit off toggle
    # still opts out unless the question itself requests outside evidence.
    return bool(user_toggle)
