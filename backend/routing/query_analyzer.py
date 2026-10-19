def requires_external_validation(query: str, user_toggle: bool) -> bool:
    
    validation_triggers = [
        "verify", "validate", "fact check", "fact-check", "is this correct", 
        "is this true", "correct?", "accurate?", "confirm", "latest", 
        "current", "recent", "according to other sources", "cross-check",
        "external", "additional context", "evidence", "corroborat", "research"
    ]
    
    query_lower = query.lower()
    for trigger in validation_triggers:
        if trigger in query_lower:
            return True
            
    # Search every enabled query, including summaries. The explicit off toggle
    # still opts out unless the question itself requests outside evidence.
    return bool(user_toggle)
