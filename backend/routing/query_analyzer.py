def requires_external_validation(query: str, user_toggle: bool) -> bool:
    if user_toggle:
        return True
    
    validation_triggers = [
        "verify", "validate", "fact check", "fact-check", "is this correct", 
        "is this true", "correct?", "accurate?", "confirm", "latest", 
        "current", "recent", "according to other sources", "cross-check"
    ]
    
    query_lower = query.lower()
    for trigger in validation_triggers:
        if trigger in query_lower:
            return True
            
    return False
