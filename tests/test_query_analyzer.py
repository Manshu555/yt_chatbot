from backend.routing.query_analyzer import requires_external_validation

def test_query_analyzer_toggle():
    assert requires_external_validation("what is this", True) == True

def test_query_analyzer_triggers():
    assert requires_external_validation("verify this fact", False) == True
    assert requires_external_validation("is this correct", False) == True
    
def test_query_analyzer_normal():
    assert requires_external_validation("what is machine learning?", False) == False
