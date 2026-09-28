from backend.rag.chunker import chunk_transcript

def test_chunker_empty():
    assert chunk_transcript("") == []

def test_chunker_sentences():
    text = "This is sentence one. This is sentence two. This is three. Four. Five."
    chunks = chunk_transcript(text)
    assert len(chunks) == 2
    assert "This is sentence one. This is sentence two. This is three. Four." in chunks[0]
