from typing import List

def chunk_transcript(text: str) -> List[str]:
    """
    Splits transcript text into chunks of 3-6 sentences.
    """
    if not text:
        return []
    
    # Simple split by period for now
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Group into chunks of 4 sentences
    chunk_size = 4
    chunks = []
    
    for i in range(0, len(sentences), chunk_size):
        chunk = ". ".join(sentences[i:i + chunk_size]) + "."
        chunks.append(chunk)
        
    return chunks
