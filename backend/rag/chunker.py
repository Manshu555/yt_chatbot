from typing import List


class ChunkText(str):
    def __new__(cls, text, metadata):
        instance = super().__new__(cls, text)
        instance.metadata = metadata
        return instance

def chunk_transcript(text: str) -> List[str]:
    """
    Splits transcript text into chunks of 3-6 sentences.
    """
    if not text:
        return []
    
    segments = getattr(text, "segments", None)
    if segments:
        chunks, words, start, end = [], [], None, None
        for segment in segments:
            incoming = segment["text"].split()
            while incoming:
                if start is None:
                    start = segment["start"]
                take = min(160 - len(words), len(incoming))
                words.extend(incoming[:take])
                incoming = incoming[take:]
                end = segment["end"]
                if len(words) == 160:
                    chunks.append(ChunkText(" ".join(words), {"start_seconds": start, "end_seconds": end}))
                    words, start = [], None
        if words:
            chunks.append(ChunkText(" ".join(words), {"start_seconds": start, "end_seconds": end}))
        return chunks

    # Keep sentence groups for text callers; bound unpunctuated captions.
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Group into chunks of 4 sentences
    chunk_size = 4
    chunks = []
    
    for i in range(0, len(sentences), chunk_size):
        chunk = ". ".join(sentences[i:i + chunk_size]) + "."
        words = chunk.split()
        chunks.extend(" ".join(words[j:j + 160]) for j in range(0, len(words), 160))
        
    return chunks
