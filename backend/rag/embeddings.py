"""One local MiniLM encoder for documents and queries; bounded text cache."""
from collections import OrderedDict
from threading import Lock

import numpy as np
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

MODEL_ID = "all-MiniLM-L6-v2:onnx:384:v1"
_encoder = None
_lock = Lock()
_vectors = OrderedDict()


def embed_texts(texts):
    global _encoder
    # Bound input before MiniLM's 256-token truncation. Callers put content first.
    texts = [" ".join(text.split())[:1600] for text in texts]
    with _lock:
        missing = list(dict.fromkeys(text for text in texts if text not in _vectors))
        if missing:
            if _encoder is None:
                _encoder = ONNXMiniLM_L6_V2()
            vectors = _encoder(missing)
            if len(vectors) != len(missing):
                raise ValueError("Incomplete embedding batch")
            for text, vector in zip(missing, vectors):
                vector = np.asarray(vector, dtype=float)
                norm = np.linalg.norm(vector)
                if vector.shape != (384,) or not np.isfinite(vector).all() or norm == 0:
                    raise ValueError("Invalid embedding")
                _vectors[text] = (vector / norm).tolist()
        output = [_vectors[text] for text in texts]
        for text in texts:
            _vectors.move_to_end(text)
        while len(_vectors) > 2048:
            _vectors.popitem(last=False)
        return output


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
