"""Small dependency-free BM25 and weighted reciprocal rank fusion."""
import math
import re
from collections import Counter
from difflib import SequenceMatcher

STOP = set("a an the is are was were be to of and or in on for by with from this that what how why does do video summarize summary explain please".split())


def terms(text):
    return [word for word in re.findall(r"[a-z0-9_]+", text.lower()) if word not in STOP]


def bm25(query, documents):
    tokens = [terms(text) for text in documents]
    counts = [Counter(doc) for doc in tokens]
    avg = sum(map(len, tokens)) / max(len(tokens), 1) or 1
    scores = [0.0] * len(tokens)
    for term in set(terms(query)):
        df = sum(term in doc for doc in tokens)
        idf = math.log(1 + (len(tokens) - df + 0.5) / (df + 0.5))
        for i, doc in enumerate(tokens):
            tf = counts[i][term]
            scores[i] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * len(doc) / avg))
    return scores


def fuse(dense, lexical, weight=0.75):
    scores = [0.0] * len(dense)
    for values, factor in [(dense, weight), (lexical, 1 - weight)]:
        order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
        for rank, i in enumerate(order, 1):
            if values[i] > 0:
                scores[i] += factor * 61 / (60 + rank)
    return scores


def near_duplicate(a, b):
    a, b = " ".join(a.lower().split()), " ".join(b.lower().split())
    return bool(a and b) and (a == b or SequenceMatcher(None, a, b).ratio() >= 0.9)


def is_summary(query):
    return bool(re.search(r"\b(summar(?:y|ize|ise)|overview|key (?:points|takeaways)|main (?:points|ideas))\b", query.lower()))
