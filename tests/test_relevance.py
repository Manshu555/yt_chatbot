import pytest
from pydantic import ValidationError

from schemas.models import ExternalEvidence, SearchResult
from search.relevance import rank_results, score_result


def source(**kwargs):
    return SearchResult(url="https://example.com", **kwargs)


@pytest.mark.parametrize("title,snippet,passage,expected", [
    ("Neural", "weights", "biases", 100),
    ("NEURAL neural!", "unrelated", "", 33.33),
    ("", "weights weights", "biases", 66.67),
    ("gardening", "", "", 0),
    (None, None, None, 0),
])
def test_score_is_unique_question_term_coverage(title, snippet, passage, expected):
    result = score_result("What are neural weights and biases?", source(title=title, snippet=snippet, passage=passage))
    assert result.relevance_score == expected
    assert result.relevance_reason
    assert score_result("What are neural weights and biases?", result) == result


@pytest.mark.parametrize("question", ["", None, "What is the?", "!!!"])
def test_empty_question_scores_zero(question):
    result = score_result(question, source(title="Neural network"))
    assert result.relevance_score == 0
    assert result.relevance_reason == "No searchable question terms to compare."


def test_reason_describes_actual_matching_fields():
    result = score_result("weights biases", source(title="Weights", passage="Biases"))
    assert result.relevance_reason == "Matched 2 of 2 question terms in title, passage: biases, weights."


def test_ranking_is_descending_and_stable_for_ties_and_zero():
    results = [source(title=title) for title in ["Garden", "Weights", "Biases", "Weights biases", "Cooking"]]
    ranked = rank_results("weights biases", results)
    assert [r.title for r in ranked] == ["Weights biases", "Weights", "Biases", "Garden", "Cooking"]
    assert [r.relevance_score for r in ranked] == [100, 50, 50, 0, 0]
    assert results[0].relevance_reason == "Relevance has not been calculated."
    assert rank_results("weights", []) == []


def test_available_evidence_selects_question_passage_without_using_claim_score():
    result = source(title="Parameters", snippet="Original snippet", domain="example.com")
    evidence = [ExternalEvidence(source_title="Source", url=result.url, domain="example.com", passage=passage, relevance_score=0.01)
                for passage in ["Menu advertising", "Weights and biases tune a neural network."]]
    ranked = rank_results("neural weights biases", [result], evidence)
    assert ranked[0].relevance_score == 100
    assert ranked[0].passage == evidence[1].passage
    assert ranked[0].snippet == result.snippet
    assert ranked[0].url == result.url
    assert all(item.relevance_score == 0.01 for item in evidence)
    assert result.passage == ""


def test_evidence_from_other_urls_is_not_used():
    evidence = [ExternalEvidence(source_title="Other", url="https://other.test", domain="other.test", passage="Weights", relevance_score=1)]
    assert rank_results("weights", [source()], evidence)[0].relevance_score == 0


@pytest.mark.parametrize("score", [-1, 101, float("nan"), float("inf")])
def test_response_score_rejects_values_outside_scale(score):
    with pytest.raises(ValidationError):
        source(relevance_score=score)


def test_missing_text_defaults_and_serialization():
    result = score_result("weights", source())
    serialized = SearchResult.model_validate_json(result.model_dump_json())
    assert serialized.relevance_score == 0
    assert serialized.relevance_reason == "No question terms matched the available title, snippet or passage."
