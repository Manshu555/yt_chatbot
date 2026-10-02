import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from backend.rag.openai_batch import (
    BATCH_ENDPOINT,
    BatchAPIError,
    BatchMetrics,
    BatchRequest,
    compare_live_vs_batch,
    create_batch,
    create_batch_jsonl,
    create_batch_request,
    download_batch_results,
    get_batch_status,
    make_custom_id,
    metrics_from_batch,
    parse_batch_results,
    upload_batch_file,
    wait_for_batch,
)


class TinyResponse(BaseModel):
    value: str


def test_make_custom_id_is_deterministic_and_sanitized():
    assert make_custom_id("YouTube Summary", 1) == "youtube_summary_001"
    assert make_custom_id("claim/validation", 12) == "claim_validation_012"


def test_create_batch_jsonl_uses_responses_endpoint():
    request = create_batch_request(
        "youtube_summary_001",
        "Summarize this.",
        model="gpt-test",
        max_output_tokens=100,
    )

    jsonl = create_batch_jsonl([request])
    line = json.loads(jsonl)

    assert line["custom_id"] == "youtube_summary_001"
    assert line["method"] == "POST"
    assert line["url"] == BATCH_ENDPOINT
    assert line["body"]["model"] == "gpt-test"
    assert line["body"]["input"] == "Summarize this."
    assert line["body"]["max_output_tokens"] == 100


def test_create_batch_jsonl_rejects_duplicate_custom_ids():
    requests = [
        BatchRequest(custom_id="same", body={"model": "m", "input": "a"}),
        BatchRequest(custom_id="same", body={"model": "m", "input": "b"}),
    ]
    with pytest.raises(ValueError, match="Duplicate custom_id"):
        create_batch_jsonl(requests)


def test_create_batch_request_adds_structured_output_schema():
    request = create_batch_request(
        "claim_extraction_001",
        "Extract claims.",
        schema_model=TinyResponse,
    )

    schema_format = request.body["text"]["format"]
    assert schema_format["type"] == "json_schema"
    assert schema_format["name"] == "TinyResponse"
    assert schema_format["strict"] is True
    assert schema_format["schema"]["properties"]["value"]["type"] == "string"


def test_upload_batch_file(tmp_path):
    path = tmp_path / "batch.jsonl"
    path.write_text('{"custom_id":"x"}\n', encoding="utf-8")
    client = MagicMock()
    client.files.create.return_value = SimpleNamespace(id="file_123")

    uploaded = upload_batch_file(str(path), client=client)

    assert uploaded.id == "file_123"
    assert client.files.create.call_args.kwargs["purpose"] == "batch"


def test_create_batch_calls_openai_batches_api():
    client = MagicMock()
    client.batches.create.return_value = SimpleNamespace(id="batch_123", status="validating")

    batch = create_batch("file_123", client=client, metadata={"suite": "unit"})

    assert batch.id == "batch_123"
    assert client.batches.create.call_args.kwargs["input_file_id"] == "file_123"
    assert client.batches.create.call_args.kwargs["endpoint"] == "/v1/responses"
    assert client.batches.create.call_args.kwargs["completion_window"] == "24h"


def test_get_batch_status():
    client = MagicMock()
    client.batches.retrieve.return_value = SimpleNamespace(id="batch_123", status="completed")

    batch = get_batch_status("batch_123", client=client)

    assert batch.status == "completed"
    client.batches.retrieve.assert_called_once_with("batch_123")


def test_wait_for_batch_rejects_aggressive_polling():
    with pytest.raises(ValueError, match="at least 10 seconds"):
        wait_for_batch("batch_123", client=MagicMock(), poll_interval_seconds=1)


def test_download_batch_results_requires_output_file():
    with pytest.raises(BatchAPIError, match="output_file_id"):
        download_batch_results(SimpleNamespace(id="batch_123", output_file_id=None), client=MagicMock())


def test_parse_batch_results_maps_by_custom_id_and_parses_schema():
    output_line = {
        "custom_id": "claim_extraction_001",
        "response": {
            "status_code": 200,
            "body": {
                "output_text": json.dumps({"value": "ok"}),
                "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
            },
        },
    }

    parsed = parse_batch_results(
        json.dumps(output_line) + "\n",
        schema_by_custom_id={"claim_extraction_001": TinyResponse},
    )

    assert parsed.outputs["claim_extraction_001"].parsed == TinyResponse(value="ok")
    assert parsed.metrics.successful_outputs == 1
    assert parsed.metrics.total_tokens == 7


def test_parse_batch_results_tracks_failed_request():
    output_line = {
        "custom_id": "bad_001",
        "response": {"status_code": 500, "body": {"error": "boom"}},
        "error": {"message": "failed"},
    }

    parsed = parse_batch_results(json.dumps(output_line) + "\n")

    assert parsed.outputs["bad_001"].error
    assert parsed.metrics.failed_outputs == 1


def test_parse_batch_results_tracks_malformed_structured_output():
    output_line = {
        "custom_id": "structured_001",
        "response": {
            "status_code": 200,
            "body": {"output_text": "not json"},
        },
    }

    parsed = parse_batch_results(json.dumps(output_line) + "\n", default_schema=TinyResponse)

    assert parsed.outputs["structured_001"].parsed is None
    assert "Structured output" in parsed.outputs["structured_001"].error
    assert parsed.metrics.failed_outputs == 1


def test_metrics_from_batch():
    batch = SimpleNamespace(
        id="batch_123",
        created_at=100.0,
        completed_at=160.0,
        request_counts=SimpleNamespace(total=3, completed=2, failed=1),
    )

    metrics = metrics_from_batch(batch)

    assert metrics.batch_id == "batch_123"
    assert metrics.request_count == 3
    assert metrics.completed_count == 2
    assert metrics.failed_count == 1
    assert metrics.total_duration_seconds == 60.0


def test_compare_live_vs_batch_summary():
    batch_metrics = BatchMetrics(
        request_count=2,
        completed_count=2,
        failed_count=0,
        total_duration_seconds=120.0,
        total_tokens=42,
    )

    comparison = compare_live_vs_batch(
        live_request_count=2,
        live_duration_seconds=5.0,
        live_successful_requests=2,
        live_failed_requests=0,
        live_total_tokens=40,
        batch_metrics=batch_metrics,
    )

    assert comparison.live_request_count == 2
    assert comparison.batch_request_count == 2
    assert comparison.batch_total_tokens == 42
    assert "asynchronous" in comparison.note
