import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type

from pydantic import BaseModel, ValidationError

try:
    from config import OPENAI_API_KEY, OPENAI_BATCH_MODEL
except ModuleNotFoundError:
    from backend.config import OPENAI_API_KEY, OPENAI_BATCH_MODEL

BATCH_ENDPOINT = "/v1/responses"
DEFAULT_COMPLETION_WINDOW = "24h"
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


@dataclass
class BatchRequest:
    custom_id: str
    body: Dict[str, Any]
    method: str = "POST"
    url: str = BATCH_ENDPOINT


@dataclass
class BatchMetrics:
    batch_id: Optional[str] = None
    request_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    created_at: Optional[float] = None
    completed_at: Optional[float] = None
    total_duration_seconds: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    successful_outputs: int = 0
    failed_outputs: int = 0


@dataclass
class BatchOutput:
    custom_id: str
    status_code: Optional[int]
    text: Optional[str] = None
    parsed: Optional[Any] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class BatchParseResult:
    outputs: Dict[str, BatchOutput]
    metrics: BatchMetrics


@dataclass
class BatchBenchmarkComparison:
    live_request_count: int
    batch_request_count: int
    live_duration_seconds: Optional[float]
    batch_duration_seconds: Optional[float]
    live_successful_requests: int
    batch_successful_requests: int
    live_failed_requests: int
    batch_failed_requests: int
    live_total_tokens: Optional[int] = None
    batch_total_tokens: Optional[int] = None
    note: str = "Batch API is asynchronous and is not expected to be faster for interactive requests."


class BatchAPIError(RuntimeError):
    pass


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def make_custom_id(prefix: str, index: int, width: int = 3) -> str:
    clean_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", prefix).strip("_").lower()
    if not clean_prefix:
        clean_prefix = "request"
    return f"{clean_prefix}_{index:0{width}d}"


def _schema_format(schema_model: Type[BaseModel]) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "name": schema_model.__name__,
        "schema": schema_model.model_json_schema(),
        "strict": True,
    }


def create_response_body(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    schema_model: Optional[Type[BaseModel]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model or OPENAI_BATCH_MODEL,
        "input": prompt,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if max_output_tokens is not None:
        body["max_output_tokens"] = max_output_tokens
    if schema_model is not None:
        body["text"] = {"format": _schema_format(schema_model)}
    if metadata:
        body["metadata"] = metadata
    return body


def create_batch_request(
    custom_id: str,
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    schema_model: Optional[Type[BaseModel]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> BatchRequest:
    return BatchRequest(
        custom_id=custom_id,
        body=create_response_body(
            prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            schema_model=schema_model,
            metadata=metadata,
        ),
    )


def create_batch_jsonl(requests: Iterable[BatchRequest], output_path: Optional[str] = None) -> str:
    lines: List[str] = []
    seen_custom_ids = set()

    for request in requests:
        if not request.custom_id:
            raise ValueError("Every batch request requires a custom_id.")
        if request.custom_id in seen_custom_ids:
            raise ValueError(f"Duplicate custom_id: {request.custom_id}")
        seen_custom_ids.add(request.custom_id)

        payload = {
            "custom_id": request.custom_id,
            "method": request.method,
            "url": request.url,
            "body": request.body,
        }
        if payload["method"] != "POST":
            raise ValueError("OpenAI Batch Responses requests must use POST.")
        if payload["url"] != BATCH_ENDPOINT:
            raise ValueError(f"Batch request url must be {BATCH_ENDPOINT}.")
        lines.append(json.dumps(payload, ensure_ascii=False))

    jsonl = "\n".join(lines)
    if lines:
        jsonl += "\n"

    if output_path:
        Path(output_path).write_text(jsonl, encoding="utf-8")
    return jsonl


def _get_openai_client(api_key: Optional[str] = None):
    key = api_key or OPENAI_API_KEY
    if not key:
        raise BatchAPIError("OPENAI_API_KEY is not configured.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise BatchAPIError("The openai package is required for Batch API calls.") from exc
    return OpenAI(api_key=key)


def upload_batch_file(jsonl_path: str, *, client: Any = None) -> Any:
    api = client or _get_openai_client()
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(jsonl_path)
    if path.suffix.lower() != ".jsonl":
        raise ValueError("Batch input file must use the .jsonl extension.")

    with path.open("rb") as file_handle:
        return api.files.create(file=file_handle, purpose="batch")


def create_batch(
    input_file_id: str,
    *,
    client: Any = None,
    endpoint: str = BATCH_ENDPOINT,
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    if endpoint != BATCH_ENDPOINT:
        raise ValueError(f"This adapter supports only {BATCH_ENDPOINT}.")
    api = client or _get_openai_client()
    return api.batches.create(
        input_file_id=input_file_id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=metadata or {},
    )


def get_batch_status(batch_id: str, *, client: Any = None) -> Any:
    api = client or _get_openai_client()
    return api.batches.retrieve(batch_id)


def wait_for_batch(
    batch_id: str,
    *,
    client: Any = None,
    poll_interval_seconds: int = 60,
    timeout_seconds: Optional[int] = None,
) -> Any:
    if poll_interval_seconds < 10:
        raise ValueError("Use a poll interval of at least 10 seconds for Batch API waits.")

    api = client or _get_openai_client()
    started = time.monotonic()
    while True:
        batch = get_batch_status(batch_id, client=api)
        status = str(_get(batch, "status", "")).lower()
        if status in TERMINAL_STATUSES:
            return batch
        if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"Timed out waiting for batch {batch_id}; latest status={status}.")
        time.sleep(poll_interval_seconds)


def download_file(file_id: str, *, client: Any = None, output_path: Optional[str] = None) -> str:
    if not file_id:
        raise BatchAPIError("Missing file id.")
    api = client or _get_openai_client()
    content = api.files.content(file_id)

    if hasattr(content, "text"):
        text = content.text
    elif hasattr(content, "content"):
        raw = content.content
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    elif hasattr(content, "read"):
        raw = content.read()
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    else:
        text = str(content)

    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
    return text


def download_batch_results(batch_or_id: Any, *, client: Any = None, output_path: Optional[str] = None) -> str:
    batch = get_batch_status(batch_or_id, client=client) if isinstance(batch_or_id, str) else batch_or_id
    output_file_id = _get(batch, "output_file_id")
    if not output_file_id:
        raise BatchAPIError("Batch has no output_file_id.")
    return download_file(output_file_id, client=client, output_path=output_path)


def download_batch_errors(batch_or_id: Any, *, client: Any = None, output_path: Optional[str] = None) -> Optional[str]:
    batch = get_batch_status(batch_or_id, client=client) if isinstance(batch_or_id, str) else batch_or_id
    error_file_id = _get(batch, "error_file_id")
    if not error_file_id:
        return None
    return download_file(error_file_id, client=client, output_path=output_path)


def _extract_response_text(body: Dict[str, Any]) -> Optional[str]:
    if body.get("output_text"):
        return body["output_text"]

    for item in body.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and content.get("text") is not None:
                return content["text"]

    choices = body.get("choices") or []
    if choices:
        message = choices[0].get("message", {})
        if message.get("content"):
            return message["content"]
    return None


def _usage_tokens(body: Dict[str, Any]) -> Dict[str, int]:
    usage = body.get("usage") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def parse_batch_results(
    jsonl_content: str,
    *,
    schema_by_custom_id: Optional[Dict[str, Type[BaseModel]]] = None,
    default_schema: Optional[Type[BaseModel]] = None,
) -> BatchParseResult:
    outputs: Dict[str, BatchOutput] = {}
    metrics = BatchMetrics()

    for line_number, raw_line in enumerate(jsonl_content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            custom_id = f"malformed_line_{line_number}"
            outputs[custom_id] = BatchOutput(
                custom_id=custom_id,
                status_code=None,
                raw={"line": raw_line},
                error=f"Malformed JSONL output: {exc}",
            )
            metrics.failed_outputs += 1
            continue

        custom_id = item.get("custom_id") or f"missing_custom_id_{line_number}"
        response = item.get("response") or {}
        error = item.get("error")
        status_code = response.get("status_code")
        body = response.get("body") or {}
        metrics.request_count += 1

        if error or not status_code or status_code >= 400:
            outputs[custom_id] = BatchOutput(
                custom_id=custom_id,
                status_code=status_code,
                raw=item,
                error=json.dumps(error or body, ensure_ascii=False),
            )
            metrics.failed_outputs += 1
            continue

        tokens = _usage_tokens(body)
        metrics.input_tokens += tokens["input_tokens"]
        metrics.output_tokens += tokens["output_tokens"]
        metrics.total_tokens += tokens["total_tokens"]

        text = _extract_response_text(body)
        schema_model = (schema_by_custom_id or {}).get(custom_id, default_schema)
        parsed = None
        parse_error = None

        if schema_model is not None:
            if not text:
                parse_error = "Structured output response did not contain text."
            else:
                try:
                    parsed = schema_model.model_validate_json(text)
                except ValidationError as exc:
                    parse_error = f"Structured output validation failed: {exc}"
                except ValueError as exc:
                    parse_error = f"Structured output JSON parsing failed: {exc}"

        outputs[custom_id] = BatchOutput(
            custom_id=custom_id,
            status_code=status_code,
            text=text,
            parsed=parsed,
            raw=item,
            error=parse_error,
        )
        if parse_error:
            metrics.failed_outputs += 1
        else:
            metrics.successful_outputs += 1

    metrics.completed_count = metrics.successful_outputs
    metrics.failed_count = metrics.failed_outputs
    return BatchParseResult(outputs=outputs, metrics=metrics)


def metrics_from_batch(batch: Any) -> BatchMetrics:
    request_counts = _get(batch, "request_counts") or {}
    created_at = _get(batch, "created_at")
    completed_at = _get(batch, "completed_at")
    duration = None
    if created_at and completed_at:
        duration = float(completed_at) - float(created_at)

    return BatchMetrics(
        batch_id=_get(batch, "id"),
        request_count=int(_get(request_counts, "total", 0) or 0),
        completed_count=int(_get(request_counts, "completed", 0) or 0),
        failed_count=int(_get(request_counts, "failed", 0) or 0),
        created_at=created_at,
        completed_at=completed_at,
        total_duration_seconds=duration,
    )


def compare_live_vs_batch(
    *,
    live_request_count: int,
    live_duration_seconds: Optional[float],
    live_successful_requests: int,
    live_failed_requests: int,
    batch_metrics: BatchMetrics,
    live_total_tokens: Optional[int] = None,
) -> BatchBenchmarkComparison:
    return BatchBenchmarkComparison(
        live_request_count=live_request_count,
        batch_request_count=batch_metrics.request_count,
        live_duration_seconds=live_duration_seconds,
        batch_duration_seconds=batch_metrics.total_duration_seconds,
        live_successful_requests=live_successful_requests,
        batch_successful_requests=batch_metrics.completed_count or batch_metrics.successful_outputs,
        live_failed_requests=live_failed_requests,
        batch_failed_requests=batch_metrics.failed_count or batch_metrics.failed_outputs,
        live_total_tokens=live_total_tokens,
        batch_total_tokens=batch_metrics.total_tokens or None,
    )


def create_summary_batch_request(
    custom_id: str,
    query: str,
    retrieved_evidence: Iterable[Any],
    *,
    model: Optional[str] = None,
) -> BatchRequest:
    context = "\n".join(getattr(chunk, "text", str(chunk)) for chunk in retrieved_evidence)
    prompt = f"""Using the following YouTube transcript evidence, answer the question.
If the answer is not in the evidence, say that it could not be found in the transcript.

QUESTION:
{query}

TRANSCRIPT EVIDENCE:
{context}
"""
    return create_batch_request(
        custom_id,
        prompt,
        model=model,
        temperature=0.7,
        max_output_tokens=1024,
        metadata={"workload": "summary_evaluation"},
    )


def create_claim_extraction_batch_request(
    custom_id: str,
    query: str,
    video_evidence: Iterable[Any],
    *,
    model: Optional[str] = None,
) -> BatchRequest:
    try:
        from prompts.claims import CLAIM_EXTRACTION_PROMPT
        from validation.claim_extractor import ClaimExtractionResponse
    except ModuleNotFoundError:
        from backend.prompts.claims import CLAIM_EXTRACTION_PROMPT
        from backend.validation.claim_extractor import ClaimExtractionResponse

    context = "\n".join(getattr(chunk, "text", str(chunk)) for chunk in video_evidence)
    prompt = CLAIM_EXTRACTION_PROMPT.format(query=query, context=context)
    return create_batch_request(
        custom_id,
        prompt,
        model=model,
        temperature=0.1,
        max_output_tokens=300,
        schema_model=ClaimExtractionResponse,
        metadata={"workload": "claim_extraction_evaluation"},
    )


def create_validation_batch_request(
    custom_id: str,
    claims: Iterable[Any],
    evidence_by_id: Dict[str, Any],
    *,
    model: Optional[str] = None,
) -> BatchRequest:
    try:
        from validation.validator import BatchValidationResponse
    except ModuleNotFoundError:
        from backend.validation.validator import BatchValidationResponse

    prompt = "You are a strict fact-checking assistant. Evaluate each claim against the provided evidence.\n\n"
    prompt += "CLAIMS:\n"
    for claim in claims:
        prompt += f"[{claim.id}] {claim.text}\n"
    prompt += "\nEVIDENCE:\n"
    for ev_id, evidence in evidence_by_id.items():
        title = getattr(evidence, "source_title", "")
        passage = getattr(evidence, "passage", str(evidence))
        prompt += f"[{ev_id}] Source: {title} - Passage: {passage}\n"
    prompt += "\nFor each claim, determine if the evidence as a whole supports it, contradicts it, or provides no verifiable information. Return the results in the requested JSON schema."

    return create_batch_request(
        custom_id,
        prompt,
        model=model,
        temperature=0.1,
        max_output_tokens=1000,
        schema_model=BatchValidationResponse,
        metadata={"workload": "batch_validation_evaluation"},
    )


def create_batch_input(
    requests: Iterable[BatchRequest],
    output_path: str,
) -> str:
    return create_batch_jsonl(requests, output_path=output_path)
