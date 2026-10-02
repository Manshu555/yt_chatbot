import argparse
import tempfile
from pathlib import Path

from backend.rag.openai_batch import (
    create_batch,
    create_batch_jsonl,
    create_batch_request,
    download_batch_errors,
    download_batch_results,
    metrics_from_batch,
    parse_batch_results,
    upload_batch_file,
    wait_for_batch,
)


def main():
    parser = argparse.ArgumentParser(description="Small OpenAI Batch API smoke test.")
    parser.add_argument("--wait", action="store_true", help="Wait for completion and download results.")
    parser.add_argument("--poll-interval", type=int, default=60, help="Polling interval in seconds when --wait is used.")
    args = parser.parse_args()

    requests = [
        create_batch_request(
            "batch_smoke_001",
            "Reply with one short sentence explaining what a transcript is.",
            max_output_tokens=80,
        ),
        create_batch_request(
            "batch_smoke_002",
            "Reply with three concise bullet points about why offline evals are useful.",
            max_output_tokens=120,
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "openai_batch_smoke.jsonl"
        create_batch_jsonl(requests, output_path=str(jsonl_path))

        uploaded = upload_batch_file(str(jsonl_path))
        batch = create_batch(uploaded.id, metadata={"purpose": "yt_chatbot_batch_smoke"})

        print(f"BATCH ID: {batch.id}")
        print(f"STATUS: {batch.status}")
        print(f"REQUEST COUNT: {len(requests)}")

        if not args.wait:
            print("Created batch only. Re-run with --wait to poll and download results.")
            return

        final_batch = wait_for_batch(batch.id, poll_interval_seconds=args.poll_interval)
        metrics = metrics_from_batch(final_batch)
        print(f"STATUS: {final_batch.status}")
        print(f"COMPLETED: {metrics.completed_count}")
        print(f"FAILED: {metrics.failed_count}")

        if final_batch.status != "completed":
            errors = download_batch_errors(final_batch)
            if errors:
                print("ERRORS:")
                print(errors)
            return

        output_jsonl = download_batch_results(final_batch)
        parsed = parse_batch_results(output_jsonl)
        print(f"SUCCESSFUL OUTPUTS: {parsed.metrics.successful_outputs}")
        print(f"FAILED OUTPUTS: {parsed.metrics.failed_outputs}")
        print(f"TOTAL TOKENS: {parsed.metrics.total_tokens}")
        print("RESULTS:")
        for custom_id, output in parsed.outputs.items():
            print(f"{custom_id}: {output.text or output.error}")


if __name__ == "__main__":
    main()
