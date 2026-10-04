import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
import webvtt
import sys
from config import TRANSCRIPT_TIMEOUT_SECONDS

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5  # seconds


def _parse_transcript(subtitle_file: str) -> str:
    # Rolling automatic captions repeat adjacent lines and overlapping words.
    words = []
    previous_line = None
    for caption in webvtt.read(subtitle_file):
        for line in caption.text.splitlines():
            line = " ".join(line.split())
            if not line or line == previous_line:
                continue
            previous_line = line
            incoming = line.split()
            overlap = 0
            for count in range(min(len(words), len(incoming)), 1, -1):
                if words[-count:] == incoming[:count]:
                    overlap = count
                    break
            words.extend(incoming[overlap:])
    return " ".join(words)


def get_transcript(video_id: str) -> str:
    """Fetch English captions with bounded downloads and optional cookie auth."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        return ""
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    # Isolated files avoid collisions between requests for the same video.
    with tempfile.TemporaryDirectory(prefix="yt_transcript_") as directory:
        subtitle_base = Path(directory) / video_id
        for attempt in range(MAX_RETRIES):
            cmd = [
                sys.executable, "-m", "yt_dlp",
                "--write-sub", "--write-auto-sub",
                # yt-dlp defaults to one preferred English track. A broad regex
                # also downloads translations, whose failure aborts the run.
                "--extractor-args", "youtube:skip=translated_subs",
                "--skip-download", "--sub-format", "vtt",
                "--socket-timeout", "10", "--retries", "0",
                "--fragment-retries", "0", "--extractor-retries", "0",
                "-o", str(subtitle_base),
            ]
            cookie_file = os.environ.get("YOUTUBE_COOKIES_FILE")
            if cookie_file and os.path.isfile(cookie_file):
                cmd.extend(["--cookies", cookie_file])
            else:
                cookie_browser = os.environ.get("YOUTUBE_COOKIES_BROWSER")
                if cookie_browser:
                    cmd.extend(["--cookies-from-browser", cookie_browser])
            cmd.append(video_url)

            try:
                result = subprocess.run(
                    cmd, check=True, capture_output=True, text=True,
                    timeout=TRANSCRIPT_TIMEOUT_SECONDS,
                )
                subtitles = sorted(Path(directory).glob(f"{video_id}.en*.vtt"))
                subtitles.sort(key=lambda path: path.name != f"{video_id}.en.vtt")
                if subtitles:
                    transcript = _parse_transcript(str(subtitles[0]))
                    print(f"Fetched transcript for video {video_id}: {len(transcript)} characters")
                    return transcript
                stderr = result.stderr or ""
                if "429" not in stderr:
                    print(f"No English subtitles found for video {video_id}")
                    return ""
            except subprocess.TimeoutExpired:
                print(f"YouTube transcript download timed out for video {video_id}.")
                return ""
            except subprocess.CalledProcessError as error:
                if "429" not in (error.stderr or ""):
                    print(f"YouTube transcript download failed for video {video_id} (exit {error.returncode}).")
                    return ""
            except Exception as error:
                print(f"Transcript extraction failed ({type(error).__name__}).")
                return ""

            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BACKOFF_BASE * (2 ** attempt)
                print(f"YouTube 429 on attempt {attempt + 1}/{MAX_RETRIES}. Retrying in {delay}s...")
                time.sleep(delay)

    print(f"YouTube rate limit exhausted after {MAX_RETRIES} attempts for video {video_id}.")
    return ""
