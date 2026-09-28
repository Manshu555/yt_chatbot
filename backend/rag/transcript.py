import os
import subprocess
import time
import webvtt

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5  # seconds

def get_transcript(video_id: str) -> str:
    """
    Downloads and parses the transcript for a given YouTube video ID.
    
    Supports multiple cookie strategies to bypass YouTube rate limits:
    1. YOUTUBE_COOKIES_FILE env var → uses a Netscape cookies.txt file
       (exported via browser extension like "Get cookies.txt LOCALLY")
    2. YOUTUBE_COOKIES_BROWSER env var → uses --cookies-from-browser 
       (requires browser to be fully closed; may fail on modern Chrome/Edge 
       due to DPAPI Application-Bound Encryption)
    3. No cookies → works for public videos when IP is not rate-limited
    
    Implements retry with exponential backoff for 429 errors.
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    subtitle_file = f"{video_id}.en.vtt"

    for attempt in range(MAX_RETRIES):
        try:
            cmd = [
                "yt-dlp",
                "--write-auto-sub",
                "--sub-lang", "en",
                "--skip-download",
                "--sub-format", "vtt",
                "-o", video_id,
            ]
            
            # Strategy 1: cookies.txt file (most reliable on Windows)
            cookie_file = os.environ.get("YOUTUBE_COOKIES_FILE")
            if cookie_file and os.path.isfile(cookie_file):
                cmd.extend(["--cookies", cookie_file])
            else:
                # Strategy 2: browser cookie extraction (fallback)
                cookie_browser = os.environ.get("YOUTUBE_COOKIES_BROWSER")
                if cookie_browser:
                    cmd.extend(["--cookies-from-browser", cookie_browser])
                
            cmd.append(video_url)
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Check if subtitle file exists
            if not os.path.exists(subtitle_file):
                # Check stderr for 429
                if "429" in (result.stderr or ""):
                    wait_time = RETRY_BACKOFF_BASE * (2 ** attempt)
                    print(f"YouTube 429 on attempt {attempt+1}/{MAX_RETRIES}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                print(f"No English subtitles found for video {video_id}")
                return ""

            # Parse the .vtt file
            transcript_text = ""
            for caption in webvtt.read(subtitle_file):
                transcript_text += caption.text + " "

            # Clean up the subtitle file
            os.remove(subtitle_file)

            print(f"Fetched transcript for video {video_id}: {len(transcript_text)} characters")
            return transcript_text.strip()

        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            if "429" in stderr:
                wait_time = RETRY_BACKOFF_BASE * (2 ** attempt)
                print(f"YouTube 429 on attempt {attempt+1}/{MAX_RETRIES}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            print(f"Error fetching subtitles for video {video_id} with yt-dlp: {stderr}")
            return ""
        except Exception as e:
            print(f"Unexpected error while fetching subtitles for video {video_id}: {e}")
            return ""

    print(f"Failed to fetch transcript for video {video_id} after {MAX_RETRIES} attempts (YouTube rate limited).")
    return ""
