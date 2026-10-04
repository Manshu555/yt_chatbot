from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import subprocess

import pytest
import rag.transcript as transcript

VIDEO_ID = "aircAruvnKk"


def test_rolling_captions_remove_only_adjacent_repetition(monkeypatch):
    captions = [
        SimpleNamespace(text="Neural networks learn\nNeural networks learn"),
        SimpleNamespace(text="networks learn from examples"),
        SimpleNamespace(text="Weights control connections"),
        SimpleNamespace(text="Neural networks learn"),
    ]
    monkeypatch.setattr(transcript.webvtt, "read", lambda path: captions)
    assert transcript._parse_transcript("unused.vtt") == "Neural networks learn from examples Weights control connections Neural networks learn"


def test_regional_english_captions_use_isolated_files_and_bounded_command(monkeypatch):
    paths = []
    def download(command, **kwargs):
        base = Path(command[command.index("-o") + 1])
        path = base.with_name(base.name + ".en-US.vtt")
        path.write_text("test captions", encoding="utf-8")
        paths.append(path)
        assert kwargs["timeout"] == transcript.TRANSCRIPT_TIMEOUT_SECONDS
        assert "--write-sub" in command and "--write-auto-sub" in command
        assert "--sub-lang" not in command
        assert command[command.index("--extractor-args") + 1] == "youtube:skip=translated_subs"
        assert command[command.index("--extractor-retries") + 1] == "0"
        return SimpleNamespace(stderr="")
    monkeypatch.setattr(transcript.subprocess, "run", download)
    monkeypatch.setattr(transcript.webvtt, "read", lambda path: [SimpleNamespace(text="Neural networks learn.")])
    assert transcript.get_transcript(VIDEO_ID) == "Neural networks learn."
    assert paths and not paths[0].parent.exists()


@pytest.mark.parametrize("manual, automatic, expected", [
    ({"en": [], "en-US": []}, {"en-orig": [], "en-ar": []}, "en"),
    ({"en-US": []}, {"en-orig": []}, "en-US"),
    ({}, {"en-orig": [], "en-ar": []}, "en-orig"),
])
def test_default_downloader_selects_only_one_preferred_english_track(manual, automatic, expected):
    from yt_dlp import YoutubeDL
    def tracks(languages):
        return {language: [{"ext": "vtt", "url": "https://captions.test/" + language}] for language in languages}
    with YoutubeDL({"quiet": True, "writesubtitles": True, "writeautomaticsub": True, "subtitlesformat": "vtt"}) as downloader:
        selected = downloader.process_subtitles(VIDEO_ID, tracks(manual), tracks(automatic))
    assert list(selected) == [expected]


def test_download_timeout_returns_without_retrying(monkeypatch):
    run = Mock(side_effect=subprocess.TimeoutExpired("yt_dlp", 20))
    sleep = Mock()
    monkeypatch.setattr(transcript.subprocess, "run", run)
    monkeypatch.setattr(transcript.time, "sleep", sleep)
    assert transcript.get_transcript(VIDEO_ID) == ""
    assert run.call_count == 1
    sleep.assert_not_called()


def test_non_english_caption_is_not_used_as_english_evidence(monkeypatch):
    def download(command, **kwargs):
        base = Path(command[command.index("-o") + 1])
        base.with_name(base.name + ".fr.vtt").write_text("French captions", encoding="utf-8")
        return SimpleNamespace(stderr="")
    parse = Mock()
    monkeypatch.setattr(transcript.subprocess, "run", download)
    monkeypatch.setattr(transcript, "_parse_transcript", parse)
    assert transcript.get_transcript(VIDEO_ID) == ""
    parse.assert_not_called()


def test_youtube_429_does_not_sleep_after_last_attempt(monkeypatch):
    run = Mock(side_effect=subprocess.CalledProcessError(1, "yt_dlp", stderr="HTTP Error 429"))
    sleep = Mock()
    monkeypatch.setattr(transcript.subprocess, "run", run)
    monkeypatch.setattr(transcript.time, "sleep", sleep)
    assert transcript.get_transcript(VIDEO_ID) == ""
    assert run.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [5, 10]


def test_cookie_file_is_used_and_never_logged(monkeypatch, tmp_path, capsys):
    cookie = tmp_path / "cookies.txt"
    cookie.write_text("private cookie content", encoding="utf-8")
    monkeypatch.setenv("YOUTUBE_COOKIES_FILE", str(cookie))
    monkeypatch.setenv("YOUTUBE_COOKIES_BROWSER", "chrome")
    run = Mock(return_value=SimpleNamespace(stderr=""))
    monkeypatch.setattr(transcript.subprocess, "run", run)
    assert transcript.get_transcript(VIDEO_ID) == ""
    command = run.call_args.args[0]
    assert command[command.index("--cookies") + 1] == str(cookie)
    assert "--cookies-from-browser" not in command
    assert "private cookie content" not in capsys.readouterr().out


@pytest.mark.parametrize("video_id", ["../cookies", "", "https://youtube.com/watch?v=aircAruvnKk"])
def test_invalid_video_id_never_runs_downloader(monkeypatch, video_id):
    run = Mock()
    monkeypatch.setattr(transcript.subprocess, "run", run)
    assert transcript.get_transcript(video_id) == ""
    run.assert_not_called()
