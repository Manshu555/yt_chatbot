from youtube_transcript_api import YouTubeTranscriptApi

video_id = "FuqNluMTIR8"
transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
print("Available transcripts:")
for transcript in transcripts:
    print(f"- Language: {transcript.language} ({transcript.language_code}), manually created: {not transcript.is_generated}")

# Then try to fetch a specific transcript explicitly:
try:
    transcript = transcripts.find_manually_created_transcript(['en'])
except:
    transcript = transcripts.find_generated_transcript(['en'])

transcript_list = transcript.fetch()
print(transcript_list)
