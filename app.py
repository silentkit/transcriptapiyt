from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from gradio_client import Client
from pytubefix import YouTube
from pytubefix.cli import on_progress
import os
import re


app = Flask(__name__)

# Whisper JAX setup
API_URL = "https://sanchit-gandhi-whisper-jax.hf.space/"
client = Client(API_URL)

def transcribe_audio(audio_path, task="transcribe", return_timestamps=False):
    """Function to transcribe an audio file using the Whisper JAX endpoint."""
    if task not in ["transcribe", "translate"]:
        raise ValueError("task should be one of 'transcribe' or 'translate'.")
    text, runtime = client.predict(
        audio_path,
        task,
        return_timestamps,
        api_name="/predict_1",
    )
    return text

def extract_video_id(url):
    youtube_regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([^\s&]+)"
    match = re.search(youtube_regex, url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except TranscriptsDisabled:
        print("Subtitles are disabled for this video. Falling back to Whisper JAX model.")
        url = f"https://www.youtube.com/watch?v={video_id}"
        return manual_transcribe(url)

def manual_transcribe(url):
    yt = YouTube(url, on_progress_callback=on_progress)
    print(f"Downloading audio for: {yt.title}")
    
    # Get the audio stream and download it
    ys = yt.streams.get_audio_only()
    audio_file = ys.download(filename_suffix=".mp3")  # Save as mp3
    
    # Transcribe the downloaded audio
    transcription = transcribe_audio(audio_file)
    
    # Clean up the downloaded file
    os.remove(audio_file)
    
    return transcription

@app.route('/transcribe', methods=['GET'])
def transcribe_endpoint():
    video_url = request.args.get('url')
    if not video_url:
        return jsonify({"error": "No URL provided"}), 400
    
    try:
        video_id = extract_video_id(video_url)
        transcript = get_transcript(video_id)
        return jsonify({"transcript": transcript})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
