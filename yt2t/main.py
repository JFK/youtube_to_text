# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
from typing import Optional

from pytube import YouTube
from moviepy.editor import AudioFileClip, VideoFileClip
import whisper
import torch
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from tqdm import tqdm
import wave
from pydub import AudioSegment


def on_progress_callback(stream, chunk, bytes_remaining) -> None:
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    tqdm_instance.update(bytes_downloaded - tqdm_instance.n)


def is_valid_youtube_url(url: str) -> bool:
    regex = re.compile(r"^(https?\:\/\/)?(www\.youtube\.com|youtu\.?be)\/.+$")
    return re.match(regex, url) is not None


def download_youtube_audio_as_mp3(url: str) -> Optional[str]:
    global tqdm_instance
    try:
        yt = YouTube(url, on_progress_callback=on_progress_callback)
        audio_stream = yt.streams.filter(only_audio=True).first()
        tqdm_instance = tqdm(
            total=audio_stream.filesize,
            unit="B",
            unit_scale=True,
            desc="Downloading youtube audio",
        )
        temp_file = audio_stream.download(
            skip_existing=False, filename=f"{yt.video_id}.mp4"
        )
        tqdm_instance.close()
        output_path = f"{yt.video_id}.mp3"
        audio_clip = AudioFileClip(temp_file)
        audio_clip.write_audiofile(output_path, codec="libmp3lame")
        os.remove(temp_file)
        return output_path
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        return None


def convert_mp3_to_wav(mp3_path: str) -> Optional[str]:
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        wav_path = mp3_path.replace(".mp3", ".wav")
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"Error converting MP3 to WAV: {e}")
        return None


def convert_mp4_to_wav(mp4_path: str) -> Optional[str]:
    try:
        video = VideoFileClip(mp4_path)
        audio = video.audio
        wav_path = mp4_path.replace(".mp4", ".wav")
        audio.write_audiofile(wav_path)
        return wav_path
    except Exception as e:
        print(f"Error converting MP4 to WAV: {e}")
        return None


def get_audio_length(audio_path: str) -> float:
    with wave.open(audio_path, "rb") as audio:
        frames = audio.getnframes()
        rate = audio.getframerate()
        duration = frames / float(rate)
        return duration


def convert_audio_to_text(audio_path: str, language: str = "ja") -> Optional[str]:
    try:
        if not audio_path.endswith(".wav"):
            if audio_path.endswith(".mp3"):
                audio_path = convert_mp3_to_wav(audio_path)
            elif audio_path.endswith(".mp4"):
                audio_path = convert_mp4_to_wav(audio_path)
            else:
                raise ValueError(
                    "Unsupported file format. Only MP3 and MP4 are supported."
                )

        audio_length = get_audio_length(audio_path)
        model_name = os.getenv("WHISPER_MODEL", "large-v3")
        model = whisper.load_model(
            model_name, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        with tqdm(
            total=audio_length, unit="s", unit_scale=True, desc="Transcribing audio"
        ) as pbar:
            result = model.transcribe(audio_path, language=language, verbose=False)
            pbar.update(audio_length)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None


def get_youtube_transcript(url: str, language: str = "ja") -> Optional[str]:
    try:
        video_id = YouTube(url).video_id
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript([language])
        print(f"Retrieving transcript for video: {video_id}")
        return " ".join([entry["text"] for entry in transcript.fetch()])
    except (TranscriptsDisabled, NoTranscriptFound):
        print(f"No transcript available for video: {video_id}")
        return None
    except Exception as e:
        print(f"Error retrieving transcript: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert audio files to text.")
    parser.add_argument(
        "source", type=str, help="YouTube video URL or local audio file path"
    )
    parser.add_argument(
        "-l", "--language", type=str, default="ja", help="Language of the audio"
    )
    parser.add_argument(
        "-ig",
        "--ignore_transcript",
        action="store_true",
        help="Ignore YouTube transcript and generate text from audio",
    )

    args = parser.parse_args()

    if os.path.isfile(args.source):
        if args.source.endswith(".mp4"):
            audio_file = convert_mp4_to_wav(args.source)
            if audio_file is None:
                print("Failed to convert MP4 to WAV.")
                return
        else:
            audio_file = args.source
        text = convert_audio_to_text(audio_file, args.language)
    elif is_valid_youtube_url(args.source):
        if args.ignore_transcript:
            text = None
        else:
            text = get_youtube_transcript(args.source, args.language)
        if text is None:
            audio_file = download_youtube_audio_as_mp3(args.source)
            if audio_file is None:
                print("Failed to download or convert YouTube video to audio.")
                return
            text = convert_audio_to_text(audio_file, args.language)
    else:
        print(
            "Invalid source. Please provide a valid YouTube URL or local audio file path."
        )
        return

    if text is not None:
        print(text)


if __name__ == "__main__":
    main()
