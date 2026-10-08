import os

import yt_dlp
from pydub import AudioSegment

# NOTE: yt-dlp and pydub both require ffmpeg to be installed and on your PATH.

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str, output_dir: str = DOWNLOAD_DIR) -> str:
    """Download audio from a URL (YouTube or any site yt-dlp supports) as WAV."""
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "restrictfilenames": True,  # avoid slashes, emojis, odd characters
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        try:
            # Most reliable: the final path after post-processing
            filename = info["requested_downloads"][0]["filepath"]
        except (KeyError, IndexError, TypeError):
            filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"

    return filename


def convert_to_wav(input_path: str, output_dir: str = DOWNLOAD_DIR) -> str:
    """Convert any audio file to mono, 16 kHz WAV (safe if input is already in output_dir)."""
    os.makedirs(output_dir, exist_ok=True)

    name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{name}.wav")

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)

    if os.path.abspath(input_path) == os.path.abspath(output_path):
        # Same file: write to a temp file, then replace, to avoid corruption
        tmp_path = os.path.join(output_dir, f"{name}.tmp.wav")
        audio.export(tmp_path, format="wav")
        os.replace(tmp_path, output_path)
    else:
        audio.export(output_path, format="wav")

    return output_path


def chunk_audio(
    wav_path: str,
    chunk_min: int = 10,
    output_dir: str = DOWNLOAD_DIR,
    overlap_sec: float = 0,
) -> list:
    """Split a WAV into fixed-length chunks. Returns the list of chunk file paths.

    overlap_sec adds extra audio to the end of each chunk (except it never
    runs past the end of the file), which helps avoid cutting words in half.
    """
    os.makedirs(output_dir, exist_ok=True)

    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = int(chunk_min * 60 * 1000)
    overlap_ms = int(overlap_sec * 1000)
    base = os.path.splitext(os.path.basename(wav_path))[0]
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms + overlap_ms]
        chunk_path = os.path.join(output_dir, f"{base}_chunk_{i}.wav")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected URL. Downloading audio...")
        downloaded_path = download_youtube_audio(source)
    else:
        print("Detected local file.")
        downloaded_path = source

    # Normalize both paths the same way: mono, 16 kHz WAV
    print("Converting to mono 16 kHz WAV...")
    wav_path = convert_to_wav(downloaded_path)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_pipeline.py <youtube_url_or_local_file>")
        sys.exit(1)

    result = process_input(sys.argv[1])
    for path in result:
        print(path)