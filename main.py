import os
import subprocess
import sys
import whisper
from uvr.separate import _audio_pre_

# MDX-Net and yt-dlp are external command-line tools or separate libs, 
# so here we'll integrate them via subprocess calls.

def run_ffmpeg_extract(input_path, output_wav):
    print(f"Extracting audio from {input_path} to {output_wav} ...")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-ac", "2", "-ar", "44100",
        "-loglevel", "error", output_wav
    ]
    subprocess.run(cmd, check=True)
    print("Audio extraction completed.")

def run_mdxnet_separation(input_wav, output_dir):
    """
    Run MDX-Net vocal separation by directly calling the separation logic from uvr/separate.py.
    """
    print(f"Running MDX-Net vocal separation on {input_wav} ...")
    
    model_path = "uvr/uvr5_weights/2_HP-UVR.pth"
    device = "cuda"
    is_half = True
    pre_fun = _audio_pre_(model_path=model_path, device=device, is_half=is_half)
    pre_fun._path_audio_(input_wav, output_dir, output_dir)
    print("Vocal separation done.")

def run_whisper_transcribe(input_wav, output_srt):
    print(f"Transcribing and timing lyrics in {input_wav} ...")
    model = whisper.load_model("base")
    result = model.transcribe(input_wav, word_timestamps=True)
    with open(output_srt, "w", encoding="utf-8") as f:
        segments = result["segments"]
        for i, segment in enumerate(segments, 1):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            f.write(f"{i}\n")
            f.write(f"{to_srt_timestamp(start)} --> {to_srt_timestamp(end)}\n")
            f.write(f"{text}\n\n")
    print(f"Transcription saved to {output_srt}")

def to_srt_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def create_karaoke_video(instr_wav, srt_file, output_mp4):
    print("Combining instrumental audio & lyrics subtitles into karaoke video...")
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", instr_wav
    ]
    duration_sec = float(subprocess.check_output(duration_cmd).decode().strip())

    video_cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=640x480:d={duration_sec}",
        "-i", instr_wav,
        "-vf", f"subtitles={srt_file}:force_style='Alignment=10,Fontsize=24,PrimaryColour=&HFFFFFF&'",
        "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_mp4
    ]
    subprocess.run(video_cmd, check=True)
    print(f"Karaoke video created at {output_mp4}")

def download_audio_with_ytdlp(youtube_url, output_path):
    print(f"Downloading audio from YouTube URL: {youtube_url}")
    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "-x", "--audio-format", "wav",
        "-o", output_path,
        youtube_url
    ]
    subprocess.run(cmd, check=True)
    print(f"Downloaded and converted audio saved to: {output_path}")

def prompt_user(prompt):
    while True:
        answer = input(prompt + " (y/n): ").strip().lower()
        if answer in ("y", "n"):
            return answer == "y"
        print("Please enter 'y' or 'n'.")

def main():
    print("=== Karaoke Version Pipeline (MDX-Net + yt-dlp) ===")

    use_yt = prompt_user("Will you provide a YouTube URL to download the audio?")

    if use_yt:
        youtube_url = input("Enter YouTube URL: ").strip()
        base_name = "downloaded_audio"
        temp_wav = f"{base_name}.wav"
        download_audio_with_ytdlp(youtube_url, temp_wav)
    else:
        input_path = input("Enter path to input audio/video file: ").strip()
        if not os.path.isfile(input_path):
            print(f"File not found: {input_path}")
            sys.exit(1)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        temp_wav = f"{base_name}_extracted.wav"
        if prompt_user("Step 1: Extract audio to WAV?"):
            run_ffmpeg_extract(input_path, temp_wav)
        else:
            print("Skipping audio extraction step.")

    output_dir = f"{base_name}_separated"
    instr_wav = os.path.join(output_dir, f"instrument_{os.path.basename(temp_wav)}")
    srt_file = f"{base_name}.srt"
    output_karaoke = f"{base_name}_karaoke.mp4"

    if prompt_user("Step 2: Perform vocal separation with MDX-Net?"):
        run_mdxnet_separation(temp_wav, output_dir)
    else:
        print("Skipping vocal separation step.")

    if prompt_user("Step 3: Transcribe lyrics and generate subtitles?"):
        run_whisper_transcribe(temp_wav, srt_file)
    else:
        print("Skipping transcription step.")

    if prompt_user("Step 4: Assemble karaoke video with subtitles?"):
        # Confirm accompaniment wav existence:
        if not os.path.isfile(instr_wav):
            print(f"Cannot find accompaniment audio at {instr_wav}. Please check separation output.")
            sys.exit(1)
        create_karaoke_video(instr_wav, srt_file, output_karaoke)
    else:
        print("Skipping video assembly step.")

    print("Pipeline complete.")

if __name__ == "__main__":
    main()
