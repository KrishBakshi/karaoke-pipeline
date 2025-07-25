import os
import subprocess
import sys
import whisper
from uvr.separate import _audio_pre_
import hashlib

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

def run_whisper_transcribe_karaoke(input_wav, output_ass):
    import whisper
    model = whisper.load_model("base")
    result = model.transcribe(input_wav, word_timestamps=True)
    segments = result["segments"]

    with open(output_ass, "w", encoding="utf-8") as f:
        # Write ASS header for Full HD, two styles, large font, center-center
        f.write("[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        # White Default, Yellow Highlight, both large, center-center (Alignment=5)
        f.write("Style: Default,Arial,60,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,0,5,10,10,10,1\n")
        f.write("Style: Highlight,Arial,60,&H0000FFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,0,5,10,10,10,1\n\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        for segment in segments:
            words = segment.get("words", [])
            # Group words into lines of <=10 chars (by word, not splitting words)
            line_words = []
            line_len = 0
            line_start_idx = 0
            for idx, word in enumerate(words):
                word_text = word["word"]
                add_len = len(word_text) + (1 if line_words else 0)
                if line_len + add_len > 10 and line_words:
                    # Process the current line
                    for w_idx, w in enumerate(line_words):
                        # Highlight only the current word
                        highlight_line = ""
                        for i, lw in enumerate(line_words):
                            if i == w_idx:
                                highlight_line += "{\\rHighlight}" + lw + "{\\rDefault}"
                            else:
                                highlight_line += lw
                            if i != len(line_words) - 1:
                                highlight_line += " "
                        w_start = words[line_start_idx + w_idx]["start"]
                        w_end = words[line_start_idx + w_idx]["end"]
                        f.write(f"Dialogue: 0,{to_ass_timestamp(w_start)},{to_ass_timestamp(w_end)},Default,,0,0,0,,{highlight_line}\n")
                    # Start new line
                    line_words = [word_text]
                    line_len = len(word_text)
                    line_start_idx = idx
                else:
                    if line_words:
                        line_len += 1  # space
                    line_words.append(word_text)
                    line_len += len(word_text)
            # Process the last line
            if line_words:
                for w_idx, w in enumerate(line_words):
                    highlight_line = ""
                    for i, lw in enumerate(line_words):
                        if i == w_idx:
                            highlight_line += "{\\rHighlight}" + lw + "{\\rDefault}"
                        else:
                            highlight_line += lw
                        if i != len(line_words) - 1:
                            highlight_line += " "
                    w_start = words[line_start_idx + w_idx]["start"]
                    w_end = words[line_start_idx + w_idx]["end"]
                    f.write(f"Dialogue: 0,{to_ass_timestamp(w_start)},{to_ass_timestamp(w_end)},Default,,0,0,0,,{highlight_line}\n")

def to_ass_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h:01d}:{m:02d}:{s:02d}.{cs:02d}"

def create_karaoke_video(instr_wav, ass_file, output_mp4):
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
        "-vf", f"ass={ass_file}",
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

def get_output_subdir(base_name, input_path):
    # Generate a 6-char hash from the input path for uniqueness
    h = hashlib.sha1(input_path.encode('utf-8')).hexdigest()[:6]
    subdir = os.path.join('output', f"{base_name}_{h}")
    os.makedirs(subdir, exist_ok=True)
    return subdir

def get_youtube_title(youtube_url):
    # Use yt-dlp to fetch the video title
    result = subprocess.run([
        "yt-dlp", "--get-title", youtube_url
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        print("Failed to fetch YouTube title, using fallback name.")
        return "downloaded_audio"

def main():
    print("=== Karaoke Version Pipeline (MDX-Net + yt-dlp) ===")

    use_yt = prompt_user("Will you provide a YouTube URL to download the audio?")

    if use_yt:
        youtube_url = input("Enter YouTube URL: ").strip()
        # Get the song title from YouTube
        yt_title = get_youtube_title(youtube_url)
        base_name = yt_title if yt_title else "downloaded_audio"
        output_subdir = get_output_subdir(base_name, youtube_url)
        temp_wav = os.path.join(output_subdir, f"{base_name}.wav")
        download_audio_with_ytdlp(youtube_url, temp_wav)
        input_path_for_hash = youtube_url
    else:
        input_path = input("Enter path to input audio/video file: ").strip()
        if not os.path.isfile(input_path):
            print(f"File not found: {input_path}")
            sys.exit(1)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_subdir = get_output_subdir(base_name, input_path)
        temp_wav = os.path.join(output_subdir, f"{base_name}_extracted.wav")
        input_path_for_hash = input_path
        if prompt_user("Step 1: Extract audio to WAV?"):
            run_ffmpeg_extract(input_path, temp_wav)
        else:
            print("Skipping audio extraction step.")

    # All outputs go in output_subdir
    output_dir = os.path.join(output_subdir, f"{base_name}_separated")
    os.makedirs(output_dir, exist_ok=True)
    instr_wav = os.path.join(output_dir, f"instrument_{os.path.basename(temp_wav)}")
    ass_file = os.path.join(output_subdir, f"{base_name}.ass")
    output_karaoke = os.path.join(output_subdir, f"{base_name}_karaoke.mp4")

    if prompt_user("Step 2: Perform vocal separation with MDX-Net?"):
        run_mdxnet_separation(temp_wav, output_dir)
    else:
        print("Skipping vocal separation step.")

    if prompt_user("Step 3: Transcribe lyrics and generate subtitles?"):
        run_whisper_transcribe_karaoke(temp_wav, ass_file)
    else:
        print("Skipping transcription step.")

    if prompt_user("Step 4: Assemble karaoke video with subtitles?"):
        # Confirm accompaniment wav existence:
        if not os.path.isfile(instr_wav):
            print(f"Cannot find accompaniment audio at {instr_wav}. Please check separation output.")
            sys.exit(1)
        create_karaoke_video(instr_wav, ass_file, output_karaoke)
    else:
        print("Skipping video assembly step.")

    print(f"Pipeline complete. All files are in: {output_subdir}")

if __name__ == "__main__":
    main()
