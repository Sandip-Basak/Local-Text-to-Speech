"""
colab_tts.py - Kokoro-82M Text-to-Speech Engine for Google Colab
================================================================
A self-contained, optimized Text-to-Speech script designed to run directly
in Google Colab (with GPU/CUDA or CPU), featuring:
  - Automatic dependency installation (kokoro, soundfile, espeak-ng)
  - Seamless inline IPython audio playback in Colab notebook cells
  - Intelligent conversational chunking preserving natural pauses & inflections
  - 50+ Kokoro-82M voices with support for voice blending (e.g. 'af_heart(0.7)+af_bella(0.3)')
  - GPU acceleration auto-detection (NVIDIA T4/A100/V100 on Colab)
  - Direct file download to your local machine

How to run in Google Colab:
--------------------------
Option 1: In a Colab Code Cell (Command line):
    !python colab_tts.py --voice af_heart --text "Hello from Google Colab!"

Option 2: In a Colab Code Cell (Interactive run):
    %run colab_tts.py

Option 3: Import as a module in Colab:
    from colab_tts import KokoroTTS
    tts = KokoroTTS()
    tts.synthesize("Hello world!")
"""

import os
import sys

# Ensure UTF-8 output encoding for terminals supporting Unicode / emojis
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import time
import re
import shutil
import subprocess
import argparse
from typing import List, Generator, Tuple, Optional
import numpy as np


# ==============================================================================
# 1. Automatic Dependency Management for Google Colab
# ==============================================================================

def is_colab_environment() -> bool:
    """Check whether code is running inside Google Colab."""
    return "google.colab" in sys.modules or os.path.exists("/content")


def ensure_colab_dependencies():
    """
    Automatically installs necessary system and python dependencies if
    running inside Google Colab and packages are missing.
    """
    if not is_colab_environment():
        return

    # 1. Install system dependency 'espeak-ng' (required for phonemization on Linux)
    if shutil.which("espeak-ng") is None:
        print("[Colab Setup] 📦 Installing system package 'espeak-ng' for phoneme synthesis...")
        try:
            subprocess.run(
                ["apt-get", "-qq", "-y", "install", "espeak-ng"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            print("[Colab Setup] ✅ 'espeak-ng' installed successfully.")
        except Exception as e:
            print(f"[Colab Setup] ⚠️ Warning: Failed to install espeak-ng: {e}")

    # 2. Check and install Python dependencies (kokoro, soundfile)
    needed_packages = []
    try:
        import kokoro
    except ImportError:
        needed_packages.append("kokoro>=0.8.4")

    try:
        import soundfile
    except ImportError:
        needed_packages.append("soundfile>=0.12.1")

    if needed_packages:
        print(f"[Colab Setup] 📦 Installing Python packages: {', '.join(needed_packages)}...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q"] + needed_packages,
                check=True
            )
            print("[Colab Setup] ✅ Python dependencies installed successfully.")
        except Exception as e:
            print(f"[Colab Setup] ❌ Error installing dependencies: {e}")


# Run the setup check immediately in Colab
ensure_colab_dependencies()

try:
    import soundfile as sf
except ImportError:
    sf = None

# Optional local playback support (via sounddevice if available on local machine)
try:
    import sounddevice as sd
    LOCAL_PLAYBACK_AVAILABLE = True
except (ImportError, OSError):
    LOCAL_PLAYBACK_AVAILABLE = False


# ==============================================================================
# 2. Kokoro Voice Catalog
# ==============================================================================

AVAILABLE_VOICES = {
    "American Female": [
        ("af_heart", "Flagship warm, natural, and expressive voice (Recommended)"),
        ("af_bella", "Energetic, clear, and friendly conversational voice"),
        ("af_nicole", "Calm, professional, soothing tone"),
        ("af_sarah", "Dynamic, articulate, and expressive"),
        ("af_sky", "Soft, gentle, and approachable"),
        ("af_alloy", "Balanced, neutral assistant tone"),
        ("af_river", "Smooth, relaxed, narrative tone"),
    ],
    "American Male": [
        ("am_adam", "Deep, warm, natural conversational male voice (Recommended)"),
        ("am_michael", "Clear, engaging, and professional tone"),
        ("am_echo", "Resonant, calm narrator voice"),
        ("am_fenrir", "Rich, confident, and dramatic voice"),
        ("am_liam", "Youthful, energetic conversational voice"),
        ("am_onyx", "Authoritative, deep broadcasting tone"),
        ("am_puck", "Playful, light-hearted, and upbeat"),
    ],
    "British Female": [
        ("bf_emma", "Warm, sophisticated British accent"),
        ("bf_isabella", "Crisp, clear, elegant British tone"),
        ("bf_alice", "Gentle, polite British voice"),
        ("bf_lily", "Expressive, youthful British tone"),
    ],
    "British Male": [
        ("bm_george", "Classic, cultured British male voice"),
        ("bm_daniel", "Refined, articulate British accent"),
        ("bm_lewis", "Modern, friendly British conversationalist"),
        ("bm_fable", "Storyteller, dramatic British tone"),
    ]
}


# ==============================================================================
# 3. Kokoro TTS Engine Class
# ==============================================================================

class KokoroTTS:
    """
    Kokoro-82M Text-to-Speech Engine with Google Colab optimizations.
    """

    def __init__(
        self,
        lang_code: str = "a",
        default_voice: str = "af_heart",
        speed: float = 1.0,
        device: Optional[str] = None
    ):
        """
        Initialize the Kokoro TTS engine.

        :param lang_code: Language code ('a' for American English, 'b' for British English).
        :param default_voice: Default voice ID (e.g., 'af_heart', 'am_adam').
        :param speed: Speech rate multiplier (1.0 = normal).
        :param device: 'cuda', 'cpu', or None for automatic detection.
        """
        import torch
        from kokoro import KPipeline

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.lang_code = lang_code
        self.default_voice = default_voice
        self.speed = speed
        self.sample_rate = 24000  # Kokoro standard output sample rate (24kHz)

        print(f"[KokoroTTS] 🚀 Initializing pipeline on device: {self.device.upper()} (lang='{lang_code}')")
        if self.device == "cpu" and is_colab_environment():
            print("💡 Tip: For 5x to 10x faster generation in Colab, enable GPU acceleration:")
            print("   Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU")

        start_time = time.time()
        self.pipeline = KPipeline(lang_code=self.lang_code, device=self.device)
        print(f"[KokoroTTS] ✨ Pipeline loaded in {time.time() - start_time:.2f}s.")

    @staticmethod
    def chunk_text(text: str, max_chars: int = 250) -> List[str]:
        """
        Intelligently split paragraphs into natural conversational chunks.

        Preserves expressive markers:
        - Ellipses (...) for contemplative or hesitant pauses
        - Exclamation marks (!) for elevated energy
        - Question marks (?) for upward cadence
        - Commas (,) and semicolons (;) for natural breath points
        - Em-dashes (--) for conversational asides

        :param text: The raw input string.
        :param max_chars: Maximum character length per chunk before forcing a clause boundary.
        :return: List of clean, punctuated sentence/clause chunks.
        """
        text = text.strip()
        if not text:
            return []

        # Split on sentence boundaries: periods, exclamations, question marks, newlines
        sentence_pattern = r'(?<=[.!?…])\s+(?=[A-Z0-9"\'“‘])|(?<=\n)\s*'
        raw_sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]

        refined_chunks = []
        for sentence in raw_sentences:
            if len(sentence) <= max_chars:
                refined_chunks.append(sentence)
            else:
                # If a sentence is unusually long, split on intermediate clause markers
                clause_pattern = r'(?<=[;,:\u2014\-])\s+'
                clauses = [c.strip() for c in re.split(clause_pattern, sentence) if c.strip()]

                current_chunk = ""
                for clause in clauses:
                    if len(current_chunk) + len(clause) + 1 <= max_chars:
                        current_chunk = f"{current_chunk} {clause}".strip()
                    else:
                        if current_chunk:
                            refined_chunks.append(current_chunk)
                        current_chunk = clause
                if current_chunk:
                    refined_chunks.append(current_chunk)

        return refined_chunks

    def generate_chunks(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None
    ) -> Generator[Tuple[int, str, np.ndarray, float], None, None]:
        """
        Generator yielding audio chunks sequentially with latency metrics.

        :param text: Input text.
        :param voice: Voice ID or blend (e.g., 'af_heart', 'am_adam', 'af_heart+af_bella').
        :param speed: Speech speed multiplier.
        :yield: (chunk_index, chunk_text, audio_numpy_array, latency_seconds).
        """
        selected_voice = voice or self.default_voice
        selected_speed = speed if speed is not None else self.speed
        chunks = self.chunk_text(text)

        for idx, chunk in enumerate(chunks, 1):
            chunk_start = time.time()
            generator = self.pipeline(chunk, voice=selected_voice, speed=selected_speed)

            chunk_audios = []
            for _, _, audio in generator:
                if audio is not None:
                    if hasattr(audio, 'cpu'):
                        audio = audio.cpu().numpy()
                    chunk_audios.append(audio)

            if chunk_audios:
                combined_chunk_audio = np.concatenate(chunk_audios)
            else:
                combined_chunk_audio = np.array([], dtype=np.float32)

            latency = time.time() - chunk_start
            yield idx, chunk, combined_chunk_audio, latency

    @staticmethod
    def voice_to_filename(voice: str) -> str:
        """
        Derive a safe .wav filename from a voice identifier or blend string.

        Examples:
            'af_heart'                        -> 'af_heart.wav'
            'af_heart(0.7)+af_bella(0.3)'      -> 'af_heart_af_bella.wav'
        """
        # Extract plain voice names, dropping blend weights like '(0.7)'
        voice_names = re.findall(r'[A-Za-z0-9_]+(?=(?:\([^)]*\))?(?:\+|$))', voice)
        safe_name = "_".join(voice_names) if voice_names else "output"
        return f"{safe_name}.wav"

    def synthesize(
        self,
        text: str,
        output_file: Optional[str] = None,
        output_dir: str = "output",
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        pause_between_chunks_ms: int = 180,
        play_inline: bool = True,
        download: bool = False,
        autoplay: bool = False
    ) -> np.ndarray:
        """
        Synthesize speech from input text, save as WAV, and render in Google Colab.

        :param text: Full input text string.
        :param output_file: Destination WAV filepath (or None to auto-name after the voice,
            e.g. 'af_heart.wav', saved inside `output_dir`).
        :param output_dir: Folder the WAV file is saved into when `output_file` is a bare
            filename or None (default: 'output').
        :param voice: Voice identifier or blend string.
        :param speed: Speech speed multiplier.
        :param pause_between_chunks_ms: Silence duration inserted between chunks.
        :param play_inline: If True and in a notebook/Colab environment, displays an HTML5 audio player.
        :param download: If True and in Colab, automatically triggers a file download to your machine.
        :param autoplay: If True, automatically autoplays the inline audio in Colab.
        :return: Full concatenated audio as float32 numpy array.
        """
        selected_voice = voice or self.default_voice
        selected_speed = speed if speed is not None else self.speed

        if output_file:
            # If a bare filename (no directory component) was given, place it in output_dir.
            if not os.path.dirname(output_file) and output_dir:
                output_file = os.path.join(output_dir, output_file)
        else:
            filename = self.voice_to_filename(selected_voice)
            output_file = os.path.join(output_dir, filename) if output_dir else filename

        chunks = self.chunk_text(text)
        total_chunks = len(chunks)

        print("\n" + "=" * 70)
        print(f"🎙️  Kokoro TTS Synthesis Started")
        print(f"   Voice: {selected_voice} | Speed: {selected_speed}x | Device: {self.device.upper()}")
        print(f"   Total text chunks to process: {total_chunks}")
        print("=" * 70 + "\n")

        all_audio_segments = []
        silence_samples = int(self.sample_rate * (pause_between_chunks_ms / 1000.0))
        silence_gap = np.zeros(silence_samples, dtype=np.float32)

        total_gen_start = time.time()

        for idx, chunk_text, audio_array, latency in self.generate_chunks(
            text, voice=selected_voice, speed=selected_speed
        ):
            audio_duration = len(audio_array) / self.sample_rate if len(audio_array) > 0 else 0.0
            rtf = latency / audio_duration if audio_duration > 0 else 0.0

            print(f"[{idx}/{total_chunks}] ⏱️ Gen: {latency:.2f}s | Audio: {audio_duration:.2f}s | RTF: {rtf:.2f}x")
            print(f"    Text: \"{chunk_text}\"")

            if len(audio_array) > 0:
                all_audio_segments.append(audio_array)
                all_audio_segments.append(silence_gap)

        total_time = time.time() - total_gen_start

        if all_audio_segments:
            final_audio = np.concatenate(all_audio_segments)
        else:
            final_audio = np.array([], dtype=np.float32)

        total_audio_duration = len(final_audio) / self.sample_rate

        print("\n" + "=" * 70)
        print(f"✅ Synthesis Complete!")
        print(f"   Total Generation Time : {total_time:.2f}s")
        print(f"   Total Audio Duration  : {total_audio_duration:.2f}s")
        if total_audio_duration > 0:
            print(f"   Overall RTF (Real-Time Factor): {total_time / total_audio_duration:.2f}x")
        print("=" * 70)

        # Save audio file
        if output_file and len(final_audio) > 0:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            sf.write(output_file, final_audio, self.sample_rate)
            abs_path = os.path.abspath(output_file)
            print(f"💾 Audio saved successfully to: {abs_path}\n")

        # Inline Audio Playback in Google Colab / Jupyter Notebooks
        displayed_player = False
        if play_inline and len(final_audio) > 0:
            try:
                from IPython.display import Audio, display
                # Check if running in interactive notebook/Colab shell
                get_ipython()
                display(Audio(data=final_audio, rate=self.sample_rate, autoplay=autoplay))
                displayed_player = True
                print("🔊 [Colab] Inline audio player displayed above.")
            except (NameError, ImportError):
                # Script is running via shell (`!python colab_tts.py`) rather than kernel cell
                if output_file:
                    print("💡 To listen directly in a Colab notebook code cell, run:")
                    print("   import IPython.display as ipd")
                    print(f"   ipd.Audio('{output_file}')\n")

        # Optional download to local machine in Google Colab
        if download and output_file and os.path.exists(output_file):
            if is_colab_environment():
                try:
                    from google.colab import files
                    print("📥 Triggering download to your computer...")
                    files.download(output_file)
                except Exception as e:
                    print(f"⚠️ Could not trigger Colab download: {e}")
            else:
                print("💡 Note: --download is only applicable when running inside Google Colab.")

        return final_audio

    @staticmethod
    def print_voices():
        """Print the catalog of recommended Kokoro voices."""
        print("\n" + "=" * 65)
        print("🎭 Available Kokoro-82M Voices")
        print("=" * 65)
        for category, voices in AVAILABLE_VOICES.items():
            print(f"\n📌 {category}:")
            for voice_id, description in voices:
                print(f"  • {voice_id:<12} : {description}")
        print("\n💡 Voice Blending Tip:")
        print("  You can blend voices together! Example: 'af_heart(0.7)+af_bella(0.3)'")
        print("=" * 65 + "\n")


# Rich conversational test paragraph showcasing expressiveness & pauses
TEST_PARAGRAPH = """
Hello there! Welcome to this local speech synthesis test powered by the Kokoro model on Google Colab.
Can you hear how smooth and natural this sounds?
I'm designed to capture all the subtle nuances of human conversation... including natural pauses, varied inflections, and emotional depth!
When I speak, commas give me a brief moment to breathe, allowing thoughts to flow effortlessly.
Did you know? Exclamation marks bring genuine excitement and energy to my words!
On the other hand... ellipses create thoughtful, contemplative pauses—just like a person pondering their next thought.
Whether you're asking questions, telling a captivating story, or building an intelligent voice assistant, expressiveness is key.
Generating speech in small chunks allows us to begin playing audio almost instantly, keeping latency astonishingly low.
No robotic monotone, no awkward robotic stutters—just crisp, fluid, twenty-four kilohertz audio.
Think about the possibilities: real-time voice agents, immersive podcast narration, interactive gaming characters, and accessible learning tools!
Pretty impressive for an eighty-two million parameter model, wouldn't you say?
If you'd like, you can easily experiment with different voices, adjust speech rates, or even blend multiple speakers together.
Thank you for listening to this demonstration!
Let's build something truly incredible together.
""".strip()


# ==============================================================================
# 4. Command-Line & Notebook Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kokoro-82M Text-to-Speech Engine (Google Colab Edition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples in Google Colab:
  !python colab_tts.py
  !python colab_tts.py --voice am_adam --speed 1.05 --output my_speech.wav
  !python colab_tts.py --text "Hello Colab! Kokoro TTS is running on GPU." --download
  !python colab_tts.py --voice "af_heart(0.6)+af_bella(0.4)"
  !python colab_tts.py --list-voices
        """
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help="Text string to synthesize (defaults to conversational test paragraph)"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to a text file to read and synthesize"
    )
    parser.add_argument(
        "--voice", "-v",
        type=str,
        default="af_heart",
        help="Voice identifier or blend string (default: af_heart)"
    )
    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save the output WAV file (default: <output-dir>/<voice>.wav, e.g. output/af_heart.wav)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Folder to save the output WAV file into (default: output)"
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        default="a",
        choices=["a", "b"],
        help="Language code: 'a' for American English, 'b' for British English (default: a)"
    )
    parser.add_argument(
        "--download", "-d",
        action="store_true",
        help="Download generated audio file to local computer via Colab files API"
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Automatically start playing audio when player renders in notebook cell"
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Disable inline audio player rendering"
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available recommended voices and exit"
    )

    args = parser.parse_args()

    if args.list_voices:
        KokoroTTS.print_voices()
        return

    # Determine text to synthesize
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text_to_speak = f.read().strip()
    elif args.text:
        text_to_speak = args.text.strip()
    else:
        text_to_speak = TEST_PARAGRAPH

    # Initialize Kokoro TTS
    tts = KokoroTTS(
        lang_code=args.lang,
        default_voice=args.voice,
        speed=args.speed
    )

    # Synthesize
    tts.synthesize(
        text=text_to_speak,
        output_file=args.output,
        output_dir=args.output_dir,
        voice=args.voice,
        speed=args.speed,
        play_inline=not args.no_play,
        download=args.download,
        autoplay=args.autoplay
    )


if __name__ == "__main__":
    main()
