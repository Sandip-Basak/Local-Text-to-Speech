"""
TTS.py - Local Text-to-Speech using Kokoro-82M (PyTorch Version)
Asynchronous Pipelined Streaming Edition
"""

import os
import sys
import time
import re
import argparse
import threading
import queue
from typing import List, Generator, Tuple, Optional
import numpy as np
import soundfile as sf

# Optional real-time playback support
try:
    import sounddevice as sd
    AUDIO_PLAYBACK_AVAILABLE = True
except ImportError:
    AUDIO_PLAYBACK_AVAILABLE = False


# Available top-tier Kokoro voices
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


class KokoroTTS:
    """
    Local Text-to-Speech engine utilizing the Kokoro-82M PyTorch model.
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

        print(f"[KokoroTTS] Initializing pipeline on device: {self.device.upper()} (lang='{lang_code}')")
        start_time = time.time()
        
        # Initialize KPipeline from the kokoro library
        self.pipeline = KPipeline(lang_code=self.lang_code, device=self.device)
        
        print(f"[KokoroTTS] Pipeline loaded in {time.time() - start_time:.2f}s.")

    @staticmethod
    def chunk_text(text: str, max_chars: int = 250) -> List[str]:
        """
        Intelligently split a large paragraph into natural conversational chunks.

        Preserves natural pause markers like:
        - Ellipses (...) for contemplative or hesitant pauses
        - Exclamation marks (!) for emotional inflection
        - Question marks (?) for interrogative cadence
        - Commas (,) and semicolons (;) for natural breathing intervals
        - Em-dashes (--) for conversational asides

        :param text: The raw input string.
        :param max_chars: Maximum character length per chunk before forcing a clause boundary.
        :return: List of clean, punctuated sentence/clause chunks.
        """
        # Normalize whitespace while preserving essential newlines
        text = text.strip()
        if not text:
            return []

        # Split on sentence boundaries: periods, exclamations, question marks, newlines
        # We look for punctuation followed by whitespace or end of string, keeping punctuation attached
        sentence_pattern = r'(?<=[.!?…])\s+(?=[A-Z0-9"\'“‘])|(?<=\n)\s*'
        raw_sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]

        refined_chunks = []
        for sentence in raw_sentences:
            if len(sentence) <= max_chars:
                refined_chunks.append(sentence)
            else:
                # If sentence is unusually long, split on intermediate clause markers (semicolons, colons, em-dashes, commas)
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
        Generator that chunks text and yields audio arrays sequentially.
        :param text: Input text (single sentence or entire multi-sentence paragraph).
        :param voice: Voice ID or blend to use (e.g., 'af_heart', 'am_adam', 'af_heart+af_bella').
        :param speed: Speech speed multiplier.
        :yield: Tuples of (chunk_index, chunk_text, audio_numpy_array, generation_latency_seconds).
        """
        selected_voice = voice or self.default_voice
        selected_speed = speed if speed is not None else self.speed
        chunks = self.chunk_text(text)

        for idx, chunk in enumerate(chunks, 1):
            chunk_start = time.time()
            # Kokoro KPipeline returns a generator yielding (graphemes, phonemes, audio)
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

    def synthesize(
        self,
        text: str,
        output_file: Optional[str] = "output.wav",
        play_audio: bool = True,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        pause_between_chunks_ms: int = 180
    ) -> np.ndarray:
        """
        Synthesize speech from input text, optionally streaming playback in real-time,
        and save the complete audio to a WAV file.

        :param text: Full input text.
        :param output_file: Path to save the final WAV file (or None to skip saving).
        :param play_audio: Whether to play audio through speakers chunk-by-chunk.
        :param voice: Voice ID or voice blend.
        :param speed: Speech speed.
        :param pause_between_chunks_ms: Natural silence duration inserted between chunks.
        :return: Full concatenated numpy audio array.
        """
        selected_voice = voice or self.default_voice
        selected_speed = speed if speed is not None else self.speed
        
        chunks = self.chunk_text(text)
        total_chunks = len(chunks)

        print("\n" + "=" * 70)
        print(f"🎙️  Kokoro TTS Pipelined Synthesis Started")
        print(f"   Voice: {selected_voice} | Speed: {selected_speed}x | Device: {self.device.upper()}")
        print(f"   Total text chunks to process: {total_chunks}")
        print("=" * 70 + "\n")

        all_audio_segments = []
        silence_samples = int(self.sample_rate * (pause_between_chunks_ms / 1000.0))
        silence_gap = np.zeros(silence_samples, dtype=np.float32)

        # Buffer queue for audio chunks awaiting playback
        playback_queue = queue.Queue()
        playback_error = []

        def playback_consumer():
            """Dedicated consumer thread to stream audio chunks sequentially."""
            while True:
                item = playback_queue.get()
                if item is None:  # Sentinel value signaling end of stream
                    playback_queue.task_done()
                    break
                try:
                    sd.play(item, samplerate=self.sample_rate)
                    sd.wait()  # Blocks only the playback worker thread, NOT model inference
                except Exception as e:
                    playback_error.append(e)
                finally:
                    playback_queue.task_done()

        playback_thread = None
        if play_audio and AUDIO_PLAYBACK_AVAILABLE:
            playback_thread = threading.Thread(target=playback_consumer, daemon=True)
            playback_thread.start()

        total_gen_start = time.time()

        # Producer Loop: Continuously generates audio without waiting on speaker output
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

                # Append silence pause to chunk for smooth playback transition
                chunk_to_play = np.concatenate([audio_array, silence_gap]) if len(silence_gap) > 0 else audio_array

                # Push to playback queue immediately (Non-blocking)
                if play_audio and AUDIO_PLAYBACK_AVAILABLE:
                    playback_queue.put(chunk_to_play)

        total_time = time.time() - total_gen_start

        # Signal consumer thread that generation is complete, then wait for buffer to drain
        if play_audio and AUDIO_PLAYBACK_AVAILABLE and playback_thread:
            playback_queue.put(None)
            playback_thread.join()

        if playback_error:
            print(f"    ⚠️ Playback warning: {playback_error[0]}")

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
            print(f"   Overall RTF (Real Time Factor): {total_time / total_audio_duration:.2f}x (lower is faster)")
        print("=" * 70)

        # Save to output file
        if output_file and len(final_audio) > 0:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            sf.write(output_file, final_audio, self.sample_rate)
            print(f"💾 Full audio saved successfully to: {os.path.abspath(output_file)}\n")

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


# 15-sentence rich conversational test paragraph
TEST_PARAGRAPH = """
Hello there! Welcome to this local speech synthesis test powered by the Kokoro model.
Can you hear how smooth and natural this sounds?
I'm designed to capture all the subtle nuances of human conversation... including natural pauses, varied inflections, and emotional depth!
When I speak, commas give me a brief moment to breathe, allowing thoughts to flow effortlessly.
Did you know? Exclamation marks bring genuine excitement and energy to my words!
On the other hand... ellipses create thoughtful, contemplative pauses—just like a person pondering their next thought.
Whether you're asking questions, telling a captivating story, or building an intelligent voice assistant, expressiveness is key.
Generating speech in small chunks allows us to begin playing audio almost instantly, keeping latency astonishingly low.
No robotic monotone, no awkward robotic stutters—just crisp, fluid, twenty-four kilohertz audio.
Think about the possibilities: real-time voice agents, immersive podcast narration, interactive gaming characters, and accessible learning tools!
Right now, all of this computation is happening entirely on your local machine, preserving privacy and eliminating costly cloud API fees.
Pretty impressive for an eighty-two million parameter model, wouldn't you say?
If you'd like, you can easily experiment with different voices, adjust speech rates, or even blend multiple speakers together.
Thank you for listening to this demonstration!
Let's build something truly incredible together.
""".strip()


def main():
    parser = argparse.ArgumentParser(
        description="Kokoro-82M Local PyTorch Text-to-Speech Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python TTS.py
  python TTS.py --voice am_adam --speed 1.05 --output my_speech.wav
  python TTS.py --text "Hello! How are you doing today?" --no-play
  python TTS.py --voice "af_heart(0.6)+af_bella(0.4)"
  python TTS.py --list-voices
        """
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help="Text to synthesize (defaults to the 15-sentence natural test paragraph)"
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
        help="Voice identifier or blend (default: af_heart)"
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
        default="output.wav",
        help="Path to save the output WAV file (default: output.wav)"
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Disable real-time speaker audio playback during generation"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="a",
        choices=["a", "b"],
        help="Language code: 'a' for American English, 'b' for British English (default: a)"
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

    # Determine input text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text_to_speak = f.read().strip()
    elif args.text:
        text_to_speak = args.text.strip()
    else:
        text_to_speak = TEST_PARAGRAPH

    # Initialize Kokoro TTS engine
    tts = KokoroTTS(
        lang_code=args.lang,
        default_voice=args.voice,
        speed=args.speed
    )

    # Synthesize and save
    tts.synthesize(
        text=text_to_speak,
        output_file=args.output,
        play_audio=not args.no_play,
        voice=args.voice,
        speed=args.speed
    )


if __name__ == "__main__":
    main()