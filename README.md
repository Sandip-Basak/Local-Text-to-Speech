# 🎙️ Local Text-to-Speech (TTS) with Kokoro-82M (PyTorch)

A fast, lightweight, and natural-sounding local Text-to-Speech engine using the **Kokoro-82M** PyTorch model. Designed specifically for low-latency conversational voice agents, podcast narration, and local AI prototyping.

---

## 🌟 Why Kokoro-82M?

**Kokoro-82M** is an open-weight, high-quality TTS model with only **82 million parameters**. Despite its compact size, it rivals large commercial TTS cloud APIs in voice quality, natural cadence, and human-like inflection.

- ⚡ **Ultra-Fast & Lightweight**: Minimal VRAM and CPU footprint; runs smoothly on consumer GPUs and modern CPUs.
- 🎧 **Studio-Quality Audio**: Native **24 kHz** sample rate output with clear pronunciation and rich vocal tone.
- 🎭 **Expressive Prosody**: Dynamically responds to punctuation (commas, ellipses, exclamation marks, question marks) to convey realistic emotion and pacing.
- 🔒 **100% Local & Private**: All speech synthesis occurs entirely on your device with zero API keys or recurring cloud costs.
- 🎛️ **Voice Blending**: Easily mix multiple voice styles together (e.g., combining warmth and energy).

---

## 📁 Project Structure

```
.
├── TTS.py              # Main Kokoro TTS engine & CLI script
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules for venv and audio outputs
└── README.md           # Documentation and usage guide
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python **3.10, 3.11, or 3.12** installed.
- (Optional) NVIDIA GPU with CUDA for faster-than-realtime synthesis.

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

> **Note for CUDA Users (Optional):**  
> If you have an NVIDIA GPU and want GPU acceleration, ensure PyTorch is installed with CUDA support:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 🏃 Running the TTS Engine

### 1. Default Run (15-Sentence Test Paragraph)
Running `TTS.py` without arguments synthesizes a rich, 15-sentence conversational paragraph demonstrating natural pauses, ellipses, and expressive inflections. It plays the audio through your speakers in real-time as chunks are generated, and saves the final full output to `output.wav`.

```bash
python TTS.py
```

---

### 2. Synthesize Custom Text
```bash
python TTS.py --text "Hello! This is a test of natural voice generation... Pretty impressive, isn't it?"
```

---

### 3. Read from a Text File
```bash
python TTS.py --file path/to/script.txt --output story.wav
```

---

### 4. Select Different Voices
```bash
# Natural male voice
python TTS.py --voice am_adam --text "Hello! I'm Adam, ready to assist you."

# Energetic female voice
python TTS.py --voice af_bella --text "Good morning! Let's get started right away!"

# British female accent
python TTS.py --voice bf_emma --lang b --text "Good afternoon, it's an absolute pleasure."
```

---

### 5. Voice Blending
Blend two or more voices by specifying their names and optional weights:
```bash
python TTS.py --voice "af_heart(0.7)+af_bella(0.3)" --text "This voice blends warmth with conversational energy."
```

---

### 6. Adjust Speed & Headless Mode
```bash
# Speed up speech (1.15x) and disable live audio playback
python TTS.py --speed 1.15 --no-play --output fast_speech.wav
```

---

### 7. List All Available Recommended Voices
```bash
python TTS.py --list-voices
```

---

## 🎭 Recommended Voice Catalog

Kokoro-82M features 50+ preset voices. Here are the top conversational choices:

| Category | Voice ID | Description / Tone |
| :--- | :--- | :--- |
| **American Female** | `af_heart` | ⭐ Flagship voice — warm, balanced, natural, and expressive |
| | `af_bella` | Energetic, clear, and upbeat conversational voice |
| | `af_nicole` | Calm, professional, and soothing |
| | `af_sarah` | Articulate, dynamic, and narrative |
| | `af_sky` | Gentle, approachable, and soft-spoken |
| **American Male** | `am_adam` | ⭐ Deep, warm, friendly conversational male voice |
| | `am_michael` | Clear, engaging, and professional |
| | `am_echo` | Resonant, calm narrator tone |
| | `am_fenrir` | Rich, bold, and dramatic |
| **British Female** | `bf_emma` | Warm, sophisticated British accent |
| | `bf_isabella` | Crisp, clear, elegant British tone |
| **British Male** | `bm_george` | Classic, cultured British male voice |
| | `bm_lewis` | Modern, friendly British conversationalist |

---

## ✍️ Natural Speech & Prompting Guide

Kokoro-82M interprets standard punctuation to control pitch, cadence, and breath timing:

- **Commas (`,`)**: Creates a short breath pause (~100-150ms). Use them to break up long sentences naturally.
- **Ellipses (`...`)**: Produces a longer, reflective pause (~300-500ms), simulating hesitation or thinking before continuing.
- **Exclamation Marks (`!`)**: Elevates vocal energy, excitement, and pitch.
- **Question Marks (`?`)**: Inflects upward at the end of the clause for an authentic inquiring tone.
- **Em-Dashes (`--`)**: Adds a crisp, conversational parenthetical aside.

**Example of Conversational Text:**
> *"Well... to be completely honest, I didn't expect that to happen! But don't worry, we can solve this together."*

---

## 🧩 Voice Agent Integration (Python API)

You can import `KokoroTTS` directly into your voice agent or LLM streaming pipeline:

```python
from TTS import KokoroTTS

# 1. Initialize engine once (caches model in memory/GPU)
tts = KokoroTTS(voice="af_heart", speed=1.0)

# 2. Stream generation for incoming LLM tokens or sentences
incoming_llm_response = (
    "I just analyzed your data! Everything looks great, "
    "and we are ready to move on to the next phase."
)

# 3. Generate and stream chunks
for chunk_idx, chunk_text, audio_numpy, latency in tts.generate_chunks(incoming_llm_response):
    print(f"Chunk {chunk_idx}: '{chunk_text}' (generated in {latency:.2f}s)")
    # Send audio_numpy directly to audio stream / WebSocket / WebRTC
```

---

## 🛠️ CLI Options Reference

| Argument | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--text` | `-t` | Built-in test paragraph | Text string to synthesize |
| `--file` | `-f` | `None` | Text file to read and synthesize |
| `--voice` | `-v` | `af_heart` | Voice identifier or blend string |
| `--speed` | `-s` | `1.0` | Speech playback rate multiplier |
| `--output` | `-o` | `output.wav` | Destination path for saved audio |
| `--no-play` | | `False` | Disable real-time speaker playback |
| `--lang` | | `a` | Language code (`a` = US English, `b` = UK English) |
| `--list-voices`| | `False` | Display the recommended voice catalog |

---

## 📄 License
This project uses the open-weight **Kokoro-82M** model created by `@hexgrad` and released under the **Apache 2.0 License**.
