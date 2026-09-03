"""
api.py - Fast WebSocket Streaming Text-to-Speech API for Kokoro-82M
Designed for ultra-low latency real-time voice agents and LLM streaming pipelines.
"""

import os
import sys

# Ensure UTF-8 output encoding across Windows/Linux terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import re
import time
import json
import asyncio
import io
import threading
from typing import List, Optional, Tuple, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from TTS import KokoroTTS, AVAILABLE_VOICES


# ---------------------------------------------------------------------------
# Global Engine & Synchronization
# ---------------------------------------------------------------------------
tts_engine: Optional[KokoroTTS] = None
engine_lock = threading.Lock()
server_start_time: float = 0.0


def get_engine() -> KokoroTTS:
    """Retrieve the initialized global Kokoro TTS engine."""
    global tts_engine
    if tts_engine is None:
        device_env = os.getenv("DEVICE", None)
        if device_env and device_env.lower() == "auto":
            device_env = None
        default_voice = os.getenv("DEFAULT_VOICE", "af_heart")
        tts_engine = KokoroTTS(
            lang_code="a",
            default_voice=default_voice,
            device=device_env
        )
    return tts_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context:
    Preloads Kokoro-82M model and warms up PyTorch CUDA/CPU kernels on startup.
    """
    global server_start_time
    server_start_time = time.time()
    print("\n🚀 [FastAPI] Initializing Kokoro-82M TTS Streaming Engine...")
    
    # Preload engine and run a tiny warm-up inference to avoid cold-start latency
    engine = get_engine()
    print(f"⚡ [FastAPI] Running warm-up synthesis on {engine.device.upper()}...")
    try:
        # Generate 1 small chunk to load Hugging Face weights and CUDA kernels
        list(engine.generate_chunks("Ready.", voice=engine.default_voice, speed=1.0))
        print("✅ [FastAPI] Engine warmed up and ready for streaming!\n")
    except Exception as e:
        print(f"⚠️ [FastAPI] Warm-up encountered error: {e}")

    yield

    print("🛑 [FastAPI] Shutting down TTS server...")


# ---------------------------------------------------------------------------
# FastAPI Application Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Kokoro-82M Streaming TTS API",
    description="Ultra-low latency bidirectional WebSocket streaming TTS API for real-time conversational voice agents.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Audio Conversion Helpers
# ---------------------------------------------------------------------------
def float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 [-1.0, 1.0] audio array to 16-bit signed PCM bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    int16_arr = (clipped * 32767.0).astype(np.int16)
    return int16_arr.tobytes()


def float32_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Package float32 audio array into standard 16-bit WAV bytes container."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Streaming Text Chunker
# ---------------------------------------------------------------------------
class StreamingTextAccumulator:
    """
    Buffers incremental LLM tokens and slices natural clause/sentence chunks
    to maximize synthesis speed while preserving human prosody and inflection.
    """

    def __init__(self, min_clause_chars: int = 20, max_chunk_chars: int = 140):
        self.buffer = ""
        self.min_clause_chars = min_clause_chars
        self.max_chunk_chars = max_chunk_chars

        # Strong sentence boundaries: . ! ? … \n followed by whitespace or quote or end
        self.sentence_regex = re.compile(r'([.!?…\n]+)([\s"\'“”]+|$)')
        # Conversational clause pause markers: , ; : — --
        self.clause_regex = re.compile(r'([,;:\u2014\-]+)([\s"\'“”]+)')

    def add_token(self, token: str) -> List[str]:
        """Add incremental token/text and return any complete chunks ready for synthesis."""
        self.buffer += token
        ready_chunks = []

        while True:
            # 1. Look for strong sentence boundary
            match = self.sentence_regex.search(self.buffer)
            if match:
                end_pos = match.end()
                chunk = self.buffer[:end_pos].strip()
                self.buffer = self.buffer[end_pos:]
                if chunk:
                    ready_chunks.append(chunk)
                continue

            # 2. Look for natural clause boundary if we have accumulated enough text for natural cadence
            if len(self.buffer) >= self.min_clause_chars:
                clause_match = self.clause_regex.search(self.buffer)
                if clause_match:
                    end_pos = clause_match.end()
                    chunk = self.buffer[:end_pos].strip()
                    self.buffer = self.buffer[end_pos:]
                    if chunk:
                        ready_chunks.append(chunk)
                    continue

            # 3. Prevent latency stall: if buffer exceeds maximum chars without punctuation, cut at word boundary
            if len(self.buffer) >= self.max_chunk_chars:
                last_space = self.buffer.rfind(" ")
                if last_space > self.min_clause_chars:
                    chunk = self.buffer[:last_space].strip()
                    self.buffer = self.buffer[last_space + 1:]
                    if chunk:
                        ready_chunks.append(chunk)
                    continue

            break

        return ready_chunks

    def flush(self) -> List[str]:
        """Flush and return any remaining text in the buffer."""
        remaining = self.buffer.strip()
        self.buffer = ""
        if remaining:
            return [remaining]
        return []

    def clear(self):
        """Clear buffer immediately (used during barge-in/interruption)."""
        self.buffer = ""


# ---------------------------------------------------------------------------
# WebSocket Streaming Endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/tts")
@app.websocket("/ws/stream")
async def websocket_tts_endpoint(
    websocket: WebSocket,
    voice: str = Query(default="af_heart", description="Voice ID or blend (e.g. af_heart, am_adam)"),
    speed: float = Query(default=1.0, description="Speech speed multiplier"),
    format: str = Query(default="pcm", pattern="^(pcm|wav|json)$", description="Audio chunk format (pcm, wav, json)"),
    lang: str = Query(default="a", description="Language code: 'a' (US) or 'b' (UK)")
):
    """
    Bidirectional WebSocket for low-latency streaming Text-to-Speech.
    
    Incoming Messages (from Client):
      - JSON text: {"type": "text", "text": "Hello "} or {"text": "Hello"}
      - Plain text string: "Hello "
      - Flush command: {"type": "flush"} or {"flush": true}
      - Interrupt / barge-in command: {"type": "interrupt"} or {"type": "clear"}
      - Config change: {"type": "config", "voice": "am_adam", "speed": 1.1}
      - Ping: {"type": "ping"} -> replies {"type": "pong"}

    Outgoing Messages (to Client):
      - Binary Frame: Raw 16-bit linear PCM (24kHz, mono) or WAV chunk (if format='pcm' or 'wav')
      - JSON Messages:
          {"type": "connected", "sample_rate": 24000, "format": "pcm"}
          {"type": "chunk_start", "chunk_id": 1, "text": "..."}
          {"type": "chunk_end", "chunk_id": 1, "latency_ms": 65, "duration_sec": 1.2}
          {"type": "done", "total_chunks": 3, "total_duration_sec": 4.1}
          {"type": "interrupted"}
    """
    await websocket.accept()
    engine = get_engine()

    current_voice = voice
    current_speed = speed
    current_format = format
    current_lang = lang

    # Send initial connection handshake
    await websocket.send_json({
        "type": "connected",
        "sample_rate": engine.sample_rate,
        "voice": current_voice,
        "speed": current_speed,
        "format": current_format,
        "device": engine.device
    })

    # Asynchronous queues and state tracking
    text_queue: asyncio.Queue[Optional[Tuple[int, str]]] = asyncio.Queue()
    accumulator = StreamingTextAccumulator()
    chunk_counter = 0
    is_interrupted = False

    async def synthesis_worker():
        """
        Background consumer task:
        Takes ready text chunks from text_queue, synthesizes them with Kokoro,
        and streams audio chunks back to the WebSocket immediately.
        """
        nonlocal is_interrupted, chunk_counter
        total_audio_samples = 0
        total_synthesis_time = 0.0

        while True:
            item = await text_queue.get()
            if item is None:
                # End-of-turn sentinel reached
                text_queue.task_done()
                if not is_interrupted and chunk_counter > 0:
                    total_dur = total_audio_samples / engine.sample_rate
                    await websocket.send_json({
                        "type": "done",
                        "total_chunks": chunk_counter,
                        "total_duration_sec": round(total_dur, 3),
                        "total_synthesis_time_sec": round(total_synthesis_time, 3),
                        "rtf": round(total_synthesis_time / total_dur, 3) if total_dur > 0 else 0.0
                    })
                # Reset counters for the next conversational turn
                chunk_counter = 0
                total_audio_samples = 0
                total_synthesis_time = 0.0
                is_interrupted = False
                continue

            chunk_id, chunk_text = item

            if is_interrupted:
                text_queue.task_done()
                continue

            # Notify client that chunk synthesis has started
            await websocket.send_json({
                "type": "chunk_start",
                "chunk_id": chunk_id,
                "text": chunk_text
            })

            # Run synchronous PyTorch Kokoro inference in a thread pool to avoid blocking the event loop
            synth_start = time.time()
            try:
                def run_inference(text: str, v: str, s: float):
                    with engine_lock:
                        # Kokoro pipeline generator
                        gen = engine.pipeline(text, voice=v, speed=s)
                        audios = []
                        for _, _, audio in gen:
                            if audio is not None:
                                if hasattr(audio, "cpu"):
                                    audio = audio.cpu().numpy()
                                audios.append(audio)
                        if audios:
                            return np.concatenate(audios)
                        return np.array([], dtype=np.float32)

                audio_np = await asyncio.to_thread(
                    run_inference, chunk_text, current_voice, current_speed
                )
            except Exception as e:
                text_queue.task_done()
                await websocket.send_json({
                    "type": "error",
                    "chunk_id": chunk_id,
                    "error": str(e)
                })
                continue

            latency_ms = (time.time() - synth_start) * 1000.0
            total_synthesis_time += (time.time() - synth_start)

            if is_interrupted:
                text_queue.task_done()
                continue

            # Send audio payload
            if len(audio_np) > 0:
                total_audio_samples += len(audio_np)
                duration_sec = len(audio_np) / engine.sample_rate

                if current_format == "pcm":
                    pcm_bytes = float32_to_pcm16_bytes(audio_np)
                    await websocket.send_bytes(pcm_bytes)
                elif current_format == "wav":
                    wav_bytes = float32_to_wav_bytes(audio_np, engine.sample_rate)
                    await websocket.send_bytes(wav_bytes)
                elif current_format == "json":
                    import base64
                    pcm_bytes = float32_to_pcm16_bytes(audio_np)
                    b64_audio = base64.b64encode(pcm_bytes).decode("ascii")
                    await websocket.send_json({
                        "type": "audio",
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "audio_base64": b64_audio,
                        "format": "pcm16",
                        "sample_rate": engine.sample_rate,
                        "duration_sec": round(duration_sec, 3),
                        "latency_ms": round(latency_ms, 1)
                    })

                # Chunk metadata event
                await websocket.send_json({
                    "type": "chunk_end",
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "duration_sec": round(duration_sec, 3),
                    "latency_ms": round(latency_ms, 1),
                    "rtf": round((latency_ms / 1000.0) / duration_sec, 3) if duration_sec > 0 else 0.0
                })

            text_queue.task_done()

    # Launch background synthesis worker
    worker_task = asyncio.create_task(synthesis_worker())

    try:
        while True:
            # Receive message from WebSocket (JSON or text)
            message = await websocket.receive()

            if "text" in message:
                raw_data = message["text"].strip()
                if not raw_data:
                    continue

                # Parse JSON if applicable
                is_json = False
                data = {}
                if raw_data.startswith("{") and raw_data.endswith("}"):
                    try:
                        data = json.loads(raw_data)
                        is_json = True
                    except json.JSONDecodeError:
                        is_json = False

                if is_json:
                    msg_type = data.get("type", "")

                    # 1. Ping / Pong
                    if msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                        continue

                    # 2. Dynamic config change
                    if msg_type == "config":
                        if "voice" in data:
                            current_voice = data["voice"]
                        if "speed" in data:
                            current_speed = float(data["speed"])
                        if "format" in data and data["format"] in ["pcm", "wav", "json"]:
                            current_format = data["format"]
                        await websocket.send_json({
                            "type": "config_updated",
                            "voice": current_voice,
                            "speed": current_speed,
                            "format": current_format
                        })
                        continue

                    # 3. Interruption / Barge-in
                    if msg_type in ["interrupt", "clear"] or data.get("clear"):
                        is_interrupted = True
                        accumulator.clear()
                        # Drain pending items from queue
                        while not text_queue.empty():
                            try:
                                text_queue.get_nowait()
                                text_queue.task_done()
                            except asyncio.QueueEmpty:
                                break
                        await websocket.send_json({"type": "interrupted"})
                        is_interrupted = False
                        continue

                    # 4. Flush (end of LLM generation)
                    if msg_type == "flush" or data.get("flush"):
                        remaining_chunks = accumulator.flush()
                        for rc in remaining_chunks:
                            chunk_counter += 1
                            await text_queue.put((chunk_counter, rc))
                        # Signal end of turn with None sentinel
                        await text_queue.put(None)
                        continue

                    # 5. Text payload
                    text_content = data.get("text", "")
                    if text_content:
                        is_interrupted = False
                        ready_chunks = accumulator.add_token(text_content)
                        for chunk in ready_chunks:
                            chunk_counter += 1
                            await text_queue.put((chunk_counter, chunk))

                else:
                    # Plain text message
                    if raw_data == "<FLUSH>":
                        remaining_chunks = accumulator.flush()
                        for rc in remaining_chunks:
                            chunk_counter += 1
                            await text_queue.put((chunk_counter, rc))
                        await text_queue.put(None)
                    elif raw_data == "<INTERRUPT>":
                        is_interrupted = True
                        accumulator.clear()
                        while not text_queue.empty():
                            try:
                                text_queue.get_nowait()
                                text_queue.task_done()
                            except asyncio.QueueEmpty:
                                break
                        await websocket.send_json({"type": "interrupted"})
                        is_interrupted = False
                    else:
                        is_interrupted = False
                        ready_chunks = accumulator.add_token(raw_data)
                        for chunk in ready_chunks:
                            chunk_counter += 1
                            await text_queue.put((chunk_counter, chunk))

            elif "bytes" in message:
                # Client sent binary data (ignore or echo)
                pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        worker_task.cancel()


# ---------------------------------------------------------------------------
# Utility and Health Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint providing engine status, hardware accelerator, and uptime."""
    import torch
    engine = get_engine()
    return {
        "status": "healthy",
        "service": "Kokoro-82M Streaming TTS",
        "device": engine.device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sample_rate": engine.sample_rate,
        "default_voice": engine.default_voice,
        "uptime_seconds": round(time.time() - server_start_time, 1) if server_start_time > 0 else 0
    }


@app.get("/voices", tags=["Voices"])
async def list_voices():
    """Retrieve the recommended catalog of Kokoro-82M voices categorized by accent and gender."""
    return {
        "recommended_flagship": "af_heart",
        "categories": AVAILABLE_VOICES,
        "blending_syntax": "voice1(weight1)+voice2(weight2), e.g. 'af_heart(0.7)+af_bella(0.3)'"
    }


# ---------------------------------------------------------------------------
# Interactive Web Playground UI
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["Playground"])
async def web_playground():
    """
    Renders a dark-mode browser playground to test real-time WebSocket
    streaming, word-by-word LLM token streaming simulation, and live PCM audio playback.
    """
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>🎙️ Kokoro-82M Real-Time Streaming TTS</title>
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --primary: #58a6ff;
      --primary-hover: #1f6feb;
      --success: #3fb950;
      --danger: #f85149;
      --warning: #d29922;
      --text: #c9d1d9;
      --text-bright: #ffffff;
      --muted: #8b949e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding: 24px; display: flex; justify-content: center; }
    .container { width: 100%; max-width: 900px; display: flex; flex-direction: column; gap: 20px; }
    header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; }
    h1 { color: var(--text-bright); font-size: 1.5rem; display: flex; align-items: center; gap: 10px; }
    .badge { background: #238636; color: white; font-size: 0.75rem; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }
    .row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
    .col { flex: 1; min-width: 200px; display: flex; flex-direction: column; gap: 6px; }
    label { font-size: 0.85rem; color: var(--muted); font-weight: 600; text-transform: uppercase; }
    select, input, textarea { background: #0d1117; border: 1px solid var(--border); border-radius: 6px; color: var(--text-bright); padding: 10px 12px; font-size: 0.95rem; }
    select:focus, input:focus, textarea:focus { outline: none; border-color: var(--primary); }
    textarea { min-height: 120px; resize: vertical; line-height: 1.5; }
    .btn-group { display: flex; gap: 12px; flex-wrap: wrap; }
    button { cursor: pointer; font-weight: 600; border: none; border-radius: 6px; padding: 10px 18px; font-size: 0.95rem; display: inline-flex; align-items: center; gap: 8px; transition: 0.2s; }
    .btn-primary { background: var(--primary-hover); color: white; }
    .btn-primary:hover { background: var(--primary); }
    .btn-success { background: #238636; color: white; }
    .btn-success:hover { background: var(--success); }
    .btn-danger { background: #da3633; color: white; }
    .btn-danger:hover { background: var(--danger); }
    .btn-outline { background: transparent; border: 1px solid var(--border); color: var(--text); }
    .btn-outline:hover { background: #21262d; border-color: var(--muted); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .status-bar { display: flex; gap: 16px; align-items: center; padding: 12px 16px; background: #0d1117; border-radius: 6px; border: 1px solid var(--border); font-size: 0.9rem; }
    .status-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--muted); }
    .status-dot.active { background: var(--success); box-shadow: 0 0 8px var(--success); }
    .metric { display: flex; gap: 6px; align-items: baseline; }
    .metric-value { color: var(--text-bright); font-weight: bold; }
    .log-box { background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 12px; max-height: 220px; overflow-y: auto; font-family: monospace; font-size: 0.85rem; line-height: 1.6; }
    .log-entry { margin-bottom: 4px; }
    .log-time { color: var(--muted); margin-right: 8px; }
    .log-success { color: var(--success); }
    .log-info { color: var(--primary); }
    .log-warn { color: var(--warning); }
    .log-danger { color: var(--danger); }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>🎙️ Kokoro-82M WebSocket Streaming</h1>
      <span class="badge" id="deviceBadge">LOADING...</span>
    </header>

    <div class="card">
      <div class="row">
        <div class="col">
          <label for="voiceSelect">Voice</label>
          <select id="voiceSelect">
            <option value="af_heart" selected>af_heart (Warm, Natural Female - Recommended)</option>
            <option value="am_adam">am_adam (Deep, Friendly Male - Recommended)</option>
            <option value="af_bella">af_bella (Energetic Female)</option>
            <option value="af_nicole">af_nicole (Calm, Soothing Female)</option>
            <option value="af_sarah">af_sarah (Expressive Female)</option>
            <option value="am_michael">am_michael (Engaging Male)</option>
            <option value="am_echo">am_echo (Calm Narrator Male)</option>
            <option value="bf_emma">bf_emma (British Female)</option>
            <option value="bm_george">bm_george (British Male)</option>
            <option value="af_heart(0.7)+af_bella(0.3)">Blend: Heart(0.7) + Bella(0.3)</option>
          </select>
        </div>
        <div class="col" style="max-width: 140px;">
          <label for="speedInput">Speed (<span id="speedVal">1.0</span>x)</label>
          <input type="range" id="speedInput" min="0.6" max="1.8" step="0.05" value="1.0" />
        </div>
        <div class="col" style="max-width: 160px;">
          <label for="tokenDelay">LLM Stream Delay</label>
          <select id="tokenDelay">
            <option value="35">35 ms / token</option>
            <option value="60">60 ms / token</option>
            <option value="15">15 ms (Fast)</option>
            <option value="0">Instant (Burst)</option>
          </select>
        </div>
      </div>

      <div class="row">
        <div class="col">
          <label for="textPrompt">Prompt / Conversational Text</label>
          <textarea id="textPrompt">Hello there! Welcome to the real-time streaming audio test powered by Kokoro-82M. As you stream words from your LLM, this engine synthesizes human speech chunk-by-chunk with ultra-low latency... Can you hear how natural and smooth this sounds?</textarea>
        </div>
      </div>

      <div class="btn-group">
        <button id="streamBtn" class="btn-success" onclick="startStreaming()">
          ⚡ Stream Text & Synthesize (LLM Mode)
        </button>
        <button id="interruptBtn" class="btn-danger" onclick="sendInterrupt()" disabled>
          ⏹️ Interrupt (Barge-in)
        </button>
        <button class="btn-outline" onclick="loadSampleText(1)">Sample 1 (Conversational)</button>
        <button class="btn-outline" onclick="loadSampleText(2)">Sample 2 (Short Prompt)</button>
      </div>
    </div>

    <!-- Status & Real-time Metrics -->
    <div class="status-bar">
      <div class="status-dot" id="wsDot"></div>
      <span id="wsStatus">Connecting WebSocket...</span>
      <div style="flex: 1;"></div>
      <div class="metric">
        <span>TTFA (First Audio):</span>
        <span class="metric-value" id="ttfaMetric">-</span>
      </div>
      <div class="metric">
        <span>Chunks Received:</span>
        <span class="metric-value" id="chunksMetric">0</span>
      </div>
    </div>

    <!-- Live Event Log -->
    <div class="card">
      <label>Live Streaming Log</label>
      <div class="log-box" id="logBox"></div>
    </div>
  </div>

  <script>
    let ws = null;
    let audioCtx = null;
    let scheduledAudioTime = 0;
    let streamStartTime = 0;
    let firstAudioReceived = false;
    let chunkCount = 0;
    let isStreaming = false;

    // Fetch health and setup device badge
    fetch('/health')
      .then(res => res.json())
      .then(data => {
        const badge = document.getElementById('deviceBadge');
        badge.innerText = `${data.device.toUpperCase()}` + (data.cuda_available ? ' (GPU)' : ' (CPU)');
        if (!data.cuda_available) {
          badge.style.background = '#8957e5';
        }
      })
      .catch(() => {});

    // Update speed display
    document.getElementById('speedInput').addEventListener('input', (e) => {
      document.getElementById('speedVal').innerText = e.target.value;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'config', speed: parseFloat(e.target.value) }));
      }
    });

    document.getElementById('voiceSelect').addEventListener('change', (e) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'config', voice: e.target.value }));
      }
    });

    function log(msg, type = 'info') {
      const box = document.getElementById('logBox');
      const time = new Date().toLocaleTimeString();
      const div = document.createElement('div');
      div.className = `log-entry log-${type}`;
      div.innerHTML = `<span class="log-time">[${time}]</span> ${msg}`;
      box.appendChild(div);
      box.scrollTop = box.scrollHeight;
    }

    function initAudioContext() {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
      }
      if (audioCtx.state === 'suspended') {
        audioCtx.resume();
      }
      scheduledAudioTime = audioCtx.currentTime;
    }

    function connectWebSocket() {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const voice = document.getElementById('voiceSelect').value;
      const speed = document.getElementById('speedInput').value;
      const wsUrl = `${proto}//${window.location.host}/ws/tts?voice=${encodeURIComponent(voice)}&speed=${speed}&format=pcm`;

      ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        document.getElementById('wsDot').classList.add('active');
        document.getElementById('wsStatus').innerText = 'Connected (Ready)';
        log('WebSocket connected to streaming TTS engine', 'success');
      };

      ws.onclose = () => {
        document.getElementById('wsDot').classList.remove('active');
        document.getElementById('wsStatus').innerText = 'Disconnected (Reconnecting...)';
        log('WebSocket closed. Retrying in 2 seconds...', 'warn');
        setTimeout(connectWebSocket, 2000);
      };

      ws.onerror = (err) => {
        log('WebSocket error occurred', 'danger');
      };

      ws.onmessage = (evt) => {
        if (typeof evt.data === 'string') {
          handleJsonMessage(JSON.parse(evt.data));
        } else if (evt.data instanceof ArrayBuffer) {
          handleBinaryAudio(evt.data);
        }
      };
    }

    function handleJsonMessage(msg) {
      if (msg.type === 'connected') {
        log(`Handshake OK | Sample Rate: ${msg.sample_rate}Hz | Device: ${msg.device.toUpperCase()}`, 'info');
      } else if (msg.type === 'chunk_start') {
        log(`⚡ Synthesizing Chunk #${msg.chunk_id}: "${msg.text}"`, 'info');
      } else if (msg.type === 'chunk_end') {
        log(`✅ Chunk #${msg.chunk_id} ready in ${msg.latency_ms}ms (Audio: ${msg.duration_sec}s, RTF: ${msg.rtf}x)`, 'success');
      } else if (msg.type === 'done') {
        log(`🎉 Stream turn complete! Total: ${msg.total_chunks} chunks, ${msg.total_duration_sec}s audio`, 'success');
        finishStreamingUI();
      } else if (msg.type === 'interrupted') {
        log(`⏹️ Interrupted / Barged in! Audio stopped.`, 'danger');
        stopAudioPlayback();
        finishStreamingUI();
      }
    }

    function handleBinaryAudio(arrayBuffer) {
      if (!firstAudioReceived) {
        firstAudioReceived = true;
        const ttfa = Math.round(performance.now() - streamStartTime);
        document.getElementById('ttfaMetric').innerText = `${ttfa} ms`;
        log(`🚀 First Audio Received! TTFA: ${ttfa} ms`, 'success');
      }

      chunkCount++;
      document.getElementById('chunksMetric').innerText = chunkCount;

      playPcmChunk(arrayBuffer);
    }

    function playPcmChunk(arrayBuffer) {
      initAudioContext();

      // Convert 16-bit PCM to float32 samples [-1.0, 1.0]
      const int16Array = new Int16Array(arrayBuffer);
      const float32Array = new Float32Array(int16Array.length);
      for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768.0;
      }

      const audioBuffer = audioCtx.createBuffer(1, float32Array.length, 24000);
      audioBuffer.copyToChannel(float32Array, 0);

      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtx.destination);

      // Seamless buffer queueing
      const now = audioCtx.currentTime;
      if (scheduledAudioTime < now) {
        scheduledAudioTime = now + 0.02; // Small buffer cushion
      }

      source.start(scheduledAudioTime);
      scheduledAudioTime += audioBuffer.duration;
    }

    function stopAudioPlayback() {
      if (audioCtx) {
        audioCtx.close().then(() => {
          audioCtx = null;
        });
      }
      scheduledAudioTime = 0;
    }

    async function startStreaming() {
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        alert('WebSocket is not connected yet. Please wait a moment.');
        return;
      }

      initAudioContext();
      isStreaming = true;
      firstAudioReceived = false;
      chunkCount = 0;
      document.getElementById('chunksMetric').innerText = '0';
      document.getElementById('ttfaMetric').innerText = 'waiting...';
      document.getElementById('streamBtn').disabled = true;
      document.getElementById('interruptBtn').disabled = false;

      const rawText = document.getElementById('textPrompt').value.trim();
      const delayMs = parseInt(document.getElementById('tokenDelay').value);

      // Split into words / token-like fragments to simulate streaming LLM output
      const tokens = rawText.match(/\\S+\\s*/g) || [rawText];

      log(`▶️ Starting streaming synthesis simulation (${tokens.length} tokens, ${delayMs}ms delay)...`, 'info');
      streamStartTime = performance.now();

      for (const token of tokens) {
        if (!isStreaming) break;
        ws.send(JSON.stringify({ type: 'text', text: token }));
        if (delayMs > 0) {
          await new Promise(r => setTimeout(r, delayMs));
        }
      }

      if (isStreaming) {
        // Signal flush (LLM finished generating)
        ws.send(JSON.stringify({ type: 'flush' }));
      }
    }

    function sendInterrupt() {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'interrupt' }));
      }
      stopAudioPlayback();
      finishStreamingUI();
    }

    function finishStreamingUI() {
      isStreaming = false;
      document.getElementById('streamBtn').disabled = false;
      document.getElementById('interruptBtn').disabled = true;
    }

    function loadSampleText(num) {
      if (num === 1) {
        document.getElementById('textPrompt').value = "Well... to be completely honest, I didn't expect that to happen! But don't worry, commas give me a brief moment to breathe, and exclamation marks bring genuine excitement!";
      } else {
        document.getElementById('textPrompt').value = "Hello! I am ready to assist you right away. What would you like to explore next?";
      }
    }

    // Connect WebSocket on page load
    connectWebSocket();
  </script>
</body>
</html>
    """


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"🎙️ Starting Kokoro-82M Streaming Server on http://{host}:{port}")
    uvicorn.run("api:app", host=host, port=port, log_level="info")
