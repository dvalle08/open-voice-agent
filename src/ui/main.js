const { Room, RoomEvent, Track, createLocalAudioTrack } = LivekitClient;

const statusEl = document.getElementById("status");
const connectBtn = document.getElementById("connect");
const disconnectBtn = document.getElementById("disconnect");
const muteBtn = document.getElementById("mute");
const canvas = document.getElementById("wave");
const remoteAudio = document.getElementById("remote-audio");
const metricsLog = document.getElementById("metrics-log");
const ctx = canvas.getContext("2d");

let room = null;
let localTrack = null;
let analyser = null;
let audioContext = null;
let animationId = null;
let muted = false;

// Metrics tracking
let metricsHistory = [];
let averages = {
  totalLatency: [],
  vadDetectionDelay: [],
  llmTtft: [],
  ttsTtfb: [],
  sttDuration: [],
  tokensPerSecond: []
};

function setStatus(text) {
  statusEl.textContent = text;
}

function clearWave() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawWave() {
  if (!analyser) {
    clearWave();
    return;
  }

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteFrequencyData(dataArray);

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const barCount = 48;
  const step = Math.floor(bufferLength / barCount);
  const barWidth = canvas.width / barCount;

  for (let i = 0; i < barCount; i += 1) {
    const value = dataArray[i * step] || 0;
    const normalized = value / 255;
    const barHeight = Math.max(6, normalized * canvas.height);
    const halfHeight = barHeight / 2;
    const centerY = canvas.height / 2;
    const y = centerY - halfHeight;
    const x = i * barWidth;
    ctx.fillStyle = "#5b8cff";
    ctx.fillRect(x + 2, y, barWidth - 4, barHeight);
  }

  animationId = window.requestAnimationFrame(drawWave);
}

function setupAnalyser(track) {
  if (!track) {
    return;
  }

  audioContext = new AudioContext();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 512;

  const stream = new MediaStream([track.mediaStreamTrack]);
  const source = audioContext.createMediaStreamSource(stream);
  source.connect(analyser);

  drawWave();
}

async function connectToRoom() {
  if (!TOKEN || !LIVEKIT_URL) {
    setStatus("Missing LiveKit token or URL.");
    return;
  }

  setStatus("Connecting to LiveKit...");
  connectBtn.disabled = true;

  room = new Room();
  room.on(RoomEvent.TrackSubscribed, (track) => {
    if (track.kind === Track.Kind.Audio) {
      track.attach(remoteAudio);
      setStatus("Connected - agent audio streaming.");
    }
  });

  room.on(RoomEvent.Disconnected, () => {
    setStatus("Disconnected.");
    connectBtn.disabled = false;
    disconnectBtn.disabled = true;
    muteBtn.disabled = true;
    cleanupWave();
  });

  room.on(RoomEvent.DataReceived, (data, participant, kind, topic) => {
    if (topic === "metrics") {
      const decoder = new TextDecoder("utf-8");
      const jsonStr = decoder.decode(data);
      try {
        const metricsData = JSON.parse(jsonStr);
        if (metricsData.type === "metrics_live_update") {
          if (metricsData.diagnostic === true) return;
          const shouldReset = metricsData.stage === "stt";
          updateLiveMetrics(metricsData, shouldReset);
        } else if (metricsData.type === "conversation_turn") {
          updateLiveMetrics(metricsData, false);
          renderMetrics(metricsData);
        }
      } catch (error) {
        console.error("Failed to parse metrics:", error);
      }
    }
  });

  await room.connect(LIVEKIT_URL, TOKEN);

  localTrack = await createLocalAudioTrack();
  await room.localParticipant.publishTrack(localTrack);

  setupAnalyser(localTrack);

  disconnectBtn.disabled = false;
  muteBtn.disabled = false;
  setStatus("Connected - mic streaming.");
}

function cleanupWave() {
  if (animationId) {
    window.cancelAnimationFrame(animationId);
    animationId = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  analyser = null;
  clearWave();
}

async function disconnectRoom() {
  if (!room) {
    return;
  }
  setStatus("Disconnecting...");
  await room.disconnect();
  room = null;
  localTrack = null;
  muted = false;
  muteBtn.textContent = "Mute mic";
  cleanupWave();
  resetMetrics();
}

function resetMetrics() {
  metricsHistory = [];
  averages = {
    totalLatency: [],
    vadDetectionDelay: [],
    llmTtft: [],
    ttsTtfb: [],
    sttDuration: [],
    tokensPerSecond: []
  };

  // Clear live metrics
  document.getElementById("live-total").textContent = "--";
  document.getElementById("live-llm").textContent = "--";
  document.getElementById("live-tts").textContent = "--";
  document.getElementById("live-stt").textContent = "--";
  document.getElementById("live-vad").textContent = "--";
  document.getElementById("live-total-bar").style.width = "0%";
  document.getElementById("live-llm-bar").style.width = "0%";
  document.getElementById("live-tts-bar").style.width = "0%";
  document.getElementById("live-stt-bar").style.width = "0%";
  document.getElementById("live-vad-bar").style.width = "0%";

  // Clear conversation history
  metricsLog.innerHTML = "";
}

async function toggleMute() {
  if (!room) {
    return;
  }
  muted = !muted;
  await room.localParticipant.setMicrophoneEnabled(!muted);
  muteBtn.textContent = muted ? "Unmute mic" : "Mute mic";
  setStatus(muted ? "Mic muted." : "Mic streaming.");
}

connectBtn.addEventListener("click", () => {
  connectToRoom().catch((error) => {
    setStatus(`Connection failed: ${error.message}`);
    connectBtn.disabled = false;
    cleanupWave();
  });
});

disconnectBtn.addEventListener("click", () => {
  disconnectRoom().catch((error) => {
    setStatus(`Disconnect failed: ${error.message}`);
  });
});

muteBtn.addEventListener("click", () => {
  toggleMute().catch((error) => {
    setStatus(`Mute failed: ${error.message}`);
  });
});

function setLiveMetric(metricId, value, maxLatency, warningThreshold, criticalThreshold) {
  const bar = document.getElementById(`live-${metricId}-bar`);
  const label = document.getElementById(`live-${metricId}`);
  if (value === undefined || value === null || Number.isNaN(value)) {
    return;
  }
  const percent = Math.min((value / maxLatency) * 100, 100);
  label.textContent = `${value.toFixed(2)}s`;
  bar.style.width = `${percent}%`;
  bar.className = "metric-bar-fill " + getLatencyClass(value, warningThreshold, criticalThreshold);
}

function clearLiveMetric(metricId) {
  document.getElementById(`live-${metricId}`).textContent = "--";
  document.getElementById(`live-${metricId}-bar`).style.width = "0%";
}

function updateLiveMetrics(turn, resetMissing = true) {
  const maxLatency = 6.0; // 6 seconds max for visualization
  const metrics = turn.metrics || {};
  const latencies = turn.latencies || {};

  if (latencies.total_latency !== undefined) {
    setLiveMetric("total", latencies.total_latency, maxLatency, 1.0, 2.0);
  } else if (resetMissing) {
    clearLiveMetric("total");
  }

  const vadDetectionDelay = latencies.vad_detection_delay ?? latencies.eou_delay;
  if (vadDetectionDelay !== undefined) {
    setLiveMetric("vad", vadDetectionDelay, maxLatency, 0.4, 0.8);
  } else if (resetMissing) {
    clearLiveMetric("vad");
  }

  if (metrics.llm && metrics.llm.ttft !== undefined) {
    setLiveMetric("llm", metrics.llm.ttft, maxLatency, 0.5, 1.0);
  } else {
    if (resetMissing) clearLiveMetric("llm");
  }

  if (metrics.tts && metrics.tts.ttfb !== undefined) {
    setLiveMetric("tts", metrics.tts.ttfb, maxLatency, 0.3, 0.6);
  } else {
    if (resetMissing) clearLiveMetric("tts");
  }

  if (metrics.stt) {
    const sttDisplayDuration = metrics.stt.display_duration ?? metrics.stt.duration;
    if (sttDisplayDuration !== undefined) {
      setLiveMetric("stt", sttDisplayDuration, maxLatency, 0.2, 0.5);
    } else if (resetMissing) {
      clearLiveMetric("stt");
    }
  } else {
    if (resetMissing) clearLiveMetric("stt");
  }
}

function getLatencyClass(value, warningThreshold, criticalThreshold) {
  if (value >= criticalThreshold) return "critical";
  if (value >= warningThreshold) return "warning";
  return "";
}

function renderMetrics(turn) {
  // Track for averages
  metricsHistory.push(turn);
  if (turn.latencies?.total_latency > 0) {
    averages.totalLatency.push(turn.latencies.total_latency);
  }
  const vadDelay = turn.latencies?.vad_detection_delay ?? turn.latencies?.eou_delay ?? 0;
  if (vadDelay > 0) {
    averages.vadDetectionDelay.push(vadDelay);
  }
  if (turn.metrics?.llm?.ttft > 0) {
    averages.llmTtft.push(turn.metrics.llm.ttft);
  }
  if (turn.metrics?.tts?.ttfb > 0) {
    averages.ttsTtfb.push(turn.metrics.tts.ttfb);
  }
  if (turn.metrics?.stt) {
    const sttDuration = turn.metrics.stt.display_duration ?? turn.metrics.stt.duration ?? 0;
    if (sttDuration > 0) averages.sttDuration.push(sttDuration);
  }
  if (turn.metrics?.llm?.tokens_per_second > 0) {
    averages.tokensPerSecond.push(turn.metrics.llm.tokens_per_second);
  }

  // Add to conversation history
  const card = document.createElement("div");
  card.className = "turn-card";

  const timestamp = new Date(turn.timestamp * 1000).toLocaleTimeString();
  const roleLabel = turn.role.toUpperCase();

  let metricsHTML = "";

  if (turn.latencies) {
    metricsHTML = `
      <div class="metric-row">
        <div class="metric">Total Latency: <span class="metric-value">${turn.latencies.total_latency.toFixed(2)}s</span></div>
        <div class="metric">VAD Detection Delay: <span class="metric-value">${vadDelay.toFixed(2)}s</span></div>
        ${
          turn.metrics?.llm
            ? `<div class="metric">LLM TTFT: <span class="metric-value">${turn.metrics.llm.ttft.toFixed(2)}s</span></div>`
            : ""
        }
        ${
          turn.metrics?.tts
            ? `<div class="metric">TTS TTFB: <span class="metric-value">${turn.metrics.tts.ttfb.toFixed(2)}s</span></div>`
            : ""
        }
        ${
          turn.metrics?.stt
            ? `<div class="metric">STT Duration: <span class="metric-value">${(turn.metrics.stt.display_duration ?? turn.metrics.stt.duration).toFixed(2)}s</span></div>`
            : ""
        }
        ${
          turn.metrics?.llm
            ? `<div class="metric">Tokens/sec: <span class="metric-value">${turn.metrics.llm.tokens_per_second.toFixed(1)}</span></div>`
            : ""
        }
      </div>
    `;
  }

  const transcript = turn.transcript || "";
  const truncatedTranscript =
    transcript.length > 200
      ? transcript.substring(0, 200) + "..."
      : transcript;

  card.innerHTML = `
    <div class="turn-header">
      <strong>${roleLabel}</strong>
      <span>${timestamp}</span>
    </div>
    <div class="turn-transcript">${truncatedTranscript || "(no transcript)"}</div>
    ${metricsHTML}
  `;

  metricsLog.insertBefore(card, metricsLog.firstChild);

  while (metricsLog.children.length > 20) {
    metricsLog.removeChild(metricsLog.lastChild);
  }
}
