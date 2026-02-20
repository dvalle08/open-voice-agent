const { Room, RoomEvent, Track, createLocalAudioTrack } = LivekitClient;

const statusEl = document.getElementById("status");
const statusDot = document.getElementById("status-dot");
const connectBtn = document.getElementById("connect");
const disconnectBtn = document.getElementById("disconnect");
const muteBtn = document.getElementById("mute");
const canvas = document.getElementById("wave");
const remoteAudio = document.getElementById("remote-audio");
const ctx = canvas.getContext("2d");

let room = null;
let localTrack = null;
let analyser = null;
let audioContext = null;
let animationId = null;
let muted = false;
let resizeObserver = null;
let currentSessionId = null;

let metricsHistory = [];
let turnCount = 0;
let averages = {
  eouDelay: [],
  sttFinalization: [],
  llmTtft: [],
  llmToTtsHandoff: [],
  ttsTtfb: [],
  totalLatency: [],
};
const LIVE_METRIC_IDS = [
  "eou",
  "stt-finalization",
  "llm-ttft",
  "llm-handoff",
  "tts-ttfb",
  "total",
];
let activeLiveSpeechId = null;

// Initialize canvas sizing on load
window.addEventListener('DOMContentLoaded', () => {
  resizeCanvas();

  // Watch for container size changes
  resizeObserver = new ResizeObserver(() => {
    resizeCanvas();
  });
  resizeObserver.observe(canvas.parentElement);
});

function setStatus(text, state) {
  statusEl.textContent = text;
  statusDot.className = "status-dot";
  if (state === "connected") statusDot.classList.add("connected");
  else if (state === "connecting") statusDot.classList.add("connecting");
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

  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const barCount = 64;
  const step = Math.floor(bufferLength / barCount);
  const gap = 3;
  const barWidth = (w - gap * (barCount - 1)) / barCount;
  const centerY = h / 2;
  const maxBarHeight = h * 0.85;

  for (let i = 0; i < barCount; i++) {
    const raw = dataArray[i * step] || 0;
    const normalized = raw / 255;
    const eased = normalized * normalized;
    const barHeight = Math.max(3, eased * maxBarHeight);
    const halfHeight = barHeight / 2;
    const x = i * (barWidth + gap);
    const y = centerY - halfHeight;

    const intensity = 0.25 + normalized * 0.75;
    const r = Math.round(108 * intensity);
    const g = Math.round(143 * intensity);
    const b = Math.round(255 * intensity);

    ctx.beginPath();
    const radius = Math.min(barWidth / 2, 3);
    roundRect(ctx, x, y, barWidth, barHeight, radius);
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.4 + normalized * 0.6})`;
    ctx.fill();

    if (normalized > 0.3) {
      ctx.shadowColor = `rgba(108, 143, 255, ${normalized * 0.4})`;
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowColor = "transparent";
      ctx.shadowBlur = 0;
    }
  }

  animationId = window.requestAnimationFrame(drawWave);
}

function roundRect(context, x, y, w, h, r) {
  if (w < 2 * r) r = w / 2;
  if (h < 2 * r) r = h / 2;
  context.moveTo(x + r, y);
  context.arcTo(x + w, y, x + w, y + h, r);
  context.arcTo(x + w, y + h, x, y + h, r);
  context.arcTo(x, y + h, x, y, r);
  context.arcTo(x, y, x + w, y, r);
  context.closePath();
}

function resizeCanvas() {
  const container = canvas.parentElement;
  const containerWidth = container.clientWidth;
  const containerHeight = container.clientHeight;

  // Maintain aspect ratio, but scale to fit container
  const maxWidth = 900;
  const aspectRatio = 900 / 200; // Original aspect ratio

  let canvasWidth = Math.min(containerWidth, maxWidth);
  let canvasHeight = Math.round(canvasWidth / aspectRatio);

  // If height exceeds container, scale down
  if (canvasHeight > containerHeight - 40) { // 40px for padding
    canvasHeight = containerHeight - 40;
    canvasWidth = Math.round(canvasHeight * aspectRatio);
  }

  // Ensure minimum size
  canvasWidth = Math.max(canvasWidth, 400);
  canvasHeight = Math.max(canvasHeight, 150);

  // Update canvas dimensions
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;
  canvas.style.width = `${canvasWidth}px`;
  canvas.style.height = `${canvasHeight}px`;
}

function setupAnalyser(track) {
  if (!track) return;

  audioContext = new AudioContext();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 512;
  analyser.smoothingTimeConstant = 0.8;

  const stream = new MediaStream([track.mediaStreamTrack]);
  const source = audioContext.createMediaStreamSource(stream);
  source.connect(analyser);

  drawWave();
}

async function connectToRoom() {
  if (!TOKEN || !LIVEKIT_URL) {
    setStatus("Missing token or URL", "");
    return;
  }

  currentSessionId = crypto.randomUUID();

  setStatus("Connecting...", "connecting");
  connectBtn.disabled = true;

  room = new Room();
  room.on(RoomEvent.TrackSubscribed, (track) => {
    if (track.kind === Track.Kind.Audio) {
      track.attach(remoteAudio);
      setStatus("Agent streaming", "connected");
    }
  });

  room.on(RoomEvent.Disconnected, () => {
    setStatus("Disconnected", "");
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
          handleLiveTurnBoundary(metricsData);
          updateLiveMetrics(metricsData);
        } else if (metricsData.type === "conversation_turn") {
          updateLiveMetrics(metricsData);
          renderTurn(metricsData);
        }
      } catch (error) {
        console.error("Failed to parse metrics:", error);
      }
    }
  });

  await room.connect(LIVEKIT_URL, TOKEN);

  try {
    await room.localParticipant.publishData(
      new TextEncoder().encode(
        JSON.stringify({
          type: "session_meta",
          session_id: currentSessionId,
          participant_id: room.localParticipant.identity
        })
      ),
      {
        reliable: true,
        topic: "session_meta"
      }
    );
  } catch (error) {
    console.warn("Failed to publish session metadata:", error);
  }

  localTrack = await createLocalAudioTrack();
  await room.localParticipant.publishTrack(localTrack);

  setupAnalyser(localTrack);

  disconnectBtn.disabled = false;
  muteBtn.disabled = false;
  setStatus("Mic streaming", "connected");
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
  if (resizeObserver) {
    resizeObserver.disconnect();
    resizeObserver = null;
  }
  analyser = null;
  clearWave();
}

async function disconnectRoom() {
  if (!room) return;
  setStatus("Disconnecting...", "connecting");
  await room.disconnect();
  room = null;
  localTrack = null;
  muted = false;
  muteBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg> Mute`;
  cleanupWave();
  resetMetrics();
}

function resetMetrics() {
  metricsHistory = [];
  turnCount = 0;
  activeLiveSpeechId = null;
  averages = {
    eouDelay: [],
    sttFinalization: [],
    llmTtft: [],
    llmToTtsHandoff: [],
    ttsTtfb: [],
    totalLatency: [],
  };

  clearAllLiveMetrics();

  [
    "avg-eou",
    "avg-stt-finalization",
    "avg-llm-ttft",
    "avg-llm-handoff",
    "avg-tts-ttfb",
    "avg-total",
  ].forEach((id) => {
    document.getElementById(id).innerHTML = '-- <span class="unit">s</span>';
  });
}

function handleLiveTurnBoundary(metricsData) {
  if (metricsData.stage !== "eou") return;

  const speechId = metricsData.speech_id;
  if (!speechId) {
    clearAllLiveMetrics();
    activeLiveSpeechId = null;
    return;
  }

  if (speechId === activeLiveSpeechId) return;
  activeLiveSpeechId = speechId;
  clearAllLiveMetrics();
}

async function toggleMute() {
  if (!room) return;
  muted = !muted;
  await room.localParticipant.setMicrophoneEnabled(!muted);
  if (muted) {
    muteBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="2" x2="22" y1="2" y2="22"/><path d="M18.89 13.23A7.12 7.12 0 0 0 19 12v-2"/><path d="M5 10v2a7 7 0 0 0 12 5"/><path d="M15 9.34V5a3 3 0 0 0-5.68-1.33"/><path d="M9 9v3a3 3 0 0 0 5.12 2.12"/><line x1="12" x2="12" y1="19" y2="22"/></svg> Unmute`;
    setStatus("Mic muted", "connected");
  } else {
    muteBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg> Mute`;
    setStatus("Mic streaming", "connected");
  }
}

connectBtn.addEventListener("click", () => {
  connectToRoom().catch((error) => {
    setStatus(`Failed: ${error.message}`, "");
    connectBtn.disabled = false;
    cleanupWave();
  });
});

disconnectBtn.addEventListener("click", () => {
  disconnectRoom().catch((error) => {
    setStatus(`Error: ${error.message}`, "");
  });
});

muteBtn.addEventListener("click", () => {
  toggleMute().catch((error) => {
    setStatus(`Error: ${error.message}`, "");
  });
});

function getLatencyClass(value, warningThreshold, criticalThreshold) {
  if (value >= criticalThreshold) return "critical";
  if (value >= warningThreshold) return "warning";
  return "";
}

function getTpsClass(value, warningThreshold, criticalThreshold) {
  if (value <= criticalThreshold) return "critical";
  if (value <= warningThreshold) return "warning";
  return "";
}

function setLiveMetric(metricId, value, maxValue, warningThreshold, criticalThreshold, options) {
  const bar = document.getElementById(`live-${metricId}-bar`);
  const label = document.getElementById(`live-${metricId}`);
  if (value === undefined || value === null || Number.isNaN(value)) return;

  const percent = Math.min((value / maxValue) * 100, 100);
  const invertedThresholds = options && options.inverted;
  const cls = invertedThresholds
    ? getTpsClass(value, warningThreshold, criticalThreshold)
    : getLatencyClass(value, warningThreshold, criticalThreshold);

  const suffix = (options && options.suffix) || "s";
  const decimals = (options && options.decimals !== undefined) ? options.decimals : 2;
  label.textContent = decimals > 0 ? `${value.toFixed(decimals)}${suffix}` : `${Math.round(value)} ${suffix}`;
  label.className = "metric-card-value" + (cls ? ` ${cls}` : "");
  bar.style.width = `${percent}%`;
  bar.className = "metric-card-fill" + (cls ? ` ${cls}` : "");
}

function clearLiveMetric(metricId) {
  const label = document.getElementById(`live-${metricId}`);
  const bar = document.getElementById(`live-${metricId}-bar`);
  label.textContent = "--";
  label.className = "metric-card-value";
  bar.style.width = "0%";
  bar.className = "metric-card-fill";
}

function clearAllLiveMetrics() {
  LIVE_METRIC_IDS.forEach((id) => clearLiveMetric(id));
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function updateLiveMetrics(turn) {
  const metrics = turn.metrics || {};
  const latencies = turn.latencies || {};

  const eouDelay = latencies.eou_delay ?? latencies.vad_detection_delay;
  if (isFiniteNumber(eouDelay)) {
    setLiveMetric("eou", eouDelay, 4.0, 0.8, 1.2);
  }

  const sttFinalizationDelay = latencies.stt_finalization_delay;
  if (isFiniteNumber(sttFinalizationDelay)) {
    setLiveMetric("stt-finalization", sttFinalizationDelay, 3.0, 0.4, 0.8);
  }

  const llmTtft = metrics.llm?.ttft;
  if (isFiniteNumber(llmTtft)) {
    setLiveMetric("llm-ttft", llmTtft, 4.0, 0.5, 1.0);
  }

  const llmToTtsHandoff = latencies.llm_to_tts_handoff_latency;
  if (isFiniteNumber(llmToTtsHandoff)) {
    setLiveMetric("llm-handoff", llmToTtsHandoff, 8.0, 1.5, 3.0);
  }

  const ttsTtfb = metrics.tts?.ttfb;
  if (isFiniteNumber(ttsTtfb)) {
    setLiveMetric("tts-ttfb", ttsTtfb, 4.0, 0.3, 0.6);
  }

  const totalLatency = latencies.total_latency;
  if (isFiniteNumber(totalLatency)) {
    setLiveMetric("total", totalLatency, 8.0, 1.5, 3.0);
  }
}

function updateAverages() {
  const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;

  const avgEou = avg(averages.eouDelay);
  const avgSttFinalization = avg(averages.sttFinalization);
  const avgLlmTtft = avg(averages.llmTtft);
  const avgLlmToTtsHandoff = avg(averages.llmToTtsHandoff);
  const avgTtsTtfb = avg(averages.ttsTtfb);
  const avgTotalLatency = avg(averages.totalLatency);

  const setAverageValue = (id, value) => {
    document.getElementById(id).innerHTML = value !== null
      ? `${value.toFixed(2)} <span class="unit">s</span>`
      : '-- <span class="unit">s</span>';
  };

  setAverageValue("avg-eou", avgEou);
  setAverageValue("avg-stt-finalization", avgSttFinalization);
  setAverageValue("avg-llm-ttft", avgLlmTtft);
  setAverageValue("avg-llm-handoff", avgLlmToTtsHandoff);
  setAverageValue("avg-tts-ttfb", avgTtsTtfb);
  setAverageValue("avg-total", avgTotalLatency);
}

function renderTurn(turn) {
  metricsHistory.push(turn);
  const latencies = turn.latencies || {};
  const metrics = turn.metrics || {};

  const eouDelay = latencies.eou_delay ?? latencies.vad_detection_delay;
  if (isFiniteNumber(eouDelay) && eouDelay > 0) averages.eouDelay.push(eouDelay);

  const sttFinalization = latencies.stt_finalization_delay;
  if (isFiniteNumber(sttFinalization) && sttFinalization > 0) {
    averages.sttFinalization.push(sttFinalization);
  }

  const llmTtft = metrics.llm?.ttft;
  if (isFiniteNumber(llmTtft) && llmTtft > 0) averages.llmTtft.push(llmTtft);

  const llmToTtsHandoff = latencies.llm_to_tts_handoff_latency;
  if (isFiniteNumber(llmToTtsHandoff) && llmToTtsHandoff > 0) {
    averages.llmToTtsHandoff.push(llmToTtsHandoff);
  }

  const ttsTtfb = metrics.tts?.ttfb;
  if (isFiniteNumber(ttsTtfb) && ttsTtfb > 0) averages.ttsTtfb.push(ttsTtfb);

  const totalLatency = latencies.total_latency;
  if (isFiniteNumber(totalLatency) && totalLatency > 0) {
    averages.totalLatency.push(totalLatency);
  }

  updateAverages();
}
