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
let remoteAudioTrack = null;
let analyser = null;
let audioContext = null;
let animationId = null;
let muted = false;
let resizeObserver = null;
let currentSessionId = null;
let currentRoomName = null;
let activeConnectionSeq = 0;
let connectionState = "idle";
const AUDIO_DIAGNOSTICS = false;

const CONNECTION_STATES = Object.freeze({
  IDLE: "idle",
  CONNECTING: "connecting",
  CONNECTED: "connected",
  DISCONNECTING: "disconnecting",
});

let averages = {
  eouDelay: [],
  llmTtft: [],
  voiceGeneration: [],
  totalLatency: [],
};
const LIVE_METRIC_IDS = [
  "eou",
  "llm-ttft",
  "voice-generation",
  "total",
];
const liveTotalLabelEl = document.getElementById("live-total-label");
const liveTotalTechTextEl = document.getElementById("live-total-tech-text");
const liveTotalTooltipTextEl = document.getElementById("live-total-tooltip-text");
const toolPhaseShellEl = document.getElementById("tool-phase-shell");
const toolPhaseToggleEl = document.getElementById("tool-phase-toggle");
const toolPhaseContentEl = document.getElementById("tool-phase-content");
const toolPhaseCountEl = document.getElementById("tool-phase-count");
const toolPhaseTagEl = document.getElementById("live-tool-tag");
const toolTitleEl = document.getElementById("live-tool-title");
const toolDescEl = document.getElementById("live-tool-desc");
const toolListEl = document.getElementById("live-tool-list");
const postToolStageRowEl = document.getElementById("post-tool-stage-row");
const totalTurnCardEl = document.getElementById("total-turn-card");
const traceDropdownEl = document.getElementById("trace-dropdown");
const traceDropdownToggleEl = document.getElementById("trace-dropdown-toggle");
const traceDropdownLabelEl = document.getElementById("trace-dropdown-label");
const traceDropdownMenuEl = document.getElementById("trace-dropdown-menu");
const traceDropdownListEl = document.getElementById("trace-dropdown-list");
const traceDropdownEmptyEl = document.getElementById("trace-dropdown-empty");
let activeLiveSpeechId = null;
let liveTurnValues = createEmptyLiveTurnValues();
let toolPhaseExpanded = true;
let toolTurnActive = false;
let langfuseEnabled = false;
let langfuseHost = null;
let langfuseProjectId = null;
let traceHistory = [];
const TRACE_HISTORY_LIMIT = 100;

function normalizeNonEmptyString(value) {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeLangfuseHost(value) {
  const normalized = normalizeNonEmptyString(value);
  if (!normalized) return null;
  const withoutSlash = normalized.replace(/\/+$/, "");
  if (!/^https?:\/\//i.test(withoutSlash)) return null;
  return withoutSlash;
}

function buildLangfuseTraceUrl(host, projectId, traceId) {
  return `${host}/project/${encodeURIComponent(projectId)}/traces/${encodeURIComponent(traceId)}`;
}

function formatTraceCreatedAtLocal(epochSeconds) {
  if (typeof epochSeconds !== "number" || !Number.isFinite(epochSeconds)) {
    return "--";
  }
  const date = new Date(epochSeconds * 1000);
  if (Number.isNaN(date.getTime())) {
    return "--";
  }
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function closeTraceDropdown() {
  if (traceDropdownMenuEl) {
    traceDropdownMenuEl.hidden = true;
  }
  if (traceDropdownToggleEl) {
    traceDropdownToggleEl.setAttribute("aria-expanded", "false");
  }
}

function renderTraceHistory() {
  if (!traceDropdownListEl || !traceDropdownEmptyEl) return;
  traceDropdownListEl.innerHTML = "";
  if (traceHistory.length === 0) {
    traceDropdownEmptyEl.hidden = false;
    return;
  }

  traceDropdownEmptyEl.hidden = true;
  traceHistory.forEach((item) => {
    const row = document.createElement("div");
    row.className = "trace-entry";

    const meta = document.createElement("div");
    meta.className = "trace-entry-meta";

    const idEl = document.createElement("span");
    idEl.className = "trace-entry-id";
    idEl.textContent = item.traceId;
    idEl.title = item.traceId;

    const timeEl = document.createElement("span");
    timeEl.className = "trace-entry-time";
    timeEl.textContent = `Created at ${formatTraceCreatedAtLocal(item.createdAt)}`;

    const openLink = document.createElement("a");
    openLink.className = "trace-entry-open";
    openLink.href = item.traceUrl;
    openLink.target = "_blank";
    openLink.rel = "noopener noreferrer";
    openLink.textContent = "Open Trace";

    meta.appendChild(idEl);
    meta.appendChild(timeEl);
    row.appendChild(meta);
    row.appendChild(openLink);
    traceDropdownListEl.appendChild(row);
  });
}

function updateTraceDropdownUI() {
  const hasLinkConfig = Boolean(langfuseEnabled && langfuseHost && langfuseProjectId);
  let label = "Connect to initialize trace links.";
  let toggleDisabled = true;

  if (!langfuseEnabled) {
    label = "Langfuse tracing is disabled for this session.";
  } else if (!hasLinkConfig) {
    label = "Langfuse host/project is missing.";
  } else if (traceHistory.length === 0) {
    label = "Waiting for first trace...";
    toggleDisabled = false;
  } else {
    label = `Traces (${traceHistory.length})`;
    toggleDisabled = false;
  }

  if (traceDropdownLabelEl) {
    traceDropdownLabelEl.textContent = label;
  }

  if (traceDropdownToggleEl) {
    traceDropdownToggleEl.disabled = toggleDisabled;
    if (toggleDisabled) {
      closeTraceDropdown();
    }
  }

  if (traceDropdownEmptyEl) {
    if (!langfuseEnabled) {
      traceDropdownEmptyEl.textContent = "Langfuse tracing is disabled.";
    } else if (!hasLinkConfig) {
      traceDropdownEmptyEl.textContent = "Langfuse host/project is missing; links unavailable.";
    } else {
      traceDropdownEmptyEl.textContent = "No traces yet.";
    }
  }

  renderTraceHistory();
}

function resetTracePanel() {
  langfuseEnabled = false;
  langfuseHost = null;
  langfuseProjectId = null;
  traceHistory = [];
  closeTraceDropdown();
  updateTraceDropdownUI();
}

function configureTracePanelFromBootstrap(bootstrap) {
  langfuseEnabled = bootstrap && bootstrap.langfuse_enabled === true;
  langfuseHost = normalizeLangfuseHost(bootstrap ? bootstrap.langfuse_host : null);
  langfuseProjectId = normalizeNonEmptyString(bootstrap ? bootstrap.langfuse_project_id : null);
  traceHistory = [];
  closeTraceDropdown();
  updateTraceDropdownUI();
}

function handleTraceUpdate(payload) {
  if (!payload || payload.type !== "trace_update") return;
  if (!currentSessionId) return;

  const payloadSessionId = normalizeNonEmptyString(payload.session_id);
  if (payloadSessionId && payloadSessionId !== currentSessionId) return;

  const traceId = normalizeNonEmptyString(payload.trace_id);
  if (!traceId) return;

  if (!(langfuseEnabled && langfuseHost && langfuseProjectId)) {
    updateTraceDropdownUI();
    return;
  }

  const timestampRaw = (typeof payload.timestamp === "number" && Number.isFinite(payload.timestamp))
    ? payload.timestamp
    : (Date.now() / 1000);
  const timestamp = timestampRaw > 1e12 ? (timestampRaw / 1000) : timestampRaw;
  const traceUrl = buildLangfuseTraceUrl(langfuseHost, langfuseProjectId, traceId);

  traceHistory = traceHistory.filter((item) => item.traceId !== traceId);
  traceHistory.push({
    traceId,
    traceUrl,
    createdAt: timestamp,
  });
  traceHistory.sort((a, b) => {
    if (a.createdAt !== b.createdAt) {
      return a.createdAt - b.createdAt;
    }
    return a.traceId.localeCompare(b.traceId);
  });
  if (traceHistory.length > TRACE_HISTORY_LIMIT) {
    traceHistory = traceHistory.slice(-TRACE_HISTORY_LIMIT);
  }
  updateTraceDropdownUI();
}

function createEmptyLiveTurnValues() {
  return {
    eouDelay: null,
    llmTtft: null,
    ttsTtfb: null,
  };
}

function setToolPhaseExpanded(expanded) {
  toolPhaseExpanded = Boolean(expanded);
  if (toolPhaseToggleEl) {
    toolPhaseToggleEl.setAttribute("aria-expanded", toolPhaseExpanded ? "true" : "false");
  }
  if (toolPhaseContentEl) {
    toolPhaseContentEl.hidden = !toolPhaseExpanded;
  }
}

function formatSeconds(seconds) {
  if (!isFiniteNumber(seconds)) return "--";
  return `${seconds.toFixed(2)}s`;
}

function setValueAndBar(valueElId, barElId, seconds, maxSeconds, options) {
  const valueEl = document.getElementById(valueElId);
  const barEl = document.getElementById(barElId);
  if (!valueEl || !barEl) return;
  const loadingOnMissing = options && options.loadingOnMissing === true;
  if (!isFiniteNumber(seconds)) {
    valueEl.textContent = loadingOnMissing ? "coming..." : "--";
    valueEl.classList.toggle("loading", loadingOnMissing);
    barEl.style.width = "0%";
    return;
  }
  const pct = Math.min((seconds / maxSeconds) * 100, 100);
  valueEl.textContent = formatSeconds(seconds);
  valueEl.classList.remove("loading");
  barEl.style.width = `${pct}%`;
}

function secondsFromPhase(phases, phaseId) {
  if (!Array.isArray(phases)) return null;
  const phase = phases.find((item) => item && item.id === phaseId);
  if (!phase || typeof phase !== "object") return null;
  if (isFiniteNumber(phase.duration_seconds)) return phase.duration_seconds;
  if (isFiniteNumber(phase.duration_ms)) return phase.duration_ms / 1000;
  return null;
}

function clearToolList() {
  if (!toolListEl) return;
  toolListEl.innerHTML = "";
  const emptyEl = document.createElement("div");
  emptyEl.className = "tool-list-empty";
  emptyEl.textContent = "No tools executed.";
  toolListEl.appendChild(emptyEl);
}

function renderToolList(tools) {
  if (!toolListEl) return;
  toolListEl.innerHTML = "";
  if (!Array.isArray(tools) || tools.length === 0) {
    clearToolList();
    return;
  }

  tools.forEach((tool) => {
    const row = document.createElement("div");
    row.className = "tool-list-item";

    const nameEl = document.createElement("span");
    nameEl.className = "tool-list-name";
    nameEl.textContent = tool && typeof tool.name === "string" && tool.name.trim()
      ? tool.name
      : "tool_call";

    const durationEl = document.createElement("span");
    durationEl.className = "tool-list-duration";
    const seconds = isFiniteNumber(tool.duration_seconds)
      ? tool.duration_seconds
      : (isFiniteNumber(tool.duration_ms) ? tool.duration_ms / 1000 : null);
    durationEl.textContent = formatSeconds(seconds);

    row.appendChild(nameEl);
    row.appendChild(durationEl);
    toolListEl.appendChild(row);
  });
}

function setTotalCardMode(hasTools) {
  if (!liveTotalLabelEl || !liveTotalTechTextEl || !liveTotalTooltipTextEl) return;
  if (hasTools) {
    liveTotalLabelEl.textContent = "First Audio";
    liveTotalTechTextEl.textContent = "EOU to first assistant audio (pre-tool acknowledgment)";
    liveTotalTooltipTextEl.textContent = "Sum of the three stages above before tool execution.";
    return;
  }
  liveTotalLabelEl.textContent = "Total Round-Trip";
  liveTotalTechTextEl.textContent = "End-to-End Latency";
  liveTotalTooltipTextEl.textContent = "Sum of the three stages above: EOU delay + Thinking (LLM TTFT) + Voice Generation (TTS TTFB).";
}

function clearToolPipelineView() {
  toolTurnActive = false;
  if (toolPhaseShellEl) {
    toolPhaseShellEl.hidden = true;
  }
  setToolPhaseExpanded(true);
  if (toolPhaseCountEl) {
    toolPhaseCountEl.textContent = "0 tools called";
  }
  if (toolPhaseTagEl) {
    toolPhaseTagEl.textContent = "Tool Call";
  }
  if (toolTitleEl) {
    toolTitleEl.textContent = "Tool execution";
  }
  if (toolDescEl) {
    toolDescEl.textContent = "Executing external function calls";
  }
  clearToolList();
  setValueAndBar("live-tool-total", "live-tool-total-bar", null, 6.0);
  setValueAndBar("live-post-llm-ttft", "live-post-llm-ttft-bar", null, 4.0);
  setValueAndBar("live-post-voice-generation", "live-post-voice-generation-bar", null, 4.0);
  setValueAndBar("live-second-audio", "live-second-audio-bar", null, 8.0);
  setValueAndBar("live-total-turn", "live-total-turn-bar", null, 20.0);
}

function renderTurnPipelineSummary(summary) {
  if (!summary || summary.type !== "turn_pipeline_summary") return;
  const hasTools = summary.has_tools === true;
  const isPartial = summary.partial === true;
  toolTurnActive = hasTools;

  const phase1 = secondsFromPhase(summary.phases, 1);
  const phase2 = secondsFromPhase(summary.phases, 2);
  const phase3 = secondsFromPhase(summary.phases, 3);

  if (isFiniteNumber(phase1)) {
    liveTurnValues.eouDelay = phase1;
    setLiveMetric("eou", phase1, 4.0, 0.8, 1.2);
  }
  if (isFiniteNumber(phase2)) {
    liveTurnValues.llmTtft = phase2;
    setLiveMetric("llm-ttft", phase2, 4.0, 0.5, 1.0);
  }
  if (isFiniteNumber(phase3)) {
    liveTurnValues.ttsTtfb = phase3;
    setLiveMetric("voice-generation", phase3, 4.0, 0.6, 1.2);
  }

  const initialResponseTotalSeconds = computeVisibleResponseTotal(
    phase1,
    phase2,
    phase3,
  );
  const totalTurnSecondsRaw = isFiniteNumber(summary.total_turn_duration_seconds)
    ? summary.total_turn_duration_seconds
    : (isFiniteNumber(summary.total_turn_duration_ms)
      ? summary.total_turn_duration_ms / 1000
      : null);
  const totalTurnSeconds = isPartial ? null : totalTurnSecondsRaw;
  const secondAudioSeconds = isFiniteNumber(summary.second_audio_latency_seconds)
    ? summary.second_audio_latency_seconds
    : (isFiniteNumber(summary.second_audio_latency_ms)
      ? summary.second_audio_latency_ms / 1000
      : null);

  setTotalCardMode(hasTools);
  setValueAndBar(
    "live-total",
    "live-total-bar",
    initialResponseTotalSeconds,
    8.0,
    { loadingOnMissing: isPartial },
  );
  const totalAvgLabel = document.getElementById("live-total-avg");
  if (totalAvgLabel && hasTools) {
    totalAvgLabel.textContent = "";
  }

  if (!hasTools) {
    clearToolPipelineView();
    return;
  }

  const toolSectionWasHidden = toolPhaseShellEl ? toolPhaseShellEl.hidden : true;
  if (toolPhaseShellEl) {
    toolPhaseShellEl.hidden = false;
  }
  if (totalTurnCardEl) {
    totalTurnCardEl.hidden = false;
  }
  if (toolSectionWasHidden) {
    setToolPhaseExpanded(true);
  }

  const toolPhase = summary.tool_phase || {};
  const tools = Array.isArray(toolPhase.tools) ? toolPhase.tools : [];
  const toolCount = tools.length;
  if (toolPhaseCountEl) {
    toolPhaseCountEl.textContent = `${toolCount} tool${toolCount === 1 ? "" : "s"} called`;
  }
  if (toolPhaseTagEl) {
    toolPhaseTagEl.textContent = toolCount === 1 ? "Tool Call" : "Tool Calls";
  }
  if (toolTitleEl) {
    if (toolCount === 1 && tools[0] && typeof tools[0].name === "string" && tools[0].name.trim()) {
      toolTitleEl.textContent = tools[0].name;
    } else {
      toolTitleEl.textContent = `${toolCount} tools`;
    }
  }
  if (toolDescEl) {
    toolDescEl.textContent = (isPartial && tools.length === 0)
      ? "Waiting for tool execution..."
      : "Executing external function calls";
  }
  renderToolList(tools);

  const toolTotalSecondsRaw = isFiniteNumber(toolPhase.total_duration_seconds)
    ? toolPhase.total_duration_seconds
    : (isFiniteNumber(toolPhase.total_duration_ms)
      ? toolPhase.total_duration_ms / 1000
      : tools.reduce((sum, tool) => {
          if (isFiniteNumber(tool.duration_seconds)) return sum + tool.duration_seconds;
          if (isFiniteNumber(tool.duration_ms)) return sum + (tool.duration_ms / 1000);
          return sum;
        }, 0));
  const toolTotalSeconds = (isPartial && tools.length === 0) ? null : toolTotalSecondsRaw;
  setValueAndBar(
    "live-tool-total",
    "live-tool-total-bar",
    toolTotalSeconds,
    8.0,
    { loadingOnMissing: isPartial },
  );

  const postPhase5 = secondsFromPhase(summary.post_tool_phases, 5);
  const postPhase6 = secondsFromPhase(summary.post_tool_phases, 6);
  setValueAndBar(
    "live-post-llm-ttft",
    "live-post-llm-ttft-bar",
    postPhase5,
    4.0,
    { loadingOnMissing: isPartial },
  );
  setValueAndBar(
    "live-post-voice-generation",
    "live-post-voice-generation-bar",
    postPhase6,
    4.0,
    { loadingOnMissing: isPartial },
  );
  setValueAndBar(
    "live-second-audio",
    "live-second-audio-bar",
    secondAudioSeconds,
    8.0,
    { loadingOnMissing: isPartial },
  );
  setValueAndBar(
    "live-total-turn",
    "live-total-turn-bar",
    totalTurnSeconds,
    20.0,
    { loadingOnMissing: isPartial },
  );

  if (postToolStageRowEl) {
    postToolStageRowEl.hidden = false;
  }
}

// Initialize canvas sizing on load
window.addEventListener('DOMContentLoaded', () => {
  resizeCanvas();

  // Watch for container size changes
  resizeObserver = new ResizeObserver(() => {
    resizeCanvas();
  });
  resizeObserver.observe(canvas.parentElement);

  if (toolPhaseToggleEl) {
    toolPhaseToggleEl.addEventListener("click", () => {
      setToolPhaseExpanded(!toolPhaseExpanded);
    });
  }
  setToolPhaseExpanded(true);
  clearToolPipelineView();
  if (traceDropdownToggleEl) {
    traceDropdownToggleEl.addEventListener("click", () => {
      if (!traceDropdownMenuEl) return;
      const nextOpen = traceDropdownMenuEl.hidden;
      traceDropdownMenuEl.hidden = !nextOpen;
      traceDropdownToggleEl.setAttribute("aria-expanded", nextOpen ? "true" : "false");
    });
  }
  document.addEventListener("click", (event) => {
    if (!traceDropdownEl) return;
    if (!traceDropdownEl.contains(event.target)) {
      closeTraceDropdown();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeTraceDropdown();
    }
  });
  if (traceDropdownListEl) {
    traceDropdownListEl.addEventListener("click", () => {
      closeTraceDropdown();
    });
  }
});

function setStatus(text, state) {
  statusEl.textContent = text;
  statusDot.className = "status-dot";
  if (state === "connected") statusDot.classList.add("connected");
  else if (state === "connecting") statusDot.classList.add("connecting");
}

function setConnectionState(nextState) {
  connectionState = nextState;
  connectBtn.disabled = connectionState !== CONNECTION_STATES.IDLE;
  disconnectBtn.disabled = connectionState !== CONNECTION_STATES.CONNECTED;
  muteBtn.disabled = connectionState !== CONNECTION_STATES.CONNECTED;
}

function resetMuteButton() {
  muteBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg> Mute`;
}

function clearRemoteAudio() {
  remoteAudio.pause();
  remoteAudio.srcObject = null;
  remoteAudio.removeAttribute("src");
  remoteAudio.load();
}

function getMediaTrackSettings(track) {
  const mediaTrack = track && track.mediaStreamTrack;
  if (!mediaTrack || typeof mediaTrack.getSettings !== "function") {
    return {};
  }
  try {
    return mediaTrack.getSettings() || {};
  } catch (_error) {
    return {};
  }
}

function logAudioDiagnostics(eventName, details = {}) {
  if (!AUDIO_DIAGNOSTICS) return;
  console.info("[audio-diagnostics]", eventName, {
    timestamp: new Date().toISOString(),
    sessionId: currentSessionId,
    roomName: currentRoomName,
    ...details,
  });
}

function detachRemoteAudioTrack(track, reason) {
  if (!track || track.kind !== Track.Kind.Audio) return;

  try {
    track.detach(remoteAudio);
  } catch (error) {
    console.warn("Failed to detach remote audio track:", error);
  }

  if (remoteAudioTrack === track) {
    remoteAudioTrack = null;
  }
  clearRemoteAudio();
  logAudioDiagnostics("remote_track_detached", {
    reason,
    trackSid: track.sid || null,
  });
}

function attachRemoteAudioTrack(track, participant) {
  if (!track || track.kind !== Track.Kind.Audio) return;

  if (remoteAudioTrack && remoteAudioTrack !== track) {
    detachRemoteAudioTrack(remoteAudioTrack, "replaced_by_new_track");
  }

  remoteAudioTrack = track;
  track.attach(remoteAudio);

  const trackSettings = getMediaTrackSettings(track);
  logAudioDiagnostics("remote_track_subscribed", {
    participantIdentity: participant && participant.identity ? participant.identity : null,
    trackSid: track.sid || null,
    trackSampleRate: trackSettings.sampleRate ?? null,
    trackSettings,
    remotePlaybackRate: remoteAudio.playbackRate,
    remoteDefaultPlaybackRate: remoteAudio.defaultPlaybackRate,
  });

  const playPromise = remoteAudio.play();
  if (playPromise && typeof playPromise.catch === "function") {
    playPromise.catch((error) => {
      console.warn("Remote audio playback did not auto-start:", error);
    });
  }
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

  const localTrackSettings = getMediaTrackSettings(track);
  logAudioDiagnostics("local_analyser_ready", {
    localTrackSampleRate: localTrackSettings.sampleRate ?? null,
    localTrackSettings,
    audioContextSampleRate: audioContext.sampleRate,
  });

  drawWave();
}

async function fetchSessionBootstrap() {
  if (!SESSION_BOOTSTRAP_URL || !LIVEKIT_URL) {
    throw new Error("Missing LiveKit configuration");
  }

  const response = await fetch(`${SESSION_BOOTSTRAP_URL}?t=${Date.now()}`, {
    method: "GET",
    cache: "no-store",
  });

  if (!response.ok) {
    let message = `bootstrap request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.message === "string" && body.message) {
        message = body.message;
      }
    } catch (_ignored) {
      // Keep default message when response is not JSON.
    }
    throw new Error(message);
  }

  return response.json();
}

async function connectToRoom() {
  if (connectionState !== CONNECTION_STATES.IDLE) return;

  const connectionSeq = ++activeConnectionSeq;
  setConnectionState(CONNECTION_STATES.CONNECTING);
  setStatus("Preparing session...", "connecting");

  let nextRoom = null;
  try {
    const bootstrap = await fetchSessionBootstrap();
    if (connectionSeq !== activeConnectionSeq) return;

    currentSessionId = bootstrap.session_id || crypto.randomUUID();
    currentRoomName = bootstrap.room_name || null;
    configureTracePanelFromBootstrap(bootstrap);

    if (!bootstrap.token) {
      throw new Error("Session bootstrap did not return a token");
    }

    setStatus(
      `Connecting to ${currentRoomName || "room"}...`,
      "connecting"
    );

    nextRoom = new Room();
    room = nextRoom;
    resetMetrics();
    if (remoteAudioTrack) {
      detachRemoteAudioTrack(remoteAudioTrack, "before_new_room_connect");
    } else {
      clearRemoteAudio();
    }

    nextRoom.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
      if (room !== nextRoom || connectionSeq !== activeConnectionSeq) return;
      if (track.kind === Track.Kind.Audio) {
        attachRemoteAudioTrack(track, participant);
        setStatus("Agent streaming", "connected");
      }
    });

    nextRoom.on(RoomEvent.TrackUnsubscribed, (track, publication, participant) => {
      if (room !== nextRoom || connectionSeq !== activeConnectionSeq) return;
      if (track.kind !== Track.Kind.Audio) return;
      detachRemoteAudioTrack(track, "track_unsubscribed");
      logAudioDiagnostics("remote_track_unsubscribed", {
        participantIdentity: participant && participant.identity ? participant.identity : null,
        trackSid: track.sid || null,
      });
    });

    nextRoom.on(RoomEvent.Disconnected, () => {
      if (room !== nextRoom || connectionSeq !== activeConnectionSeq) return;
      room = null;
      if (localTrack) {
        localTrack.stop();
      }
      localTrack = null;
      currentSessionId = null;
      currentRoomName = null;
      resetTracePanel();
      muted = false;
      resetMuteButton();
      if (remoteAudioTrack) {
        detachRemoteAudioTrack(remoteAudioTrack, "room_disconnected");
      } else {
        clearRemoteAudio();
      }
      cleanupWave();
      resetMetrics();
      setConnectionState(CONNECTION_STATES.IDLE);
      setStatus("Disconnected", "");
    });

    nextRoom.on(RoomEvent.DataReceived, (data, participant, kind, topic) => {
      if (room !== nextRoom || connectionSeq !== activeConnectionSeq) return;
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
            if (metricsData.role === "agent") {
              updateLiveMetrics(metricsData);
            }
            renderTurn(metricsData);
          } else if (metricsData.type === "turn_pipeline_summary") {
            renderTurnPipelineSummary(metricsData);
          } else if (metricsData.type === "trace_update") {
            handleTraceUpdate(metricsData);
          }
        } catch (error) {
          console.error("Failed to parse metrics:", error);
        }
      }
    });

    await nextRoom.connect(LIVEKIT_URL, bootstrap.token);
    if (room !== nextRoom || connectionSeq !== activeConnectionSeq) return;

    localTrack = await createLocalAudioTrack();
    if (room !== nextRoom || connectionSeq !== activeConnectionSeq) {
      localTrack.stop();
      localTrack = null;
      return;
    }

    await nextRoom.localParticipant.publishTrack(localTrack);
    setupAnalyser(localTrack);

    muted = false;
    resetMuteButton();
    setConnectionState(CONNECTION_STATES.CONNECTED);
    setStatus(`Mic streaming (${currentRoomName || "connected"})`, "connected");
  } catch (error) {
    if (localTrack) {
      localTrack.stop();
      localTrack = null;
    }
    if (room === nextRoom) {
      try {
        await room.disconnect();
      } catch (disconnectError) {
        console.warn("Failed to disconnect after connect error:", disconnectError);
      }
      room = null;
    }
    currentSessionId = null;
    currentRoomName = null;
    resetTracePanel();
    muted = false;
    resetMuteButton();
    if (remoteAudioTrack) {
      detachRemoteAudioTrack(remoteAudioTrack, "connect_error");
    } else {
      clearRemoteAudio();
    }
    cleanupWave();
    resetMetrics();
    setConnectionState(CONNECTION_STATES.IDLE);
    throw error;
  }
}

function cleanupWave() {
  if (animationId) {
    window.cancelAnimationFrame(animationId);
    animationId = null;
  }
  if (audioContext) {
    const closeResult = audioContext.close();
    if (closeResult && typeof closeResult.catch === "function") {
      closeResult.catch(() => {});
    }
    audioContext = null;
  }
  analyser = null;
  clearWave();
}

async function disconnectRoom() {
  if (!room || connectionState !== CONNECTION_STATES.CONNECTED) return;

  const disconnectingRoom = room;
  const disconnectSeq = ++activeConnectionSeq;
  setConnectionState(CONNECTION_STATES.DISCONNECTING);
  setStatus("Disconnecting...", "connecting");

  try {
    if (localTrack) {
      try {
        await disconnectingRoom.localParticipant.unpublishTrack(localTrack);
      } catch (error) {
        console.warn("Failed to unpublish local track during disconnect:", error);
      }
      localTrack.stop();
      localTrack = null;
    }
    if (remoteAudioTrack) {
      detachRemoteAudioTrack(remoteAudioTrack, "manual_disconnect");
    } else {
      clearRemoteAudio();
    }
    await disconnectingRoom.disconnect();
  } finally {
    if (room === disconnectingRoom) {
      room = null;
    }
    if (disconnectSeq === activeConnectionSeq) {
      currentSessionId = null;
      currentRoomName = null;
      resetTracePanel();
      muted = false;
      resetMuteButton();
      cleanupWave();
      resetMetrics();
      setConnectionState(CONNECTION_STATES.IDLE);
      setStatus("Disconnected", "");
    }
  }
}

function resetMetrics() {
  activeLiveSpeechId = null;
  liveTurnValues = createEmptyLiveTurnValues();
  toolTurnActive = false;
  averages = {
    eouDelay: [],
    llmTtft: [],
    voiceGeneration: [],
    totalLatency: [],
  };

  clearAllLiveMetrics();
  clearToolPipelineView();
  setTotalCardMode(false);

  updateLiveMetricAverages();
}

function handleLiveTurnBoundary(metricsData) {
  if (metricsData.stage !== "eou") return;

  const speechId = metricsData.speech_id;
  if (!speechId) {
    clearAllLiveMetrics();
    clearToolPipelineView();
    setTotalCardMode(false);
    activeLiveSpeechId = null;
    liveTurnValues = createEmptyLiveTurnValues();
    return;
  }

  if (speechId === activeLiveSpeechId) return;
  activeLiveSpeechId = speechId;
  liveTurnValues = createEmptyLiveTurnValues();
  clearToolPipelineView();
  setTotalCardMode(false);
  setAllLiveMetricsLoading();
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

resetMuteButton();
setConnectionState(CONNECTION_STATES.IDLE);
setTotalCardMode(false);
clearToolPipelineView();
resetTracePanel();

connectBtn.addEventListener("click", () => {
  connectToRoom().catch((error) => {
    setStatus(`Failed: ${error.message}`, "");
    setConnectionState(CONNECTION_STATES.IDLE);
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

function getLiveMetricValueBaseClass(metricId) {
  return metricId === "total"
    ? "metric-card-value pipeline-total-value"
    : "metric-card-value";
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
  label.className = getLiveMetricValueBaseClass(metricId) + (cls ? ` ${cls}` : "");
  bar.style.width = `${percent}%`;
  bar.className = "metric-card-fill" + (cls ? ` ${cls}` : "");
}

function setLiveMetricAverage(metricId, value) {
  const averageLabel = document.getElementById(`live-${metricId}-avg`);
  if (!averageLabel) return;
  averageLabel.textContent = value !== null ? `avg ${value.toFixed(2)}s` : "";
}

function setLiveMetricLoading(metricId) {
  const label = document.getElementById(`live-${metricId}`);
  const bar = document.getElementById(`live-${metricId}-bar`);
  label.textContent = "coming...";
  label.className = `${getLiveMetricValueBaseClass(metricId)} loading`;
  bar.style.width = "0%";
  bar.className = "metric-card-fill";
}

function clearLiveMetric(metricId) {
  const label = document.getElementById(`live-${metricId}`);
  const bar = document.getElementById(`live-${metricId}-bar`);
  const averageLabel = document.getElementById(`live-${metricId}-avg`);
  label.textContent = "--";
  label.className = getLiveMetricValueBaseClass(metricId);
  bar.style.width = "0%";
  bar.className = "metric-card-fill";
  if (averageLabel) averageLabel.textContent = "";
}

function clearAllLiveMetrics() {
  LIVE_METRIC_IDS.forEach((id) => clearLiveMetric(id));
}

function setAllLiveMetricsLoading() {
  LIVE_METRIC_IDS.forEach((id) => setLiveMetricLoading(id));
  updateLiveMetricAverages();
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function avg(values) {
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function computeVisibleResponseTotal(eouDelay, llmTtft, ttsTtfb) {
  if (
    !isFiniteNumber(eouDelay) ||
    !isFiniteNumber(llmTtft) ||
    !isFiniteNumber(ttsTtfb)
  ) {
    return null;
  }
  return eouDelay + llmTtft + ttsTtfb;
}

function updateLiveMetricAverages() {
  setLiveMetricAverage("eou", avg(averages.eouDelay));
  setLiveMetricAverage("llm-ttft", avg(averages.llmTtft));
  setLiveMetricAverage("voice-generation", avg(averages.voiceGeneration));
  if (toolTurnActive) {
    setLiveMetricAverage("total", null);
  } else {
    setLiveMetricAverage("total", avg(averages.totalLatency));
  }
}

function shouldApplyUserLatency(turn, nextValue, currentValue) {
  if (!isFiniteNumber(nextValue)) return false;

  const stage = turn.stage;
  const isUserStage = stage === "eou" || stage === "stt" || turn.role === "user";
  if (isUserStage) return true;
  if (nextValue > 0) return true;

  if (isFiniteNumber(currentValue)) {
    return false;
  }

  // Instruction-only startup turn has no EOU boundary and can legitimately be zero.
  return activeLiveSpeechId === null;
}

function updateLiveMetrics(turn) {
  const metrics = turn.metrics || {};
  const latencies = turn.latencies || {};

  const eouDelay = latencies.eou_delay ?? latencies.vad_detection_delay;
  if (shouldApplyUserLatency(turn, eouDelay, liveTurnValues.eouDelay)) {
    liveTurnValues.eouDelay = eouDelay;
    setLiveMetric("eou", eouDelay, 4.0, 0.8, 1.2);
  }

  const llmTtft = metrics.llm?.ttft;
  if (isFiniteNumber(llmTtft)) {
    liveTurnValues.llmTtft = llmTtft;
    setLiveMetric("llm-ttft", llmTtft, 4.0, 0.5, 1.0);
  }

  const ttsTtfb = metrics.tts?.ttfb;
  if (isFiniteNumber(ttsTtfb)) {
    liveTurnValues.ttsTtfb = ttsTtfb;
    setLiveMetric("voice-generation", ttsTtfb, 4.0, 0.6, 1.2);
  }

  const computedTotal = computeVisibleResponseTotal(
    liveTurnValues.eouDelay,
    liveTurnValues.llmTtft,
    liveTurnValues.ttsTtfb,
  );
  if (isFiniteNumber(computedTotal) && !toolTurnActive) {
    setLiveMetric("total", computedTotal, 8.0, 1.5, 3.0);
  }
}

function renderTurn(turn) {
  const latencies = turn.latencies || {};
  const metrics = turn.metrics || {};

  const eouDelay = latencies.eou_delay ?? latencies.vad_detection_delay;
  if (isFiniteNumber(eouDelay) && eouDelay > 0) averages.eouDelay.push(eouDelay);

  const llmTtft = metrics.llm?.ttft;
  if (isFiniteNumber(llmTtft) && llmTtft > 0) averages.llmTtft.push(llmTtft);

  const ttsTtfb = metrics.tts?.ttfb;
  if (isFiniteNumber(ttsTtfb) && ttsTtfb > 0) {
    averages.voiceGeneration.push(ttsTtfb);
  }

  const totalLatency = computeVisibleResponseTotal(eouDelay, llmTtft, ttsTtfb);
  if (isFiniteNumber(totalLatency) && totalLatency > 0) {
    averages.totalLatency.push(totalLatency);
  }

  updateLiveMetricAverages();
}
