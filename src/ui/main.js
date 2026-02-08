const { Room, RoomEvent, Track, createLocalAudioTrack } = LivekitClient;

const statusEl = document.getElementById("status");
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
    const x = i * barWidth;
    const y = canvas.height - barHeight;
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
