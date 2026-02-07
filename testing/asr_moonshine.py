import io
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "usefulsensors/moonshine-streaming-small"

model = MoonshineStreamingForConditionalGeneration.from_pretrained(model_id).to(
    device, torch_dtype
)
processor = AutoProcessor.from_pretrained(model_id)

# Read audio file
with open("dev/kokoro_tts.wav", "rb") as f:
    audio_bytes = f.read()

# Load audio using soundfile
audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

if audio_np.ndim > 1:
    audio_np = np.mean(audio_np, axis=1)

if sr != 16000:
    ratio_gcd = math.gcd(sr, 16000)
    up = 16000 // ratio_gcd
    down = sr // ratio_gcd
    print(f"Resampling from {sr}Hz to 16000Hz")
    audio_np = resample_poly(audio_np, up=up, down=down)

inputs = processor(
    audio_np,
    return_tensors="pt",
    sampling_rate=16000,
).to(device, torch_dtype)

token_limit_factor = 6.5 / 16000
max_length = int((inputs.attention_mask.sum() * token_limit_factor).max().item())

generated_ids = model.generate(**inputs, max_length=max_length)
transcription = processor.decode(generated_ids[0], skip_special_tokens=True)

print(f"Transcription: {transcription}")
