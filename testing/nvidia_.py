import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

transcriptions = asr_model.transcribe(["dev/kokoro_tts.wav"])