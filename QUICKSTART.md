# 🚀 Quick Start Guide

Get your Open Voice Agent up and running in minutes!

## ✅ Pre-flight Checklist

- [ ] Python 3.13+ installed
- [ ] Gradium API key ([sign up here](https://gradium.ai))
- [ ] NVIDIA API key or Hugging Face token
- [ ] Dependencies installed

## 📦 Installation Steps

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Configure Environment

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add:
- `GRADIUM_API_KEY` - Your Gradium API key
- `NVIDIA_API_KEY` - Your NVIDIA API key (or `HF_TOKEN` for Hugging Face)

### 3. Run the Application

**Easiest way - Run both server and UI:**
```bash
python main.py both
```

**Or run separately:**

Terminal 1 - Start FastAPI server:
```bash
python main.py api
```

Terminal 2 - Start Streamlit UI:
```bash
python main.py streamlit
```

### 4. Open the UI

Navigate to: `http://localhost:8501`

### 5. Start Talking!

1. Click "Test Connection" in the sidebar to verify server is running
2. Select your preferred voice
3. Click the audio input widget to start recording
4. Speak your message clearly
5. Click again to stop recording
6. Click "Send Audio" to process
7. See the AI's response in the chat!

## 🎯 What You Can Do

- **Have natural conversations** with AI using voice
- **Select different voices** from multiple languages
- **Download transcripts** of your conversations
- **Switch between LLM providers** (NVIDIA/Hugging Face)
- **Customize settings** via environment variables

## 🔧 Testing the API

### Check Health
```bash
curl http://localhost:8000/health
```

### Test WebSocket (using wscat)
```bash
# Install wscat
npm install -g wscat

# Connect
wscat -c ws://localhost:8000/ws/voice

# Send start message
{"type": "start_conversation"}
```

## 📊 System Requirements

- **CPU**: Modern multi-core processor
- **RAM**: 4GB minimum (8GB recommended)
- **Internet**: Stable connection for API calls
- **Browser**: Chrome, Firefox, or Edge (for WebRTC audio)

## ⚡ Performance Tips

1. **Use NVIDIA provider** for fastest LLM responses
2. **Select closest region** in Gradium settings (EU or US)
3. **Reduce LLM_MAX_TOKENS** for quicker responses
4. **Enable streaming** for real-time feedback

## 🐛 Common Issues

### "ModuleNotFoundError: streamlit_realtime_audio"
- Run `uv sync` to update dependencies
- The app now uses built-in `st.audio_input`

### "API key is required for the hosted NIM"
- Add `NVIDIA_API_KEY` to `.env` file (get from https://build.nvidia.com/)
- Or switch to Hugging Face: `LLM_PROVIDER=huggingface` with `HF_TOKEN`

### "WebSocket connection failed"
- Ensure FastAPI server is running: check `http://localhost:8000/health`
- Click "Test Connection" button in Streamlit sidebar
- Check firewall settings

### "No audio detected" or recording issues
- Grant microphone permissions in browser
- Click audio input widget to start/stop recording
- Try a different browser (Chrome works best)

### "Import errors"
- Run `uv sync` or `pip install -e .` again
- Check Python version: `python --version` (must be 3.13+)

## 🎓 Next Steps

- Explore the [full README](README.md) for advanced configuration
- Review the architecture documentation
- Try different voices and LLM models
- Customize the system prompt in `src/agent/graph.py`
- Add your own voice providers

## 💡 Tips for Best Results

1. **Speak clearly** and at a normal pace
2. **Wait for the response** before speaking again
3. **Use natural language** - the AI understands context
4. **Keep sentences concise** for voice interaction
5. **Check the transcript** to see what was understood

## 🆘 Need Help?

- Check the [README](README.md) for detailed documentation
- Review logs in the terminal for error messages
- Ensure all environment variables are set correctly
- Test each component separately (API, then UI)

---

**Ready to go? Run `python main.py both` and start talking!** 🎤
