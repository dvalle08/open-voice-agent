# Troubleshooting Guide

## Common Issues and Solutions

### 1. "ModuleNotFoundError: No module named 'streamlit_realtime_audio'"

**Problem:** The old audio package is no longer used.

**Solution:**
```bash
# Sync dependencies
uv sync

# Or reinstall
pip install -e .
```

The app now uses Streamlit's built-in `st.audio_input` widget.

---

### 2. "An API key is required for the hosted NIM"

**Problem:** Missing NVIDIA API key.

**Solutions:**

**Option A: Use NVIDIA (Recommended)**
1. Get an API key from https://build.nvidia.com/
2. Add to your `.env` file:
   ```env
   NVIDIA_API_KEY=nvapi-your-key-here
   ```

**Option B: Use Hugging Face**
1. Get a token from https://huggingface.co/settings/tokens
2. Update your `.env` file:
   ```env
   LLM_PROVIDER=huggingface
   HF_TOKEN=hf_your_token_here
   HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
   ```

---

### 3. "WebSocket connection failed"

**Problem:** Cannot connect to FastAPI server.

**Diagnosis:**
```bash
# Check if server is running
curl http://localhost:8000/health
```

**Solutions:**
- Ensure FastAPI server is running: `python main.py api` or `python main.py both`
- Check if port 8000 is available: `lsof -i :8000`
- Verify firewall settings
- Check CORS settings in `src/core/settings.py`

---

### 4. Audio Not Recording

**Problem:** Microphone not working in browser.

**Solutions:**
- Grant microphone permissions in browser
- Check browser console for errors (F12)
- Try a different browser (Chrome recommended)
- Ensure microphone is not in use by another application
- Test microphone: `navigator.mediaDevices.getUserMedia({audio: true})`

---

### 5. "No transcript generated from audio"

**Problem:** STT returns empty result.

**Possible Causes:**
- Audio too short (speak for at least 1-2 seconds)
- Audio quality too low
- Background noise too high
- Gradium API error

**Solutions:**
- Speak louder and clearer
- Reduce background noise
- Check Gradium API status
- Verify `GRADIUM_API_KEY` is valid
- Check Gradium credits: https://gradium.ai/dashboard

---

### 6. Gradium API Errors

**Problem:** Voice processing fails.

**Common Errors:**

**"Invalid API key"**
- Verify `GRADIUM_API_KEY` in `.env` file
- Ensure no extra spaces or quotes
- Get new key from https://gradium.ai/

**"Insufficient credits"**
- Check your credit balance at https://gradium.ai/dashboard
- Add more credits or upgrade plan

**"Rate limit exceeded"**
- Wait a few seconds between requests
- Consider upgrading your Gradium plan

---

### 7. Streamlit Reloads Constantly

**Problem:** App keeps rerunning.

**Solutions:**
- Don't use `st.experimental_rerun()` unnecessarily
- Check for infinite loops in session state updates
- Verify button states are managed correctly
- Review Streamlit's execution model

---

### 8. WebSocket Timeout

**Problem:** "WebSocket receive timeout" in logs.

**Solutions:**
- Increase timeout in `src/streamlit_app.py` (line with `timeout=30.0`)
- Check network stability
- Verify server is processing requests (check API logs)
- Ensure LLM API is responding (check LLM provider status)

---

### 9. Import Errors

**Problem:** `ModuleNotFoundError` or `ImportError`.

**Solutions:**
```bash
# Clean install
rm -rf .venv
uv sync

# Or with pip
pip install -e . --force-reinstall

# Verify Python version
python --version  # Should be 3.13+
```

---

### 10. Conversation State Issues

**Problem:** Agent forgets context or repeats responses.

**Solutions:**
- Clear conversation with "Clear Conversation" button
- Check LangGraph memory is working (logs should show conversation history)
- Verify session IDs are consistent
- Review `src/agent/graph.py` for state management

---

## Debug Mode

Enable detailed logging:

```bash
# Set environment variable
export OVA_STAGE=dev

# Run with verbose logging
python main.py both
```

Check logs in terminal for detailed error messages.

---

## Performance Issues

### Slow Response Times

**Check:**
1. LLM provider response time
2. Network latency
3. Audio processing time
4. Token generation limits

**Optimize:**
- Use faster LLM models
- Reduce `LLM_MAX_TOKENS`
- Select closest Gradium region (EU/US)
- Enable streaming: `LLM_STREAMING=true`

---

## Getting Help

1. **Check logs:** Review terminal output for errors
2. **Test components:** Test API, LLM, and Gradium separately
3. **Review settings:** Verify all `.env` variables
4. **Check status:** 
   - Gradium: https://status.gradium.ai/
   - NVIDIA: https://status.nvidia.com/
5. **GitHub Issues:** Report bugs with logs and configuration

---

## Useful Commands

```bash
# Check health
curl http://localhost:8000/health

# Test WebSocket
wscat -c ws://localhost:8000/ws/voice

# View logs
tail -f logs/app.log  # if logging to file

# Check dependencies
uv tree

# Verify environment
env | grep -E "GRADIUM|NVIDIA|HF_"
```

---

## Still Need Help?

- Review the [README.md](README.md) for detailed documentation
- Check the [QUICKSTART.md](QUICKSTART.md) for setup instructions
- Ensure all prerequisites are installed
- Verify API keys are valid and have credits
- Test each component separately
