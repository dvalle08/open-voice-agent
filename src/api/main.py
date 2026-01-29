from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.api.websocket import VoiceWebSocketHandler
from src.core.logger import logger
from src.core.settings import settings

app = FastAPI(
    title="Open Voice Agent API",
    description="Real-time voice conversation agent with WebSocket support",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "open-voice-agent",
        "version": "0.1.0",
    }


@app.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket):
    handler = VoiceWebSocketHandler(websocket)
    
    try:
        await handler.connect()
        await handler.handle_conversation()
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await handler.send_error(str(e))
    finally:
        await handler.disconnect()


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Open Voice Agent API...")
    logger.info(f"Voice provider: {settings.voice.VOICE_PROVIDER}")
    logger.info(f"LLM provider: {settings.llm.LLM_PROVIDER}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Open Voice Agent API...")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api.API_HOST,
        port=settings.api.API_PORT,
        workers=settings.api.API_WORKERS,
        reload=True,
    )
