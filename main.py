"""Main entry point for Open Voice Agent.

Supports running:
- FastAPI WebSocket server
- Streamlit UI
- Both simultaneously
"""

import argparse
import multiprocessing
import subprocess
import sys

from src.core.logger import logger


def run_api():
    """Run the FastAPI WebSocket server."""
    logger.info("Starting FastAPI server...")
    import uvicorn
    from src.core.settings import settings
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api.API_HOST,
        port=settings.api.API_PORT,
        workers=settings.api.API_WORKERS,
        reload=True,
        log_level="info",
    )


def run_streamlit():
    """Run the Streamlit UI."""
    logger.info("Starting Streamlit UI...")
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/streamlit_app.py",
        "--server.port=8501",
        "--server.address=localhost",
    ])


def run_both():
    """Run both FastAPI and Streamlit in parallel."""
    logger.info("Starting both FastAPI server and Streamlit UI...")
    
    # Create processes
    api_process = multiprocessing.Process(target=run_api, name="FastAPI")
    streamlit_process = multiprocessing.Process(target=run_streamlit, name="Streamlit")
    
    try:
        # Start both processes
        api_process.start()
        streamlit_process.start()
        
        # Wait for both to complete
        api_process.join()
        streamlit_process.join()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        api_process.terminate()
        streamlit_process.terminate()
        api_process.join()
        streamlit_process.join()
        logger.info("Shutdown complete")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Open Voice Agent - Real-time AI voice conversations"
    )
    parser.add_argument(
        "mode",
        choices=["api", "streamlit", "both"],
        default="both",
        nargs="?",
        help="Run mode: 'api' (FastAPI server), 'streamlit' (UI), or 'both' (default)",
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting Open Voice Agent in '{args.mode}' mode...")
    
    if args.mode == "api":
        run_api()
    elif args.mode == "streamlit":
        run_streamlit()
    else:  # both
        run_both()


if __name__ == "__main__":
    main()
