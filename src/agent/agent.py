"""CLI entrypoint for running the LiveKit agent server."""

from livekit import agents

from src.agent.runtime.session import server
from src.core.logger import detach_default_root_handler


if __name__ == "__main__":
    detach_default_root_handler()
    agents.cli.run_app(server)
