"""JuniperCascor API server entry point.

Usage:
    python server.py
    # or:
    uvicorn api.app:app --host 127.0.0.1 --port 8200
"""

import uvicorn

from api.app import create_app
from api.settings import get_settings


def main() -> None:
    """Start the JuniperCascor API server."""
    settings = get_settings()
    app = create_app(settings)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
