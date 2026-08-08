"""JuniperCascor API server entry point.

Usage:
    python server.py
    python server.py --config conf/experiments/<name>.yaml
    # or:
    uvicorn api.app:app --host 127.0.0.1 --port 8200
"""

import argparse
import os

import uvicorn

from api.app import create_app
from api.settings import get_settings


def main() -> None:
    """Start the JuniperCascor API server."""
    parser = argparse.ArgumentParser(description="JuniperCascor API server")
    parser.add_argument(
        "--config",
        default=None,
        help="Experiment YAML whose service: block overrides env (sets JUNIPER_CASCOR_CONFIG_FILE; Wave-3 operator convenience -- the experiment stack threads the env var itself, plan SS5.2/SS6.1)",
    )
    args = parser.parse_args()
    if args.config:
        # Must land before the first (lru_cache'd) get_settings() call (settings.py SS5.2).
        os.environ["JUNIPER_CASCOR_CONFIG_FILE"] = args.config
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
