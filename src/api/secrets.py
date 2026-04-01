"""Docker secrets utility for reading secrets from file-based mounts.

In containerized environments, Docker (and Docker Compose) can mount secret
files at ``/run/secrets/``.  A companion ``_FILE`` environment variable
points to the mounted path.  ``get_secret()`` reads the secret from that
file when available, falling back to the plain environment variable.
"""

import os
from pathlib import Path


def get_secret(env_var: str, file_env_var: str | None = None) -> str | None:
    """Read a secret value, preferring file-based Docker secrets over env vars.

    Args:
        env_var: Name of the environment variable holding the secret value
            directly (e.g. ``JUNIPER_DATA_API_KEY``).
        file_env_var: Name of the environment variable whose value is the
            *path* to a file containing the secret.  Defaults to
            ``<env_var>_FILE`` when not supplied.

    Returns:
        The secret string (stripped of surrounding whitespace), or ``None``
        if neither the file nor the plain env var is set.
    """
    if file_env_var is None:
        file_env_var = f"{env_var}_FILE"

    file_path = os.environ.get(file_env_var)
    if file_path:
        path = Path(file_path)
        if path.is_file():
            return path.read_text().strip()

    return os.environ.get(env_var)
