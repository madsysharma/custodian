"""Download issue screenshots and pass them to the model as inline data.

GitHub serves uploaded images (github.com/user-attachments/...) through a redirect to a
short-lived signed S3 link. Some model providers, including Fireworks, refuse to follow
redirects when they fetch an image themselves, and the request then fails. Downloading the
image here and sending its bytes inline avoids that, and also works with expiring links.
"""

from __future__ import annotations

import base64
from urllib.parse import urlparse

import httpx

MAX_IMAGE_BYTES = 4 * 1024 * 1024
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


def fetch_image_data_url(url: str, github_token: str = "", client: httpx.Client | None = None) -> str | None:
    """Return the image as a base64 data URL, or None if it cannot be used.

    Failures are expected (deleted uploads, private assets, huge files, non-images), so this
    never raises: the issue is still triaged from its text.
    """
    headers = {"User-Agent": "custodian-maintainer-agent"}
    if github_token and urlparse(url).hostname in {"github.com", "user-images.githubusercontent.com"}:
        # httpx drops this header automatically if the redirect goes to another host (S3).
        headers["Authorization"] = f"Bearer {github_token}"
    own_client = client is None
    client = client or httpx.Client(timeout=20, follow_redirects=True)
    try:
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code != 200:
                return None
            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type not in ALLOWED_TYPES:
                return None
            data = bytearray()
            for chunk in response.iter_bytes():
                data.extend(chunk)
                if len(data) > MAX_IMAGE_BYTES:
                    return None
        return f"data:{content_type};base64,{base64.b64encode(bytes(data)).decode('ascii')}"
    except httpx.HTTPError:
        return None
    finally:
        if own_client:
            client.close()