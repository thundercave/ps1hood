"""Small HTTP helper with retries."""

from __future__ import annotations

import time
from typing import Any

import httpx

_DEFAULT_HEADERS = {
    "User-Agent": "ps1-hood/0.1 (personal 3d reconstruction; +https://github.com/)",
}


class HttpError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def request(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    retries: int = 4,
    raw: bool = False,
) -> httpx.Response:
    merged = dict(_DEFAULT_HEADERS)
    if headers:
        merged.update(headers)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = httpx.request(
                method,
                url,
                params=params,
                headers=merged,
                timeout=timeout,
                follow_redirects=True,
            )
            if resp.status_code in {429, 500, 502, 503, 504}:
                time.sleep(0.6 * (2**attempt))
                last = HttpError(f"{resp.status_code} from {url}", resp.status_code)
                continue
            if resp.status_code >= 400:
                raise HttpError(
                    f"{resp.status_code} from {url}: {resp.text[:300]}", resp.status_code
                )
            return resp
        except httpx.HTTPError as exc:
            last = exc
            time.sleep(0.4 * (2**attempt))
    raise HttpError(f"request failed after {retries} tries: {last}") from last


def get_json(url: str, **kwargs: Any) -> Any:
    return request("GET", url, **kwargs).json()


def get_bytes(url: str, **kwargs: Any) -> bytes:
    return request("GET", url, **kwargs).content
