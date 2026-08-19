"""Small rate-limited HTTP client for public dataset acquisition helpers."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str]


class NetworkError(RuntimeError):
    """A request failed before an HTTP response was received."""


class HttpStatusError(RuntimeError):
    def __init__(self, url: str, status: int, body: bytes = b"") -> None:
        snippet = body[:160].decode("utf-8", errors="replace").replace("\n", " ")
        detail = f": {snippet}" if snippet else ""
        super().__init__(f"HTTP {status} for {url}{detail}")
        self.url = url
        self.status = status


Transport = Callable[[str, Mapping[str, str], float, int], HttpResponse]


def urllib_transport(
    url: str,
    headers: Mapping[str, str],
    timeout: float,
    max_response_bytes: int,
) -> HttpResponse:
    request = Request(url, headers=dict(headers), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(max_response_bytes + 1)
            if len(body) > max_response_bytes:
                raise NetworkError(
                    f"response exceeded {max_response_bytes} bytes for {url}"
                )
            return HttpResponse(
                int(response.status), body, dict(response.headers.items())
            )
    except HTTPError as exc:
        body = exc.read(max_response_bytes + 1)
        headers_out = dict(exc.headers.items()) if exc.headers is not None else {}
        return HttpResponse(int(exc.code), body[:max_response_bytes], headers_out)
    except URLError as exc:
        raise NetworkError(f"request failed for {url}: {exc.reason}") from exc


def _header(headers: Mapping[str, str], name: str) -> str | None:
    expected = name.lower()
    for key, value in headers.items():
        if key.lower() == expected:
            return value
    return None


def retry_after_seconds(value: str | None, now: datetime | None = None) -> float | None:
    if not value:
        return None
    stripped = value.strip()
    if stripped.isdigit():
        return float(stripped)
    try:
        retry_at = parsedate_to_datetime(stripped)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return max((retry_at - current).total_seconds(), 0.0)


class RateLimitedClient:
    """Sequential HTTPS client with retry, backoff, and a minimum request interval."""

    def __init__(
        self,
        *,
        user_agent: str = "UltraIR-data-preparation/1.0",
        min_interval: float = 1.0,
        timeout: float = 30.0,
        retries: int = 4,
        backoff: float = 1.0,
        max_response_bytes: int = 10 * 1024 * 1024,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("user_agent must not be empty")
        if min_interval < 0 or timeout <= 0 or retries < 0 or backoff < 0:
            raise ValueError("invalid HTTP timing or retry settings")
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self.user_agent = user_agent.strip()
        self.min_interval = float(min_interval)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        self.max_response_bytes = int(max_response_bytes)
        self._transport = transport or urllib_transport
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None

    def _wait_for_slot(self) -> None:
        if self._last_request_at is None:
            return
        remaining = self.min_interval - (self._clock() - self._last_request_at)
        if remaining > 0:
            self._sleep(remaining)

    def get(self, url: str, *, accept: str = "*/*") -> HttpResponse:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(f"only absolute HTTPS URLs are supported: {url!r}")
        headers = {"User-Agent": self.user_agent, "Accept": accept}
        last_network_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self._wait_for_slot()
            self._last_request_at = self._clock()
            try:
                response = self._transport(
                    url, headers, self.timeout, self.max_response_bytes
                )
            except (NetworkError, OSError, TimeoutError) as exc:
                last_network_error = exc
                if attempt >= self.retries:
                    raise NetworkError(
                        f"request failed after {attempt + 1} attempts: {url}: {exc}"
                    ) from exc
                delay = self.backoff * (2**attempt)
                if delay > 0:
                    self._sleep(delay)
                continue

            if 200 <= response.status < 300:
                return response
            retryable = response.status == 429 or 500 <= response.status < 600
            if retryable and attempt < self.retries:
                retry_after = retry_after_seconds(
                    _header(response.headers, "Retry-After")
                )
                delay = (
                    retry_after
                    if retry_after is not None
                    else self.backoff * (2**attempt)
                )
                if delay > 0:
                    self._sleep(delay)
                continue
            raise HttpStatusError(url, response.status, response.body)
        raise NetworkError(f"request failed for {url}: {last_network_error}")

    def get_text(self, url: str, *, accept: str = "text/plain") -> str:
        response = self.get(url, accept=accept)
        return response.body.decode("utf-8", errors="replace")

    def get_json(self, url: str) -> Any:
        response = self.get(url, accept="application/json")
        try:
            return json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON response from {url}") from exc
