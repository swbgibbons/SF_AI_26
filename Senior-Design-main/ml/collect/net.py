"""Polite HTTP: one session, a descriptive User-Agent, per-host rate limits, robots.txt."""

import time
import urllib.robotparser
from urllib.parse import urlparse

import requests

USER_AGENT = "RoofInSight-collector/0.1 (student research dataset; +https://github.com/swbgibbons/SF_AI_26)"

# Minimum seconds between requests to the same host. Image CDNs get a short gap,
# APIs and HTML pages a longer one.
DEFAULT_GAP = 1.0
HOST_GAPS = {
    "www.reddit.com": 6.0,        # unauthenticated limit is ~10 requests/minute
    "oauth.reddit.com": 1.0,      # OAuth limit is 100 requests/minute
    "api.openverse.org": 3.5,     # anonymous limit is 20 requests/minute
    "www.pinterest.com": 3.0,
    "commons.wikimedia.org": 1.0,
}


class Http:
    def __init__(self, user_agent=USER_AGENT, timeout=30):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.timeout = timeout
        self._last = {}
        self._robots = {}

    def _wait(self, url):
        host = urlparse(url).netloc
        gap = HOST_GAPS.get(host, DEFAULT_GAP)
        elapsed = time.monotonic() - self._last.get(host, 0)
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last[host] = time.monotonic()

    def allowed(self, url):
        """robots.txt check for HTML pages we fetch (not for APIs, feeds or image files)."""
        p = urlparse(url)
        root = f"{p.scheme}://{p.netloc}"
        if root not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self.get(root + "/robots.txt")
                rp.parse(r.text.splitlines() if r.ok else [])
            except requests.RequestException:
                rp.parse([])
            self._robots[root] = rp
        return self._robots[root].can_fetch(self.session.headers["User-Agent"], url)

    def get(self, url, retries=3, **kw):
        kw.setdefault("timeout", self.timeout)
        for attempt in range(retries):
            self._wait(url)
            r = self.session.get(url, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                try:
                    wait = int(r.headers.get("Retry-After", 0) or 0)
                except ValueError:
                    wait = 0
                if attempt + 1 < retries:  # capped: one slow host shouldn't stall the whole run
                    time.sleep(min(wait or 10 * (attempt + 1), 60))
                continue
            return r
        return r

    def post(self, url, **kw):
        kw.setdefault("timeout", self.timeout)
        self._wait(url)
        return self.session.post(url, **kw)
