"""Where candidate photos come from. Every source yields Candidate records that point
straight at an image file, never at the web page around it."""

import html
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from urllib.parse import urlparse

IMAGE_EXT = re.compile(r"\.(jpe?g|png|webp)(\?.*)?$", re.I)


@dataclass
class Candidate:
    source: str            # reddit | pinterest | openverse | wikimedia | url
    source_id: str         # stable id within the source, used in file names
    image_url: str         # direct link to the image file
    page_url: str = ""     # where a person can see it in context (credit / license check)
    title: str = ""
    author: str = ""
    license: str = ""
    query: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- relevance

ROOF_WORDS = ("roof", "shingle", "shingles", "reroof", "re-roof", "ridge cap", "flashing")
TYPE_WORDS = {
    "missing_shingles": ("missing shingle", "missing shingles", "shingles missing", "shingles gone",
                         "blown off", "blew off", "ripped off", "torn off", "broken shingle",
                         "cracked shingle", "exposed felt", "exposed underlayment", "tarp", "bare deck"),
    "wind": ("wind", "storm", "hurricane", "tornado", "gust", "lifted", "creased", "flapping",
             "uplift", "blown"),
    "hail": ("hail",),
}
# Things that are almost always off-topic for this dataset.
NEGATIVE_WORDS = ("meme", "invoice", "quote me", "estimate", "price", "how much", "solar panel",
                  "metal roof", "tile roof", "car ", "windshield", "skylight leak")


DAMAGE_WORDS = ("damage", "damaged", "storm", "hurricane", "tornado", "hail", "wind", "missing", "blown",
                "torn", "broken", "repair", "tarp", "leak", "destroyed", "lifted", "creased")


def is_damage_photo_title(text):
    """Stricter check for library searches (Openverse, Wikimedia), whose full-text search also
    matches old book scans and scenery: needs a roof word AND a damage word."""
    t = " " + (text or "").lower() + " "
    return any(w in t for w in ROOF_WORDS) and any(w in t for w in DAMAGE_WORDS) \
        and not re.search(r"\((1[5-9]\d\d)\)", t)  # "Title (1905)" = Internet Archive book scan


def guess_type(text):
    """Best-guess damage type from a title/description. Returns (type, relevant)."""
    t = " " + (text or "").lower() + " "
    relevant = any(w in t for w in ROOF_WORDS) and not any(w in t for w in NEGATIVE_WORDS)
    for kind in ("missing_shingles", "hail", "wind"):
        if any(w in t for w in TYPE_WORDS[kind]):
            return kind, relevant
    return "unsorted", relevant


# ---------------------------------------------------------------- Reddit

# r/Roofing first: it has by far the most close-up damage photos.
REDDIT_SUBS = ["Roofing", "HomeImprovement", "homeowners", "Insurance", "HomeInspections"]
REDDIT_QUERIES = ["missing shingles", "wind damage", "shingles blown off", "broken shingles",
                  "creased shingles", "lifted shingles", "storm damage roof", "hail damage"]


def parse_reddit_search_url(url):
    """https://www.reddit.com/r/Roofing/search/?q=missing+shingles&type=media -> ("Roofing", "missing shingles")"""
    from urllib.parse import parse_qs
    p = urlparse(html.unescape(url))
    m = re.search(r"/r/([^/]+)", p.path)
    q = parse_qs(p.query).get("q", [""])[0]
    if not m or not q:
        raise ValueError(f"expected a reddit.com/r/<sub>/search/?q=... URL: {url}")
    return m.group(1), q


def reddit_images(post):
    """Direct image URLs from a Reddit post (listing JSON 'data'); links to other sites are skipped."""
    urls = []
    if post.get("is_video") or "v.redd.it" in (post.get("url") or "") or post.get("post_hint") in (
            "hosted:video", "rich:video"):
        return []  # no videos
    if post.get("is_gallery") and post.get("media_metadata"):
        order = [i["media_id"] for i in (post.get("gallery_data") or {}).get("items", [])]
        meta = post["media_metadata"]
        for mid in order or list(meta):
            m = meta.get(mid) or {}
            if m.get("status") == "valid" and m.get("e") == "Image" and m.get("s", {}).get("u"):
                urls.append(html.unescape(m["s"]["u"]))
        return urls
    url = post.get("url_overridden_by_dest") or post.get("url") or ""
    host = urlparse(url).netloc
    if host in ("i.redd.it", "i.imgur.com") and IMAGE_EXT.search(url):  # .gif/.gifv not matched
        return [url]
    if post.get("post_hint") == "image":
        imgs = (post.get("preview") or {}).get("images") or []
        if imgs and imgs[0].get("source", {}).get("url"):
            return [html.unescape(imgs[0]["source"]["url"])]
    return []


class Reddit:
    """Reddit search. Uses the official OAuth API when REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are
    set (create a free 'script' app at https://www.reddit.com/prefs/apps); otherwise the public
    .json listing, which is much slower (about 10 requests a minute)."""

    def __init__(self, net, subs=None, queries=None, searches=None, per_query=250, time_filter="all"):
        """searches: explicit (sub, query) pairs; otherwise every sub x every query."""
        self.net, self.per_query, self.time_filter = net, per_query, time_filter
        self.searches = searches or [(s, q) for s in (subs or REDDIT_SUBS) for q in (queries or REDDIT_QUERIES)]
        self.base, self.headers = "https://www.reddit.com", {}
        cid, secret = os.environ.get("REDDIT_CLIENT_ID"), os.environ.get("REDDIT_CLIENT_SECRET")
        if cid and secret:
            r = net.post("https://www.reddit.com/api/v1/access_token",
                         auth=(cid, secret), data={"grant_type": "client_credentials"})
            r.raise_for_status()
            self.base = "https://oauth.reddit.com"
            self.headers = {"Authorization": "bearer " + r.json()["access_token"]}

    def __iter__(self):
        for sub, q in self.searches:
            if True:
                after, fetched = None, 0
                while fetched < self.per_query:
                    suffix = "" if self.base.startswith("https://oauth") else ".json"
                    r = self.net.get(f"{self.base}/r/{sub}/search{suffix}", headers=self.headers, params={
                        "q": q, "restrict_sr": 1, "sort": "relevance", "t": self.time_filter,
                        "limit": 100, "after": after, "raw_json": 1, "type": "link"})
                    if not r.ok:
                        hint = " (Reddit blocks anonymous cloud requests: set REDDIT_CLIENT_ID/SECRET)" \
                            if r.status_code == 403 and not self.headers else ""
                        print(f"  reddit r/{sub} '{q}': HTTP {r.status_code}, skipping{hint}")
                        break
                    data = r.json()["data"]
                    for child in data["children"]:
                        post = child["data"]
                        if post.get("over_18"):
                            continue
                        text = f"{post.get('title', '')} {post.get('link_flair_text') or ''}"
                        for i, url in enumerate(reddit_images(post)):
                            yield Candidate(
                                source="reddit", source_id=f"{post['id']}_{i}" if i else post["id"],
                                image_url=url, page_url="https://www.reddit.com" + post["permalink"],
                                title=post.get("title", ""), author="u/" + post.get("author", ""),
                                license="unknown (Reddit user content)", query=f"r/{sub}: {q}",
                                extra={"text": text, "subreddit": sub})
                    fetched += len(data["children"])
                    after = data.get("after")
                    if not after or not data["children"]:
                        break


# ---------------------------------------------------------------- Pinterest

PIN_SIZE = re.compile(r"(https://i\.pinimg\.com/)(?:\d+x\d*|originals)(/.+)")


def pinimg_original(url):
    """https://i.pinimg.com/236x/aa/bb/cc/x.jpg -> https://i.pinimg.com/originals/aa/bb/cc/x.jpg"""
    m = PIN_SIZE.match(url or "")
    return f"{m.group(1)}originals{m.group(2)}" if m else url


def pinimg_fallback(url):
    """Some pins have no 'originals' copy; the 736x size always exists."""
    m = PIN_SIZE.match(url or "")
    return f"{m.group(1)}736x{m.group(2)}" if m else url


def board_rss_url(board_url):
    """https://www.pinterest.com/user/board/ -> https://www.pinterest.com/user/board.rss"""
    p = urlparse(board_url if "://" in board_url else "https://" + board_url)
    parts = [x for x in p.path.split("/") if x]
    if len(parts) < 2:
        raise ValueError(f"not a board URL (need pinterest.com/<user>/<board>/): {board_url}")
    return f"https://www.pinterest.com/{parts[0]}/{parts[1].removesuffix('.rss')}.rss"


def parse_board_rss(xml_text):
    """Yield (pin_url, title, image_url) from a Pinterest board RSS feed."""
    root = ET.fromstring(xml_text)
    for item in root.iter("item"):
        link = (item.findtext("link") or "").strip()
        title = (item.findtext("title") or "").strip()
        desc = item.findtext("description") or ""
        m = re.search(r'<img[^>]+src="([^"]+)"', desc)
        if m:
            yield link, title, html.unescape(m.group(1))


OG_IMAGE = re.compile(r'<meta[^>]+(?:property|name)="og:image"[^>]+content="([^"]+)"', re.I)
OG_TITLE = re.compile(r'<meta[^>]+(?:property|name)="og:(?:title|description)"[^>]+content="([^"]*)"', re.I)


def pin_id(url):
    m = re.search(r"/pin/(?:[^/]*--)?(\d+)", url)
    return m.group(1) if m else re.sub(r"\W+", "_", urlparse(url).path).strip("_")[-40:]


class Pinterest:
    """Pinterest without scraping search results (which Pinterest's terms forbid).

    Pinterest's robots.txt currently disallows every unlisted bot from the whole site (RSS feeds and
    pin pages included), and the collector honours that, so in practice the route that works is the
    official API: set PINTEREST_ACCESS_TOKEN and pass boards you own. The RSS and pin-page paths
    stay for the case where Pinterest allows them again. Inputs:

    * boards: public board URLs. Pinterest publishes an RSS feed per board (the most recent ~25-50
      pins). Save pins you like to a board while browsing, then run this to fetch them all.
    * pins: individual pin URLs (one per line in a text file, pin.it short links work). The pin
      page's preview image is read and the full-size file is downloaded from Pinterest's image
      server, so you never land on the website the pin links to.

    With PINTEREST_ACCESS_TOKEN set (Pinterest API v5, boards you own), --pinterest-board also
    pages through every pin on the board instead of just the RSS window.
    """

    def __init__(self, net, boards=(), pins=()):
        self.net, self.boards, self.pins = net, list(boards), list(pins)
        self.token = os.environ.get("PINTEREST_ACCESS_TOKEN")

    def __iter__(self):
        for board in self.boards:
            yield from (self._board_api(board) if self.token else self._board_rss(board))
        for pin in self.pins:
            c = self._pin(pin)
            if c:
                yield c

    def _board_rss(self, board):
        rss = board_rss_url(board)
        if not self.net.allowed(rss):
            print(f"  pinterest: robots.txt disallows automated reads of {rss}. Set PINTEREST_ACCESS_TOKEN "
                  "(official API, boards you own) instead; see collect/README.md")
            return
        r = self.net.get(rss)
        if not r.ok:
            print(f"  pinterest board {board}: HTTP {r.status_code} (is the board public?)")
            return
        for link, title, img in parse_board_rss(r.text):
            yield Candidate("pinterest", pin_id(link), pinimg_original(img), link, title,
                            license="unknown (see pin source)", query=f"board: {board}",
                            extra={"fallback": pinimg_fallback(img), "text": title})

    def _board_api(self, board):
        parts = [x for x in urlparse(board if "://" in board else "https://" + board).path.split("/") if x]
        headers = {"Authorization": "Bearer " + self.token}
        r = self.net.get("https://api.pinterest.com/v5/boards", headers=headers, params={"page_size": 250})
        r.raise_for_status()
        match = [b for b in r.json().get("items", [])
                 if b.get("name", "").lower().replace(" ", "-") == parts[-1].lower()]
        if not match:
            print(f"  pinterest API: no board of yours named '{parts[-1]}', trying RSS")
            yield from self._board_rss(board)
            return
        bookmark = None
        while True:
            r = self.net.get(f"https://api.pinterest.com/v5/boards/{match[0]['id']}/pins",
                             headers=headers, params={"page_size": 100, "bookmark": bookmark})
            r.raise_for_status()
            data = r.json()
            for pin in data.get("items", []):
                images = (pin.get("media") or {}).get("images") or {}
                best = max(images.values(), key=lambda i: i.get("width", 0), default=None)
                if best:
                    text = f"{pin.get('title') or ''} {pin.get('description') or ''}"
                    yield Candidate("pinterest", pin["id"], best["url"],
                                    f"https://www.pinterest.com/pin/{pin['id']}/", pin.get("title") or "",
                                    license="unknown (see pin source)", query=f"board: {board}",
                                    extra={"text": text, "outside_link": pin.get("link") or ""})
            bookmark = data.get("bookmark")
            if not bookmark:
                break

    def _pin(self, url):
        url = url.strip()
        if PIN_SIZE.match(url):  # already a direct image link
            return Candidate("pinterest", pin_id(url), pinimg_original(url), "", query="pin list",
                             extra={"fallback": pinimg_fallback(url)})
        if not self.net.allowed(url):
            print(f"  pinterest: robots.txt disallows {url}; add it to a board and use --pinterest-board")
            return None
        # pin.it short links redirect to pinterest.com; anything that leaves Pinterest is ignored.
        r = self.net.get(url, allow_redirects=True)
        if not r.ok or "pinterest." not in urlparse(r.url).netloc:
            print(f"  pinterest: could not open {url} (HTTP {r.status_code}, ended at {urlparse(r.url).netloc})")
            return None
        m = OG_IMAGE.search(r.text)
        if not m:
            print(f"  pinterest: no image found on {url}")
            return None
        t = OG_TITLE.search(r.text)
        title = html.unescape(t.group(1)) if t else ""
        img = html.unescape(m.group(1))
        return Candidate("pinterest", pin_id(r.url), pinimg_original(img), r.url, title,
                         license="unknown (see pin source)", query="pin list",
                         extra={"fallback": pinimg_fallback(img), "text": title})


# ---------------------------------------------------------------- Openverse (Flickr, Wikimedia, ...)

OPENVERSE_QUERIES = ["roof wind damage", "missing shingles", "storm damaged roof shingles",
                     "hurricane roof damage", "roof shingles blown off", "damaged roof shingles",
                     "hail damage roof", "tornado roof damage", "roof tarp storm"]


class Openverse:
    """Openly licensed photos (mostly Flickr and Wikimedia) via the Openverse API. No key needed;
    anonymous use is limited to 20 results per page and about 20 requests a minute."""

    API = "https://api.openverse.org/v1/images/"

    def __init__(self, net, queries=None, pages=3):
        self.net, self.queries, self.pages = net, queries or OPENVERSE_QUERIES, pages

    def __iter__(self):
        for q in self.queries:
            for page in range(1, self.pages + 1):
                r = self.net.get(self.API, params={"q": q, "page": page, "page_size": 20,
                                                   "mature": "false", "category": "photograph"})
                if not r.ok:
                    print(f"  openverse '{q}' page {page}: HTTP {r.status_code}")
                    break
                results = r.json().get("results", [])
                for it in results:
                    lic = f"CC {it.get('license', '').upper()} {it.get('license_version') or ''}".strip()
                    if it.get("license") in ("cc0", "pdm"):
                        lic = it["license"].upper()
                    text = f"{it.get('title') or ''} {' '.join(t['name'] for t in it.get('tags') or [])}"
                    yield Candidate("openverse", f"{it.get('source', 'ov')}_{it['id'][:12]}", it["url"],
                                    it.get("foreign_landing_url") or "", it.get("title") or "",
                                    it.get("creator") or "", lic, q,
                                    extra={"text": text, "strict": True,
                                           "license_url": it.get("license_url") or ""})
                if len(results) < 20:
                    break


# ---------------------------------------------------------------- Wikimedia Commons

WIKIMEDIA_QUERIES = ["roof wind damage shingles", "missing shingles roof", "hurricane damage roof shingles",
                     "storm damaged roof", "tornado damage roof", "hail damage roof"]


class Wikimedia:
    """Wikimedia Commons file search (all files are freely licensed)."""

    API = "https://commons.wikimedia.org/w/api.php"

    def __init__(self, net, queries=None, per_query=50):
        self.net, self.queries, self.per_query = net, queries or WIKIMEDIA_QUERIES, per_query

    def __iter__(self):
        for q in self.queries:
            r = self.net.get(self.API, params={
                "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
                "gsrsearch": f"{q} filetype:bitmap", "gsrlimit": self.per_query, "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1280})
            if not r.ok:
                print(f"  wikimedia '{q}': HTTP {r.status_code}")
                continue
            for page in (r.json().get("query") or {}).get("pages", {}).values():
                info = (page.get("imageinfo") or [{}])[0]
                if info.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
                    continue
                meta = info.get("extmetadata") or {}
                strip = lambda k: re.sub(r"<[^>]+>", "", (meta.get(k) or {}).get("value", "")).strip()
                title = page.get("title", "").removeprefix("File:")
                yield Candidate("wikimedia", str(page.get("pageid")), info.get("thumburl") or info["url"],
                                info.get("descriptionurl", ""), title, strip("Artist"),
                                strip("LicenseShortName"), q,
                                extra={"text": f"{title} {strip('Categories')}", "strict": True})


# ---------------------------------------------------------------- plain URL list

class UrlList:
    """Direct image URLs, one per line (anything that isn't an image file is skipped)."""

    def __init__(self, urls):
        self.urls = [u.strip() for u in urls if u.strip() and not u.startswith("#")]

    def __iter__(self):
        for u in self.urls:
            if PIN_SIZE.match(u):
                yield Candidate("pinterest", pin_id(u), pinimg_original(u), query="url list",
                                extra={"fallback": pinimg_fallback(u)})
            else:
                name = os.path.splitext(os.path.basename(urlparse(u).path))[0] or "img"
                yield Candidate("url", re.sub(r"\W+", "_", name)[:40], u, query="url list")
