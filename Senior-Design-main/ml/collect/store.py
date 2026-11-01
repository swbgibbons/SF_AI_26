"""Download, check, dedupe and save photos as JPEG, with a manifest row per photo."""

import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MANIFEST_FIELDS = ["file", "guessed_type", "source", "source_id", "title", "page_url", "image_url",
                   "author", "license", "query", "width", "height", "sha256", "dhash",
                   "model_label", "model_damaged", "downloaded_at"]
MAX_BYTES = 25 * 1024 * 1024


def dhash(image, size=8):
    """64-bit difference hash: near-identical photos (resized, recompressed) get close hashes."""
    g = ImageOps.exif_transpose(image).convert("L").resize((size + 1, size), Image.LANCZOS)
    px = g.tobytes()
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (px[row * (size + 1) + col] > px[row * (size + 1) + col + 1])
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


class DedupeIndex:
    """Exact (sha256 of the bytes) and near-duplicate (dHash) lookup over photos we already have."""

    def __init__(self, max_distance=5):
        self.max_distance = max_distance
        self.sha = {}
        self.hashes = []  # (dhash, label)

    def add(self, sha, dh, label):
        self.sha.setdefault(sha, label)
        self.hashes.append((dh, label))

    def match(self, sha, dh):
        if sha in self.sha:
            return self.sha[sha]
        for h, label in self.hashes:
            if hamming(h, dh) <= self.max_distance:
                return label
        return None

    def index_folder(self, folder, cache_path=None):
        """Hash every image under folder (recursively)."""
        return self.index_files(sorted(Path(folder).rglob("*")), cache_path)

    def index_files(self, paths, cache_path=None):
        """Hash the given image files. Results are cached by (path, size, mtime)."""
        cache = {}
        if cache_path and Path(cache_path).exists():
            cache = json.loads(Path(cache_path).read_text())
        changed, n = False, 0
        for p in map(Path, paths):
            if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
                continue
            st = p.stat()
            key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
            if key not in cache:
                try:
                    data = p.read_bytes()
                    with Image.open(io.BytesIO(data)) as im:
                        cache[key] = [hashlib.sha256(data).hexdigest(), dhash(im)]
                    changed = True
                except Exception:
                    continue
            sha, dh = cache[key]
            self.add(sha, dh, str(p))
            n += 1
        if cache_path and changed:
            Path(cache_path).write_text(json.dumps(cache))
        return n


def safe_name(text, n=60):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:n] or "img"


class Store:
    def __init__(self, out_dir, net, index, min_short_side=300, classifier=None):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.net, self.index, self.min_short_side, self.classifier = net, index, min_short_side, classifier
        self.manifest = self.out / "manifest.csv"
        self.seen_path = self.out / ".seen_urls.txt"
        self.seen = set(self.seen_path.read_text().split()) if self.seen_path.exists() else set()
        self.rejected = self.out / "skipped.csv"
        self.stats = {}
        # Photos saved by earlier runs into this folder count as "already have".
        if self.manifest.exists():
            with open(self.manifest, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("sha256") and row.get("dhash"):
                        self.index.add(row["sha256"], int(row["dhash"], 16), row["file"])

    def _count(self, key):
        self.stats[key] = self.stats.get(key, 0) + 1

    def _skip(self, cand, reason):
        self._count(reason.split(":")[0])
        new = not self.rejected.exists()
        with open(self.rejected, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["reason", "source", "image_url", "page_url", "title"])
            w.writerow([reason, cand.source, cand.image_url, cand.page_url, cand.title[:200]])

    def _mark_seen(self, url):
        self.seen.add(url)
        with open(self.seen_path, "a") as f:
            f.write(url + "\n")

    def fetch_image(self, url):
        """Return image bytes, or (None, reason). Redirects are followed only while they stay
        image responses; a link that lands on a web page is dropped, not opened."""
        try:
            r = self.net.get(url, stream=True, allow_redirects=True)
        except Exception as e:  # network errors, bad TLS, etc.
            return None, f"error: {type(e).__name__}"
        ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if not r.ok:
            return None, f"http_{r.status_code}"
        if not ctype.startswith("image/"):
            r.close()
            return None, f"not_an_image: {ctype or 'unknown'} at {urlparse(r.url).netloc}"
        data = b""
        for chunk in r.iter_content(65536):
            data += chunk
            if len(data) > MAX_BYTES:
                r.close()
                return None, "too_large"
        return data, None

    def save(self, cand, guessed_type):
        """Download one candidate. Returns the saved path or None."""
        if cand.image_url in self.seen:
            self._count("already_seen")
            return None
        data, why = self.fetch_image(cand.image_url)
        if data is None and cand.extra.get("fallback") and cand.extra["fallback"] != cand.image_url:
            data, why = self.fetch_image(cand.extra["fallback"])
        self._mark_seen(cand.image_url)
        if data is None:
            self._skip(cand, why)
            return None
        try:
            with Image.open(io.BytesIO(data)) as im:
                if getattr(im, "is_animated", False):
                    self._skip(cand, "animated")
                    return None
                image = ImageOps.exif_transpose(im).convert("RGB")
        except Exception:
            self._skip(cand, "unreadable_image")
            return None
        w, h = image.size
        if min(w, h) < self.min_short_side:
            self._skip(cand, f"too_small: {w}x{h}")
            return None

        sha = hashlib.sha256(data).hexdigest()
        dh = dhash(image)
        dup = self.index.match(sha, dh)
        if dup:
            self._skip(cand, f"duplicate: {dup}")
            return None

        model_label, model_damaged = "", ""
        if self.classifier is not None:
            res = self.classifier.predict(image)
            model_label = res["label"]
            model_damaged = f"{res['scores']['damaged']:.3f}" if res.get("scores") else ""

        folder = self.out / guessed_type
        folder.mkdir(exist_ok=True)
        path = folder / f"{guessed_type}__{cand.source}_{safe_name(cand.source_id)}.jpg"
        k = 2
        while path.exists():
            path = folder / f"{guessed_type}__{cand.source}_{safe_name(cand.source_id)}_{k}.jpg"
            k += 1
        image.save(path, "JPEG", quality=92)
        self.index.add(sha, dh, str(path))

        new = not self.manifest.exists()
        with open(self.manifest, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            if new:
                wr.writeheader()
            wr.writerow({
                "file": str(path.relative_to(self.out)), "guessed_type": guessed_type,
                "source": cand.source, "source_id": cand.source_id, "title": cand.title[:300],
                "page_url": cand.page_url, "image_url": cand.image_url, "author": cand.author,
                "license": cand.license, "query": cand.query, "width": w, "height": h,
                "sha256": sha, "dhash": f"{dh:016x}", "model_label": model_label,
                "model_damaged": model_damaged,
                "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
        self._count("saved")
        return path
