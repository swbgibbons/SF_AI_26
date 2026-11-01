"""Offline tests for the photo collector (no network: HTTP is faked)."""

import csv
import io

import numpy as np
from PIL import Image

from collect import sources as S
from collect.store import DedupeIndex, Store, dhash, hamming


def photo(seed=0, size=(640, 480)):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (12, 16, 3), dtype=np.uint8)  # blocky so resizes keep structure
    return Image.fromarray(small).resize(size, Image.NEAREST)


def jpeg_bytes(im, **kw):
    buf = io.BytesIO()
    im.save(buf, "JPEG", **kw)
    return buf.getvalue()


class FakeResp:
    def __init__(self, body=b"", ctype="image/jpeg", status=200, url="https://i.redd.it/x.jpg"):
        self.content, self.status_code, self.url = body, status, url
        self.headers = {"Content-Type": ctype}
        self.ok = status < 400
        self.text = body.decode("utf-8", "ignore")

    def iter_content(self, n):
        for i in range(0, len(self.content), n):
            yield self.content[i:i + n]

    def close(self):
        pass


class FakeNet:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, **kw):
        self.calls.append(url)
        return self.routes.get(url, FakeResp(status=404))

    def allowed(self, url):
        return True


# ---------------------------------------------------------------- reddit

def test_reddit_single_image_gallery_and_video():
    assert S.reddit_images({"url": "https://i.redd.it/abc.jpeg"}) == ["https://i.redd.it/abc.jpeg"]
    gallery = {"is_gallery": True, "gallery_data": {"items": [{"media_id": "a"}, {"media_id": "b"}]},
               "media_metadata": {"a": {"status": "valid", "e": "Image", "s": {"u": "https://preview.redd.it/a.jpg?w=1&amp;s=2"}},
                                  "b": {"status": "valid", "e": "AnimatedImage", "s": {"gif": "x"}}}}
    assert S.reddit_images(gallery) == ["https://preview.redd.it/a.jpg?w=1&s=2"]
    assert S.reddit_images({"is_video": True, "url": "https://v.redd.it/zzz"}) == []
    assert S.reddit_images({"url": "https://i.redd.it/anim.gif"}) == []
    # A link post pointing at a roofing company's site is not followed.
    assert S.reddit_images({"url": "https://someroofer.com/blog/wind-damage"}) == []


def test_parse_reddit_search_url():
    url = "https://www.reddit.com/r/Roofing/search/?q=missing+shingles&amp;type=media&amp;cId=33"
    assert S.parse_reddit_search_url(url) == ("Roofing", "missing shingles")


def test_guess_type():
    assert S.guess_type("Wind blew off half my shingles") == ("missing_shingles", True)
    assert S.guess_type("Is this hail damage on my roof?") == ("hail", True)
    assert S.guess_type("Creased shingles after last night's storm") == ("wind", True)
    assert S.guess_type("Wind damage to my fence")[1] is False


def test_library_titles_need_roof_and_damage_words():
    assert S.is_damage_photo_title("Algiers Hurricane Isaac Damaged Roof 1.JPG")
    assert not S.is_damage_photo_title("American homes and gardens (1905) roof")
    assert not S.is_damage_photo_title("Coast watch (1979) storm roof damage")
    assert not S.is_damage_photo_title("New roof on church")


# ---------------------------------------------------------------- pinterest

def test_pinimg_original_and_board_rss():
    assert S.pinimg_original("https://i.pinimg.com/236x/aa/bb/cc/f.jpg") == "https://i.pinimg.com/originals/aa/bb/cc/f.jpg"
    assert S.pinimg_fallback("https://i.pinimg.com/originals/aa/bb/cc/f.jpg") == "https://i.pinimg.com/736x/aa/bb/cc/f.jpg"
    assert S.board_rss_url("https://www.pinterest.com/hugo/roof-damage/") == "https://www.pinterest.com/hugo/roof-damage.rss"
    rss = """<rss><channel><item><title>Wind damaged roof</title>
      <link>https://www.pinterest.com/pin/123456/</link>
      <description>&lt;a href="/pin/123456/"&gt;&lt;img src="https://i.pinimg.com/236x/aa/bb/cc/f.jpg"&gt;&lt;/a&gt;</description>
      </item></channel></rss>"""
    assert list(S.parse_board_rss(rss)) == [
        ("https://www.pinterest.com/pin/123456/", "Wind damaged roof", "https://i.pinimg.com/236x/aa/bb/cc/f.jpg")]


def test_pin_page_reads_image_without_leaving_pinterest():
    page = b'<html><meta property="og:image" content="https://i.pinimg.com/736x/aa/bb/cc/f.jpg">' \
           b'<meta property="og:title" content="Missing shingles"></html>'
    net = FakeNet({"https://www.pinterest.com/pin/42/": FakeResp(page, "text/html", url="https://www.pinterest.com/pin/42/")})
    (c,) = list(S.Pinterest(net, pins=["https://www.pinterest.com/pin/42/"]))
    assert c.image_url == "https://i.pinimg.com/originals/aa/bb/cc/f.jpg"
    assert c.extra["fallback"].startswith("https://i.pinimg.com/736x/")
    assert c.source_id == "42" and c.title == "Missing shingles"


# ---------------------------------------------------------------- store

def test_dhash_matches_resized_recompressed_copy():
    a = photo(1)
    b = Image.open(io.BytesIO(jpeg_bytes(a.resize((320, 240)), quality=60)))
    assert hamming(dhash(a), dhash(b)) <= 5
    assert hamming(dhash(a), dhash(photo(2))) > 10


def test_store_saves_jpeg_dedupes_and_skips_web_pages(tmp_path):
    existing = tmp_path / "have"
    existing.mkdir()
    photo(1).save(existing / "old.jpg")
    index = DedupeIndex()
    assert index.index_folder(existing) == 1

    png = io.BytesIO()
    photo(3).save(png, "PNG")
    net = FakeNet({
        "https://i.redd.it/new.png": FakeResp(png.getvalue(), "image/png"),
        "https://i.redd.it/dup.jpg": FakeResp(jpeg_bytes(photo(1).resize((800, 600)))),
        "https://i.redd.it/tiny.jpg": FakeResp(jpeg_bytes(photo(4, (200, 150)))),
        "https://redirects.example/img.jpg": FakeResp(b"<html>", "text/html", url="https://shop.example/"),
    })
    store = Store(tmp_path / "out", net, index)
    mk = lambda u, i: S.Candidate("reddit", i, u, "https://reddit.com/x", "t")
    saved = store.save(mk("https://i.redd.it/new.png", "n1"), "wind")
    assert saved and saved.suffix == ".jpg" and Image.open(saved).format == "JPEG"
    assert saved.parent.name == "wind"
    assert store.save(mk("https://i.redd.it/dup.jpg", "d1"), "wind") is None
    assert store.save(mk("https://i.redd.it/tiny.jpg", "t1"), "wind") is None
    assert store.save(mk("https://redirects.example/img.jpg", "r1"), "wind") is None
    assert store.stats == {"saved": 1, "duplicate": 1, "too_small": 1, "not_an_image": 1}

    rows = list(csv.DictReader(open(tmp_path / "out" / "manifest.csv")))
    assert len(rows) == 1 and rows[0]["page_url"] == "https://reddit.com/x" and rows[0]["file"] == "wind/" + saved.name

    # A second run into the same folder neither re-downloads nor re-saves.
    store2 = Store(tmp_path / "out", net, DedupeIndex())
    n = len(net.calls)
    assert store2.save(mk("https://i.redd.it/new.png", "n1"), "wind") is None
    assert len(net.calls) == n and store2.stats == {"already_seen": 1}


def test_pinterest_falls_back_to_736x(tmp_path):
    net = FakeNet({"https://i.pinimg.com/736x/aa/f.jpg": FakeResp(jpeg_bytes(photo(5, (736, 552))))})
    store = Store(tmp_path, net, DedupeIndex())
    c = S.Candidate("pinterest", "9", "https://i.pinimg.com/originals/aa/f.jpg",
                    extra={"fallback": "https://i.pinimg.com/736x/aa/f.jpg"})
    assert store.save(c, "unsorted") is not None
