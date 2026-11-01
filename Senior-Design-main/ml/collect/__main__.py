"""Collect roof damage photos for review.

Downloads image files directly (JPEG out), never the web page around them, drops videos,
small images and anything we already have (exact or near-duplicate), and writes one
manifest row per photo with its source page, author and license.

Examples:
    # Reddit + openly licensed sources, default searches
    python -m collect --sources reddit openverse wikimedia

    # Exactly the Reddit searches you were browsing
    python -m collect --sources reddit \\
        --reddit-url "https://www.reddit.com/r/Roofing/search/?q=missing+shingles&type=media" \\
        --reddit-url "https://www.reddit.com/r/Roofing/search/?q=wind+damage&type=media"

    # Pinterest: a board you saved pins to, and/or a text file of pin links
    python -m collect --sources pinterest --pinterest-board https://www.pinterest.com/you/roof-damage/ \\
        --pin-list pins.txt
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import requests

from . import sources as S
from .net import Http
from .store import DedupeIndex, Store

PROJECT = Path("/mnt/project-files")
DEFAULT_DEDUPE = [PROJECT / "data", PROJECT / "dataset", PROJECT]  # project root holds the iStock originals


def build_sources(args, net):
    out = []
    for name in args.sources:
        if name == "reddit":
            searches = [S.parse_reddit_search_url(u) for u in args.reddit_url] or None
            out.append(("reddit", S.Reddit(net, subs=args.reddit_sub or None, queries=args.query or None,
                                           searches=searches, per_query=args.per_query)))
        elif name == "pinterest":
            pins = []
            for f in args.pin_list:
                pins += [l.strip() for l in Path(f).read_text().splitlines() if l.strip() and not l.startswith("#")]
            if not args.pinterest_board and not pins:
                print("pinterest: give --pinterest-board and/or --pin-list (search pages aren't scraped)")
                continue
            out.append(("pinterest", S.Pinterest(net, boards=args.pinterest_board, pins=pins)))
        elif name == "openverse":
            out.append(("openverse", S.Openverse(net, queries=args.query or None)))
        elif name == "wikimedia":
            out.append(("wikimedia", S.Wikimedia(net, queries=args.query or None)))
        elif name == "urls":
            urls = []
            for f in args.url_list:
                urls += Path(f).read_text().splitlines()
            out.append(("urls", S.UrlList(urls)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=["reddit", "openverse", "wikimedia"],
                    choices=["reddit", "pinterest", "openverse", "wikimedia", "urls"])
    ap.add_argument("--out", default=None,
                    help="output folder (default: /mnt/project-files/data/collected/<today> if that "
                         "folder exists, else ./collected/<today>)")
    ap.add_argument("--dedupe-against", nargs="*", default=None,
                    help="folders of photos we already have (default: the project's data/, dataset/ "
                         "and iStock originals)")
    ap.add_argument("--query", action="append", default=[], help="search terms (repeatable; replaces defaults)")
    ap.add_argument("--reddit-sub", action="append", default=[], help="subreddit (repeatable)")
    ap.add_argument("--reddit-url", action="append", default=[],
                    help="a reddit search URL to run as-is (repeatable), e.g. r/Roofing/search/?q=wind+damage")
    ap.add_argument("--per-query", type=int, default=250, help="Reddit posts to read per search")
    ap.add_argument("--pinterest-board", action="append", default=[], help="public board URL (repeatable)")
    ap.add_argument("--pin-list", action="append", default=[], help="text file of pin URLs, one per line")
    ap.add_argument("--url-list", action="append", default=[], help="text file of direct image URLs")
    ap.add_argument("--min-side", type=int, default=300, help="drop images whose short side is smaller")
    ap.add_argument("--keep-off-topic", action="store_true",
                    help="keep search results whose title never mentions a roof/shingles")
    ap.add_argument("--max", type=int, default=0, help="stop after saving this many photos (0 = no limit)")
    ap.add_argument("--model", default=None,
                    help="optional: roof classifier .onnx; adds model_label / model_damaged columns")
    args = ap.parse_args(argv)

    today = date.today().isoformat()
    out = Path(args.out) if args.out else (
        PROJECT / "data" / "collected" / today if (PROJECT / "data").is_dir() else Path("collected") / today)

    index = DedupeIndex()
    dedupe = [Path(p) for p in args.dedupe_against] if args.dedupe_against is not None else \
        [p for p in DEFAULT_DEDUPE if p.is_dir()]
    out.mkdir(parents=True, exist_ok=True)
    for folder in dedupe:
        # The project root is only scanned at the top level; data/ and dataset/ are scanned fully.
        if folder == PROJECT:
            n = index.index_files(sorted(folder.glob("*")), out.parent / ".hash_cache_root.json")
        else:
            n = index.index_folder(folder, out.parent / f".hash_cache_{folder.name}.json")
        print(f"dedupe: {n} existing photos in {folder}")

    classifier = None
    if args.model:
        from roofml.predict import RoofClassifier  # needs onnxruntime + numpy
        classifier = RoofClassifier(args.model)

    net = Http()
    store = Store(out, net, index, min_short_side=args.min_side, classifier=classifier)
    print(f"saving to {out}")
    try:
        for name, source in build_sources(args, net):
            print(f"== {name}")
            try:
                for cand in source:
                    kind, relevant = S.guess_type(cand.extra.get("text") or cand.title)
                    # Pinterest boards / URL lists are hand-picked, so they skip the topic filter.
                    if not relevant and cand.source in ("reddit", "openverse", "wikimedia") \
                            and not args.keep_off_topic:
                        store._skip(cand, "off_topic")
                        continue
                    path = store.save(cand, kind)
                    if path:
                        print(f"  + {path.relative_to(out)}")
                    if args.max and store.stats.get("saved", 0) >= args.max:
                        raise KeyboardInterrupt
            except requests.RequestException as e:
                print(f"  {name}: network error, moving on ({type(e).__name__})")
    except KeyboardInterrupt:
        print("stopped")
    print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(store.stats.items())) or "nothing found")
    print(f"manifest: {store.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
