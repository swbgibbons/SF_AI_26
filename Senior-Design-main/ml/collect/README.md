# Roof damage photo collector

Finds roof damage photos (priority: **wind** and **missing/broken shingles**, plus hail) on Reddit,
Pinterest and openly licensed photo libraries, and downloads **the image files themselves** as JPEG.
It never opens the website a pin or post links to, skips videos and GIFs, drops small images and
anything you already have, and records where every photo came from.

```bash
cd Senior-Design-main/ml
pip install requests pillow numpy
python -m collect --help
```

## What it does with each photo

1. Gets a **direct image link** from the source (Reddit `i.redd.it` / gallery images, Pinterest
   `i.pinimg.com` originals, Openverse / Wikimedia file URLs). Link posts to other sites are ignored.
2. Downloads it. If a link redirects to a web page instead of an image, it is dropped
   (`not_an_image` in `skipped.csv`) rather than opened.
3. Drops videos, animated GIFs, broken files and images under 300px on the short side (`--min-side`).
4. Drops **duplicates** of photos you already have: exact (SHA-256) and near-identical (perceptual
   hash, catches resized / recompressed copies like the iStock `612x612` previews).
5. Guesses the damage type from the post title (`wind`, `missing_shingles`, `hail`, or `unsorted`),
   saves it as `<type>/<type>__<source>_<id>.jpg`, and appends a row to `manifest.csv`.

Output is a **review folder**, not training data. Look through it, delete what's wrong, and then
label the keepers the usual way (`dataset/README.md`).

```
collected/2026-11-01/
  wind/  missing_shingles/  hail/  unsorted/
  manifest.csv     file, guessed_type, source, title, page_url, image_url, author, license, ...
  skipped.csv      everything that was dropped, and why
```

Re-running into the same folder is safe: URLs already tried are not downloaded again.

## Sources

| source | how | needs |
|---|---|---|
| `reddit` | Reddit search API, image posts only. Default: r/Roofing (then HomeImprovement, homeowners, Insurance, HomeInspections) for "missing shingles", "wind damage", "shingles blown off", ... | Nothing, but unauthenticated Reddit allows ~10 requests/min and may block. **Recommended:** create a free *script* app at <https://www.reddit.com/prefs/apps> and set `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` (100 requests/min). |
| `pinterest` | Your boards (RSS feed, or the official API) and/or a list of pin links | See below |
| `openverse` | Openverse API: Creative Commons photos, mostly from Flickr | Nothing |
| `wikimedia` | Wikimedia Commons search | Nothing |
| `urls` | A text file of direct image links | Nothing |

### Reddit: run the exact searches you were browsing

Paste the search page URL; the subreddit and search words are read from it.

```bash
python -m collect --sources reddit \
  --reddit-url "https://www.reddit.com/r/Roofing/search/?q=missing+shingles&type=media" \
  --reddit-url "https://www.reddit.com/r/Roofing/search/?q=wind+damage&type=media"
```

### Pinterest: save pins to a board, then pull the board

Pinterest's terms forbid scraping search results, so the tool doesn't crawl Pinterest search.
Instead it removes the tedious part: while browsing, **save pins to a board** (one click each), then

```bash
python -m collect --sources pinterest --pinterest-board https://www.pinterest.com/<you>/<board>/
```

- A public board is read from its RSS feed, which only lists the most recent ~25–50 pins, so run
  it every so often or use a fresh board per batch.
- To get every pin on a board you own, create an app at <https://developers.pinterest.com/>,
  generate an access token with `boards:read` and `pins:read`, and set `PINTEREST_ACCESS_TOKEN`.
- Or put pin links (including `pin.it` short links) in a text file, one per line, and pass
  `--pin-list pins.txt`. The tool reads the image from the pin and downloads the full-size file
  from Pinterest's image server; it never follows the pin to the original website.

## Common options

- `--query "roof wind damage"` (repeatable) replaces the default search terms for every source.
- `--max 200` stops after 200 saved photos.
- `--dedupe-against path/to/photos other/folder` points the duplicate check at your local copies of
  the dataset. In the project's cloud workspace it defaults to `/mnt/project-files/{data,dataset}`
  and the iStock originals; on your own computer, point it at wherever your photos are.
- `--out folder` sets the output folder.
- `--model path/to/best.onnx` scores each new photo with the roof classifier and fills the
  `model_label` / `model_damaged` columns, so you can sort the review by how damaged it looks
  (needs `onnxruntime`).
- `--keep-off-topic` keeps results whose title never mentions a roof or shingles (dropped by default).

## Licensing

`manifest.csv` keeps the source page, author and license for every photo. Openverse and Wikimedia
photos are Creative Commons or public domain (check the exact license: some are NonCommercial or
require attribution). Reddit and Pinterest photos belong to whoever posted them; they are marked
`unknown`. Fine for reviewing and for a class project's training set, but don't republish them
(slides, a public dataset, the app) without checking the source page.

The collector identifies itself with a descriptive User-Agent, waits between requests to each site
(6s for unauthenticated Reddit, 3.5s for Openverse), backs off on HTTP 429, and checks robots.txt
before opening any pin page.
