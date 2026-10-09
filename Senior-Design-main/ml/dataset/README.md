# Roof damage dataset v1 (first-pass labels, 2026-10-09)

`manifest.csv` covers 239 unique roof photos in `images/` (iStock previews, ~612px wide; exact duplicates removed). The labels are Claude's first pass and
need a human check, starting with everything not marked `high`.

## Columns

| column | meaning |
|---|---|
| path | relative to this folder (use `--images-root dataset`) |
| label | damaged / undamaged / unusable (what the model trains on) |
| group | photos of the same roof or the same iStock upload batch share a group, so they never split across train/test |
| split | train / val / test, assigned by group |
| confidence | high / medium / low: how sure the first pass is. Review low and medium first |
| material | asphalt, clay_tile, concrete_tile, slate, wood_shake, metal, flat, mixed |
| damage_hint | wind, hail, missing_broken, none: a head start on phase 2, not checked carefully |
| damage_type | hail / wind / missing_shingles for damaged photos (one per photo, see below) |
| notes | what the labeler saw |
| idx | photo number used on the review contact sheets in the project files |

## Rules used

- **damaged**: visible missing, torn, lifted, broken or cracked roofing, holes, exposed deck or
  trusses. Any material.
- **undamaged**: roof surface clearly visible and intact. Moss, stains and debris alone don't count
  as damage.
- **unusable**: the photo can't answer "is this roof damaged?". Examples: scenery with no
  inspectable roof, a collage, a roof under construction or being re-roofed, a flat or metal-sheet
  roof (out of scope), or a surface too dark to see. Technically bad photos (tiny, blurry, dark)
  are caught by the quality gate in code and don't need this label.

## Review

The project files have contact sheets with the photo number, label and confidence (green high, amber medium,
red low): `undamaged_*.jpg`, `unusable_*.jpg`, `damaged_to_check_*.jpg`. To fix a label, edit the
`label` column in `manifest.csv` (find the row by `idx`), or tell Claude the numbers.

## Counts

| | damaged | undamaged | unusable |
|---|---|---|---|
| train | 147 | 13 | 10 |
| val | 29 | 4 | 2 |
| test | 29 | 3 | 2 |
| total | 205 | 20 | 14 |

Damaged material: 134 asphalt, 47 clay tile, the rest concrete tile, slate and other.
Damage hints: 127 wind, 61 missing/broken, only 5 possible hail (all low confidence).

## Damage types (phase 2, first pass 2026-10-09)

Each damaged photo has exactly one damage type, in `damage_type` (hail, wind or missing_shingles),
mirrored in three 0/1 columns `hail`, `wind`, `missing_shingles`.

| type | tagged when |
|---|---|
| wind | lifted, creased, folded or peeled-back shingles, or storm stripping. **Wind wins:** when wind tore shingles off, the photo is wind, not missing_shingles |
| missing_shingles | shingles, tiles or slates missing, broken or cracked with no sign of wind (age, rot, impact, neglect) |
| hail | round bruises, granule loss or impact marks with no torn edges |

`types_confidence` is high / medium / low for the type (all hail tags are low).
Undamaged and unusable rows leave these blank.

| type | total | train | val | test |
|---|---|---|---|---|
| wind | 130 | 99 | 14 | 17 |
| missing_shingles | 70 | 47 | 13 | 10 |
| hail | 5 | 1 | 2 | 2 |

Most missing_shingles photos are old or broken clay tile, slate or concrete tile; most wind photos
are hurricane-stripped asphalt. Hail is too rare to train on; it needs its own photos.

These are whole-photo tags. Locating the damage (phase 3) needs boxes drawn around each damaged
area, for example in CVAT, Label Studio or Roboflow, exported as COCO or YOLO.

## Training

    cd Senior-Design-main/ml
    python -m roofml.train --manifest dataset/manifest.csv --images-root dataset --model convnext_tiny --out runs/convnext_tiny

## License note

The images are iStock preview downloads. Check iStock's terms before sharing this dataset publicly
or using it beyond the class project.
