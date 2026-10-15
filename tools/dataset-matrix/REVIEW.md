# Review: roof dataset comparison matrix

## Verdict on the original script

It is clean, runnable code (parameterised SQL, validated 1-5 input, sensible weights), but it is a
**hand-typed scorecard, not a dataset tool**. It never opens a single image or label file, so it
cannot "take in" any file or quality type: resolution, format and annotation type are free-text
notes, and every score is someone's guess. Fine as a shared spreadsheet; not enough to pick training
data for this detector.

### Gaps that matter for this project
1. **No per-class counts.** The goal is hail / wind / missing shingles. A dataset with 10,000 hail
   images and zero missing-shingle boxes scores the same as a balanced one. (The backend's
   `core/services.py` currently only knows `["wind", "hail"]`, so missing shingles is the class most
   at risk of having no data.)
2. **No hard gates.** RF-DETR needs boxes (polygons/masks convert). Image-level "hail / no hail"
   folders can't train it, but can still rank #1. An unknown or non-commercial license is also just
   a note.
3. **"Drone similarity" is hard-coded** to one deployment assumption; it doesn't record what the
   viewpoint actually is (drone / satellite / ground).
4. **Box size vs. model input is ignored.** The backend's normal mode squashes the whole photo to the
   model input (560px). A hail hit that is 0.2% of a 4000px drone photo becomes ~1px. That decides
   whether a dataset is usable, and whether you must use the tiled mode.
5. **Nothing about data hygiene:** corrupt files, duplicates (inflate counts and leak between
   train/test), blur, exposure, EXIF rotation (phone/drone photos that look upright but whose
   pixels are sideways, so boxes don't line up), HEIC from iPhones.

### Smaller bugs
- Delete and Update print "deleted"/"updated" even when the ID doesn't exist; delete has no confirm.
- Only scores can be edited; a typo in any other field means delete and re-add.
- Changing weights never recomputes stored totals.
- Unknown image count silently becomes 0. No export (CSV) to share with the team.
- "Damaged / undamaged balance" is text, so it can't be compared or ranked.

## What v2 (`roof_dataset_matrix.py`) adds
- `profile <folder>` scans a downloaded dataset: JPG/PNG/TIFF/WebP/BMP (+HEIC with
  `pillow-heif`), corrupt files, resolution spread and % low-res, % blurry, % badly exposed,
  exact and near duplicates, EXIF rotation, drone/phone hints from EXIF.
- Auto-detects **COCO JSON, YOLO txt (boxes and segmentation), Pascal VOC XML, and
  class-per-folder** layouts, counts instances, maps raw names (e.g. `hail_damage`, `Missing_Shingle`)
  to hail / wind / missing_shingles via `CLASS_ALIASES`, and reports unmapped labels.
- Reports % of boxes too small to detect in normal vs. tiled inference.
- Suggests 1-5 scores from those measurements; you confirm or override (viewpoint stays a human call).
- Hard gates: no boxes or license marked unusable → score 0 and flagged in the list.
- Per-class counts and undamaged-image counts in the ranked table, plus totals across usable datasets.
- Edit any field, confirm deletes, `rescore` after changing weights, `export` to CSV.
- Opens an existing v1 `roof_dataset_matrix.db` and migrates it in place.

Run: `pip install pillow numpy` then `python roof_dataset_matrix.py` (menu) or
`python roof_dataset_matrix.py profile path/to/dataset --save`.

Tested on synthetic YOLO (mixed formats, a corrupt file, a duplicate, blur, tiny image, EXIF-rotated
DJI photo), COCO-polygon, VOC and class-folder datasets, plus a v1 database. Not yet run on a real
public roof dataset.
