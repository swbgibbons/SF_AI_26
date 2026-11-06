# Dataset count and pie chart

Counts the licensing-safe roof photos per damage type and draws the pie chart for slides.
Re-run it every time new photos arrive.

```bash
pip install matplotlib pillow          # pillow is only needed for the test
cd Senior-Design-main/ml/dataset_count

# today's dataset (Roboflow hail batch, RF_Clean parts 5-6)
python count_dataset.py --labels labels/rf_clean_parts5-6.csv --out out/

# current combined count (Nov 6): parts 5-6, a teammate's parts 7-8, and the counts
# other teammates reported for photos they haven't sent yet
python count_dataset.py --labels labels/rf_clean_parts5-6.csv \
    --labels /mnt/project-files/dataset/incoming/rf_clean_part7/dataset/image_labels.csv \
    --labels /mnt/project-files/dataset/incoming/rf_clean_part8/dataset/image_labels.csv \
    --pending labels/pending_teammates.csv --out out/

# when a teammate's photos arrive: add their folder (or a Roboflow export) with --images,
# or the image_labels.csv inside their export with --labels, and lower their row in
# labels/pending_teammates.csv so nothing is counted twice
```

`--pending` takes `class,count[,who]` rows. The chart shows one total per damage type (in hand + pending);
`counts.csv` and `summary.md` keep the in-hand and pending numbers apart.
They can't be checked for V1 copies or duplicates until they arrive.

`labels/rf_clean_part7.csv` and `rf_clean_part8.csv` are copies of the sheets inside the
teammate's zips (counts reproduce without the photos; pass the copies in `incoming/` to also hash
the photos).

Outputs in `out/`: `pie.png` (slides), `pie.svg`, `counts.csv`, `summary.md`
(per-split table when the photos sit in train/valid/test folders), and `images.csv`
(every photo, its class, and why it was excluded if it was).

## What it reads

| Input | Layout |
|---|---|
| `--labels sheet.csv` | one row per photo: `image_id`, `file` or `image`, and `label` or `damage_type` (hail, wind, missing_shingles, no_damage, unusable). Optional `labels` = `hail;wind` for photos with two types, `source`, `license`, `split`. |
| `--images DIR` | a Roboflow **COCO** export (`_annotations.coco.json`), a Roboflow **YOLO** export (`data.yaml`), or plain folders named by type (`hail/`, `Wind Damage/`, `missing_shingles/`, `no_damage/` … at any depth, inside train/valid/test or not). |

Folder and class names are matched loosely (`Hail_Damage`, `Missing_Shingle`, `undamaged` all work;
see `ALIASES`). Photos it can't place are counted as "Not labeled yet". A Roboflow photo with
boxes of two types is counted under the type with the most boxes, and `summary.md` reports how
many photos have more than one type. A Roboflow photo with no boxes counts as no damage.

## V1 iStock photos are never counted

The V1 photos were iStock previews and can't be used for licensing reasons. Every photo is
checked against `v1_blocklist_sha256.txt` and skipped if:

- its SHA-256 matches one of the 1,153 V1 files (originals, training copies, review copies, zips), or
- its file name contains `istock` or one of the 239 V1 iStock photo ids (catches renamed or
  resized copies such as `hail__2234328050.jpg`).

Exact duplicates across sources are counted once. To add more files to the blocklist:
`python count_dataset.py blocklist FOLDER_OR_ZIP ... >> v1_blocklist_sha256.txt`

## Current labels

`labels/rf_clean_parts5-6.csv` holds the 263 Roboflow Universe photos from the hail batch
(RF_Clean parts 5 and 6): 51 validated by Hugo in the Roof Damage Validation page, 212 drafted by
Claude in the Hail Batch AI Review page, with Hugo's two flags applied. The license column says
CC BY 4.0; confirm it on the source dataset's Roboflow page and credit it in the report.

Test: `python -m pytest test_count_dataset.py -q`
