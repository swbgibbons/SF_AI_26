# RoofInsight final dataset

Reviewed roof-damage dataset in YOLO format: 733 images from 356 original photos.

## Layout

```text
final_dataset/
    Hail/              Training/ Validation/ Testing/   (each with images/ and labels/)
    Wind/              Training/ Validation/ Testing/
    Missing Shingles/  Training/ Validation/ Testing/
    No Damage/         Training/ Validation/ Testing/
    data.yaml          YOLO config listing all four category folders per split
    manifest.csv       every file with its source, original photo, split and box counts
```

Class ids: `0 = hail`, `1 = wind`, `2 = missing_shingles`. Each label line is
`class x_center y_center width height` (normalized). No Damage images have an empty label file.
An image sits in the folder of its most common damage type; a few images also contain boxes of a second type,
each with its own correct id.

| Category | Total | Training | Validation | Testing |
|---|---|---|---|---|
| Hail | 228 | 169 | 31 | 28 |
| Wind | 18 | 14 | 2 | 2 |
| Missing Shingles | 361 | 252 | 54 | 55 |
| No Damage | 126 | 79 | 22 | 25 |
| **All** | **733** | 514 | 109 | 110 |

## How it was prepared

- Several versions of one original photo (Roboflow crops/zooms, pieces of one drone photo, nearby drone frames,
  one claim's photo set, one video's screenshots) may appear, but always in one split, never across
  Training/Validation/Testing.
- Every image was reviewed by eye. Wind vs missing-shingle boxes were labeled by hand; photos with tile, slate,
  metal or wood roofs, severe damage (decking exposed), watermarks or no visible damage were removed.
- Images and box coordinates are unchanged from the source files; only class ids were set.

## Known limits

- Wind is small (18 images, 46 wind boxes in the whole dataset). More photos of lifted, creased or folded
  shingles are needed.
- Some chalk-circled hail hits in the drone pieces have no box yet.
- `training/train.py` expects a COCO export with `0 = wind, 1 = hail` and no missing-shingle class. This dataset
  needs a reviewed conversion (and a team decision on the missing-shingle class) before that script can use it.

## Sources and licenses

- RF_Clean hail / no-damage drone pieces (team dataset, Roboflow export).
- roof-wind-damage v26 by deldetect, CC BY 4.0: https://universe.roboflow.com/deldetect/roof-wind-damage
- Roof Damage Project v3 by aaron-davis-dvipp, CC BY 4.0:
  https://universe.roboflow.com/aaron-davis-dvipp/roof-damage-project

CC BY 4.0 data may be used, including commercially, with credit to the authors above.
