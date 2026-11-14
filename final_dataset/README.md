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

