# v2: teaching the detector hail without losing wind and missing shingles

Branch `v2-hail`. Status: **labels merged into the working dataset; not frozen, not trained yet.**

## Goal

Make the damage detector confident on **hail** while keeping **wind** and **missing/broken
shingles** at least as good as v1. v1 scored 0 on hail because it had one hail photo to learn from
(see `reports/yolo_vs_rfdetr.md`).

## What this branch adds

| Change | Where |
|---|---|
| RF-DETR training and comparison scripts (merged from `claude/project-thread-hb78eu`, unchanged) | `detect/train_rfdetr.py`, `detect/compare.py`, `reports/compare_v1_*` |
| Importer for the Roboflow batch | `detect/import_rf_clean.py` |
| 1003 Roboflow photos and 3238 boxes added to the **working** labels (v1 rows untouched) | `dataset/manifest.csv`, `dataset/boxes.csv` |
| This write-up | `V2_HAIL_APPROACH.md` |

`dataset/versions/v1/` is not touched, so v1 can still be rebuilt and verified exactly.

## Where the new data comes from

Two Roboflow exports of roof photos, cleaned before import (the "RF_Clean" batch):

- **testset2** (hail-damage boxes) and **testset1** (broken / loose / missing shingle polygons).
  Both annotate the same 189 drone photos, and testset1's "broken shingle" class on those photos
  is actually hail hits, so those boxes are imported as `hail`.
- Removed before import: photos under 15 KB, thin letterboxed strips (picture aspect < 0.5 or
  short side < 150 px), near-duplicates, and hail photos whose only hits are under 10 px or cut
  off at the edge.
- Every hail photo in `hail_damage` passed a visual check on the full image (`confidence` high).
  `poten_hail` photos have annotator hail boxes but the roof looks undamaged at full-image view
  (`confidence` low). They are kept as hail training data on purpose.

The importer reads an already-converted folder (`RF_Clean_Annotated`, kept outside git with the
photos) whose `manifest.csv` records, for every photo, its source file, split, roof group, box
counts and any change made.

## Labeling rules applied

Same classes and order as v1: `wind=0, hail=1, missing_shingles=2`, one damage type per photo,
undamaged photos as background. On top of that:

1. **Missing/broken shingles were recovered.** An earlier sort had merged testset1's broken,
   loose and missing shingle polygons (32 web photos) into a single "wind" folder. Each box's
   original class was restored from testset1 (same polygons, same order, verified), and the boxes
   became `missing_shingles`.
2. **Wind wins** (rule from `dataset/README.md`). Four of those photos show shingles stripped over
   an area down to the underlayment, like v1's wind photos. After review they are `wind` (16
   boxes): `10-reasons-to-repair-roof-damage-asap`, `Storm-Damage-2`,
   `storm-damage-722-furnace-hills-pike-1`, `wind-roof-damage-repair`.
3. **One missed photo fixed.** `Unhealthy-Roof` sat in no-damage with no boxes but shows a hole and
   broken clay tiles. It has 2 hand-drawn `missing_shingles` boxes (`status` = `manual`).
4. Polygons were converted to boxes, and boxes thinner than 3 px (edge slivers) were dropped (38).
5. One photo is still open for review: `DJIw_391` (wood shake, a few slipped shakes) is kept as
   undamaged.

`status` in `boxes.csv` is `roboflow` for imported boxes and `manual` for the hand-drawn ones.

## Splits

Same method as v1: every photo of a roof goes to one split, about 70 / 15 / 15.

- Roof groups are `rf-<group>`. The claim-numbered close-up series (`05FW7131_*`, `05F1LHD1_*`)
  are one roof each. Copies of one source photo (Roboflow `.rf.<hash>` augmentations, and the
  testset1/testset2 copy of the same drone photo) share a group.
- Roboflow's own train/test split was **not** kept: it put crops of the same photo on both sides
  (30 source photos leaked). The cleaned hail split was already by roof and was kept.
- All v1 rows keep their v1 split; every v1 test photo stays in test.

| split | photos | hail boxes | wind boxes | missing_shingles boxes | background photos |
|---|---|---|---|---|---|
| train | 864 | 2030 | 138 | 140 | 54 |
| val | 194 | 583 | 15 | 24 | 17 |
| test | 184 | 507 | 22 | 37 | 29 |

(Counts include v1. Unusable photos are in the manifest but left out of training: 10 / 2 / 2.)

## Severe structural damage (sponsor rule)

On branch `claude/project-thread-wcqdvo`, the sponsor rule marks photos with holes through the
roof structure or collapse as `unusable`. That rule moves **60 v1 photos** out of training, and
**40 of them are missing_shingles**, leaving 30 of the original 70. It would also remove **7 of
the 10 missing-shingles test photos**, leaving 3, too few to measure anything.

Decision for v2: **keep those photos as v1 labeled them** (this branch does not merge
`wcqdvo`), and measure whether they help:

- Train v2 with them (the default) and an ablation **without** them.
- Report missing-shingles scores on v1 test for all photos and for non-severe photos only.
- If the photos don't improve missing-shingles detection on non-severe roofs, drop them and adopt
  the sponsor rule for v3.

Some RF_Clean web photos would likely also be severe by the same rule (holes showing battens or
darkness): `Screen-Shot-2021-12-15`, `7-15-signs-of-roof-damage`, `Clay-roof-tiles-missing`,
`cost-of-roof-repair`, `image4`, `roof-repair`, `shutterstock_20822053`, `Unhealthy-Roof`. They
are labeled `missing_shingles` and not yet tagged severe.

<details>
<summary>The 60 v1 photos the sponsor rule marks severe (kept as damaged in v2)</summary>

| photo | v1 type | split | v1 note |
|---|---|---|---|
| `istock-1289183330.jpg` | missing_shingles | test | missing tiles exposing battens |
| `istock-1292591243.jpg` | missing_shingles | test | same roof as 49 |
| `istock-1368251260.jpg` | missing_shingles | test | missing tiles, same roof as 49 |
| `istock-1464549794.jpg` | missing_shingles | test | dormer roof broken open |
| `istock-2166416857.jpg` | missing_shingles | test | missing tiles |
| `istock-2287227974.jpg` | missing_shingles | test | missing tiles |
| `istock-963180436.jpg` | missing_shingles | test | several missing tiles |
| `istock-104207957.jpg` | missing_shingles | train | slate roof; broken and missing slates exposing battens |
| `istock-1135068321.jpg` | missing_shingles | train | holes, abandoned building |
| `istock-1163541715.jpg` | missing_shingles | train | large hole |
| `istock-1214762496.jpg` | missing_shingles | train | aerial/drone view, collapsed roof section |
| `istock-1316463534.jpg` | missing_shingles | train | collapsed derelict building |
| `istock-1326039993.jpg` | missing_shingles | train | holes and broken shingles, moss |
| `istock-1365774784.jpg` | missing_shingles | train | missing slates near ridge |
| `istock-1368251153.jpg` | missing_shingles | train | collapsed roof with plants growing |
| `istock-1388131267.jpg` | missing_shingles | train | broken tiles |
| `istock-1388919750.jpg` | missing_shingles | train | partly collapsed |
| `istock-1621206938.jpg` | missing_shingles | train | missing tiles exposing battens |
| `istock-2158183855.jpg` | missing_shingles | train | hole punched through roof |
| `istock-2161302398.jpg` | missing_shingles | train | hole exposing rafters |
| `istock-2161302404.jpg` | missing_shingles | train | aerial, holes in tile roof |
| `istock-2189140775.jpg` | missing_shingles | train | collapsed old roof |
| `istock-2273401386.jpg` | missing_shingles | train | hole by chimney |
| `istock-2278810575.jpg` | missing_shingles | train | holes along roof |
| `istock-497853989.jpg` | missing_shingles | train | many missing tiles exposing battens |
| `istock-519530560.jpg` | missing_shingles | train | derelict pyramid roof, many missing/loose shakes |
| `istock-669240502.jpg` | missing_shingles | train | clay tile; tiles missing exposing battens |
| `istock-870471802.jpg` | missing_shingles | train | many broken tiles, holes |
| `istock-976623908.jpg` | missing_shingles | train | collapsed section, whole building in frame |
| `istock-1205391887.jpg` | missing_shingles | val | displaced tiles |
| `istock-1247229250.jpg` | missing_shingles | val | holes in tile roof |
| `istock-1253783035.jpg` | missing_shingles | val | small holes near ridge, distant shot |
| `istock-2183136122.jpg` | missing_shingles | val | broken tiles |
| `istock-2278550588.jpg` | missing_shingles | val | collapsed |
| `istock-2278550602.jpg` | missing_shingles | val | collapsed, same as 213 |
| `istock-492965351.jpg` | missing_shingles | val | broken/slipped tiles; same photographer batch as 492968089 |
| `istock-492968089.jpg` | missing_shingles | val | clay tile; many broken/missing tiles; SAME ROOF as 493070879 |
| `istock-493070879.jpg` | missing_shingles | val | clay tile; closer shot of the same roof as 492968089 |
| `istock-960214058.jpg` | missing_shingles | val | hole with exposed battens |
| `istock-966403608.jpg` | missing_shingles | val | partially collapsed roof |
| `istock-2148752521.jpg` | wind | test | aerial, collapsed roof |
| `istock-1254608275.jpg` | wind | train | storm damage around roof windows |
| `istock-1254608276.jpg` | wind | train | same building as 39, wider view |
| `istock-1254608277.jpg` | wind | train | same building as 39 |
| `istock-1254608279.jpg` | wind | train | same building as 39 |
| `istock-1254608283.jpg` | wind | train | same building as 39 |
| `istock-1431078234.jpg` | wind | train | aerial, roof torn off, structural damage |
| `istock-1512134652.jpg` | wind | train | roof deck gone, trusses exposed |
| `istock-1612631850.jpg` | wind | train | roof deck gone, trusses exposed |
| `istock-1621912024.jpg` | wind | train | roof deck gone, trusses exposed, same as 134 |
| `istock-1860386162.jpg` | wind | train | roof deck gone, trusses exposed |
| `istock-2124072165.jpg` | wind | train | roof deck gone, trusses exposed |
| `istock-2136050586.jpg` | wind | train | aerial, collapsed roof |
| `istock-2150829405.jpg` | wind | train | roof deck gone, trusses exposed |
| `istock-2151920306.jpg` | wind | train | deck panels torn off |
| `istock-2156501281.jpg` | wind | train | deck panels torn off |
| `istock-2157989679.jpg` | wind | train | aerial, collapsed roof and debris |
| `istock-2284106202.jpg` | wind | train | roof deck gone, trusses exposed |
| `istock-2294973638.jpg` | wind | train | aerial, holes through roof |
| `istock-468992927.jpg` | wind | train | coastal storm, missing slates near ridge, building in wider  |
</details>

## Model and training plan

**RF-DETR Nano**, per `reports/yolo_vs_rfdetr.md`: better boxes than YOLO11n on v1 (missing-shingles
AP50 0.28 vs 0.08 to 0.16), Apache-2.0, and the backend already serves an RF-DETR ONNX model.
YOLO11n v1 (`models/v1/best.pt`) stays frozen as the fallback.

The risk is imbalance: hail has about 15 times more boxes than wind or missing_shingles, which
could pull the model toward hail and cost the other two. Planned countermeasures:

1. **Repeat wind and missing-shingles training photos** in the train export so each class
   contributes a similar number of boxes per round (about 4x). Val and test are not repeated.
2. Keep the 54 undamaged training photos as background so hail-trained features don't fire on
   clean roofs.
3. Choose the best round on **val** by the lowest of the three class AP50s, not the average.

Runs, all on the same frozen v2 and seed:

| run | data |
|---|---|
| baseline | v1 only (RF-DETR retrained here, same settings, for a like-for-like comparison) |
| v2 | v1 + RF_Clean |
| v2-no-severe | v2 without the 60 severe v1 photos |

On CPU, expect roughly 1 h for the baseline and 5 to 7 h per v2 run at 40 rounds; a GPU makes this
minutes.

## How v2 is judged

Scored with `detect/compare.py` at the confidence cutoff used for v1 (0.25):

- **Hail improves:** hail AP50 and hail photos found on the RF_Clean test (visibly damaged hail
  roofs), versus 0 for v1.
- **Wind and missing shingles not compromised** (on the locked v1 test): each class AP50 no more than
  0.05 below the baseline, and correct damage type on at least 19 of 29 damaged photos. 0.05 is
  about the seed-to-seed swing seen in v1.
- **Clean roofs stay clean:** no more false boxes on the test background photos than the baseline.

## Still needed before training

1. **The exact v1 photos.** 223 of the 239 local copies match `versions/v1/SHA256SUMS`; 15 differ
   and 1 is missing, so `detect.build_dataset --version v1` and `detect.freeze v2` will refuse to
   run. Use `dataset/roof_damage_dataset_v1.zip` from the project files.
2. Freeze: `python -m detect.freeze v2`, then commit `dataset/versions/v2/`.
3. Go-ahead to train.

## Reproduce

```bash
cd Senior-Design-main/ml
# v1 photos from the project zip into dataset/images/ (see dataset/README.md)
python -m detect.import_rf_clean --src <path>/RF_Clean_Annotated   # already done on this branch
python -m detect.freeze v2
python -m detect.build_dataset --version v2 --out data/det-v2
python -m detect.train_rfdetr --data data/det-v2/coco --size nano --epochs 40 --name rfdetr-n-v2
# wind / missing shingles on the locked v1 test, hail on the v2 test
python -m detect.compare --version v1 --split test --out reports/compare_v2_on_v1_test \
    --model rfdetr-nano:runs/rfdetr/rfdetr-n-v1/checkpoint_best_total.pth \
    --model rfdetr-nano:runs/rfdetr/rfdetr-n-v2/checkpoint_best_total.pth
python -m detect.compare --version v2 --split test --out reports/compare_v2_test \
    --model rfdetr-nano:runs/rfdetr/rfdetr-n-v1/checkpoint_best_total.pth \
    --model rfdetr-nano:runs/rfdetr/rfdetr-n-v2/checkpoint_best_total.pth
```

The RF_Clean photos are Roboflow exports. Check the source datasets' licenses before sharing them
outside the class project.
