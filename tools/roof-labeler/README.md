# Roof Damage Labeler

A single-page tool for drawing and checking damage boxes on roof photos. It writes YOLO labels in the
same format as `final_dataset/`, so everyone's output drops straight into training.

Nothing to install. Open `labeler.html` in Chrome, Edge or Firefox. The photos never leave your computer.

## Quick start

1. Unzip the batch you were given, for example `A_NewRoofImages_Annotated/not_used.zip`.
2. If the batch has starter labels, copy them into the same folder and name the file `annotations.json`.
   The 50 `not_used` photos have one: `starter/nr_not_used_annotations.json`.
3. Open `labeler.html`, type your name at the top right, click **Open folder** and pick the photo folder.
4. Work through the photos (see the guide below).
5. Click **Export labels**. You get `<folder>_<your name>.zip`. Send it back or commit it, as agreed.

Progress autosaves in your browser. If you close the tab, reopen the same folder in the same browser and
your edits come back. Autosave is not a backup, so export at the end of every session.

The tool also reads plain YOLO `.txt` files, boxes or polygons, that sit next to the photos with the same name
(`nr014.webp` + `nr014.txt`). When an `annotations.json` is present, its boxes win.

## Controls

| Action | Mouse | Key |
|---|---|---|
| Next / previous photo | ‹ › buttons, or click in the list | → / ← |
| Box tool / Shape tool | Box, Shape buttons | B / P |
| Draw a box (Box tool) | Drag on an empty part of the photo | |
| Draw a shape (Shape tool) | Click each corner around the damage, then click the first point | Enter closes, Backspace drops the last point, Esc cancels |
| Select, move | Click a shape, drag it | |
| Reshape | Drag any corner of the selected shape. Corners move freely, so a box can be skewed to follow the roof | |
| Add a corner | Drag one of the small diamonds on the edges | |
| Remove a corner | Double-click it (or right-click) | |
| Set class (new shape, or the selected one) | Class buttons | 1 hail, 2 wind, 3 missing shingles |
| Delete selected shape | × on the shape, or Delete shape | Del / Backspace |
| Undo | Undo | Ctrl+Z / Cmd+Z |
| Flag / unflag photo | Flag | F |
| Mark reviewed | Mark reviewed | R |
| Deselect | | Esc |

**Undo my changes** puts the current photo back to how it was when you opened the folder.

## Labeling guide

### Classes

| id | class | Box it when you see | Not this class |
|---|---|---|---|
| 0 | `hail` | Round dark bruises or dents where granules are knocked off, scattered hits on shingles, dented vents or flashing | General granule wear across the whole roof, blistering, algae streaks |
| 1 | `wind` | Lifted, creased or folded tabs, torn tabs still attached, broken or lifted ridge caps, starter strip peeled at the eave | Shingles that are simply a different colour |
| 2 | `missing_shingles` | Shingles gone and underlayment, felt, deck or the course below showing, ridge cap gone, tarps covering a hole | Tear-off in progress where a crew is re-roofing on purpose (flag it instead) |

If a spot fits two classes, pick the one that a roofer would write on the claim: a tab that is torn off
and gone is `missing_shingles`, a tab that is creased but still there is `wind`.

### Drawing shapes

- Photos are taken at an angle, so damage rarely sits in an upright rectangle. Fit the outline to the
  damaged area: start with a box and drag its corners onto the damage, or trace it with the Shape tool.
  Four to eight corners is plenty.

- One box per damaged area. Separate hits that are clearly apart get separate boxes, a cluster of hail
  hits on the same few shingles can share one.
- Fit the outline tight to the damage, with a small margin. Don't box the whole roof unless the whole
  visible slope really is stripped.
- Box only roof covering. Gutters, fascia, siding and soffits are out of scope.
- Leave a photo with no boxes when there is no roof damage. An empty label file is a valid
  "No Damage" example.
- Very small damage (a few pixels in a far-away shot) is still worth a box, but flag the photo.

### When to flag

Flag a photo and write the reason in **Note** when any of these is true:

- You are not sure the damage is real, or not sure of the class.
- The photo is unusable for training: screenshot, heavy blur, snow covering the roof, the roof is a tiny
  part of the picture, or it shows a re-roof in progress.
- The damage is only to gutters, fascia or trim.
- The roof has patches of replacement shingles that could be mistaken for damage.

Flags are questions for the reviewer, not rejections. Keep the boxes you think are right on a flagged
photo.

### Review (validation)

Every photo is checked by a second person before it goes into `final_dataset`.

1. **Labeler** draws boxes, flags anything doubtful with a note, and exports.
2. **Reviewer** opens the same photos with the labeler's `annotations.json` in the folder, fixes boxes,
   and presses **R** on each photo they agree with. Filter by **Flagged** first, then **Not reviewed**.
3. The reviewer clears the flag when the question is settled. A photo that should not be used stays
   flagged with a note saying so, and is left out of training.
4. The reviewer exports. That zip is the one that gets merged.

The tool records who labeled each photo (the first person to change its boxes), who last edited it
and who reviewed it in `annotations.json`, and shows a warning when the labeler and reviewer are the
same person. A photo is done when it is reviewed by someone other than its
labeler and not flagged.

## What the export contains

```text
labels/<photo>.txt       YOLO boxes: class x_center y_center width height (normalized), empty = no damage.
                         Each shape becomes the upright box around it, so this works with final_dataset as is.
labels_seg/<photo>.txt   YOLO polygons: class x1 y1 x2 y2 ... (normalized), the exact outlines you drew
classes.txt              hail, wind, missing_shingles (ids 0, 1, 2)
flagged.txt              flagged photos and their notes
annotations.json         every photo: shapes (points and bounding box), flag, reviewed, note, labeledBy, editedBy, reviewedBy, updatedAt
```

If the tool is used without internet, the zip library can't load and the export falls back to
`annotations.json` alone. It holds everything, and opening it again in the tool with internet gives
you the full zip.
