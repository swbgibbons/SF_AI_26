"""Run: python -m pytest test_count_dataset.py -q"""
import json
import shutil
from pathlib import Path

from PIL import Image

import count_dataset as cd


def _img(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def test_counts_folders_coco_and_blocks_v1(tmp_path):
    team = tmp_path / "teammate"
    _img(team / "hail" / "IMG_0001.jpg", (10, 0, 0))
    _img(team / "Wind Damage" / "train" / "IMG_0002.jpg", (0, 10, 0))
    _img(team / "no_damage" / "IMG_0003.jpg", (0, 0, 10))
    _img(team / "hail" / "istockphoto-111111111-612x612.jpg", (1, 1, 1))  # V1 by name
    _img(team / "wind" / "wind__222222222.jpg", (2, 2, 2))                # V1 by id
    _img(team / "hail" / "renamed_copy.jpg", (3, 3, 3))                   # V1 by hash
    shutil.copy(team / "hail" / "IMG_0001.jpg", team / "hail" / "dup.jpg")  # duplicate

    rf = tmp_path / "rf" / "train"
    _img(rf / "a.jpg", (20, 0, 0))
    _img(rf / "b.jpg", (0, 20, 0))
    _img(rf / "c.jpg", (0, 0, 20))
    (rf / "_annotations.coco.json").write_text(json.dumps({
        "categories": [{"id": 0, "name": "roof"}, {"id": 1, "name": "Hail_Damage"}, {"id": 2, "name": "Missing_Shingle"}],
        "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}, {"id": 3, "file_name": "c.jpg"}],
        "annotations": [{"image_id": 1, "category_id": 1}, {"image_id": 1, "category_id": 2}, {"image_id": 1, "category_id": 1},
                        {"image_id": 2, "category_id": 2}],
    }))

    bl = tmp_path / "bl.txt"
    bl.write_text(cd.sha256(team / "hail" / "renamed_copy.jpg") + "\nid:222222222\n")
    labels = tmp_path / "labels.csv"
    labels.write_text("image_id,file,label\np1,x.jpg,hail\np2,y.jpg,unusable\np3,z.jpg,Missing\n")

    rows = cd.collect([labels], [team, tmp_path / "rf"], cd.Blocklist(bl))
    counts, total, excluded = cd.write_outputs(rows, tmp_path / "out", "t")

    assert counts["hail"] == 1 + 1 + 1          # folder + coco(a: 2 hail vs 1 missing) + csv
    assert counts["wind"] == 1
    assert counts["missing_shingles"] == 1 + 1  # coco b + csv
    assert counts["no_damage"] == 1 + 1         # folder + coco c (no boxes)
    assert counts["unusable"] == 1
    assert total == 8
    reasons = sorted(r["excluded"].split(" (")[0] for r in excluded)
    assert len(excluded) == 4 and reasons.count("V1 photo") == 3
    for f in ("pie.png", "pie.svg", "counts.csv", "images.csv", "summary.md"):
        assert (tmp_path / "out" / f).exists()


def test_teammate_sheet_and_pending(tmp_path):
    ds = tmp_path / "part7" / "dataset"
    _img(ds / "train" / "p7_000.jpg", (30, 0, 0))
    sheet = ds / "image_labels.csv"
    sheet.write_text(
        "image,split,damage_type,hail_boxes_original,hail_boxes_added,in_training_set\n"
        "p7_000.jpg,train,Hail,3,1,yes\n"
        "p7_001.jpg,train,Unclear/unusable,2,0,\n"
        "p7_002.jpg,test,No visible damage,0,0,\n"
        "p7_003.jpg,train,Wind,0,0,\n")
    pend = tmp_path / "pending.csv"
    pend.write_text("class,count\nhail,5\nwind damage,7\nno damage,2\n")
    pending, _ = cd.read_pending(pend)
    rows = cd.collect([sheet], [], cd.Blocklist(None))
    assert rows[0]["sha256"]  # photo found next to the sheet was hashed
    counts, total, _ = cd.write_outputs(rows, tmp_path / "out", "t", pending)
    assert (counts["hail"], counts["unusable"], counts["no_damage"], counts["wind"]) == (1, 1, 1, 1)
    assert pending == {"hail": 5, "wind": 7, "no_damage": 2}
    text = (tmp_path / "out" / "counts.csv").read_text()
    assert "hail,1,5,6," in text and ",4" in text.splitlines()[1]  # 4 boxes, unusable photo's boxes ignored
