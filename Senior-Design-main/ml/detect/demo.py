"""Local demo: upload a roof photo in the browser and see the damage boxes.

    pip install gradio
    python -m detect.demo --weights runs/detect/v1/weights/best.pt

Opens http://127.0.0.1:7860. Add --share for a temporary public link.
"""

import argparse

import gradio as gr
from ultralytics import YOLO

from detect.build_dataset import CLASSES

NAMES = {"wind": "Wind", "hail": "Hail", "missing_shingles": "Missing / broken shingles"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="runs/detect/v1/weights/best.pt")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    model = YOLO(args.weights)

    def detect(image, conf):
        # Gradio hands over RGB; Ultralytics reads numpy arrays as BGR.
        result = model.predict(image[:, :, ::-1], conf=conf, verbose=False)[0]
        rows = sorted(((CLASSES[int(c)], float(s)) for c, s in zip(result.boxes.cls, result.boxes.conf)),
                      key=lambda r: -r[1])
        verdict = f"{NAMES[rows[0][0]]} ({rows[0][1]:.0%})" if rows else "No damage found"
        return result.plot()[:, :, ::-1], verdict, [[NAMES[c], f"{s:.0%}"] for c, s in rows]

    gr.Interface(
        fn=detect,
        inputs=[gr.Image(type="numpy", label="Roof photo"),
                gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Minimum confidence")],
        outputs=[gr.Image(label="Detected damage"), gr.Textbox(label="Damage type"),
                 gr.Dataframe(headers=["Box", "Confidence"], label="Boxes")],
        title="RoofInSight Detector",
        flagging_mode="never",
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
