from pathlib import Path
import tempfile
import requests

from .model_utils import RFDETR_ONNX, run_rfdetr_inference, run_rfdetr_inference_tiled

# This service class encapsulates the loading and inference logic for the RFDETR model through OnnxRuntime. It provides a clean interface for making predictions on images, whether they are local files or accessible via URLs. The class handles model loading, image downloading, and inference execution, abstracting away the complexities of these operations from the rest of the application.


class RFDETRService:
    _model = None
    _class_names = ["wind", "hail"]
    _model_type = "ONNX"
    _ram_estimate_mb = None

    @classmethod
    def _load_model(cls):
        if cls._model is not None:
            return cls._model, cls._class_names, cls._model_type

        # ----------------------------
        # BASE DIR CHECK
        # ----------------------------
        BASE_DIR = Path(__file__).resolve().parent
        if not BASE_DIR.exists():
            raise RuntimeError(f"BASE_DIR does not exist: {BASE_DIR}")

        # ----------------------------
        # ONNX MODEL PATH CHECK
        # ----------------------------
        # Prefer the local exported_models_cpu copy; otherwise fetch from HF Hub.
        BASE_DIR = Path(__file__).resolve().parent
        onnx_path = Path("exported_models", "inference_model.onnx")

        if not onnx_path.exists():
            raise RuntimeError(
                f"No ONNX model found , check docker image copy step., {onnx_path}"
            )
        # Instantiate ONNX-only model
        cls._model = RFDETR_ONNX(str(onnx_path))
        cls._run_normal = run_rfdetr_inference
        cls._run_tiled = run_rfdetr_inference_tiled

        return cls._model, cls._class_names, cls._model_type
    # ----------------------------
    # Utility: download image
    # ----------------------------

    @staticmethod
    def _download_image(url: str) -> str:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to download image: {url}") from e

        suffix = Path(url).suffix or ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(resp.content)
        tmp.close()
        return tmp.name

    # ----------------------------
    # Public inference API
    # ----------------------------

    @classmethod
    def predict(
        cls,
        image_path_or_url: str,
        mode: str = "normal",
        threshold: float = 0.4,
        tile_size: int = 560,
    ):
        model, class_names, _ = cls._load_model()

        image_path = (
            cls._download_image(image_path_or_url)
            if image_path_or_url.startswith("http")
            else image_path_or_url
        )

        if mode == "tiled":
            detections, pred_path = cls._run_tiled(
                model=model,
                image_path=image_path,
                class_names=class_names,
                tile_size=tile_size,
                overlap=0.4,
                conf_thres=threshold,
            )
        else:
            detections, pred_path = cls._run_normal(
                model=model,
                image_path=image_path,
                class_names=class_names,
                threshold=threshold,
            )

        return detections, pred_path
