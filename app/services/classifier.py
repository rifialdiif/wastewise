import io
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image, ImageOps, UnidentifiedImageError

from app.schemas.prediction import PredictionStatus

IMAGE_SIZE = (224, 224)
# Pillow's JPEG opener also handles MPO, the multi-picture JPEG many phone cameras produce.
ALLOWED_IMAGE_FORMATS = ("JPEG", "PNG", "WEBP", "BMP")
# Checked from the header before decoding, to reject decompression bombs early.
# JPEGs can be decoded at reduced scale; other formats are decoded in full, so their
# limit is lower to fit a 512 MB server.
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_PIXELS_NON_JPEG = 12_000_000


class ModelLoadError(RuntimeError):
    """Raised when the model or class mapping cannot be loaded or is inconsistent."""


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a decodable image."""


TOP_K = 3


@dataclass(frozen=True)
class ClassificationResult:
    class_name: str
    confidence: float
    # Top-K (class, confidence) pairs, highest first; the first one is the prediction.
    top_k: tuple[tuple[str, float], ...]


def floor_confidence(probability: float) -> float:
    """Round down to 4 decimals so a reported confidence never overstates the model's certainty."""
    return math.floor(probability * 10_000) / 10_000


def load_class_names(class_mapping_path: Path) -> list[str]:
    """Return class names ordered by model output index."""
    try:
        mapping = json.loads(class_mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelLoadError(f"Cannot read class mapping at {class_mapping_path}: {exc}") from exc

    expected_keys = [str(i) for i in range(len(mapping))]
    if set(mapping) != set(expected_keys):
        raise ModelLoadError(f"Class mapping keys must be 0..{len(mapping) - 1}, got {list(mapping)}")
    return [mapping[key] for key in expected_keys]


def _interpolation_weights(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source indices and weights for bilinear resizing with half-pixel centers."""
    positions = (np.arange(out_size, dtype=np.float32) + 0.5) * (in_size / out_size) - 0.5
    floor = np.floor(positions)
    lower = np.maximum(floor, 0).astype(np.int64)
    upper = np.minimum(np.ceil(positions), in_size - 1).astype(np.int64)
    return lower, upper, (positions - floor).astype(np.float32)


def resize_bilinear(pixels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an (H, W, C) image to float32, matching tf.image.resize(method="bilinear").

    That is the resize Keras' image_dataset_from_directory used during training
    (half-pixel centers, no antialiasing). Reimplemented in NumPy so the API does
    not need TensorFlow at runtime.
    """
    top_idx, bottom_idx, y_lerp = _interpolation_weights(pixels.shape[0], size[0])
    left_idx, right_idx, x_lerp = _interpolation_weights(pixels.shape[1], size[1])

    # Gather the four neighbours first, then convert: only output-sized arrays become
    # float32, so a large phone photo never gets a full-resolution float copy.
    rows_top, rows_bottom = pixels[top_idx], pixels[bottom_idx]
    top_left = rows_top[:, left_idx].astype(np.float32)
    top_right = rows_top[:, right_idx].astype(np.float32)
    bottom_left = rows_bottom[:, left_idx].astype(np.float32)
    bottom_right = rows_bottom[:, right_idx].astype(np.float32)

    x_lerp = x_lerp[None, :, None]
    top = top_left + (top_right - top_left) * x_lerp
    bottom = bottom_left + (bottom_right - bottom_left) * x_lerp
    return top + (bottom - top) * y_lerp[:, None, None]


MIN_DECODED_SIDE = 1024


def _jpeg_reduction(width: int, height: int) -> int:
    """Largest JPEG decode reduction (1, 2, 4 or 8) that keeps the shorter side >= MIN_DECODED_SIDE."""
    reduction = 1
    while reduction < 8 and min(width, height) // (reduction * 2) >= MIN_DECODED_SIDE:
        reduction *= 2
    return reduction


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes into a (1, 224, 224, 3) float32 batch of raw [0, 255] RGB pixels.

    Resizing matches the bilinear resize used during training. Pixel scaling is
    NOT applied here because the model's own Rescaling layer does it.

    Very large JPEGs (shorter side over 2048 px, e.g. 12 MP camera photos) are
    decoded at a reduced scale to keep memory low on small servers. Their shorter
    side stays at least 1024 px, so the image is still much larger than the model
    input. Smaller images, including the training data and messaging-app photos,
    are decoded at full resolution.
    """
    if not image_bytes:
        raise InvalidImageError("Uploaded file is empty.")
    try:
        image = Image.open(io.BytesIO(image_bytes), formats=ALLOWED_IMAGE_FORMATS)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise InvalidImageError(
            "File is not a supported image. Please upload a JPEG, PNG, WebP or BMP photo."
        ) from exc

    with image:
        width, height = image.size
        max_pixels = MAX_IMAGE_PIXELS if image.format == "JPEG" else MAX_IMAGE_PIXELS_NON_JPEG
        if width * height > max_pixels:
            raise InvalidImageError(
                f"Image is too large ({width}x{height}). Maximum for {image.format} is {max_pixels:,} pixels."
            )
        try:
            reduction = _jpeg_reduction(width, height)
            if reduction > 1:
                # JPEG-only, no-op for other formats: the decoder downscales by 1/2, 1/4 or 1/8.
                image.draft("RGB", (width // reduction, height // reduction))
            image = ImageOps.exif_transpose(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            pixels = np.asarray(image)
        except (Image.DecompressionBombError, OSError, ValueError) as exc:
            raise InvalidImageError("Image file is corrupted or incomplete.") from exc

    return np.expand_dims(resize_bilinear(pixels, IMAGE_SIZE), axis=0)


def apply_confidence_gate(confidence: float, threshold: float) -> PredictionStatus:
    """Accept a prediction only when its confidence reaches the threshold."""
    return PredictionStatus.ACCEPTED if confidence >= threshold else PredictionStatus.UNCERTAIN


class WasteClassifier:
    """Runs the MobileNetV2 waste classifier (TFLite) with the LiteRT runtime.

    The model contains its own Rescaling layer (pixels / 127.5 - 1), so inputs
    must be raw RGB pixel values in [0, 255].
    """

    def __init__(self, model_path: Path, class_mapping_path: Path):
        self.class_names = load_class_names(class_mapping_path)
        self.interpreter = self._load_model(model_path)
        self._input = self.interpreter.get_input_details()[0]
        self._output = self.interpreter.get_output_details()[0]
        # An interpreter is not thread-safe; FastAPI runs sync endpoints in a thread pool.
        self._lock = threading.Lock()
        self._validate()

    @staticmethod
    def _load_model(model_path: Path) -> Interpreter:
        if not model_path.is_file():
            raise ModelLoadError(f"Model file not found: {model_path}")
        try:
            interpreter = Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
        except Exception as exc:
            raise ModelLoadError(f"Failed to load model from {model_path}: {exc}") from exc
        return interpreter

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(int(d) for d in self._input["shape"])

    @property
    def num_outputs(self) -> int:
        return int(self._output["shape"][-1])

    def _validate(self) -> None:
        if self.input_shape != (1, *IMAGE_SIZE, 3):
            raise ModelLoadError(f"Unexpected model input shape {self.input_shape}")

        if self.num_outputs != len(self.class_names):
            raise ModelLoadError(
                f"Model has {self.num_outputs} outputs but class mapping has {len(self.class_names)} classes"
            )

        probabilities = self._run(np.zeros((1, *IMAGE_SIZE, 3), dtype=np.float32))
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-3):
            raise ModelLoadError("Model output is not a probability distribution (softmax)")

    def _run(self, batch: np.ndarray) -> np.ndarray:
        """Run one (1, 224, 224, 3) batch and return its class probabilities."""
        with self._lock:
            self.interpreter.set_tensor(self._input["index"], batch)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self._output["index"])[0].copy()

    def predict(self, image_bytes: bytes) -> ClassificationResult:
        """Classify one image and return the top class, its confidence and the top-K candidates."""
        probabilities = self._run(preprocess_image(image_bytes))
        ranked = np.argsort(probabilities)[::-1][:TOP_K]
        top_k = tuple(
            (self.class_names[i], floor_confidence(float(probabilities[i]))) for i in ranked
        )
        class_name, confidence = top_k[0]
        return ClassificationResult(class_name=class_name, confidence=confidence, top_k=top_k)
