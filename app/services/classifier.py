import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

from app.schemas.prediction import PredictionStatus

IMAGE_SIZE = (224, 224)


class ModelLoadError(RuntimeError):
    """Raised when the model or class mapping cannot be loaded or is inconsistent."""


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a decodable image."""


@dataclass(frozen=True)
class ClassificationResult:
    class_name: str
    confidence: float


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


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes into a (1, 224, 224, 3) float32 batch of raw [0, 255] RGB pixels.

    Resizing uses tf.image.resize (bilinear), the same as Keras'
    image_dataset_from_directory used during training. Pixel scaling is NOT
    applied here because the model's own Rescaling layer does it.
    """
    if not image_bytes:
        raise InvalidImageError("Uploaded file is empty")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            pixels = np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        raise InvalidImageError(
            "File is not a valid image. Please upload a photo such as JPEG or PNG."
        ) from exc

    resized = tf.image.resize(pixels, IMAGE_SIZE, method="bilinear")
    return np.expand_dims(resized.numpy().astype(np.float32), axis=0)


def apply_confidence_gate(confidence: float, threshold: float) -> PredictionStatus:
    """Accept a prediction only when its confidence reaches the threshold."""
    return PredictionStatus.ACCEPTED if confidence >= threshold else PredictionStatus.UNCERTAIN


class WasteClassifier:
    """Loads the MobileNetV2 waste classifier and classifies uploaded images.

    The model contains its own Rescaling layer (pixels / 127.5 - 1), so inputs
    must be raw RGB pixel values in [0, 255].
    """

    def __init__(self, model_path: Path, class_mapping_path: Path):
        self.class_names = load_class_names(class_mapping_path)
        self.model = self._load_model(model_path)
        self._validate()

    @staticmethod
    def _load_model(model_path: Path) -> keras.Model:
        if not model_path.is_file():
            raise ModelLoadError(f"Model file not found: {model_path}")
        try:
            return keras.models.load_model(model_path, compile=False)
        except Exception as exc:
            raise ModelLoadError(f"Failed to load model from {model_path}: {exc}") from exc

    def _validate(self) -> None:
        input_shape = tuple(self.model.input_shape)
        if input_shape != (None, *IMAGE_SIZE, 3):
            raise ModelLoadError(f"Unexpected model input shape {input_shape}")

        num_outputs = self.model.output_shape[-1]
        if num_outputs != len(self.class_names):
            raise ModelLoadError(
                f"Model has {num_outputs} outputs but class mapping has {len(self.class_names)} classes"
            )

        probabilities = self.model.predict(np.zeros((1, *IMAGE_SIZE, 3), dtype=np.float32), verbose=0)
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-3):
            raise ModelLoadError("Model output is not a probability distribution (softmax)")

    def predict(self, image_bytes: bytes) -> ClassificationResult:
        """Classify one image and return the top class with its confidence."""
        probabilities = self.model.predict(preprocess_image(image_bytes), verbose=0)[0]
        index = int(np.argmax(probabilities))
        # Round down so the reported confidence never overstates the model's certainty.
        confidence = math.floor(float(probabilities[index]) * 10_000) / 10_000
        return ClassificationResult(class_name=self.class_names[index], confidence=confidence)
