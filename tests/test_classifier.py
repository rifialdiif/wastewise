"""Phase 3 & 4 checks: image preprocessing, classification and the confidence gate."""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config import settings
from app.schemas.prediction import PredictionStatus
from app.services.classifier import (
    InvalidImageError,
    WasteClassifier,
    apply_confidence_gate,
    preprocess_image,
)

FIXTURES = Path(__file__).parent / "fixtures"
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def image_bytes(mode: str, size=(300, 200), fmt="PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def classifier():
    return WasteClassifier(
        settings.resolve(settings.model_path), settings.resolve(settings.class_mapping_path)
    )


# --- Preprocessing ---------------------------------------------------------


def test_preprocess_returns_raw_pixel_batch():
    batch = preprocess_image((FIXTURES / "glass.jpg").read_bytes())
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    # Raw [0, 255] pixels: the model's Rescaling layer normalizes them.
    assert batch.min() >= 0.0 and batch.max() <= 255.0 and batch.max() > 1.0


@pytest.mark.parametrize("mode", ["RGBA", "L", "P", "CMYK"])
def test_preprocess_converts_any_mode_to_rgb(mode):
    fmt = "JPEG" if mode == "CMYK" else "PNG"
    assert preprocess_image(image_bytes(mode, fmt=fmt)).shape == (1, 224, 224, 3)


@pytest.mark.parametrize(
    "payload",
    [b"", b"definitely not an image", b"%PDF-1.7 fake pdf", image_bytes("RGB", fmt="PNG")[:60]],
    ids=["empty", "text", "pdf", "truncated-png"],
)
def test_preprocess_rejects_invalid_images(payload):
    with pytest.raises(InvalidImageError):
        preprocess_image(payload)


# --- Classification --------------------------------------------------------


@pytest.mark.parametrize("class_name", CLASS_NAMES)
def test_known_images_are_classified_correctly(classifier, class_name):
    result = classifier.predict((FIXTURES / f"{class_name}.jpg").read_bytes())
    assert result.class_name == class_name
    assert 0.0 <= result.confidence <= 1.0


def test_confidence_has_at_most_four_decimals(classifier):
    result = classifier.predict((FIXTURES / "plastic.jpg").read_bytes())
    assert result.confidence == round(result.confidence, 4)


# --- Confidence gate -------------------------------------------------------


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.9612, PredictionStatus.ACCEPTED),
        (0.90, PredictionStatus.ACCEPTED),
        (0.8999, PredictionStatus.UNCERTAIN),
        (0.61, PredictionStatus.UNCERTAIN),
    ],
)
def test_confidence_gate(confidence, expected):
    assert apply_confidence_gate(confidence, threshold=0.90) == expected
