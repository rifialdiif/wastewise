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
    resize_bilinear,
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


def test_concurrent_predictions_match_sequential(classifier):
    from concurrent.futures import ThreadPoolExecutor

    images = [(FIXTURES / f"{name}.jpg").read_bytes() for name in CLASS_NAMES] * 4
    sequential = [classifier.predict(data) for data in images]
    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent = list(pool.map(classifier.predict, images))
    assert concurrent == sequential


# --- Resize (must match training's tf.image.resize) -----------------------


def test_resize_bilinear_half_pixel_centers():
    # 1x2 -> 1x4: hand-computed values of tf.image.resize(method="bilinear").
    row = np.array([[[0.0], [10.0]]])
    assert resize_bilinear(row, (1, 4))[0, :, 0].tolist() == [0.0, 2.5, 7.5, 10.0]


def test_resize_bilinear_matches_tensorflow():
    tf = pytest.importorskip("tensorflow")  # dev-only dependency
    pixels = np.asarray(Image.open(FIXTURES / "paper.jpg").convert("RGB"))
    expected = tf.image.resize(pixels, (224, 224), method="bilinear").numpy()
    np.testing.assert_allclose(resize_bilinear(pixels, (224, 224)), expected, atol=1e-3)


def test_top_k_is_ranked_and_starts_with_prediction(classifier):
    result = classifier.predict((FIXTURES / "metal.jpg").read_bytes())

    assert len(result.top_k) == 3
    assert result.top_k[0] == (result.class_name, result.confidence)
    confidences = [confidence for _, confidence in result.top_k]
    assert confidences == sorted(confidences, reverse=True)
    assert sum(confidences) <= 1.0


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
