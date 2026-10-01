"""Phase 9 checks: upload limits, image validation, unexpected errors and config validation."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from app.config import Settings, settings
from app.main import app
from app.services.classifier import InvalidImageError, preprocess_image

FIXTURES = Path(__file__).parent / "fixtures"


def encode(image: Image.Image, fmt: str) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def client():
    # raise_server_exceptions=False lets us see the 500 response instead of the raw exception.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def upload(client, data: bytes, name="photo.jpg", content_type="image/jpeg"):
    return client.post("/predict", files={"file": (name, data, content_type)})


# --- Image validation ------------------------------------------------------


@pytest.mark.parametrize("fmt", ["GIF", "TIFF", "ICO"])
def test_unsupported_formats_are_rejected(fmt):
    with pytest.raises(InvalidImageError, match="not a supported image"):
        preprocess_image(encode(Image.new("RGB", (64, 64)), fmt))


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "BMP"])
def test_supported_formats_are_accepted(fmt):
    assert preprocess_image(encode(Image.new("RGB", (64, 64)), fmt)).shape == (1, 224, 224, 3)


def test_oversized_dimensions_are_rejected_before_decoding():
    # 1-bit 8000x6000 PNG: tiny file, 48M pixels.
    with pytest.raises(InvalidImageError, match="too large"):
        preprocess_image(encode(Image.new("1", (8000, 6000)), "PNG"))


def test_truncated_jpeg_is_reported_as_corrupted():
    data = (FIXTURES / "paper.jpg").read_bytes()
    with pytest.raises(InvalidImageError, match="corrupted"):
        preprocess_image(data[: len(data) // 2])


# --- API error responses ---------------------------------------------------


def test_upload_over_limit_returns_413(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 0.005)  # ~5 KB; fixture is ~16 KB
    response = upload(client, (FIXTURES / "cardboard.jpg").read_bytes())
    assert response.status_code == 413
    assert "too large" in response.json()["detail"]


def test_declared_content_length_over_limit_returns_413(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 0.005)
    response = upload(client, b"\0" * 200_000)
    assert response.status_code == 413


def test_unsupported_format_returns_400(client):
    response = upload(client, encode(Image.new("RGB", (64, 64)), "GIF"), "anim.gif", "image/gif")
    assert response.status_code == 400
    assert "JPEG, PNG, WebP or BMP" in response.json()["detail"]


def test_empty_upload_returns_400(client):
    response = upload(client, b"")
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."


def test_unexpected_error_returns_clean_500(client, monkeypatch):
    def explode(_):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(app.state.classifier, "predict", explode)
    response = upload(client, (FIXTURES / "glass.jpg").read_bytes())

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error. Please try again later."}
    assert "secret" not in response.text


# --- Configuration ---------------------------------------------------------


@pytest.mark.parametrize("field, value", [("confidence_threshold", 1.5), ("confidence_threshold", -0.1), ("max_upload_mb", 0)])
def test_invalid_settings_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(**{field: value})
