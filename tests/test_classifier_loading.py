"""Phase 2 checks: the real model loads and matches the class mapping."""

import json

import pytest

from app.config import settings
from app.services.classifier import ModelLoadError, WasteClassifier, load_class_names

MODEL_PATH = settings.resolve(settings.model_path)
CLASS_MAPPING_PATH = settings.resolve(settings.class_mapping_path)


@pytest.fixture(scope="module")
def classifier():
    return WasteClassifier(MODEL_PATH, CLASS_MAPPING_PATH)


def test_model_loads_with_expected_shapes(classifier):
    assert classifier.input_shape == (1, 224, 224, 3)
    assert classifier.num_outputs == 6


def test_class_names_follow_output_index_order(classifier):
    assert classifier.class_names == ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def test_missing_model_file_raises_clear_error(tmp_path):
    with pytest.raises(ModelLoadError, match="Model file not found"):
        WasteClassifier(tmp_path / "missing.tflite", CLASS_MAPPING_PATH)


def test_corrupt_model_file_raises_clear_error(tmp_path):
    bad_model = tmp_path / "bad.tflite"
    bad_model.write_bytes(b"not a tflite file")
    with pytest.raises(ModelLoadError, match="Failed to load model"):
        WasteClassifier(bad_model, CLASS_MAPPING_PATH)


def test_invalid_class_mapping_raises_clear_error(tmp_path):
    bad_mapping = tmp_path / "mapping.json"
    bad_mapping.write_text(json.dumps({"1": "glass", "2": "metal"}), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="Class mapping keys"):
        load_class_names(bad_mapping)
