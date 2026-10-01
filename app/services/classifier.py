import json
from pathlib import Path

import keras
import numpy as np

IMAGE_SIZE = (224, 224)


class ModelLoadError(RuntimeError):
    """Raised when the model or class mapping cannot be loaded or is inconsistent."""


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


class WasteClassifier:
    """Loads the MobileNetV2 waste classifier and checks it matches the class mapping.

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
