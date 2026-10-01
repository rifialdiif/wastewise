"""Convert the Keras model to TFLite for the lightweight LiteRT runtime used in deployment.

Float32, no quantization: the weights are unchanged, so predictions should match
the Keras model. Requires TensorFlow (requirements-dev.txt), not needed at runtime.

Usage:
    python scripts/convert_to_tflite.py
"""

from pathlib import Path

import keras
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "models" / "mobilenetv2_waste_classifier.keras"
TARGET = ROOT / "models" / "mobilenetv2_waste_classifier.tflite"


def main() -> None:
    model = keras.models.load_model(SOURCE, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    TARGET.write_bytes(converter.convert())
    print(f"Wrote {TARGET.relative_to(ROOT)} ({TARGET.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
