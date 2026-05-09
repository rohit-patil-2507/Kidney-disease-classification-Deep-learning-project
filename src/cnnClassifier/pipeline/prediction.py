import os
from pathlib import Path

import numpy as np


class PredictionPipeline:
    """Prediction pipeline."""

    def __init__(self, filename, model_path: str = "artifacts/training/model.h5"):
        self.filename = filename
        self.model_path = Path(model_path)

    def predict(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Train the model first with `python main.py`."
            )

        # TensorFlow import is done lazily.
        # On some Windows setups TF tries to load optional plugins (e.g. tfdml_plugin.dll).
        # If that fails, we raise a clear error so the Flask endpoint returns a useful 500.
        try:
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.image import load_img, img_to_array
        except Exception as e:
            raise RuntimeError(
                "TensorFlow import failed. This is commonly caused by missing Windows TF plugin DLLs "
                "(e.g., tensorflow-plugins/tfdml_plugin.dll).\n"
                f"Original error: {e}"
            )

        model = load_model(str(self.model_path), compile=False)

        test_image = load_img(self.filename, target_size=(224, 224))
        test_image = img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        result = np.argmax(model.predict(test_image), axis=1)

        prediction = "Tumor" if result[0] == 1 else "Normal"
        return [{"image": prediction}]

