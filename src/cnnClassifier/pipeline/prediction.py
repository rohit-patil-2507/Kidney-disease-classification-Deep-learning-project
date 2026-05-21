import os
from pathlib import Path

import numpy as np
from PIL import Image


import base64
import io


class PredictionPipeline:
    """Prediction pipeline."""


    class_names = ("Normal", "Tumor")

    def __init__(self, filename, model_path: str = "artifacts/training/model.h5"):
        self.filename = filename
        self.model_path = self._resolve_model_path(model_path)
        self._model = None

    @staticmethod
    def _resolve_model_path(model_path: str) -> Path:
        preferred = Path(model_path)
        if preferred.exists():
            return preferred

        fallback = Path("model/model.h5")
        if fallback.exists():
            return fallback

        try:
            import gdown
            import os
            os.makedirs("model", exist_ok=True)
            print(f"Model not found locally. Downloading from Google Drive to {fallback}...")
            file_id = "10AzyxdAYIkA0MT5ITLKYEse0xrINX_r3"
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix + file_id, str(fallback), quiet=False)
            
            if fallback.exists():
                return fallback
        except ImportError:
            print("gdown is not installed. Skipping automatic model download.")

        return preferred

    def _load_tensorflow(self):
        # TensorFlow import is done lazily.
        # On some Windows setups TF tries to load optional plugins (e.g. tfdml_plugin.dll).
        # If that fails, we raise a clear error so apps return a useful message.
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
        except Exception as e:
            raise RuntimeError(
                "TensorFlow import failed. This is commonly caused by missing Windows TF plugin DLLs "
                "(e.g., tensorflow-plugins/tfdml_plugin.dll).\n"
                f"Original error: {e}"
            )

        return tf, load_model

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path} and automatic download failed."
            )

        if self._model is None:
            _, load_model = self._load_tensorflow()
            self._model = load_model(str(self.model_path), compile=False)

        return self._model

    def validate_image_quality(self, filename=None):
        """Validates if the image is a suitable candidate for prediction (e.g., not blurry/corrupted)."""
        image_path = filename or self.filename
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            raise ValueError("Instructions: Invalid or corrupted image format. Please upload a supported image type (e.g., JPEG, PNG).")

        try:
            with Image.open(image_path) as img:
                # Check if the image is a colorful photo instead of a grayscale scan
                img_rgb = img.convert('RGB')
                img_array_rgb = np.asarray(img_rgb, dtype=np.float32)
                r, g, b = img_array_rgb[:,:,0], img_array_rgb[:,:,1], img_array_rgb[:,:,2]
                color_diff = np.mean(np.abs(r - g) + np.abs(r - b) + np.abs(g - b))
                
                if color_diff > 10.0:
                    raise ValueError("Instructions: Uploaded image appears to be a colorful photo. Please upload a grayscale Kidney CT scan.")

                img_gray = img.convert('L')
                img_array = np.asarray(img_gray, dtype=np.float32)

                # Check if image is blank/solid color (low variance)
                std_dev = np.std(img_array)
                if std_dev < 10:
                    raise ValueError("Instructions: The uploaded image appears to be blank or a solid color. Please upload a valid, clear Kidney CT/MRI scan.")

                # Check for extreme blurriness using Laplacian variance
                try:
                    import cv2
                    img_cv = np.asarray(img_gray, dtype=np.uint8)
                    blur_metric = cv2.Laplacian(img_cv, cv2.CV_64F).var()
                except ImportError:
                    import scipy.ndimage
                    blur_metric = scipy.ndimage.laplace(img_array).var()

                if blur_metric < 50:
                    raise ValueError("Uploaded image is too blurry. Please ensure the kidney scan is in focus and clear.")
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise ValueError(f"Instructions: Failed to process image for validation. Ensure it is a valid Kidney scan. Error: {str(e)}")

    def verify_ct_scan(self, filename=None):
        """Verifies if the uploaded image is a valid CT scan using a secondary model."""
        image_path = filename or self.filename
        
        verification_model_path = Path("model/ct_verification_model.h5")
        if not verification_model_path.exists():
            print(f"Warning: CT verification model not found at {verification_model_path}. Skipping verification.")
            return

        tf, load_model = self._load_tensorflow()
        try:
            verification_model = load_model(str(verification_model_path), compile=False)
        except Exception as e:
            print(f"Failed to load CT verification model: {e}")
            return

        processed = self.preprocess_image(filename)
        probabilities = verification_model.predict(processed["batched_image"], verbose=0)[0]
        
        # Assuming index 0 is non-CT and index 1 is CT, or single sigmoid output > 0.5 is CT
        confidence = float(probabilities[0] if len(probabilities) == 1 else probabilities[1])
        
        if confidence < 0.5:
            raise ValueError("Uploaded image is not a valid kidney CT scan.")

    def preprocess_image(self, filename=None):
        image_path = filename or self.filename
        resized_image = Image.open(image_path).convert("RGB").resize((224, 224))
        image_array = np.asarray(resized_image, dtype=np.float32)
        normalized = image_array / 255.0
        batched = np.expand_dims(normalized, axis=0)

        return {
            "resized_image": resized_image,
            "image_array": image_array,
            "normalized_image": normalized,
            "batched_image": batched,
        }

    def predict(self):
        # Frontend expects top-level: prediction/confidence/probabilities[/heatmap/preprocess]
        return self.predict_detailed()

    def predict_detailed(self, filename=None):
        model = self.load_model()
        processed = self.preprocess_image(filename)
        probabilities = model.predict(processed["batched_image"], verbose=0)[0]
        probabilities = np.asarray(probabilities, dtype=float)

        if len(probabilities) == 1:
            prob_tumor = probabilities[0]
            prob_normal = 1.0 - prob_tumor
            predicted_index = 1 if prob_tumor > 0.5 else 0
            confidence = prob_tumor if predicted_index == 1 else prob_normal
            probs_dict = {
                "Normal": float(prob_normal),
                "Tumor": float(prob_tumor)
            }
        else:
            predicted_index = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_index])
            probs_dict = {
                class_name: float(probabilities[index])
                for index, class_name in enumerate(self.class_names)
            }

        prediction = self.class_names[predicted_index]

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probs_dict,
            "predicted_index": predicted_index,
        }

    def get_last_conv_layer_name(self):
        tf, _ = self._load_tensorflow()
        model = self.load_model()

        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

        raise ValueError("Could not find a Conv2D layer for Grad-CAM.")

    @staticmethod
    def _to_data_url_png(pil_img: Image.Image) -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def make_gradcam_heatmap(self, filename=None, class_index=None, layer_name=None):

        tf, _ = self._load_tensorflow()
        model = self.load_model()
        processed = self.preprocess_image(filename)
        image_batch = processed["batched_image"]
        target_layer_name = layer_name or self.get_last_conv_layer_name()

        grad_model = tf.keras.models.Model(
            model.inputs,
            [model.get_layer(target_layer_name).output, model.output],
        )

        with tf.GradientTape() as tape:
            outputs = grad_model(image_batch)
            if isinstance(outputs, dict):
                outputs = list(outputs.values())
                
            conv_outputs, predictions = outputs[0], outputs[1]
            
            if isinstance(predictions, list):
                predictions = predictions[0]
            if isinstance(conv_outputs, list):
                conv_outputs = conv_outputs[0]
                
            predictions = tf.convert_to_tensor(predictions)
            conv_outputs = tf.convert_to_tensor(conv_outputs)

            selected_index = class_index
            if selected_index is None:
                if predictions.shape[-1] == 1:
                    selected_index = 0
                else:
                    selected_index = tf.argmax(predictions[0])
            class_channel = predictions[:, selected_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

        return heatmap.numpy()

    def make_gradcam_overlay_base64(
        self,
        filename=None,
        class_index=None,
        layer_name=None,
        alpha: float = 0.38,
        cmap: str = "jet",
    ) -> str:
        """Return a base64 PNG data-URL of Grad-CAM overlay for the given image."""
        tf, _ = self._load_tensorflow()
        _ = tf  # keep TF lazy/import consistent

        # Get raw heatmap (0..1)
        heatmap = self.make_gradcam_heatmap(filename=filename, class_index=class_index, layer_name=layer_name)

        # Load original image and match model input size
        image_path = filename or self.filename
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        img_arr = np.asarray(img).astype(np.float32) / 255.0

        # Colorize heatmap
        heatmap_uint8 = np.uint8(255 * heatmap)
        
        # Resize heatmap to match image size (224, 224)
        if heatmap_uint8.shape[:2] != (224, 224):
            if hasattr(Image, "Resampling"):
                resample_mode = Image.Resampling.BILINEAR
            else:
                resample_mode = Image.BILINEAR
            heatmap_img = Image.fromarray(heatmap_uint8).resize((224, 224), resample=resample_mode)
            heatmap_uint8 = np.array(heatmap_img)

        cmap_fn = None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            cmap_fn = plt.get_cmap(cmap)
        except Exception as e:
            print(f"Matplotlib import error: {e}")
            cmap_fn = None

        if cmap_fn is None:
            # Fallback: grayscale -> red channel only
            colored = np.zeros((heatmap_uint8.shape[0], heatmap_uint8.shape[1], 3), dtype=np.float32)
            colored[..., 0] = heatmap_uint8.astype(np.float32) / 255.0
        else:
            rgba = cmap_fn(heatmap_uint8)
            colored = rgba[..., :3].astype(np.float32)

        overlay = (1 - alpha) * img_arr + alpha * colored
        overlay = np.clip(overlay, 0, 1)
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

        return self._to_data_url_png(overlay_img)

    def make_preprocess_previews_base64(self, filename=None) -> dict:
        """Return model input previews as base64 PNG data-URLs."""
        processed = self.preprocess_image(filename)
        resized = processed["resized_image"]

        # normalized preview: show as image where [0..1] -> [0..255]
        normalized = processed["normalized_image"]
        normalized_img = (np.clip(normalized, 0, 1) * 255).astype(np.uint8)
        normalized_pil = Image.fromarray(normalized_img)

        return {
            "resized": self._to_data_url_png(resized),
            "normalized_preview": self._to_data_url_png(normalized_pil),
        }
