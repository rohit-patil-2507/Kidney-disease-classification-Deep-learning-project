from flask import Flask, request, jsonify, render_template
import os
import subprocess
import sys
import uuid
from flask_cors import CORS, cross_origin

from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline


def _warn_if_tf_plugins_missing():
    # Helps explain startup failures to the user (common on some Windows setups).
    # Some TF builds on Windows attempt to load optional DML plugins.
    # If the DLL is missing, importing TF can raise NotFoundError.
    try:
        import tensorflow  # noqa: F401
    except Exception as e:
        msg = str(e)
        if "tensorflow-plugins" in msg and "tfdml_plugin.dll" in msg:
            print(
                "[WARN] TensorFlow DML plugin DLL is missing: tfdml_plugin.dll.\n"
                "       This is usually safe to ignore unless you require DML acceleration.\n"
                "       Suggested fix: reinstall/upgrade tensorflow-plugins or reinstall TensorFlow.\n"
            )
        else:
            # Re-raise unexpected errors
            raise



os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)

# Cache model pipeline (keeps UI responsive). The pipeline loads the model inside predict(),
# so we only cache the instance/config.
predictor = PredictionPipeline(filename="inputImage.jpg")

# Keep CORS enabled for local dev / AWS deployments.





@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/status", methods=["GET"])
@cross_origin()
def statusRoute():
    """Simple health/status endpoint for the frontend."""
    # Model path is defined in PredictionPipeline.
    # We surface a hint if model artifacts are missing.
    try:
        pipeline = PredictionPipeline(filename="inputImage.jpg")
        model_exists = str(pipeline.model_path.exists())
    except Exception:
        # If pipeline construction fails for any reason, still return status.
        model_exists = "unknown"

    return jsonify({
        "status": "ok",
        "model": {"ready": model_exists},
        "routes": {"predict": "/predict", "train": "/train"}
    })





@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    subprocess.run([sys.executable, "main.py"], check=True)
    # os.system("dvc repro")
    return "Training done successfully!"



@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image")

    if not image_b64 or not isinstance(image_b64, str):
        return jsonify({"error": "Missing or invalid image payload"}), 400

    # Save a unique decoded file to avoid race conditions across requests.
    filename = f"inputImage_{uuid.uuid4().hex}.jpg"

    try:
        decodeImage(image_b64, filename)
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {e}"}), 400

    try:
        # Run prediction. PredictionPipeline uses self.filename.
        predictor.filename = filename
        result = predictor.predict()

        # Normalize response shape.
        # Current PredictionPipeline returns: [{"image": "Tumor"}] or [{"image": "Normal"}]
        prediction = None
        if isinstance(result, list) and result:
            prediction = result[0].get("image")
        elif isinstance(result, dict):
            prediction = result.get("image") or result.get("prediction")

        if not prediction:
            return jsonify({"error": "Model returned an unexpected result."}), 500

        return jsonify({"prediction": prediction})

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Temporary upload cleanup to avoid filling disk over time.
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except Exception:
            # Don't fail the request if cleanup fails.
            pass





if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080) #for AWS

