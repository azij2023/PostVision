"""
Postal Envelope OCR Web Application.
Upload an envelope image → Extract structured address fields using our custom scratch CRNN.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import uuid
import base64
import io
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory, send_file
from PIL import Image
import numpy as np

# Ensure parent directory is in python path for model imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from inference import PostalAddressOCRSystem
from postprocessing.postal_geocoder import PostalIntelligenceGeocoder

app = Flask(__name__, static_folder="static", static_url_path="")

# Create uploads directory
UPLOAD_DIR = os.path.join(current_dir, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize OCR system once at startup
print("[WebApp] Initializing Postal Address OCR System...")
checkpoint_path = os.path.join(parent_dir, "checkpoints", "best_postal_crnn.pt")
ocr_system = PostalAddressOCRSystem(
    checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None
)
model_info = ocr_system.model.get_architecture_summary()
print(f"[WebApp] OCR System ready. Model: {model_info['model_name']}, "
      f"Params: {model_info['total_parameters']:,}, Device: {ocr_system.device}")


def image_to_base64(img):
    """Convert a PIL Image or numpy array to a base64-encoded PNG string."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


@app.route("/")
def index():
    """Serve the main HTML page."""
    return send_from_directory("static", "index.html")


@app.route("/style.css")
def serve_css():
    """Explicit fallback for stylesheet."""
    return send_from_directory("static", "style.css", mimetype="text/css")


@app.route("/script.js")
def serve_js():
    """Explicit fallback for client script."""
    return send_from_directory("static", "script.js", mimetype="application/javascript")


@app.route("/api/sample-image/<int:sample_id>")
def get_sample_image(sample_id):
    """Serve the raw PNG image of a sample envelope for UI preview."""
    sample_path = os.path.join(parent_dir, "sample_data", f"sample_envelope_{sample_id}.png")
    if not os.path.exists(sample_path):
        return jsonify({"error": f"Sample envelope #{sample_id} not found."}), 404
    return send_file(sample_path, mimetype="image/png")


def format_ocr_response(result, envelope_img, file_id, filename):
    """Formats full OCR result including dual-zone parsing and postage stamp diagnostics."""
    annotated_b64 = image_to_base64(result["annotated_visualization"])
    original_b64 = image_to_base64(envelope_img)

    vision_stages_b64 = {}
    for stage_name, stage_img in result.get("vision_stages", {}).items():
        vision_stages_b64[stage_name] = image_to_base64(stage_img)

    # Format postage stamp telemetry and base64 crop
    stamp_raw = result.get("postage_stamp", {})
    stamp_crop_b64 = None
    if stamp_raw.get("crop_rgb") is not None:
        try:
            stamp_crop_b64 = image_to_base64(stamp_raw["crop_rgb"])
        except Exception:
            stamp_crop_b64 = None

    postage_stamp_data = {
        "stamp_detected": stamp_raw.get("stamp_detected", False),
        "confidence": stamp_raw.get("confidence", 0.0),
        "cancellation_detected": stamp_raw.get("cancellation_detected", False),
        "cancellation_type": stamp_raw.get("cancellation_type", "NONE"),
        "status": stamp_raw.get("status", "NO_STAMP_DETECTED"),
        "global_bbox": stamp_raw.get("global_bbox"),
        "stamp_crop_image": stamp_crop_b64
    }

    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "file_id": file_id,
        "original_filename": filename,
        "original_image": original_b64,
        "annotated_image": annotated_b64,
        "num_lines_detected": result["num_lines_detected"],
        "line_details": result["line_details"],
        "structured_address": result["structured_address"],
        "dual_zone": result.get("dual_zone", {
            "recipient": result["structured_address"],
            "sender": {"detected": False, "raw_text": "Not detected on front face"}
        }),
        "postage_stamp": postage_stamp_data,
        "vision_diagnostics": result.get("vision_diagnostics", {}),
        "vision_stages": vision_stages_b64,
        "model_info": {
            "name": result["model_info"]["model_name"],
            "parameters": result["model_info"]["total_parameters"],
            "design": "100% Custom Scratch (No ResNet/DenseNet/MobileNet)"
        }
    }


@app.route("/api/upload", methods=["POST"])
def upload_and_process():
    """
    Upload an envelope image (via multipart form OR live webcam base64 JSON)
    and process it through the OCR and vision intelligence pipeline.
    """
    try:
        # Check if live webcam base64 was sent
        if request.is_json:
            json_data = request.get_json() or {}
            raw_b64 = json_data.get("image_base64", "")
            if not raw_b64:
                return jsonify({"error": "No image data received from webcam."}), 400

            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(raw_b64)
            envelope_img = Image.open(io.BytesIO(img_bytes))
            file_id = f"cam_{str(uuid.uuid4())[:8]}"
            filename = f"live_webcam_scan_{file_id}.png"

        elif "envelope" in request.files:
            file = request.files["envelope"]
            if file.filename == "":
                return jsonify({"error": "No file selected."}), 400

            allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in allowed_extensions:
                return jsonify({"error": f"Unsupported file type '{ext}'. Supported: {', '.join(allowed_extensions)}"}), 400

            img_bytes = file.read()
            envelope_img = Image.open(io.BytesIO(img_bytes))
            file_id = str(uuid.uuid4())[:8]
            filename = file.filename

            # Save upload for audit trail
            save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
            with open(save_path, "wb") as f:
                f.write(img_bytes)
        else:
            return jsonify({"error": "No file or webcam data provided. Please provide an envelope image."}), 400

        # Run OCR pipeline
        result = ocr_system.process_envelope(envelope_img)
        response_data = format_ocr_response(result, envelope_img, file_id, filename)
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/api/sample/<int:sample_id>", methods=["GET"])
def process_sample(sample_id):
    """Process one of the pre-existing sample envelopes."""
    sample_path = os.path.join(parent_dir, "sample_data", f"sample_envelope_{sample_id}.png")
    if not os.path.exists(sample_path):
        return jsonify({"error": f"Sample envelope #{sample_id} not found."}), 404

    try:
        envelope_img = Image.open(sample_path)
        result = ocr_system.process_envelope(envelope_img)
        response_data = format_ocr_response(
            result, envelope_img, f"sample_{sample_id}", f"sample_envelope_{sample_id}.png"
        )
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    """Return model architecture information."""
    return jsonify({
        "model_name": model_info["model_name"],
        "total_parameters": model_info["total_parameters"],
        "stages": model_info["stages"],
        "weights_loaded": ocr_system.model_loaded,
        "design_constraint": "100% Custom Scratch Deep Neural Network"
    })


@app.route("/api/geocode", methods=["POST"])
def geocode_postal_code():

    """Live PIN Geocoding & Validation endpoint for interactive UI edits."""
    data = request.get_json() or {}
    pin = data.get("postal_code", "")
    state = data.get("state", "")
    city = data.get("city", "")
    locality = data.get("locality", "")
    enrichment = PostalIntelligenceGeocoder.validate_and_enrich(
        postal_code=pin,
        detected_state=state,
        detected_city=city,
        detected_locality=locality
    )
    return jsonify({
        "success": True,
        "postal_validation": enrichment
    })


@app.route("/api/export-csv", methods=["POST"])
def export_postal_csv():
    """
    Generate an RFC-4180 compliant CSV manifest from one or multiple
    postal address records, formatted for postal logistics software and Excel.
    """
    import csv
    data = request.get_json() or {}
    records = data.get("records", [])
    if not isinstance(records, list):
        records = [records]

    output = io.StringIO()
    # Add UTF-8 BOM so Excel opens with proper encoding on Windows
    output.write("\ufeff")

    fieldnames = [
        "Tracking_ID",
        "Timestamp",
        "Original_Filename",
        "Recipient_Name",
        "Premise_Building",
        "Street_Locality",
        "City_District",
        "State",
        "PIN_Code",
        "Assigned_Post_Office",
        "Delivery_Type",
        "RMS_Sorting_Hub",
        "Postal_Circle",
        "Validation_Status",
        "Validation_Score",
        "Model_Confidence_Pct",
        "Lines_Detected",
        "Sender_Address",
        "Stamp_Detected",
        "Cancellation_Type",
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()

    for item in records:
        addr = item.get("structured_address", {})
        val = addr.get("postal_validation", {})
        dual = item.get("dual_zone", {})
        sender = dual.get("sender", {}).get("raw_text", "N/A")
        stamp = item.get("postage_stamp", {})
        conf = addr.get("overall_confidence", 0.0)

        writer.writerow({
            "Tracking_ID": item.get("file_id", "N/A"),
            "Timestamp": item.get("timestamp", datetime.now().isoformat()),
            "Original_Filename": item.get("original_filename", "N/A"),
            "Recipient_Name": addr.get("recipient_name", ""),
            "Premise_Building": addr.get("premise_building", ""),
            "Street_Locality": addr.get("street_locality", ""),
            "City_District": addr.get("city", ""),
            "State": addr.get("state", ""),
            "PIN_Code": addr.get("postal_code", ""),
            "Assigned_Post_Office": val.get("post_office", "Unverified"),
            "Delivery_Type": val.get("delivery_type", "Standard Mail"),
            "RMS_Sorting_Hub": val.get("routing_hub", "Regional RMS Hub"),
            "Postal_Circle": val.get("circle", "General Circle"),
            "Validation_Status": val.get("status", "UNVERIFIED"),
            "Validation_Score": f"{val.get('verification_score', 100)}%",
            "Model_Confidence_Pct": f"{round(conf * 100, 1)}%",
            "Lines_Detected": item.get("num_lines_detected", 0),
            "Sender_Address": sender,
            "Stamp_Detected": "YES" if stamp.get("stamp_detected") else "NO",
            "Cancellation_Type": stamp.get("cancellation_type", "NONE"),
        })

    csv_data = output.getvalue()
    filename = f"postal_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    response = app.response_class(
        response=csv_data,
        status=200,
        mimetype="text/csv"
    )
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

