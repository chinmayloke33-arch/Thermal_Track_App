import os
import tempfile
import numpy as np


def analyze_thermal_image_radiometric(
    file_bytes: bytes,
    threshold: float = 5.0
) -> dict:
    """
    Extracts per-pixel radiometric temperature matrix from FLIR JPG metadata.
    Falls back gracefully if ExifTool or raw metadata is missing.
    """
    # Write bytes to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    thermal_matrix = None

    try:
        # Try flirimageextractor (requires exiftool system binary)
        from flirimageextractor import FlirImageExtractor
        flir = FlirImageExtractor()
        flir.process_image(tmp_path)
        thermal_matrix = flir.get_thermal_np()
    except Exception as e:
        # If exiftool fails or isn't installed, raise a clear user error
        raise RuntimeError(
            "ExifTool binary missing or image lacks raw radiometric metadata. "
            "Please ensure packages.txt includes 'exiftool' and the file is an original RJPG."
        ) from e
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if thermal_matrix is None or thermal_matrix.size == 0:
        raise ValueError("No valid radiometric matrix found in this file.")

    # Core equipment matrix (strip 5% border)
    h, w = thermal_matrix.shape
    core = thermal_matrix[int(h * 0.05):int(h * 0.95), int(w * 0.05):int(w * 0.95)]

    max_temp = float(np.max(core))
    baseline_temp = float(np.percentile(core, 90))
    min_temp = float(np.min(core))
    mean_temp = float(np.mean(core))

    temp_diff = float(max_temp - baseline_temp)

    status = "FAULT" if temp_diff >= threshold else "NO FAULT"
    action = "Attention required within 2 days." if status == "FAULT" else "Normal operation."

    return {
        "min_temperature": round(min_temp, 2),
        "max_temperature": round(max_temp, 2),
        "temperature_difference": round(temp_diff, 2),
        "mean_temperature": round(mean_temp, 2),
        "threshold": threshold,
        "status": status,
        "action": action
    }
