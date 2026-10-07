import numpy as np
from flirimageextractor import FlirImageExtractor
import tempfile
import os


def analyze_thermal_image_radiometric(
    file_bytes: bytes,
    threshold: float = 5.0
) -> dict:
    """
    Extracts raw per-pixel radiometric temperature matrix from FLIR JPG metadata.
    Does not rely on color scales, grayscale mapping, or image legends.
    """
    # Save uploaded bytes to a temporary file for flirimageextractor
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        flir = FlirImageExtractor()
        flir.process_image(tmp_path)
        
        # Extract true temperature matrix in Celsius
        thermal_matrix = flir.get_thermal_np()

        if thermal_matrix is None or thermal_matrix.size == 0:
            raise ValueError("No radiometric metadata found in this file.")

        # Ignore outer 5% border to prevent casing/frame interference
        h, w = thermal_matrix.shape
        core_matrix = thermal_matrix[int(h * 0.05):int(h * 0.95), int(w * 0.05):int(w * 0.95)]

        # Hotspot vs conductor baseline extraction
        max_temp = float(np.max(core_matrix))
        
        # Baseline = 90th percentile of the thermal matrix (active component region)
        baseline_temp = float(np.percentile(core_matrix, 90))
        min_temp = float(np.min(core_matrix))
        mean_temp = float(np.mean(core_matrix))

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

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
