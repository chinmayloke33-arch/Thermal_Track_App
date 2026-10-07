import tempfile
from flirimageextractor import FlirImageExtractor
import numpy as np


def analyze_thermal_image(
    uploaded_file,
    threshold: float = 5.0
) -> dict:
    """
    Extracts raw per-pixel temperatures directly from FLIR radiometric metadata.
    """
    # Write uploaded stream to a temporary file for ExifTool processing
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    flir = FlirImageExtractor()
    flir.process_image(tmp_path)

    # Extract 2D array of actual temperatures in Celsius
    thermal_matrix = flir.get_thermal_np()

    if thermal_matrix is None or thermal_matrix.size == 0:
        raise ValueError(
            "No radiometric metadata found in this image. "
            "Please ensure the image was captured directly by a supported radiometric camera."
        )

    # Calculate exact temperatures
    min_temp = float(np.min(thermal_matrix))
    max_temp = float(np.max(thermal_matrix))
    mean_temp = float(np.mean(thermal_matrix))
    temp_diff = float(max_temp - min_temp)

    if temp_diff < threshold:
        status = "NO FAULT"
        action = "Operating within normal thermal parameters."
    else:
        status = "FAULT"
        action = "Attention required within 2 days."

    return {
        "min_temperature": round(min_temp, 2),
        "max_temperature": round(max_temp, 2),
        "temperature_difference": round(temp_diff, 2),
        "mean_temperature": round(mean_temp, 2),
        "threshold": threshold,
        "status": status,
        "action": action
    }
