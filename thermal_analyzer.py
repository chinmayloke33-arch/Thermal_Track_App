import cv2
import numpy as np


def analyze_thermal_image_radiometric(
    file_bytes: bytes,
    threshold: float = 5.0,
    known_cold_temp: float = 7.0,
    known_hot_temp: float = 40.0
) -> dict:
    """
    Analyzes standard thermal JPEGs without requiring ExifTool or radiometric metadata.
    Uses Otsu masking to strip background air/UI and calculates hotspot vs healthy conductor ΔT.
    """
    # 1. Decode image bytes into OpenCV format
    file_array = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to decode uploaded image file.")

    # 2. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Crop inner 80% to strip away camera side legends, text, and UI scale bars
    h, w = gray.shape
    roi = gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]

    # 4. Otsu Thresholding: Separate active equipment from cold background air/shadows
    _, object_mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    equipment_pixels = roi[object_mask > 0]

    if len(equipment_pixels) < 10:
        equipment_pixels = roi.flatten()

    # 5. Extract hotspot (top 0.5% hottest) vs carrying component baseline (20th percentile of active region)
    pixel_hotspot = np.percentile(equipment_pixels, 99.5)
    pixel_baseline = np.percentile(equipment_pixels, 20.0)
    pixel_mean = np.mean(equipment_pixels)

    # 6. Map pixel intensity directly to OEM Celsius bounds
    scale_range = known_hot_temp - known_cold_temp
    temp_per_pixel = scale_range / 255.0

    max_temp = float(known_cold_temp + (pixel_hotspot * temp_per_pixel))
    min_temp = float(known_cold_temp + (pixel_baseline * temp_per_pixel))
    mean_temp = float(known_cold_temp + (pixel_mean * temp_per_pixel))

    # Calculate target Delta T
    temp_diff = float(max_temp - min_temp)

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
